#include "ApolloMiniI8.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>

namespace {
struct ProjPC { std::uint32_t out_len, inner, cols, mode; };
struct StatPC {
    std::uint32_t n;
    float b1, b2, m_corr, v_corr, eps;
    std::uint32_t pad0, pad1;
};
struct GNormPC { std::uint32_t n, pad0, pad1, pad2; };
struct ApplyPC { std::uint32_t rows, cols, seed, mode; float lr, s, pad0, pad1; };
struct MoeApplyPC {
    std::uint32_t E, F, D, kind;
    float lr, s;
    std::uint32_t seed, mode;
};
static_assert(sizeof(ProjPC) == 16, "proj push");
static_assert(sizeof(StatPC) == 32, "stat push");
static_assert(sizeof(GNormPC) == 16, "gnorm push");
static_assert(sizeof(ApplyPC) == 32, "apply_i8 push");
static_assert(sizeof(MoeApplyPC) == 32, "apply_moe_i8 push");
}  // namespace

ApolloMiniI8::ApolloMiniI8(vk张量& 权重i8, vk张量& 行尺度, vk张量& 梯度,
                           std::uint32_t rows, std::uint32_t cols, 配置 配置, std::uint32_t seed,
                           std::uint32_t G, bool 梯度bf16, MoE直写布局 moe布局)
    : 权重_(&权重i8), 尺度_(&行尺度), 梯度_(&梯度),
      rows_(rows), cols_(cols), seed_(seed), G_(G), 梯度bf16_(梯度bf16),
      moe布局_(moe布局), 配置_(配置) {
    if (rows_ == 0 || cols_ == 0) throw std::runtime_error("ApolloMiniI8: 形状为 0");
    if (cols_ % 4u != 0u) throw std::runtime_error("ApolloMiniI8: 线性 apply 要求 cols 为 4 的倍数");
    if (moe布局_.kind != 0u) {
        if (moe布局_.kind > 2u || moe布局_.E == 0u || moe布局_.F == 0u || moe布局_.D == 0u)
            throw std::runtime_error("ApolloMiniI8: MoE 布局参数无效");
        const std::uint32_t 期望rows = moe布局_.kind == 1u ? moe布局_.E * moe布局_.D
                                                          : moe布局_.E * moe布局_.F;
        const std::uint32_t 期望cols = moe布局_.kind == 1u ? moe布局_.F
                                                          : moe布局_.D;
        const std::uint32_t 期望G = moe布局_.kind == 1u ? (moe布局_.D + 127u) / 128u
                                                        : (moe布局_.F + 127u) / 128u;
        if (rows_ != 期望rows || cols_ != 期望cols || G_ != 期望G)
            throw std::runtime_error("ApolloMiniI8: MoE 布局与梯度/scale 形状不匹配");
    }
    mode_ = (rows_ >= cols_) ? 0u : 1u;
    R_len_ = (mode_ == 0u) ? rows_ : cols_;
    inner_ = (mode_ == 0u) ? cols_ : rows_;
    P_ = vk张量(张量布局::连续({inner_}), 张量类型::浮点32, "mini_i8_P");
    M_ = vk张量(张量布局::连续({R_len_}), 张量类型::浮点32, "mini_i8_M");
    V_ = vk张量(张量布局::连续({R_len_}), 张量类型::浮点32, "mini_i8_V");
    R_ = vk张量(张量布局::连续({R_len_}), 张量类型::浮点32, "mini_i8_R");
    归约_ = vk张量(张量布局::连续({2u}), 张量类型::浮点32, "mini_i8_out");
    行范数_ = vk张量(张量布局::连续({R_len_}), 张量类型::浮点32, "mini_row_nrm");
    梯度范数_ = vk张量(张量布局::连续({1u}), 张量类型::浮点32, "mini_i8_gnorm");
    重置();
}

void ApolloMiniI8::刷新投影() {
    std::mt19937 随机(seed_ + 步数_);
    std::normal_distribution<float> 分布(0.0f, 1.0f);
    Phost_.assign(inner_, 0.0f);
    for (auto& v : Phost_) v = 分布(随机);
    P_.上传(Phost_.data(), Phost_.size() * sizeof(float));
}

void ApolloMiniI8::重置() {
    std::vector<float> 零(R_len_, 0.0f);
    M_.上传(零.data(), R_len_ * sizeof(float));
    V_.上传(零.data(), R_len_ * sizeof(float));
    R_.上传(零.data(), R_len_ * sizeof(float));
    步数_ = 0;
    上一缩放范数_ = 0.0f;
    刷新投影();
}

void ApolloMiniI8::派发归约() {
    ++步数_;
    if (配置_.投影更新间隔 > 0 && (步数_ % 配置_.投影更新间隔) == 0) 刷新投影();

    auto& 调度器 = 流水线调度器::获取();
    const char* 后缀 = 梯度bf16_ ? "_bf16" : "";
    const std::string 名proj = std::string("apollo_mini_proj") + 后缀;
    const std::string 名gnorm = std::string("apollo_mini_gnorm") + 后缀;
    const float t = float(步数_);
    const float m_corr = 1.0f / (1.0f - std::pow(配置_.beta1, t));
    const float v_corr = 1.0f / (1.0f - std::pow(配置_.beta2, t));

    const ProjPC ppc{R_len_, inner_, cols_, mode_};
    调度器.录制自定义派发算子(
        名proj.c_str(), {P_.取句柄(), 梯度_->取句柄(), R_.取句柄(), 行范数_.取句柄()},
        &ppc, sizeof(ppc), 派发规格{R_len_, 1u, 1u});

    struct NormSumPC { std::uint32_t n, p0, p1, p2; };
    static_assert(sizeof(NormSumPC) == 16, "normsum push");
    const NormSumPC npc{R_len_, 0u, 0u, 0u};
    调度器.录制自定义派发算子(
        "apollo_mini_normsum", {行范数_.取句柄(), 梯度范数_.取句柄()},
        &npc, sizeof(npc), 派发规格{1u, 1u, 1u});

    const StatPC spc{R_len_, 配置_.beta1, 配置_.beta2, m_corr, v_corr, 配置_.eps, 0u, 0u};
    调度器.录制自定义派发算子(
        "apollo_mini_stat", {R_.取句柄(), M_.取句柄(), V_.取句柄(), 归约_.取句柄()},
        &spc, sizeof(spc), 派发规格{1u, 1u, 1u});


}

void ApolloMiniI8::结算() {
    auto& 调度器 = 流水线调度器::获取();
    (void)调度器;
    float 范数[2] = {0.0f, 0.0f};
    归约_.下载(范数, sizeof(范数));
    float 梯度平方 = 0.0f;
    梯度范数_.下载(&梯度平方, sizeof(梯度平方));

    float factor = (范数[0] > 1e-20f)
                       ? std::sqrt(std::max(范数[1], 0.0f) / 范数[0])
                       : 0.0f;
    float 缩放范数 = factor * std::sqrt(std::max(梯度平方, 0.0f));
    if (上一缩放范数_ > 0.0f) {
        const float limiter = std::max(缩放范数 / (上一缩放范数_ + 1e-8f), 1.01f) / 1.01f;
        factor /= limiter;
        缩放范数 /= limiter;
    }
    上一缩放范数_ = 缩放范数;

    const float s = std::min(factor * std::sqrt(std::max(配置_.scale, 0.0f)), 1.0e3f);
    本步缩放_ = s;
    if (std::getenv("ASM_APOLLO_MINI_DEBUG"))
        std::printf("[mini_i8] step=%u %ux%u mode=%u layout=%u s=%.4g\n",
                    步数_, rows_, cols_, mode_, moe布局_.kind, s);

}

void ApolloMiniI8::派发应用() {
    auto& 调度器 = 流水线调度器::获取();
    const char* 后缀 = 梯度bf16_ ? "_bf16" : "";
    const std::string 名apply = std::string("apollo_mini_apply_i8") + 后缀;
    const float s = 本步缩放_;
    if (moe布局_.kind != 0u) {
        const std::string 名apply = std::string("apollo_mini_apply_moe_i8") +
                                    (梯度bf16_ ? "_bf16" : "");
        const MoeApplyPC apc{moe布局_.E, moe布局_.F, moe布局_.D, moe布局_.kind,
                             配置_.学习率, 本步缩放_, 步数_, 1u};
        调度器.录制自定义派发算子(
            名apply.c_str(), {权重_->取句柄(), 尺度_->取句柄(), 梯度_->取句柄()},
            &apc, sizeof(apc), 派发规格{(rows_ * cols_ + 255u) / 256u, 1u, 1u});
        return;
    }

    const ApplyPC apc{rows_, cols_, G_, 步数_, 配置_.学习率, 本步缩放_, 1u, 0u};
    // 内核每线程处理 4 个 int8（col 必须是 4 的倍数，构造时已校验）
    const std::uint32_t 线程数 = (rows_ * cols_ + 3u) / 4u;
    调度器.录制自定义派发算子(
        名apply.c_str(), {权重_->取句柄(), 尺度_->取句柄(), 梯度_->取句柄()},
        &apc, sizeof(apc), 派发规格{(线程数 + 255u) / 256u, 1u, 1u});
}

void ApolloMiniI8::步骤() {
    派发归约();
    结算();
    派发应用();
}
