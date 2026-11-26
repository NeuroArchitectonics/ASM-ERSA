#include "ApolloMini.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>

namespace {
struct ProjPC {
    std::uint32_t out_len, inner, cols, mode;
};
struct StatPC {
    std::uint32_t n;
    float b1, b2, m_corr, v_corr, eps;
    std::uint32_t pad0, pad1;
};
struct ApplyPC {
    std::uint32_t n;
    float lr, s, pad0;
};
struct GNormPC {
    std::uint32_t n, pad0, pad1, pad2;
};
static_assert(sizeof(ProjPC) == 16, "apollo_mini_proj push");
static_assert(sizeof(StatPC) == 32, "apollo_mini_stat push");
static_assert(sizeof(ApplyPC) == 16, "apollo_mini_apply push");
static_assert(sizeof(GNormPC) == 16, "apollo_mini_gnorm push");
}  // namespace

ApolloMini::ApolloMini(vk张量& 参数, vk张量& 梯度, std::uint32_t rows, std::uint32_t cols,
                       配置 配置, std::uint32_t seed, bool 梯度bf16)
    : 参数_(&参数),
      梯度_(&梯度),
      rows_(rows),
      cols_(cols),
      seed_(seed),
      配置_(配置),
      梯度bf16_(梯度bf16) {
    if (rows_ == 0 || cols_ == 0) throw std::runtime_error("ApolloMini: 形状为 0");
    // 官方 proj_type='std'：投影到较小的一维（rows>=cols 用 right，否则 left）
    mode_ = (rows_ >= cols_) ? 0u : 1u;
    R_len_ = (mode_ == 0u) ? rows_ : cols_;
    inner_ = (mode_ == 0u) ? cols_ : rows_;
    P_ = vk张量(张量布局::连续({inner_}), 张量类型::浮点32, "apollo_mini_P");
    M_ = vk张量(张量布局::连续({R_len_}), 张量类型::浮点32, "apollo_mini_M");
    V_ = vk张量(张量布局::连续({R_len_}), 张量类型::浮点32, "apollo_mini_V");
    R_ = vk张量(张量布局::连续({R_len_}), 张量类型::浮点32, "apollo_mini_R");
    归约_ = vk张量(张量布局::连续({2u}), 张量类型::浮点32, "apollo_mini_out");
    行范数_ = vk张量(张量布局::连续({R_len_}), 张量类型::浮点32, "mini_row_nrm");
    梯度范数_ = vk张量(张量布局::连续({1u}), 张量类型::浮点32, "apollo_mini_gnorm");
    重置();
}

void ApolloMini::刷新投影() {
    std::mt19937 随机(seed_ + 步数_);
    // 官方 proj_type='std'：标准正态（不按 1/sqrt(rows) 缩放）。
    // 缩放会同时压小 R 的范数、把 s=||R~||/||R|| 放大到 1e5+，更新步长失控。
    std::normal_distribution<float> 分布(0.0f, 1.0f);
    Phost_.assign(inner_, 0.0f);
    for (auto& v : Phost_) v = 分布(随机);
    P_.上传(Phost_.data(), Phost_.size() * sizeof(float));
}

void ApolloMini::重置() {
    std::vector<float> 零(R_len_, 0.0f);
    M_.上传(零.data(), R_len_ * sizeof(float));
    V_.上传(零.data(), R_len_ * sizeof(float));
    R_.上传(零.data(), R_len_ * sizeof(float));
    步数_ = 0;
    上一缩放范数_ = 0.0f;
    刷新投影();
}

void ApolloMini::步骤() {
    ++步数_;
    if (配置_.投影更新间隔 > 0 && (步数_ % 配置_.投影更新间隔) == 0) 刷新投影();

    auto& 调度器 = 流水线调度器::获取();
    const float t = float(步数_);
    const float m_corr = 1.0f / (1.0f - std::pow(配置_.beta1, t));
    const float v_corr = 1.0f / (1.0f - std::pow(配置_.beta2, t));

    const ProjPC ppc{R_len_, inner_, cols_, mode_};
    const char* 后缀 = 梯度bf16_ ? "_bf16" : "";
    const std::string 名proj = std::string("apollo_mini_proj") + 后缀;
    const std::string 名gnorm = std::string("apollo_mini_gnorm") + 后缀;
    const std::string 名apply = std::string("apollo_mini_apply") + 后缀;
    调度器.录制自定义派发算子(
        名proj.c_str(),
        {P_.取句柄(), 梯度_->取句柄(), R_.取句柄(), 行范数_.取句柄()},
        &ppc, sizeof(ppc), 派发规格{R_len_, 1u, 1u});

    const StatPC spc{R_len_, 配置_.beta1, 配置_.beta2, m_corr, v_corr, 配置_.eps, 0u, 0u};
    调度器.录制自定义派发算子(
        "apollo_mini_stat",
        {R_.取句柄(), M_.取句柄(), V_.取句柄(), 归约_.取句柄()},
        &spc, sizeof(spc), 派发规格{1u, 1u, 1u});

    struct NormSumPC { std::uint32_t n, p0, p1, p2; };
    static_assert(sizeof(NormSumPC) == 16, "normsum push");
    const NormSumPC npc{R_len_, 0u, 0u, 0u};
    调度器.录制自定义派发算子(
        "apollo_mini_normsum",
        {行范数_.取句柄(), 梯度范数_.取句柄()},
        &npc, sizeof(npc), 派发规格{1u, 1u, 1u});

    float 范数[2] = {0.0f, 0.0f};
    归约_.下载(范数, sizeof(范数));
    float 梯度平方 = 0.0f;
    梯度范数_.下载(&梯度平方, sizeof(梯度平方));

    // 官方：scaled_grad = G·factor（factor=||R~||/||R||），先过 Norm-Growth Limiter，再乘 sqrt(scale)
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
    if (std::getenv("ASM_APOLLO_MINI_DEBUG"))
        std::printf("[mini] step=%u %ux%u mode=%u ||R||2=%.4g ||Rt||2=%.4g ||G||2=%.4g factor=%.4g s=%.4g\n",
                    步数_, rows_, cols_, mode_, std::sqrt(范数[0]), std::sqrt(范数[1]),
                    std::sqrt(std::max(梯度平方, 0.0f)), factor, s);

    const ApplyPC apc{rows_ * cols_, 配置_.学习率, s, 0.0f};
    调度器.录制自定义派发算子(
        名apply.c_str(),
        {参数_->取句柄(), 梯度_->取句柄()},
        &apc, sizeof(apc), 派发规格{(rows_ * cols_ + 255u) / 256u, 1u, 1u});
}
