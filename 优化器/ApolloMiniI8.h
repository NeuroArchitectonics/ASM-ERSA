#pragma once
// APOLLO-Mini INT8 直写：参数只有一份 int8 + per-row scale，无 fp32 master。
// 低秩状态与 ApolloMini 相同（P/M/V/R），更新走 apollo_mini_apply_i8（随机舍入）。

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "vk张量.hpp"

using namespace vk张量系统;

class ApolloMiniI8 {
public:
    struct 配置 {
        float 学习率 = 1e-3f;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = 1e-8f;
        float scale = 128.0f;
        std::uint32_t 投影更新间隔 = 200;
    };

    struct MoE直写布局 {
        std::uint32_t E = 0;
        std::uint32_t F = 0;
        std::uint32_t D = 0;
        std::uint32_t kind = 0;
    };

    ApolloMiniI8(vk张量& 权重i8, vk张量& 行尺度, vk张量& 梯度,
                 std::uint32_t rows, std::uint32_t cols, 配置 配置, std::uint32_t seed,
                 std::uint32_t G, bool 梯度bf16,
                 MoE直写布局 moe布局);

    void 步骤();        // = 派发归约 + 结算 + 派发应用（单参数）
    // 相位批量化：A 阶段一次性派发所有参数的 proj/stat/gnorm，B 阶段一次性派发 apply。
    void 派发归约();    // proj + stat + gnorm（只录制，不同步）
    void 结算();        // 下载归约结果 + NGL，得到 本步缩放_
    void 派发应用();    // 用 本步缩放_ 派发 apply（只录制）
    void 重置();
    std::uint32_t 行数() const { return rows_; }
    std::uint32_t 列数() const { return cols_; }
    std::uint32_t 低秩长度() const { return R_len_; }
    std::uint32_t 步数() const { return 步数_; }
    std::uint32_t 种子() const { return seed_; }
    std::uint32_t 模式() const { return mode_; }
    std::uint32_t 尺度组数() const { return G_; }
    float 上一缩放范数() const { return 上一缩放范数_; }
    const std::vector<float>& 投影() const { return Phost_; }
    void 导出状态(std::vector<float>& P, std::vector<float>& M,
                 std::vector<float>& V) const {
        P.resize(inner_);
        M.resize(R_len_);
        V.resize(R_len_);
        if (!P.empty()) P_.下载(P.data(), P.size() * sizeof(float));
        if (!M.empty()) M_.下载(M.data(), M.size() * sizeof(float));
        if (!V.empty()) V_.下载(V.data(), V.size() * sizeof(float));
    }
    void 导入状态(const std::vector<float>& P, const std::vector<float>& M,
                 const std::vector<float>& V, std::uint32_t 步数, float 上一缩放范数) {
        if (P.size() != inner_ || M.size() != R_len_ || V.size() != R_len_)
            throw std::runtime_error("APOLLO-Mini-I8 状态尺寸不匹配");
        if (!P.empty()) P_.上传(P.data(), P.size() * sizeof(float));
        if (!M.empty()) M_.上传(M.data(), M.size() * sizeof(float));
        if (!V.empty()) V_.上传(V.data(), V.size() * sizeof(float));
        Phost_ = P;
        步数_ = 步数;
        上一缩放范数_ = 上一缩放范数;
    }

private:
    void 刷新投影();

    vk张量* 权重_;
    vk张量* 尺度_;
    vk张量* 梯度_;
    std::uint32_t rows_, cols_;
    std::uint32_t R_len_ = 0;
    std::uint32_t inner_ = 0;
    std::uint32_t mode_ = 0;
    std::uint32_t 步数_ = 0;
    std::uint32_t seed_;
    std::uint32_t G_ = 1;
    bool 梯度bf16_ = false;
    float 本步缩放_ = 0.0f;   // 结算() 产出，派发应用() 使用
    MoE直写布局 moe布局_;
    配置 配置_;
    vk张量 P_, M_, V_, R_, 归约_, 梯度范数_;
    vk张量 行范数_;   // [R_len] 每行 Σg²（proj 写，normsum 归约）
    float 上一缩放范数_ = 0.0f;
    std::vector<float> Phost_;
};
