#pragma once
// APOLLO-Mini INT8 直写：参数只有一份 int8 + per-row scale，无 fp32 master。
// 低秩状态与 ApolloMini 相同（P/M/V/R），更新走 apollo_mini_apply_i8（随机舍入）。

#include <cstdint>
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

    void 步骤();
    void 重置();
    std::uint32_t 行数() const { return rows_; }
    std::uint32_t 列数() const { return cols_; }
    std::uint32_t 低秩长度() const { return R_len_; }
    const std::vector<float>& 投影() const { return Phost_; }

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
    MoE直写布局 moe布局_;
    配置 配置_;
    vk张量 P_, M_, V_, R_, 归约_, 梯度范数_;
    float 上一缩放范数_ = 0.0f;
    std::vector<float> Phost_;
};
