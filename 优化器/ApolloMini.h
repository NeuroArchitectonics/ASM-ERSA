#pragma once
// APOLLO-Mini（低秩辅助梯度缩放，MLSys'25）VK 实现：
// 每个参数矩阵 [rows, cols] 只用 rank-1 随机投影状态：
//   R = PᵀG              (cols)
//   M_R, V_R  ← 低秩动量/二阶矩 (cols)
//   s = ||M_R/(sqrt(V_R)+eps)|| / ||R|| * scale
//   W -= lr · s · G
// 优化器状态约 2·cols + rows 个 float，与参数量相比可忽略（SGD 级）。
#include <cstddef>
#include <cstdint>
#include <vector>

#include "vk张量.hpp"

using namespace vk张量系统;

class ApolloMini {
public:
    struct 配置 {
        float 学习率 = 1e-3f;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = 1e-8f;
        float scale = 128.0f;   // rank-1 官方推荐 128
        std::uint32_t 投影更新间隔 = 200;   // 每 N 步换一次随机投影
    };

    // 参数/梯度是同一形状 [rows, cols] 的 fp32 张量（非拥有）
    ApolloMini(vk张量& 参数, vk张量& 梯度, std::uint32_t rows, std::uint32_t cols,
               配置 配置 = {}, std::uint32_t seed = 1234u, bool 梯度bf16 = false);

    void 步骤();
    void 重置();
    std::uint32_t 行数() const { return rows_; }
    std::uint32_t 列数() const { return cols_; }
    std::uint32_t 低秩长度() const { return R_len_; }
    const std::vector<float>& 投影() const { return Phost_; }

private:
    void 刷新投影();

    vk张量* 参数_;
    vk张量* 梯度_;
    std::uint32_t rows_, cols_;
    std::uint32_t R_len_ = 0;   // 低秩状态长度 = min(rows, cols)
    std::uint32_t inner_ = 0;   // 投影向量长度 = max(rows, cols)
    std::uint32_t mode_ = 0;    // 0 = right（rows>=cols），1 = left
    std::uint32_t 步数_ = 0;
    std::uint32_t seed_;
    配置 配置_;
    vk张量 P_;    // [inner_]
    vk张量 M_;    // [R_len_]
    vk张量 V_;    // [R_len_]
    vk张量 R_;    // [R_len_]
    vk张量 归约_; // [2]
    vk张量 梯度范数_; // [1]
    float 上一缩放范数_ = 0.0f;   // Norm-Growth Limiter 的上一步 ||scaled_grad||
    std::vector<float> Phost_;
    bool 梯度bf16_ = false;   // 梯度缓冲为 bf16（uint16 位模式）
};
