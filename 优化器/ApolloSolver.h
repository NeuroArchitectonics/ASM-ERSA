#pragma once
// Apollo 求解器：非拥有（non-owning）绑定外部参数/梯度张量，内部常驻 m/B/d 状态
// 生命周期约定：参数/梯度必须比 ApolloSolver 活得久；析构顺序请让 solver 先销毁。
#include <cstddef>
#include <cstdint>

#include "vk张量.hpp"

using namespace vk张量系统;

class ApolloSolver {
public:
    struct 配置 {
        float beta = 0.9f;
        float 学习率 = 1e-3f;
        float sigma = 1e-4f;
        float epsilon = 1e-8f;
        float b_max = 3.402823466e38f;   // |B| 上界；FLT_MAX 等价不启用
    };

    ApolloSolver(vk张量& 参数, vk张量& 梯度, std::size_t 元素数, 配置 配置 = {});

    void 步骤();
    void 设置学习率(float 学习率) { 配置_.学习率 = 学习率; }
    bool 参数有效() const;
    void 重置();

    std::size_t 元素数() const { return 元素数_; }

private:
    vk张量* 参数_;
    vk张量* 梯度_;
    std::size_t 元素数_;
    std::uint32_t 当前步数_ = 0;
    配置 配置_;
    vk张量 m_;
    vk张量 b_;
    vk张量 d_;
    vk张量 归约_;
};
