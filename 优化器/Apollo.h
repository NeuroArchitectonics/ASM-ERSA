#pragma once
// Apollo（对角拟牛顿，arXiv:2009.13586）Vulkan 实现，fp32 状态
#include <cstddef>
#include <cstdint>

#include "vk张量.hpp"

using namespace vk张量系统;

class Apollo {
public:
    struct 配置 {
        float beta = 0.9f;
        float 学习率 = 1e-3f;
        float sigma = 1e-4f;
        float epsilon = 1e-8f;
    };

    explicit Apollo(std::size_t 元素数, 配置 配置 = {});

    void 上传参数(const float* 数据);
    void 上传梯度(const float* 数据);
    void 步骤();
    void 下载参数(float* 输出) const;
    void 下载状态(float* m_out, float* b_out, float* d_out) const;

    std::size_t 元素数() const { return 元素数_; }
    std::uint32_t 当前步数() const { return 当前步数_; }

private:
    std::size_t 元素数_;
    std::uint32_t 当前步数_ = 0;
    配置 配置_;
    vk张量 p_;
    vk张量 g_;
    vk张量 m_;
    vk张量 b_;
    vk张量 d_;
    vk张量 归约_;
};
