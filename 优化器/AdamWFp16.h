#pragma once
// AdamW 混合精度：参数/梯度 fp16，master 权重与 m/v 为 fp32
#include <cstddef>
#include <cstdint>

#include "vk张量.hpp"

using namespace vk张量系统;

class AdamWFp16 {
public:
    struct 配置 {
        float 学习率 = 3e-4f;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float epsilon = 1e-8f;
        float 权重衰减 = 0.1f;
        bool 启用偏差修正 = true;
        bool 更新后清零梯度 = true;
    };

    explicit AdamWFp16(std::size_t 元素数, 配置 配置 = {});

    void 上传参数(const float* 数据);
    void 上传梯度(const float* 数据);
    void 步骤();
    void 下载参数(float* 输出) const;
    void 下载状态(float* m_out, float* v_out) const;

    std::size_t 元素数() const { return 元素数_; }

private:
    std::size_t 元素数_;
    std::uint32_t 当前步数_ = 0;
    配置 配置_;
    vk张量 p_;       // fp16 参数
    vk张量 g_;       // fp16 梯度
    vk张量 master_;  // fp32 主权重
    vk张量 m_;
    vk张量 v_;
};
