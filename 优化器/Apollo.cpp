#include "Apollo.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace {
struct 归约推送 {
    std::uint32_t 元素数;
    float beta;
    float c1;
    float c2;
};

struct 更新推送 {
    std::uint32_t 元素数;
    float alpha;
    float 学习率;
    float sigma;
};
}  // namespace

Apollo::Apollo(std::size_t 元素数, 配置 配置)
    : 元素数_(元素数),
      配置_(配置),
      p_(张量布局::连续({static_cast<std::uint32_t>(元素数)}), 张量类型::浮点32, "apollo_p"),
      g_(张量布局::连续({static_cast<std::uint32_t>(元素数)}), 张量类型::浮点32, "apollo_g"),
      m_(张量布局::连续({static_cast<std::uint32_t>(元素数)}), 张量类型::浮点32, "apollo_m"),
      b_(张量布局::连续({static_cast<std::uint32_t>(元素数)}), 张量类型::浮点32, "apollo_b"),
      d_(张量布局::连续({static_cast<std::uint32_t>(元素数)}), 张量类型::浮点32, "apollo_d"),
      归约_(张量布局::连续({3u}), 张量类型::浮点32, "apollo_reduce") {
    if (元素数_ == 0) throw std::runtime_error("Apollo: 元素数为 0");
    std::vector<float> 零(元素数_, 0.0f);
    m_.上传(零.data(), 零.size() * sizeof(float));
    b_.上传(零.data(), 零.size() * sizeof(float));
    d_.上传(零.data(), 零.size() * sizeof(float));
}

void Apollo::上传参数(const float* 数据) {
    p_.上传(数据, 元素数_ * sizeof(float));
}

void Apollo::上传梯度(const float* 数据) {
    g_.上传(数据, 元素数_ * sizeof(float));
}

void Apollo::步骤() {
    当前步数_++;
    const float t = static_cast<float>(当前步数_);
    const float beta_t = std::pow(配置_.beta, t);
    const float beta_t1 = std::pow(配置_.beta, t - 1.0f);
    const float c1 = 配置_.beta * (1.0f - beta_t1) / (1.0f - beta_t);
    const float c2 = (1.0f - 配置_.beta) / (1.0f - beta_t);

    const 归约推送 归约参数{
        static_cast<std::uint32_t>(元素数_), 配置_.beta, c1, c2,
    };
    流水线调度器::获取().录制自定义派发算子(
        "apollo_reduce",
        {g_.取句柄(), m_.取句柄(), d_.取句柄(), b_.取句柄(), 归约_.取句柄()},
        &归约参数, sizeof(归约参数), 派发规格{1u, 1u, 1u});

    float 归约结果[3] = {0.0f, 0.0f, 0.0f};
    归约_.下载(归约结果, sizeof(归约结果));
    const float 范数4 = std::pow(std::max(归约结果[2], 0.0f), 0.25f);
    const float 分母 = std::pow(范数4 + 配置_.epsilon, 4.0f);
    const float alpha = (归约结果[0] + 归约结果[1]) / 分母;

    const std::uint32_t 组数 = static_cast<std::uint32_t>((元素数_ + 255) / 256);
    const 更新推送 更新参数{
        static_cast<std::uint32_t>(元素数_), alpha, 配置_.学习率, 配置_.sigma,
    };
    流水线调度器::获取().录制自定义派发算子(
        "apollo_update",
        {p_.取句柄(), b_.取句柄(), m_.取句柄(), d_.取句柄()},
        &更新参数, sizeof(更新参数), 派发规格{组数, 1u, 1u});
}

void Apollo::下载参数(float* 输出) const {
    p_.下载(输出, 元素数_ * sizeof(float));
}

void Apollo::下载状态(float* m_out, float* b_out, float* d_out) const {
    m_.下载(m_out, 元素数_ * sizeof(float));
    b_.下载(b_out, 元素数_ * sizeof(float));
    d_.下载(d_out, 元素数_ * sizeof(float));
}
