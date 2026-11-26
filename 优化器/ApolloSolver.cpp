#include "ApolloSolver.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
struct 归约推送 {
    std::uint32_t 元素数;
    float beta;
    float c1;
    float c2;
    std::uint32_t init_b;
    std::uint32_t pad0, pad1, pad2;
};
static_assert(sizeof(归约推送) == 32, "apollo_reduce push");

struct 更新推送 {
    std::uint32_t 元素数;
    float alpha;
    float 学习率;
    float sigma;
    float b_max;
    float pad0, pad1, pad2;
};
static_assert(sizeof(更新推送) == 32, "apollo_update push");
}  // namespace

ApolloSolver::ApolloSolver(vk张量& 参数, vk张量& 梯度, std::size_t 元素数, 配置 配置)
    : 参数_(&参数),
      梯度_(&梯度),
      元素数_(元素数),
      配置_(配置),
      m_(张量布局::连续({static_cast<std::uint32_t>(元素数)}), 张量类型::浮点32, "apollo_m"),
      b_(张量布局::连续({static_cast<std::uint32_t>(元素数)}), 张量类型::浮点32, "apollo_b"),
      d_(张量布局::连续({static_cast<std::uint32_t>(元素数)}), 张量类型::浮点32, "apollo_d"),
      归约_(张量布局::连续({3u}), 张量类型::浮点32, "apollo_reduce") {
    if (元素数_ == 0) throw std::runtime_error("ApolloSolver: 元素数为 0");
    重置();
}

void ApolloSolver::重置() {
    std::vector<float> 零(元素数_, 0.0f);
    m_.上传(零.data(), 零.size() * sizeof(float));
    b_.上传(零.data(), 零.size() * sizeof(float));
    d_.上传(零.data(), 零.size() * sizeof(float));
    当前步数_ = 0;
}

bool ApolloSolver::参数有效() const {
    return 参数_ != nullptr && 梯度_ != nullptr &&
           参数_->有效() && 梯度_->有效();
}

void ApolloSolver::步骤() {
    if (!参数有效()) {
        throw std::runtime_error("ApolloSolver: 参数或梯度张量无效（可能已销毁）");
    }
    当前步数_++;
    const float t = static_cast<float>(当前步数_);
    const float beta_t = std::pow(配置_.beta, t);
    const float beta_t1 = std::pow(配置_.beta, t - 1.0f);
    const float c1 = 配置_.beta * (1.0f - beta_t1) / (1.0f - beta_t);
    const float c2 = (1.0f - 配置_.beta) / (1.0f - beta_t);

    const 归约推送 归约参数{
        static_cast<std::uint32_t>(元素数_), 配置_.beta, c1, c2,
        (当前步数_ == 1u ? 1u : 0u), 0u, 0u, 0u,
    };
    流水线调度器::获取().录制自定义派发算子(
        "apollo_reduce",
        {梯度_->取句柄(), m_.取句柄(), d_.取句柄(), b_.取句柄(), 归约_.取句柄()},
        &归约参数, sizeof(归约参数), 派发规格{1u, 1u, 1u});

    float 结果[3] = {0.0f, 0.0f, 0.0f};
    归约_.下载(结果, sizeof(结果));
    const float 范数4 = std::pow(std::max(结果[2], 0.0f), 0.25f);
    const float 分母 = std::pow(范数4 + 配置_.epsilon, 4.0f);
    const float alpha = (结果[0] + 结果[1]) / 分母;

    const std::uint32_t 组数 = static_cast<std::uint32_t>((元素数_ + 255) / 256);
    const 更新推送 更新参数{
        static_cast<std::uint32_t>(元素数_), alpha, 配置_.学习率, 配置_.sigma,
        配置_.b_max, 0.0f, 0.0f, 0.0f,
    };
    流水线调度器::获取().录制自定义派发算子(
        "apollo_update",
        {参数_->取句柄(), b_.取句柄(), m_.取句柄(), d_.取句柄()},
        &更新参数, sizeof(更新参数), 派发规格{组数, 1u, 1u});
}
