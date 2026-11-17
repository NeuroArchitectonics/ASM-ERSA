#include <algorithm>
#include <cmath>
#include <string>
#include <cstdio>
#include <random>
#include <vector>

#include "流水线调度器.hpp"
#include "vk张量.hpp"
#include "apollo_reduce.spv.hpp"
#include "apollo_update.spv.hpp"
#include "Apollo.h"

using namespace vk张量系统;

int main(int argc, char** argv) {
    const std::string spv = argc > 1 ? argv[1] : "spv";
    流水线调度器& 调度器 = 流水线调度器::获取();
    设备注册信息 注册;
    注册.应用名 = "apollo_test";
    阶段配置 图形;
    阶段配置 计算{阶段配置::自动队列族, 0, 1};
    阶段配置 传输;
    调度器.启动(注册, {图形, 计算, 传输}, spv);
    std::printf("设备: %s | 已加载算子: %zu\n", 调度器.获取设备().名称(), 调度器.算子数量());

    const std::size_t N = 1000;
    const int 步数 = 5;
    Apollo::配置 配置;
    配置.学习率 = 1e-3f;
    配置.sigma = 1e-4f;
    配置.epsilon = 1e-8f;

    std::mt19937 随机(1234);
    std::normal_distribution<float> 权重分布(0.0f, 1.0f);
    std::normal_distribution<float> 梯度分布(0.0f, 0.1f);

    std::vector<float> 初始(N);
    for (auto& x : 初始) x = 权重分布(随机);
    std::vector<std::vector<float>> 梯度(步数, std::vector<float>(N));
    for (auto& g : 梯度) for (auto& x : g) x = 梯度分布(随机);

    Apollo 优化器(N, 配置);
    优化器.上传参数(初始.data());
    for (int s = 0; s < 步数; ++s) {
        优化器.上传梯度(梯度[s].data());
        优化器.步骤();
    }
    std::vector<float> 结果(N);
    优化器.下载参数(结果.data());

    // fp64 参考
    std::vector<double> p(初始.begin(), 初始.end()), m(N, 0.0), b(N, 0.0), d(N, 0.0);
    for (int s = 1; s <= 步数; ++s) {
        const double beta = 配置.beta;
        const double c1 = beta * (1.0 - std::pow(beta, s - 1.0)) / (1.0 - std::pow(beta, s));
        const double c2 = (1.0 - beta) / (1.0 - std::pow(beta, s));
        std::vector<double> mn(N);
        double s1 = 0, s2 = 0, s3 = 0;
        for (std::size_t i = 0; i < N; ++i) {
            mn[i] = c1 * m[i] + c2 * 梯度[s - 1][i];
            const double delta = mn[i] - m[i];
            s1 += d[i] * delta;
            s2 += d[i] * d[i] * b[i];
            s3 += d[i] * d[i] * d[i] * d[i];
        }
        const double norm4 = std::pow(std::max(s3, 0.0), 0.25);
        const double alpha = (s1 + s2) / std::pow(norm4 + 配置.epsilon, 4.0);
        for (std::size_t i = 0; i < N; ++i) {
            b[i] -= alpha * d[i] * d[i];
            const double D = std::max(std::fabs(b[i]), (double)配置.sigma);
            d[i] = mn[i] / D;
            p[i] -= 配置.学习率 * d[i];
            m[i] = mn[i];
        }
    }

    double 最大误差 = 0.0;
    std::size_t 最大下标 = 0;
    for (std::size_t i = 0; i < N; ++i) {
        const double 差 = std::fabs((double)结果[i] - p[i]);
        if (差 > 最大误差) { 最大误差 = 差; 最大下标 = i; }
    }
    std::printf("Apollo N=%zu 步数=%d：max|Δ|=%.6e @i=%zu (gpu=%.6f ref=%.6f)\n",
                N, 步数, 最大误差, 最大下标, 结果[最大下标], (float)p[最大下标]);
    return 0;
}
