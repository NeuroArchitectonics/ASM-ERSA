// APOLLO-Mini VK 对拍：随机矩阵 [rows, cols]，N 步，GPU vs fp64 参考
// 参考用与 GPU 相同的投影向量 P（ApolloMini::投影()）。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "流水线调度器.hpp"
#include "vk张量.hpp"
#include "apollo_mini_proj.spv.hpp"
#include "apollo_mini_stat.spv.hpp"
#include "apollo_mini_apply.spv.hpp"
#include "apollo_mini_gnorm.spv.hpp"
#include "ApolloMini.h"

using namespace vk张量系统;

int main(int argc, char** argv) {
    const std::string spv = argc > 1 ? argv[1] : "spv";
    流水线调度器& 调度器 = 流水线调度器::获取();
    设备注册信息 注册;
    注册.应用名 = "apollo_mini_test";
    阶段配置 图形;
    阶段配置 计算{阶段配置::自动队列族, 0, 1};
    阶段配置 传输;
    调度器.启动(注册, {图形, 计算, 传输}, spv);
    std::printf("设备: %s | 已加载算子: %zu\n", 调度器.获取设备().名称(), 调度器.算子数量());

    const std::uint32_t rows = 64, cols = 32;
    const int 步数 = 10;
    ApolloMini::配置 配置;
    配置.学习率 = 1e-2f;
    配置.beta1 = 0.9f;
    配置.beta2 = 0.999f;
    配置.eps = 1e-8f;
    配置.scale = 128.0f;
    配置.投影更新间隔 = 1000;   // 测试内不换投影

    std::mt19937 随机(2026);
    std::normal_distribution<float> 权重分布(0.0f, 1.0f);
    std::normal_distribution<float> 梯度分布(0.0f, 0.1f);
    std::vector<float> 初始(rows * cols);
    for (auto& v : 初始) v = 权重分布(随机);
    std::vector<std::vector<float>> 梯度(步数, std::vector<float>(rows * cols));
    for (auto& g : 梯度)
        for (auto& v : g) v = 梯度分布(随机);

    vk张量 参数(张量布局::连续({rows, cols}), 张量类型::浮点32, "mini_p");
    vk张量 梯度张量(张量布局::连续({rows, cols}), 张量类型::浮点32, "mini_g");
    参数.上传(初始.data(), 初始.size() * sizeof(float));

    ApolloMini 优化器(参数, 梯度张量, rows, cols, 配置, 77u);
    for (int s = 0; s < 步数; ++s) {
        梯度张量.上传(梯度[s].data(), 梯度[s].size() * sizeof(float));
        优化器.步骤();
    }
    std::vector<float> 结果(rows * cols);
    参数.下载(结果.data(), 结果.size() * sizeof(float));

    // fp64 参考（对齐官方 APOLLO-Mini：right/left 投影 + sqrt(scale) + Norm-Growth Limiter）
    const auto& P = 优化器.投影();
    const bool right = rows >= cols;
    const std::uint32_t R_len = right ? rows : cols;
    std::vector<double> p(初始.begin(), 初始.end());
    std::vector<double> Mv(R_len, 0.0), Vv(R_len, 0.0);
    double prev_norm = 0.0;
    for (int t = 1; t <= 步数; ++t) {
        const auto& g = 梯度[t - 1];
        std::vector<double> R(R_len, 0.0);
        if (right) {
            for (std::uint32_t i = 0; i < rows; ++i) {
                double acc = 0.0;
                for (std::uint32_t j = 0; j < cols; ++j)
                    acc += double(P[j]) * double(g[size_t(i) * cols + j]);
                R[i] = acc;
            }
        } else {
            for (std::uint32_t j = 0; j < cols; ++j) {
                double acc = 0.0;
                for (std::uint32_t i = 0; i < rows; ++i)
                    acc += double(P[i]) * double(g[size_t(i) * cols + j]);
                R[j] = acc;
            }
        }
        const double b1 = 配置.beta1, b2 = 配置.beta2;
        const double mc = 1.0 / (1.0 - std::pow(b1, t));
        const double vc = 1.0 / (1.0 - std::pow(b2, t));
        double a1 = 0.0, a2 = 0.0, g2 = 0.0;
        for (std::uint32_t i = 0; i < R_len; ++i) {
            Mv[i] = b1 * Mv[i] + (1.0 - b1) * R[i];
            Vv[i] = b2 * Vv[i] + (1.0 - b2) * R[i] * R[i];
            const double rt = (Mv[i] * mc) / (std::sqrt(Vv[i] * vc) + 配置.eps);
            a1 += R[i] * R[i];
            a2 += rt * rt;
        }
        for (std::size_t i = 0; i < p.size(); ++i) g2 += double(g[i]) * double(g[i]);
        double factor = a1 > 1e-20 ? std::sqrt(a2 / a1) : 0.0;
        double scaled = factor * std::sqrt(g2);
        if (prev_norm > 0.0) {
            const double limiter = std::max(scaled / (prev_norm + 1e-8), 1.01) / 1.01;
            factor /= limiter;
            scaled /= limiter;
        }
        prev_norm = scaled;
        const double s = factor * std::sqrt(double(配置.scale));
        for (std::size_t i = 0; i < p.size(); ++i) p[i] -= double(配置.学习率) * s * double(g[i]);
    }

    double 最大误差 = 0.0;
    std::size_t 最大下标 = 0;
    for (std::size_t i = 0; i < 结果.size(); ++i) {
        const double 差 = std::fabs(double(结果[i]) - p[i]);
        if (差 > 最大误差) { 最大误差 = 差; 最大下标 = i; }
    }
    std::printf("APOLLO-Mini [%u,%u] 步数=%d：max|Δ|=%.6e @i=%zu (gpu=%.6f ref=%.6f)\n",
                rows, cols, 步数, 最大误差, 最大下标, 结果[最大下标], float(p[最大下标]));
    return 最大误差 <= 3e-5 ? 0 : 1;   // fp32 10 步累积，相对误差 ~3e-6
}
