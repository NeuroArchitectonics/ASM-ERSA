#include <cmath>
#include <cstdio>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "流水线调度器.hpp"
#include "vk张量.hpp"
#include "adamw_update_fp16.spv.hpp"
#include "AdamWFp16.h"

using namespace vk张量系统;

int main(int argc, char** argv) {
    const std::string spv = argc > 1 ? argv[1] : "spv";
    流水线调度器& 调度器 = 流水线调度器::获取();
    设备注册信息 注册;
    注册.应用名 = "adamw_fp16_test";
    阶段配置 图形;
    阶段配置 计算{阶段配置::自动队列族, 0, 1};
    阶段配置 传输;
    调度器.启动(注册, {图形, 计算, 传输}, spv);
    std::fprintf(stderr, "[dbg] started\n");
    std::printf("设备: %s | 已加载算子: %zu\n", 调度器.获取设备().名称(), 调度器.算子数量());

    const std::size_t N = 1000;
    const int 步数 = 5;
    AdamWFp16::配置 配置;
    配置.学习率 = 1e-3f;
    配置.权重衰减 = 0.01f;

    std::mt19937 随机(1234);
    std::normal_distribution<float> 权重分布(0.0f, 1.0f);
    std::normal_distribution<float> 梯度分布(0.0f, 0.1f);

    std::vector<float> 初始(N);
    for (auto& x : 初始) x = 权重分布(随机);
    std::vector<std::vector<float>> 梯度(步数, std::vector<float>(N));
    for (auto& g : 梯度) for (auto& x : g) x = 梯度分布(随机);

    AdamWFp16 优化器(N, 配置);
    std::fprintf(stderr, "[dbg] opt created\n");
    优化器.上传参数(初始.data());
    for (int s = 0; s < 步数; ++s) {
        优化器.上传梯度(梯度[s].data());
        try { 优化器.步骤(); } catch (const std::exception& e) { std::fprintf(stderr, "[dbg] step err: %s\n", e.what()); return 1; }
        std::fprintf(stderr, "[dbg] step %d done\n", s);
    }
    std::vector<float> 结果(N);
    优化器.下载参数(结果.data());

    std::vector<double> 参考(初始.begin(), 初始.end()), m(N, 0.0), v(N, 0.0);
    const double b1 = 配置.beta1, b2 = 配置.beta2, lr = 配置.学习率,
                 eps = 配置.epsilon, wd = 配置.权重衰减;
    for (int s = 1; s <= 步数; ++s) {
        const double m_corr = 1.0 / (1.0 - std::pow(b1, s));
        const double v_corr = 1.0 / (1.0 - std::pow(b2, s));
        for (std::size_t i = 0; i < N; ++i) {
            const double g = 梯度[s - 1][i];
            m[i] = b1 * m[i] + (1.0 - b1) * g;
            v[i] = b2 * v[i] + (1.0 - b2) * g * g;
            参考[i] -= lr * (m[i] * m_corr) / (std::sqrt(v[i] * v_corr) + eps);
            参考[i] -= lr * wd * 参考[i];
        }
    }

    double 最大误差 = 0.0;
    for (std::size_t i = 0; i < N; ++i)
        最大误差 = std::max(最大误差, std::fabs((double)结果[i] - 参考[i]));
    std::printf("AdamW fp16 混合精度 N=%zu 步数=%d：max|Δ|=%.6e\n", N, 步数, 最大误差);
    std::printf("常驻：参数/梯度 fp16（4 B）+ master/m/v fp32（12 B）= 16 B/参数\n");
    return 0;
}
