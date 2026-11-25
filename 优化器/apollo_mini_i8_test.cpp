// APOLLO-Mini INT8 直写对拍 + 随机舍入累积测试
// 1) 确定性模式（pad0=0）逐位对拍 fp64 参考
// 2) 随机模式（pad0=1）用同一 hash 逐位对拍
// 3) 小更新累积：round 会冻结、随机舍入会累积（训练不冻结的动机验证）
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "流水线调度器.hpp"
#include "vk张量.hpp"
#include "apollo_mini_apply_i8.spv.hpp"

using namespace vk张量系统;

struct 推送 {
    std::uint32_t rows, cols, G, seed;
    float lr, s;
    std::uint32_t mode, pad;
};
static_assert(sizeof(推送) == 32, "apollo_mini_apply_i8 push");

static std::uint32_t 哈希(std::uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}
static float 随机01(std::uint32_t idx, std::uint32_t seed) {
    return float(哈希(idx ^ (seed * 0x9E3779B9u)) & 0xFFFFFFu) / 16777216.0f;
}

static void 派发(vk张量& w, vk张量& sc, vk张量& g, 推送 pc, std::uint32_t n) {
    流水线调度器::获取().录制自定义派发算子(
        "apollo_mini_apply_i8", {w.取句柄(), sc.取句柄(), g.取句柄()},
        &pc, sizeof(pc), 派发规格{(n + 255u) / 256u, 1u, 1u});
}

int main(int argc, char** argv) {
    const std::string spv = argc > 1 ? argv[1] : "spv";
    流水线调度器& 调度器 = 流水线调度器::获取();
    设备注册信息 注册;
    注册.应用名 = "apollo_mini_i8_test";
    阶段配置 图形;
    阶段配置 计算{阶段配置::自动队列族, 0, 1};
    阶段配置 传输;
    调度器.启动(注册, {图形, 计算, 传输}, spv);
    std::printf("设备: %s | 已加载算子: %zu\n", 调度器.获取设备().名称(), 调度器.算子数量());

    const std::uint32_t rows = 64, cols = 32, n = rows * cols;
    const float lr = 1e-2f;
    const float s = 1.0f;

    std::mt19937 随机(2026);
    std::uniform_int_distribution<int> 权重分布(-127, 127);
    std::uniform_real_distribution<float> 梯度分布(-1e-3f, 1e-3f);
    std::vector<std::int8_t> w0(n);
    std::vector<float> sc(rows), g(n);
    for (std::uint32_t r = 0; r < rows; ++r) {
        int 最大 = 0;
        for (std::uint32_t c = 0; c < cols; ++c) {
            w0[r * cols + c] = static_cast<std::int8_t>(权重分布(随机));
            最大 = std::max(最大, std::abs(int(w0[r * cols + c])));
        }
        sc[r] = 最大 > 0 ? float(最大) / 127.0f : 1.0f;
    }
    for (auto& v : g) v = 梯度分布(随机);

    vk张量 W(张量布局::连续({n}), 张量类型::整数8, "i8_W");
    vk张量 SC(张量布局::连续({rows}), 张量类型::浮点32, "i8_sc");
    vk张量 G(张量布局::连续({n}), 张量类型::浮点32, "i8_g");
    W.上传(w0.data(), n);
    SC.上传(sc.data(), rows * sizeof(float));
    G.上传(g.data(), n * sizeof(float));

    // 参考：wf = w*sc - lr*s*g；q = floor(wf/sc + u)，u=0.5 或随机
    auto 参考 = [&](std::uint32_t seed, bool 随机舍入) {
        std::vector<std::int8_t> out(n);
        for (std::uint32_t i = 0; i < n; ++i) {
            const std::uint32_t r = i / cols;
            const double wf = double(w0[i]) * double(sc[r]) - double(lr) * double(s) * double(g[i]);
            const double u = 随机舍入 ? double(随机01(i, seed)) : 0.5;
            double q = std::floor(wf / double(sc[r]) + u);
            q = std::min(127.0, std::max(-127.0, q));
            out[i] = static_cast<std::int8_t>(q);
        }
        return out;
    };

    std::vector<std::int8_t> got(n);
    // 1) 确定性模式
    派发(W, SC, G, 推送{rows, cols, 1u, 0u, lr, s, 0u, 0u}, n);
    W.下载(got.data(), n);
    auto ref_det = 参考(0, false);
    int 不一致 = 0;
    for (std::uint32_t i = 0; i < n; ++i) 不一致 += got[i] != ref_det[i];
    std::printf("确定性模式：不一致 %d / %u\n", 不一致, n);

    // 2) 随机模式（同一 seed，逐位对拍）
    W.上传(w0.data(), n);
    派发(W, SC, G, 推送{rows, cols, 1u, 12345u, lr, s, 1u, 0u}, n);
    W.下载(got.data(), n);
    auto ref_rnd = 参考(12345u, true);
    不一致 = 0;
    for (std::uint32_t i = 0; i < n; ++i) 不一致 += got[i] != ref_rnd[i];
    std::printf("随机模式：不一致 %d / %u\n", 不一致, n);
    const bool 对拍通过 = (不一致 == 0);

    // 3) 小更新累积：w=0、sc=1、每步更新 0.5 个量化步
    std::vector<std::int8_t> 零(n, 0);
    std::vector<float> 单位(rows, 1.0f), 半步(n, 0.5f);
    W.上传(零.data(), n);
    SC.上传(单位.data(), rows * sizeof(float));
    G.上传(半步.data(), n * sizeof(float));
    for (int t = 0; t < 200; ++t)
        派发(W, SC, G, 推送{rows, cols, 1u, std::uint32_t(1000 + t), 1.0f, 1.0f, 1u, 0u}, n);
    W.下载(got.data(), n);
    double 随机均值 = 0.0;
    for (auto v : got) 随机均值 += v;
    随机均值 /= n;

    W.上传(零.data(), n);
    for (int t = 0; t < 200; ++t)
        派发(W, SC, G, 推送{rows, cols, 1u, 0u, 1.0f, 1.0f, 0u, 0u}, n);
    W.下载(got.data(), n);
    double 确定均值 = 0.0;
    for (auto v : got) 确定均值 += v;
    确定均值 /= n;

    std::printf("小更新 200 步（每步 0.5 步长）：round 均值=%.1f（应 0，冻结）；随机舍入均值=%.1f（应 -100）\n",
                确定均值, 随机均值);
    const bool 累积通过 = std::abs(确定均值) < 0.01 && std::abs(随机均值 + 100.0) < 3.0;

    if (对拍通过 && 累积通过) {
        std::printf("APOLLO-Mini INT8 PASS\n");
        return 0;
    }
    std::printf("APOLLO-Mini INT8 FAIL\n");
    return 1;
}
