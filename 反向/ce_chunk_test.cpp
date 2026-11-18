// 分块交叉熵对拍：GPU 结果 vs float64 CPU 参考。
// 覆盖：词表非 64 倍数、多行块、target 在不同词表块、-100 忽略行。
// 用法：ce_chunk_test.exe <spv目录>
#include "交叉熵分块.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vk张量系统;

namespace {

struct 案例 {
    std::uint32_t R;
    std::uint32_t V;
    const char* 名称;
    std::uint32_t 行块 = 64;
    std::uint32_t 词块 = 64;
};

void 跑一个案例(const 案例& 例) {
    const std::uint32_t R = 例.R, V = 例.V;
    std::mt19937 随机(20261118u + R * 131u + V);
    std::uniform_real_distribution<float> 分布(-4.0f, 4.0f);

    std::vector<float> 主机logits(static_cast<std::size_t>(R) * V);
    for (auto& x : 主机logits) x = 分布(随机);

    std::vector<std::int32_t> 主机targets(R);
    for (std::uint32_t r = 0; r < R; ++r) {
        // 留出忽略行、首块、末块、中间块的目标位置
        if (r % 7 == 3) 主机targets[r] = -100;
        else if (r % 5 == 0) 主机targets[r] = 0;
        else if (r % 5 == 1) 主机targets[r] = static_cast<std::int32_t>(V - 1);
        else 主机targets[r] = static_cast<std::int32_t>(随机() % V);
    }

    vk张量 logits(张量布局::连续({R, V}), 张量类型::浮点32, "ce.logits");
    vk张量 targets(张量布局::连续({R}), 张量类型::整数32, "ce.targets");
    logits.上传(主机logits.data(), 主机logits.size() * sizeof(float));
    targets.上传(主机targets.data(), 主机targets.size() * sizeof(std::int32_t));

    分块交叉熵 ce(R, V, "ce");
    ce.计算(logits, targets, 分块交叉熵配置{例.行块, 例.词块});

    std::vector<float> 主机loss, 主机dlogits;
    ce.下载损失(主机loss);
    ce.下载dlogits(主机dlogits);

    double loss误差 = 0.0, dlogits误差 = 0.0;
    for (std::uint32_t r = 0; r < R; ++r) {
        if (主机targets[r] < 0) {
            loss误差 = std::max(loss误差, std::fabs(static_cast<double>(主机loss[r])));
            for (std::uint32_t v = 0; v < V; ++v)
                dlogits误差 = std::max(dlogits误差,
                    std::fabs(static_cast<double>(主机dlogits[static_cast<std::size_t>(r) * V + v])));
            continue;
        }
        double 最大 = -1e300, 和 = 0.0;
        for (std::uint32_t v = 0; v < V; ++v)
            最大 = std::max(最大, static_cast<double>(主机logits[static_cast<std::size_t>(r) * V + v]));
        for (std::uint32_t v = 0; v < V; ++v)
            和 += std::exp(static_cast<double>(主机logits[static_cast<std::size_t>(r) * V + v]) - 最大);
        const double lse = 最大 + std::log(和);
        const double 参考loss = -(static_cast<double>(主机logits[static_cast<std::size_t>(r) * V + 主机targets[r]]) - lse);
        loss误差 = std::max(loss误差, std::fabs(static_cast<double>(主机loss[r]) - 参考loss));
        for (std::uint32_t v = 0; v < V; ++v) {
            const double 参考 = std::exp(static_cast<double>(主机logits[static_cast<std::size_t>(r) * V + v]) - lse)
                              - (v == static_cast<std::uint32_t>(主机targets[r]) ? 1.0 : 0.0);
            const double 实际 = 主机dlogits[static_cast<std::size_t>(r) * V + v];
            dlogits误差 = std::max(dlogits误差, std::fabs(实际 - 参考));
        }
    }

    const double loss容差 = 2e-3;     // fp32 logsumexp 在 V=200 级的误差
    const double dlogits容差 = 2e-5;
    const bool 通过 = loss误差 <= loss容差 && dlogits误差 <= dlogits容差;
    std::printf("%-28s R=%u V=%u 行块=%u 词块=%u  loss max|Δ|=%.3e  dlogits max|Δ|=%.3e  %s\n",
                例.名称, R, V, 例.行块, 例.词块, loss误差, dlogits误差,
                通过 ? "PASS" : "FAIL");
    if (!通过)
        throw std::runtime_error(std::string("案例失败: ") + 例.名称);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "用法: %s <spv目录>\n", argv[0]);
        return 2;
    }

    流水线调度器& 调度器 = 流水线调度器::获取();
    设备注册信息 注册;
    注册.应用名 = "ce_chunk_test";
    阶段配置 图形;
    阶段配置 计算{阶段配置::自动队列族, 0, 1};
    阶段配置 传输;
    try {
        调度器.启动(注册, {图形, 计算, 传输}, argv[1]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "启动失败: %s\n", e.what());
        return 1;
    }

    int 失败 = 0;
    try {
        跑一个案例({64, 64, "单块 V=64"});
        跑一个案例({7, 200, "非 64 倍数 V=200"});
        跑一个案例({130, 129, "多行块 R=130 V=129"});
        跑一个案例({64, 257, "目标跨末块 V=257"});
        跑一个案例({130, 129, "行块 32", 32, 64});
        跑一个案例({64, 257, "词块 32", 64, 32});
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        失败 = 1;
    }

    调度器.停止();
    if (失败 == 0) std::printf("== 分块交叉熵对拍 PASS ==\n");
    return 失败;
}
