// 真实词表分块 CE 冒烟：真实 SFT 样本的 next-token 目标 + 248k 词表。
// 目的不是看 loss 好不好，而是验证 64x64 分块在完整词表上不产生长 dispatch。
// 用法：real_vocab_ce_test.exe [jsonl路径]
#include "ce_chunk_host.hpp"
#include "sft_data_stream.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vk张量系统;

int main(int argc, char** argv) {
    std::string 数据路径 = (argc >= 2) ? argv[1] : ".cache/sft_zh_head.jsonl";

    sft数据::配置 数据配置;
    数据配置.数据路径 = 数据路径;
    数据配置.分词器目录 = "C:/分词器/分词器";
    数据配置.分词器DLL = "D:/智能体工作区/ASM-ERSA/.cache/tok_dll.dll";
    数据配置.最大序列长度 = 512;
    数据配置.最大样本数 = 32;
    数据配置.分词线程数 = 1;

    std::uint32_t R = 0, V = 0;
    std::vector<std::int32_t> targets;
    try {
        sft数据::Sft数据流 流(数据配置);
        sft数据::样本 样本;
        // 找一条答案段至少 65 token 的样本，凑满 64 行（短样本只覆盖单工作组）
        sft数据::样本 备选;
        std::size_t 备选答案长 = 0;
        bool 找到 = false;
        for (int 尝试 = 0; 尝试 < 32; ++尝试) {
            sft数据::样本 候选;
            if (!流.取样本(候选)) break;
            std::size_t 候选前缀 = 0;
            while (候选前缀 < 候选.labels.size() && 候选.labels[候选前缀] == -100)
                ++候选前缀;
            const std::size_t 候选答案长 = 候选.labels.size() - 候选前缀;
            if (候选答案长 >= 65) {
                样本 = std::move(候选);
                找到 = true;
                break;
            }
            if (候选答案长 > 备选答案长) {
                备选 = std::move(候选);
                备选答案长 = 候选答案长;
            }
        }
        if (!找到) {
            if (备选答案长 == 0) throw std::runtime_error("没有取到可用样本");
            样本 = std::move(备选);
        }
        V = static_cast<std::uint32_t>(流.词表大小());
        std::size_t 前缀长 = 0;
        while (前缀长 < 样本.labels.size() && 样本.labels[前缀长] == -100) ++前缀长;
        if (前缀长 + 1 >= 样本.labels.size())
            throw std::runtime_error("样本没有可监督的答案 token");
        // 从答案段前一位开始取 64 行，保证 target 落在真实 answer token 上
        const std::size_t 起点 = (前缀长 > 0) ? (前缀长 - 1) : 0;
        R = static_cast<std::uint32_t>(std::min<std::size_t>(64, 样本.labels.size() - 起点 - 1));
        if (R == 0) throw std::runtime_error("样本太短");
        targets.resize(R);
        std::uint32_t 有效目标 = 0;
        for (std::uint32_t r = 0; r < R; ++r) {
            const std::int32_t 目标 = 样本.labels[起点 + r + 1];
            targets[r] = (目标 >= 0) ? 目标 : -100;
            if (targets[r] >= 0) ++有效目标;
        }
        if (有效目标 == 0) throw std::runtime_error("答案段没有有效 target");
        std::printf("样本行=%llu 长度=%zu 前缀=%zu  R=%u V=%u 有效target=%u\n",
                    static_cast<unsigned long long>(样本.行号), 样本.input_ids.size(),
                    前缀长, R, V, 有效目标);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "数据侧失败: %s\n", e.what());
        return 1;
    }

    流水线调度器& 调度器 = 流水线调度器::获取();
    设备注册信息 注册;
    注册.应用名 = "real_vocab_ce_test";
    阶段配置 图形;
    阶段配置 计算{阶段配置::自动队列族, 0, 1};
    阶段配置 传输;
    try {
        调度器.启动(注册, {图形, 计算, 传输}, "D:/6F38~1/ASM-ERSA/spv");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "启动失败: %s\n", e.what());
        return 1;
    }

    int 返回 = 0;
    try {
        // 真实尺寸的 logits 用随机值填，只为验证分块 CE 的数值与派发上限。
        std::vector<float> 主机logits(static_cast<std::size_t>(R) * V);
        std::mt19937 随机(20261118u);
        std::uniform_real_distribution<float> 分布(-2.0f, 2.0f);
        for (auto& x : 主机logits) x = 分布(随机);

        vk张量 logits(张量布局::连续({R, V}), 张量类型::浮点32, "real.logits");
        vk张量 目标(张量布局::连续({R}), 张量类型::整数32, "real.targets");
        logits.上传(主机logits.data(), 主机logits.size() * sizeof(float));
        目标.上传(targets.data(), targets.size() * sizeof(std::int32_t));

        分块交叉熵 ce(R, V, "real.ce");
        const auto 开始 = std::chrono::steady_clock::now();
        ce.计算(logits, 目标, 分块交叉熵配置{64, 64});
        const float 平均loss = ce.取平均损失(targets);
        const double 秒 = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - 开始).count();

        std::vector<float> 主机dlogits;
        ce.下载dlogits(主机dlogits);
        double 最大行和 = 0.0;
        for (std::uint32_t r = 0; r < R; ++r) {
            if (targets[r] < 0) continue;
            double 和 = 0.0;
            for (std::uint32_t v = 0; v < V; ++v)
                和 += 主机dlogits[static_cast<std::size_t>(r) * V + v];
            最大行和 = std::max(最大行和, std::fabs(和));
        }

        std::printf("分块 CE: R=%u V=%u 平均loss=%.6f  随机基线ln(V)=%.6f\n",
                    R, V, 平均loss, std::log(static_cast<double>(V)));
        std::printf("dlogits 行和 max|Σ|=%.3e（softmax-onehot 应接近 0）  用时=%.3fs\n",
                    最大行和, 秒);
        if (!std::isfinite(平均loss) || 最大行和 > 2e-3) {
            std::fprintf(stderr, "真实词表 CE 冒烟 FAIL\n");
            返回 = 1;
        } else {
            std::printf("== 真实词表分块 CE PASS ==\n");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "CE 失败: %s\n", e.what());
        返回 = 1;
    }

    调度器.停止();
    return 返回;
}
