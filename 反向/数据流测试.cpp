// 异步流式 SFT 数据测试：真实 Belle 记录 + 真实分词器。
// 用法：sft_stream_test.exe [jsonl路径] [样本数]
#include "数据流.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <thread>

int main(int argc, char** argv) {
    sft数据::配置 配置;
    配置.数据路径 = (argc >= 2) ? argv[1] : ".cache/sft_zh_head.jsonl";
    配置.分词器目录 = "C:/分词器/分词器";
    配置.分词器DLL = "D:/智能体工作区/ASM-ERSA/.cache/tok_dll.dll";
    配置.最大序列长度 = 512;
    配置.最大样本数 = (argc >= 3) ? static_cast<std::uint64_t>(std::atoll(argv[2])) : 50;
    配置.分词线程数 = 2;
    配置.加系统提示 = true;

    try {
        sft数据::Sft数据流 流(配置);
        std::printf("分词器词表=%d  pad=%d  eos=%d\n",
                    流.词表大小(), 流.填充id(), 流.结束id());

        for (int i = 0; i < 3; ++i) {
            sft数据::样本 s;
            if (!流.取样本(s)) {
                std::printf("没有更多样本\n");
                break;
            }
            std::size_t 前缀长 = 0;
            while (前缀长 < s.labels.size() && s.labels[前缀长] == -100) ++前缀长;
            std::printf("样本%d 行=%llu 用户字节=%zu 回答字节=%zu 总token=%zu 前缀=%zu 回答token=%zu\n",
                        i + 1,
                        static_cast<unsigned long long>(s.行号),
                        s.用户.size(), s.回答.size(),
                        s.input_ids.size(), 前缀长,
                        s.input_ids.size() - 前缀长);
            std::printf("  首token前8=");
            for (std::size_t k = 0; k < std::min<std::size_t>(8, s.input_ids.size()); ++k)
                std::printf("%d ", s.input_ids[k]);
            std::printf("\n  监督首8=");
            for (std::size_t k = 前缀长; k < std::min(前缀长 + 8, s.labels.size()); ++k)
                std::printf("%d ", s.labels[k]);
            std::printf("\n  末token=%d（期望 eos=%d）\n",
                        s.input_ids.empty() ? -1 : s.input_ids.back(), 流.结束id());

            if (s.input_ids.size() != s.labels.size())
                throw std::runtime_error("input_ids 与 labels 长度不一致");
            if (前缀长 >= s.labels.size())
                throw std::runtime_error("没有监督 token");
            for (std::size_t k = 前缀长; k < s.labels.size(); ++k)
                if (s.labels[k] < 0)
                    throw std::runtime_error("回答段出现 -100");
        }

        sft数据::批次 批;
        if (流.取批次(批, 2, 256)) {
            std::printf("批次 B=%u T=%u 有效token=%u 首样本标签首8=",
                        批.批大小, 批.序列长度, 批.有效token数);
            for (std::size_t k = 0; k < 8; ++k) std::printf("%d ", 批.labels[k]);
            std::printf("\n");
        }

        sft数据::统计 终态 = 流.取统计();
        // 等后台把已解析的样本都处理完，吞吐数字才有意义
        for (int i = 0; i < 1000; ++i) {
            终态 = 流.取统计();
            if (终态.已解析 > 0 && 终态.已分词 + 终态.跳过 >= 终态.已解析) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        std::printf("统计：读行=%llu 解析=%llu 分词=%llu 跳过=%llu token=%llu 用时=%.3fs 速度=%.1f 样本/s\n",
                    static_cast<unsigned long long>(终态.已读行),
                    static_cast<unsigned long long>(终态.已解析),
                    static_cast<unsigned long long>(终态.已分词),
                    static_cast<unsigned long long>(终态.跳过),
                    static_cast<unsigned long long>(终态.总token),
                    终态.已用秒, 终态.已用秒 > 0 ? 终态.已分词 / 终态.已用秒 : 0.0);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "失败: %s\n", e.what());
        return 1;
    }
    std::printf("== 数据流测试完成 ==\n");
    return 0;
}
