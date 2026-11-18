// SFT 数据流 + M0 VK 训练图：embedding → linear → rmsnorm → linear → 分块 CE
// 数据：反向/数据流.hpp（异步读 JSONL + 分词线程 + 有界队列）
// CE：反向/交叉熵分块.hpp（真实词表、-100 忽略、64x64 分块）
// 用法：train_sft_m0.exe <spv目录> [jsonl路径] [steps]
#include "流水线调度器.hpp"
#include "vk张量.hpp"
#include "embedding_fwd.spv.hpp"
#include "linear_fwd.spv.hpp"
#include "rmsnorm_fwd.spv.hpp"
#include "linear_bwd_dx.spv.hpp"
#include "linear_bwd_dwdb.spv.hpp"
#include "rmsnorm_bwd_dx.spv.hpp"
#include "rmsnorm_bwd_dw.spv.hpp"
#include "embedding_bwd.spv.hpp"
#include "adamw_update.spv.hpp"
#include "交叉熵分块.hpp"
#include "数据流.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vk张量系统;

namespace {

struct 规格4 { std::uint32_t a, b, c, d; };
struct 规格F { std::uint32_t a, b; float eps; float pad; };
struct 规格Adam {
    float lr, beta1, beta2, eps, m_corr, v_corr;
    std::uint32_t size;
    float wd;
    std::uint32_t zero_grad;
};
static_assert(sizeof(规格Adam) == 36, "adamw_update 推送常量必须 36 字节");

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "用法: %s <spv目录> [jsonl] [steps]\n", argv[0]);
        return 2;
    }
    const std::string spv目录 = argv[1];
    const std::string 数据路径 = (argc >= 3) ? argv[2] : ".cache/sft_110k_head100.jsonl";
    const int 总步数 = (argc >= 4) ? std::atoi(argv[3]) : 100;

    const std::uint32_t B = 1;
    const std::uint32_t T = 128;
    const std::uint32_t N = B * T;
    const std::uint32_t D = 32;
    const std::uint32_t Hx = 32;
    const float lr = 3e-4f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f, wd = 0.01f;

    sft数据::配置 数据配置;
    数据配置.数据路径 = 数据路径;
    数据配置.分词器目录 = "C:/分词器/分词器";
    数据配置.分词器DLL = "D:/智能体工作区/ASM-ERSA/.cache/tok_dll.dll";
    数据配置.最大序列长度 = T;
    数据配置.批大小 = B;
    数据配置.分词线程数 = 2;
    数据配置.最大样本数 = 0;
    数据配置.加系统提示 = true;

    sft数据::Sft数据流 流(数据配置);
    const std::uint32_t V = static_cast<std::uint32_t>(流.词表大小());
    std::printf("SFT 数据流: V=%u pad=%d eos=%d | T=%u B=%u N=%u\n",
                V, 流.填充id(), 流.结束id(), T, B, N);

    流水线调度器& 调度器 = 流水线调度器::获取();
    设备注册信息 注册; 注册.应用名 = "train_sft_m0";
    阶段配置 图形; 阶段配置 计算{阶段配置::自动队列族, 0, 1}; 阶段配置 传输;
    try { 调度器.启动(注册, {图形, 计算, 传输}, spv目录); }
    catch (const std::exception& e) { std::fprintf(stderr, "启动失败: %s\n", e.what()); return 1; }
    std::printf("设备: %s | 算子: %zu\n", 调度器.获取设备().名称(), 调度器.算子数量());

    auto 建 = [](std::vector<std::uint32_t> 形状, 张量类型 类型, const char* 名) {
        return vk张量(张量布局::连续(std::move(形状)), 类型, 名);
    };
    std::uint32_t 随机状态 = 20261118u;
    auto 随机数 = [&]() {
        随机状态 = 随机状态 * 1664525u + 1013904223u;
        return (static_cast<float>((随机状态 >> 8) & 0xFFFFFFu) / 16777216.0f) * 2.0f - 1.0f;
    };
    auto 上传随机 = [&](vk张量& t, float 缩放) {
        std::vector<float> v(t.布局().元素总数);
        for (auto& x : v) x = 随机数() * 缩放;
        t.上传(v.data(), v.size() * 4);
    };
    auto 提交 = [&](const char* op, std::vector<张量句柄> 句柄, const void* pc,
                   std::uint32_t pcBytes, std::uint32_t threads) {
        const uint64_t id = 调度器.提交自定义算子(op, std::move(句柄), pc, pcBytes,
                                                  {(threads + 63u) / 64u, 1u, 1u});
        调度器.等待(id);
        if (!调度器.已完成(id))
            throw std::runtime_error(std::string("算子失败: ") + op + " | " +
                                     调度器.查询失败原因(id));
    };

    vk张量 table = 建({V, D}, 张量类型::浮点32, "table");
    vk张量 W1 = 建({Hx, D}, 张量类型::浮点32, "W1");
    vk张量 b1 = 建({Hx}, 张量类型::浮点32, "b1");
    vk张量 g = 建({Hx}, 张量类型::浮点32, "g");
    vk张量 W2 = 建({V, Hx}, 张量类型::浮点32, "W2");
    vk张量 b2 = 建({V}, 张量类型::浮点32, "b2");
    上传随机(table, 0.02f); 上传随机(W1, 0.2f); 上传随机(b1, 0.0f);
    std::vector<float> 一(Hx, 1.0f); g.上传(一.data(), 一.size() * 4);
    上传随机(W2, 0.02f); 上传随机(b2, 0.0f);

    vk张量 h0 = 建({N, D}, 张量类型::浮点32, "h0");
    vk张量 h1 = 建({N, Hx}, 张量类型::浮点32, "h1");
    vk张量 h2 = 建({N, Hx}, 张量类型::浮点32, "h2");
    vk张量 logits = 建({N, V}, 张量类型::浮点32, "logits");
    vk张量 dx2 = 建({N, Hx}, 张量类型::浮点32, "dx2");
    vk张量 dx1 = 建({N, Hx}, 张量类型::浮点32, "dx1");
    vk张量 dh0 = 建({N, D}, 张量类型::浮点32, "dh0");
    vk张量 dW1 = 建({Hx, D}, 张量类型::浮点32, "dW1");
    vk张量 db1 = 建({Hx}, 张量类型::浮点32, "db1");
    vk张量 dg = 建({Hx}, 张量类型::浮点32, "dg");
    vk张量 dW2 = 建({V, Hx}, 张量类型::浮点32, "dW2");
    vk张量 db2 = 建({V}, 张量类型::浮点32, "db2");
    vk张量 dTable = 建({V, D}, 张量类型::浮点32, "dTable");

    vk张量 ids = 建({N}, 张量类型::整数32, "ids");
    vk张量 targets = 建({N}, 张量类型::整数32, "targets");

    分块交叉熵 ce(N, V, "sft.ce");

    struct 项 { vk张量* p; vk张量* gr; vk张量 m; vk张量 v; };
    std::vector<项> 参数;
    auto 注册参数 = [&](vk张量& p, vk张量& gr, const char* 名) {
        vk张量 m = 建(std::vector<std::uint32_t>{p.布局().形状}, 张量类型::浮点32,
                     (std::string(名) + "_m").c_str());
        vk张量 v = 建(std::vector<std::uint32_t>{p.布局().形状}, 张量类型::浮点32,
                     (std::string(名) + "_v").c_str());
        std::vector<float> 零(p.布局().元素总数, 0.0f);
        m.上传(零.data(), 零.size() * 4);
        v.上传(零.data(), 零.size() * 4);
        参数.push_back(项{&p, &gr, std::move(m), std::move(v)});
    };
    注册参数(table, dTable, "table"); 注册参数(W1, dW1, "W1"); 注册参数(b1, db1, "b1");
    注册参数(g, dg, "g"); 注册参数(W2, dW2, "W2"); 注册参数(b2, db2, "b2");

    std::printf("M0 SFT 图: V=%u D=%u H=%u N=%u | 参数张量 %zu\n",
                V, D, Hx, N, 参数.size());

    const auto 开始 = std::chrono::steady_clock::now();
    int 完成步数 = 0;
    for (int step = 0; step < 总步数; ++step) {
        sft数据::批次 批;
        if (!流.取批次(批, B, T)) {
            std::printf("数据流提前结束（step %d）\n", step);
            break;
        }
        if (批.input_ids.size() != N || 批.labels.size() != N) {
            throw std::runtime_error("批次形状与 N 不一致");
        }
        ids.上传(批.input_ids.data(), 批.input_ids.size() * 4);
        targets.上传(批.labels.data(), 批.labels.size() * 4);

        // 前向
        规格4 s_emb{N, V, D, 0};
        提交("embedding_fwd", {table.取句柄(), ids.取句柄(), h0.取句柄()},
             &s_emb, sizeof(s_emb), N * D);
        规格4 s_l1{N, D, Hx, 0};
        提交("linear_fwd", {h0.取句柄(), W1.取句柄(), b1.取句柄(), h1.取句柄()},
             &s_l1, sizeof(s_l1), N * Hx);
        规格F s_rms{N, Hx, 1e-6f, 0.0f};
        提交("rmsnorm_fwd", {h1.取句柄(), g.取句柄(), h2.取句柄()},
             &s_rms, sizeof(s_rms), N);
        规格4 s_l2{N, Hx, V, 0};
        提交("linear_fwd", {h2.取句柄(), W2.取句柄(), b2.取句柄(), logits.取句柄()},
             &s_l2, sizeof(s_l2), N * V);

        // 分块 CE（真实词表；-100 忽略 prompt/padding）
        ce.计算(logits, targets, 分块交叉熵配置{64, 64});
        const float 平均损失 = ce.取平均损失(批.labels);

        // 反向
        提交("linear_bwd_dx", {ce.dlogits().取句柄(), W2.取句柄(), dx2.取句柄()},
             &s_l2, sizeof(s_l2), N * Hx);
        提交("linear_bwd_dwdb", {h2.取句柄(), ce.dlogits().取句柄(),
                                dW2.取句柄(), db2.取句柄()},
             &s_l2, sizeof(s_l2), V * Hx);
        提交("rmsnorm_bwd_dx", {h1.取句柄(), dx2.取句柄(), g.取句柄(), dx1.取句柄()},
             &s_rms, sizeof(s_rms), N);
        提交("rmsnorm_bwd_dw", {h1.取句柄(), dx2.取句柄(), dg.取句柄()},
             &s_rms, sizeof(s_rms), Hx);
        提交("linear_bwd_dx", {dx1.取句柄(), W1.取句柄(), dh0.取句柄()},
             &s_l1, sizeof(s_l1), N * D);
        提交("linear_bwd_dwdb", {h0.取句柄(), dx1.取句柄(), dW1.取句柄(), db1.取句柄()},
             &s_l1, sizeof(s_l1), Hx * D);
        提交("embedding_bwd", {dh0.取句柄(), ids.取句柄(), dTable.取句柄()},
             &s_emb, sizeof(s_emb), V);

        // AdamW
        const float b1t = std::pow(beta1, float(step + 1));
        const float b2t = std::pow(beta2, float(step + 1));
        for (auto& 项 : 参数) {
            规格Adam pc{lr, beta1, beta2, eps,
                        1.0f / (1.0f - b1t), 1.0f / (1.0f - b2t),
                        static_cast<std::uint32_t>(项.p->布局().元素总数), wd, 1u};
            提交("adamw_update",
                 {项.gr->取句柄(), 项.p->取句柄(), 项.m.取句柄(), 项.v.取句柄()},
                 &pc, sizeof(pc), static_cast<std::uint32_t>(项.p->布局().元素总数));
        }

        完成步数 = step + 1;
        if (step % 10 == 0 || step == 总步数 - 1) {
            std::printf("step %3d loss=%.4f 有效token=%u\n",
                        step, 平均损失, 批.有效token数);
        }
    }

    const double 秒 = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - 开始).count();
    std::printf("完成 %d 步 | %.1f ms/step | %.2f 步/s\n",
                完成步数, 完成步数 ? 秒 * 1000.0 / 完成步数 : 0.0,
                完成步数 ? 完成步数 / 秒 : 0.0);

    const auto 终态 = 流.取统计();
    std::printf("数据流统计: 读行=%llu 解析=%llu 分词=%llu 跳过=%llu token=%llu\n",
                static_cast<unsigned long long>(终态.已读行),
                static_cast<unsigned long long>(终态.已解析),
                static_cast<unsigned long long>(终态.已分词),
                static_cast<unsigned long long>(终态.跳过),
                static_cast<unsigned long long>(终态.总token));

    调度器.停止();
    return 0;
}
