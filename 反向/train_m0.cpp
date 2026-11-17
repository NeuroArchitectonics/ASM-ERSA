// P4-M0：最小图 Vulkan 训练器
// 图：ids → embedding → linear(W1,b1) → rmsnorm(g) → linear(W2,b2) → CE
// 反向硬编码顺序；优化器为 adamw_update（融合 m/v + 偏差修正 + 解耦衰减 + 梯度清零）
// 用法：train_m0.exe <spv目录> [steps]
#include "流水线调度器.hpp"
#include "vk张量.hpp"
#include "embedding_fwd.spv.hpp"
#include "linear_fwd.spv.hpp"
#include "rmsnorm_fwd.spv.hpp"
#include "linear_bwd_dx.spv.hpp"
#include "linear_bwd_dwdb.spv.hpp"
#include "rmsnorm_bwd_dx.spv.hpp"
#include "rmsnorm_bwd_dw.spv.hpp"
#include "fused_ce_bwd.spv.hpp"
#include "embedding_bwd.spv.hpp"
#include "adamw_update.spv.hpp"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vk张量系统;

namespace {

std::uint32_t g_lcg = 20261116u;
float 随机数() {
    g_lcg = g_lcg * 1664525u + 1013904223u;
    return (static_cast<float>((g_lcg >> 8) & 0xFFFFFFu) / 16777216.0f) * 2.0f - 1.0f;
}

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
        std::fprintf(stderr, "用法: %s <spv目录> [steps]\n", argv[0]);
        return 2;
    }
    const std::string 着色器目录 = argv[1];
    const int 总步数 = argc >= 3 ? std::atoi(argv[2]) : 50;

    流水线调度器& 调度器 = 流水线调度器::获取();
    设备注册信息 注册; 注册.应用名 = "train_m0";
    阶段配置 图形; 阶段配置 计算{阶段配置::自动队列族, 0, 1}; 阶段配置 传输;
    try { 调度器.启动(注册, {图形, 计算, 传输}, 着色器目录); }
    catch (const std::exception& e) { std::fprintf(stderr, "启动失败: %s\n", e.what()); return 1; }
    std::printf("设备: %s | 算子: %zu\n", 调度器.获取设备().名称(), 调度器.算子数量());

    // ── 维度 ──
    const uint32_t V = 64, D = 32, Hx = 32, N = 16;
    const float lr = 3e-3f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f, wd = 0.01f;

    auto 建 = [](std::vector<uint32_t> 形状, 张量类型 类型, const char* 名) {
        return vk张量(张量布局::连续(std::move(形状)), 类型, 名);
    };
    auto 上传随机 = [](vk张量& t, float 缩放) {
        std::vector<float> v(t.布局().元素总数);
        for (auto& x : v) x = 随机数() * 缩放;
        t.上传(v.data(), v.size() * 4);
    };
    auto 提交 = [&](const char* op, std::vector<张量句柄> 句柄, const void* pc,
                   std::uint32_t pcBytes, std::uint32_t threads) {
        uint64_t id = 调度器.提交自定义算子(op, std::move(句柄), pc, pcBytes,
                                            {(threads + 63u) / 64u, 1u, 1u});
        调度器.等待(id);
        if (!调度器.已完成(id))
            throw std::runtime_error(std::string("算子失败: ") + op + " | " + 调度器.查询失败原因(id));
    };

    // ── 参数与状态 ──
    vk张量 table = 建({V, D}, 张量类型::浮点32, "table");
    vk张量 W1 = 建({Hx, D}, 张量类型::浮点32, "W1");
    vk张量 b1 = 建({Hx}, 张量类型::浮点32, "b1");
    vk张量 g  = 建({Hx}, 张量类型::浮点32, "g");
    vk张量 W2 = 建({V, Hx}, 张量类型::浮点32, "W2");
    vk张量 b2 = 建({V}, 张量类型::浮点32, "b2");
    上传随机(table, 0.1f); 上传随机(W1, 0.2f); 上传随机(b1, 0.1f);
    std::vector<float> 一(Hx, 1.0f); g.上传(一.data(), 一.size() * 4);
    上传随机(W2, 0.2f); 上传随机(b2, 0.1f);

    vk张量 h0 = 建({N, D}, 张量类型::浮点32, "h0");
    vk张量 h1 = 建({N, Hx}, 张量类型::浮点32, "h1");
    vk张量 h2 = 建({N, Hx}, 张量类型::浮点32, "h2");
    vk张量 logits = 建({N, V}, 张量类型::浮点32, "logits");
    vk张量 loss = 建({N}, 张量类型::浮点32, "loss");
    vk张量 dlogits = 建({N, V}, 张量类型::浮点32, "dlogits");
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
    std::vector<std::int32_t> idh(N), th(N);
    for (uint32_t i = 0; i < N; ++i) { idh[i] = int32_t((i * 7 + 3) % V); th[i] = int32_t((idh[i] + 1) % V); }
    ids.上传(idh.data(), idh.size() * 4); targets.上传(th.data(), th.size() * 4);

    // 优化器状态（m/v）
    struct 项 { vk张量* p; vk张量* gr; vk张量 m; vk张量 v; };
    std::vector<项> 参数;
    auto 注册参数 = [&](vk张量& p, vk张量& gr, const char* 名) {
        vk张量 m = 建(std::vector<uint32_t>{p.布局().形状}, 张量类型::浮点32, (std::string(名) + "_m").c_str());
        vk张量 v = 建(std::vector<uint32_t>{p.布局().形状}, 张量类型::浮点32, (std::string(名) + "_v").c_str());
        std::vector<float> 零(p.布局().元素总数, 0.0f);
        m.上传(零.data(), 零.size() * 4); v.上传(零.data(), 零.size() * 4);
        参数.push_back(项{&p, &gr, std::move(m), std::move(v)});
    };
    注册参数(table, dTable, "table"); 注册参数(W1, dW1, "W1"); 注册参数(b1, db1, "b1");
    注册参数(g, dg, "g"); 注册参数(W2, dW2, "W2"); 注册参数(b2, db2, "b2");

    std::printf("M0 图: V=%u D=%u H=%u N=%u | 参数张量 %zu 个\n", V, D, Hx, N, 参数.size());

    // M1 对拍导出：初始参数 + 逐步 loss
    std::ofstream 导出;
    const bool 要导出 = argc >= 4;
    if (要导出) {
        导出.open(argv[3], std::ios::binary);
        if (!导出.is_open()) { std::fprintf(stderr, "无法写导出文件\n"); return 1; }
        auto 写张量 = [&](vk张量& t) {
            std::vector<float> buf(t.布局().元素总数);
            t.下载(buf.data(), buf.size() * 4);
            导出.write(reinterpret_cast<const char*>(buf.data()), buf.size() * 4);
        };
        写张量(table); 写张量(W1); 写张量(b1); 写张量(g); 写张量(W2); 写张量(b2);
    }
    std::vector<float> loss历史;

    std::vector<float> lossh(N);
    for (int step = 0; step < 总步数; ++step) {
        // ── 前向 ──
        规格4 s_emb{N, V, D, 0};
        提交("embedding_fwd", {table.取句柄(), ids.取句柄(), h0.取句柄()}, &s_emb, sizeof(s_emb), N * D);
        规格4 s_l1{N, D, Hx, 0};
        提交("linear_fwd", {h0.取句柄(), W1.取句柄(), b1.取句柄(), h1.取句柄()}, &s_l1, sizeof(s_l1), N * Hx);
        规格F s_rms{N, Hx, 1e-6f, 0.0f};
        提交("rmsnorm_fwd", {h1.取句柄(), g.取句柄(), h2.取句柄()}, &s_rms, sizeof(s_rms), N);
        规格4 s_l2{N, Hx, V, 0};
        提交("linear_fwd", {h2.取句柄(), W2.取句柄(), b2.取句柄(), logits.取句柄()}, &s_l2, sizeof(s_l2), N * V);
        规格4 s_ce{N, V, 0, 0};
        提交("fused_ce_bwd", {logits.取句柄(), targets.取句柄(), dlogits.取句柄(), loss.取句柄()},
             &s_ce, sizeof(s_ce), N);
        loss.下载(lossh.data(), lossh.size() * 4);
        double 平均 = 0.0; for (float x : lossh) 平均 += x; 平均 /= N;
        loss历史.push_back(float(平均));

        // ── 反向（固定顺序）──
        提交("linear_bwd_dx", {dlogits.取句柄(), W2.取句柄(), dx2.取句柄()}, &s_l2, sizeof(s_l2), N * Hx);
        提交("linear_bwd_dwdb", {h2.取句柄(), dlogits.取句柄(), dW2.取句柄(), db2.取句柄()},
             &s_l2, sizeof(s_l2), V * Hx);
        提交("rmsnorm_bwd_dx", {h1.取句柄(), dx2.取句柄(), g.取句柄(), dx1.取句柄()},
             &s_rms, sizeof(s_rms), N);
        提交("rmsnorm_bwd_dw", {h1.取句柄(), dx2.取句柄(), dg.取句柄()},
             &s_rms, sizeof(s_rms), Hx);
        提交("linear_bwd_dx", {dx1.取句柄(), W1.取句柄(), dh0.取句柄()}, &s_l1, sizeof(s_l1), N * D);
        提交("linear_bwd_dwdb", {h0.取句柄(), dx1.取句柄(), dW1.取句柄(), db1.取句柄()},
             &s_l1, sizeof(s_l1), Hx * D);
        提交("embedding_bwd", {dh0.取句柄(), ids.取句柄(), dTable.取句柄()},
             &s_emb, sizeof(s_emb), V);

        // ── AdamW（一步一参，zero_grad=1）──
        const float b1t = std::pow(beta1, float(step + 1));
        const float b2t = std::pow(beta2, float(step + 1));
        for (auto& 项 : 参数) {
            规格Adam pc{lr, beta1, beta2, eps,
                        1.0f / (1.0f - b1t), 1.0f / (1.0f - b2t),
                        uint32_t(项.p->布局().元素总数), wd, 1u};
            提交("adamw_update",
                 {项.gr->取句柄(), 项.p->取句柄(), 项.m.取句柄(), 项.v.取句柄()},
                 &pc, sizeof(pc), 项.p->布局().元素总数);
        }

        if (step % 10 == 0 || step == 总步数 - 1)
            std::printf("step %3d  loss=%.4f\n", step, 平均);
    }

    if (要导出) {
        std::uint32_t 步数 = static_cast<std::uint32_t>(loss历史.size());
        导出.write(reinterpret_cast<const char*>(&步数), sizeof(步数));
        导出.write(reinterpret_cast<const char*>(loss历史.data()), loss历史.size() * 4);
        导出.close();
    }

    调度器.停止();
    return 0;
}
