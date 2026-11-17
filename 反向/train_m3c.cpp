// P4-M3 第 4 步：多模态接入（冻结编码器嵌入）+ ERSA 块复合训练环
// 结构：x(冻结, 音频2+视觉2+文本2) -> Wq/Wk/Wv 投影(linear_fwd) -> 块摘要
//      -> ersa_attn_fused -> MSE(loss 主机算) -> 逐 query 反传 -> linear_bwd_dwdb
//      -> AdamW(adamw_update 内核，设备端 m/v，解耦衰减，W* 衰减/sink 不衰减)
// 说明：CPU 版 asm_ersa.ERSA 是“参考版”语义（q 归一化/JL sketch/无因果），
//       与 ersa_attn_fused 内核不同口径；本对拍按内核精确语义在 PyTorch 中复算。
// 用法：train_m3c.exe <spv目录> <导出.bin>
#include "流水线调度器.hpp"
#include "vk张量.hpp"
#include "ersa_block_sum.spv.hpp"
#include "ersa_attn_fused.spv.hpp"
#include "ersa_attn_bwd.spv.hpp"
#include "linear_fwd.spv.hpp"
#include "linear_bwd_dwdb.spv.hpp"
#include "adamw_update.spv.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vk张量系统;

namespace {
std::uint32_t g_lcg = 31337u;
float 随机数(float 幅 = 1.0f) {
    g_lcg = g_lcg * 1664525u + 1013904223u;
    return 幅 * ((static_cast<float>((g_lcg >> 8) & 0xFFFFFFu) / 16777216.0f) * 2.0f - 1.0f);
}
struct 推送块和 {
    std::uint32_t B, H, N, DK, NB, BLOCK, pad0, pad1;
};
struct 推送前向 {
    std::uint32_t B, H, T, N, DK, DV, NB, BLOCK, TOPB, WINDOW, pad0, pad1;
    float tau, sink;
};
struct 推送反向 {
    std::uint32_t B, H, T, DK, DV, pad;
};
struct 推送线性 {
    std::uint32_t N, K, M, pad;
};
struct 推送AdamW {
    float lr, beta1, beta2, eps, m_corr, v_corr;
    std::uint32_t size;
    float weight_decay;
    std::uint32_t zero_grad;
};
static_assert(sizeof(推送AdamW) == 36, "adamw push");
} // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "用法: %s <spv目录> <导出.bin>\n", argv[0]); return 2; }
    流水线调度器& 调度器 = 流水线调度器::获取();
    设备注册信息 注册; 注册.应用名 = "train_m3c";
    阶段配置 图形; 阶段配置 计算{阶段配置::自动队列族, 0, 1}; 阶段配置 传输;
    try { 调度器.启动(注册, {图形, 计算, 传输}, argv[1]); }
    catch (const std::exception& e) { std::fprintf(stderr, "启动失败: %s\n", e.what()); return 1; }

    const std::uint32_t T = 6, B = 1, H = 1, DX = 3, DK = 4, DV = 4;
    const std::uint32_t BLOCK = 2, NB = (T + BLOCK - 1) / BLOCK, TOPB = 2, WINDOW = 2;
    const std::uint32_t STEPS = 20;
    const float tau = 0.8f, 初始sink = -1.3f;
    const float lr = 0.02f, wd = 0.01f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;

    auto 建 = [](std::vector<std::uint32_t> 形状, const char* 名) {
        return vk张量(张量布局::连续(std::move(形状)), 张量类型::浮点32, 名);
    };
    auto 提交 = [&](const char* op, std::vector<张量句柄> 句柄, const void* pc,
                   std::uint32_t pcB, std::uint32_t gx, std::uint32_t gy, std::uint32_t gz) {
        std::uint64_t id = 调度器.提交自定义算子(op, std::move(句柄), pc, pcB, {gx, gy, gz});
        调度器.等待(id);
        if (!调度器.已完成(id))
            throw std::runtime_error(std::string("算子失败: ") + op + " | " + 调度器.查询失败原因(id));
    };
    auto 零张量 = [&](std::vector<std::uint32_t> 形状, std::uint32_t 个数, const char* 名) {
        vk张量 t = 建(std::move(形状), 名);
        std::vector<float> 零(个数, 0.0f);
        t.上传(零.data(), 零.size() * 4);
        return t;
    };

    // 冻结输入（模拟音频/视觉/文本编码器输出）与目标
    std::vector<float> x(T * DX), 目标(T * DV);
    for (auto& v : x) v = 随机数();
    for (auto& v : 目标) v = 随机数();

    std::vector<float> Wq0(DK * DX), Wk0(DK * DX), Wv0(DV * DX);
    for (auto& v : Wq0) v = 随机数(0.5f);
    for (auto& v : Wk0) v = 随机数(0.5f);
    for (auto& v : Wv0) v = 随机数(0.5f);

    vk张量 X = 建({T, DX}, "x"); X.上传(x.data(), x.size() * 4);
    vk张量 Qp = 建({T, DK}, "q");
    vk张量 Kp = 建({B, H, T, DK}, "K");
    vk张量 Vp = 建({B, H, T, DV}, "V");
    vk张量 BS = 建({B, H, NB, DK}, "bsum");
    vk张量 Out = 建({B, H, T, DV}, "out");
    vk张量 Dy = 建({B, H, T, DV}, "dy");
    vk张量 dQd = 建({T, DK}, "dQ");
    vk张量 dKd = 建({T, DK}, "dK");
    vk张量 dVd = 建({T, DV}, "dV");

    vk张量 Wq = 建({DK, DX}, "W_q"); Wq.上传(Wq0.data(), Wq0.size() * 4);
    vk张量 Wk = 建({DK, DX}, "W_k"); Wk.上传(Wk0.data(), Wk0.size() * 4);
    vk张量 Wv = 建({DV, DX}, "W_v"); Wv.上传(Wv0.data(), Wv0.size() * 4);
    vk张量 Sink = 建({1}, "sink_bias"); Sink.上传(&初始sink, 4);

    vk张量 零偏DK = 零张量({DK}, DK, "b0_dk");
    vk张量 零偏DV = 零张量({DV}, DV, "b0_dv");
    vk张量 dbk = 建({DK}, "db_dk");
    vk张量 dbv = 建({DV}, "db_dv");
    vk张量 dS = 建({1}, "dsink");

    vk张量 dWq = 建({DK, DX}, "dWq");
    vk张量 dWk = 建({DK, DX}, "dWk");
    vk张量 dWv = 建({DV, DX}, "dWv");
    // AdamW 状态
    vk张量 mWq = 零张量({DK, DX}, DK * DX, "mWq");
    vk张量 vWq = 零张量({DK, DX}, DK * DX, "vWq");
    vk张量 mWk = 零张量({DK, DX}, DK * DX, "mWk");
    vk张量 vWk = 零张量({DK, DX}, DK * DX, "vWk");
    vk张量 mWv = 零张量({DV, DX}, DV * DX, "mWv");
    vk张量 vWv = 零张量({DV, DX}, DV * DX, "vWv");
    vk张量 mS = 零张量({1}, 1, "mS");
    vk张量 vS = 零张量({1}, 1, "vS");

    vk张量 q1 = 建({B, H, DK}, "q1");
    vk张量 m1 = 建({B, H, T}, "mask1");
    vk张量 sm = 建({B, H}, "sinkbuf");
    vk张量 dy1 = 建({B, H, DV}, "dy1");
    vk张量 dq1 = 建({B, H, DK}, "dq1");
    vk张量 dk1 = 建({T, H, DK}, "dk1");
    vk张量 dv1 = 建({T, H, DV}, "dv1");
    vk张量 ds1 = 建({B, H}, "ds1");

    const float scale = 1.0f / std::sqrt(float(DK));
    std::vector<float> 损耗(STEPS, 0.0f);
    std::vector<float> 缓冲, qh(T * DK), bsh(NB * DK), outh(T * DV);
    std::vector<float> dQh(T * DK, 0.f), dKh(T * DK, 0.f), dVh(T * DV, 0.f);
    std::vector<float> masks(T * T, 0.f);

    for (std::uint32_t s = 0; s < STEPS; ++s) {
        // 投影前向
        推送线性 pcl{T, DX, DK, 0};
        提交("linear_fwd", {X.取句柄(), Wq.取句柄(), 零偏DK.取句柄(), Qp.取句柄()}, &pcl, sizeof(pcl),
             (T * DK + 63) / 64, 1, 1);
        提交("linear_fwd", {X.取句柄(), Wk.取句柄(), 零偏DK.取句柄(), Kp.取句柄()}, &pcl, sizeof(pcl),
             (T * DK + 63) / 64, 1, 1);
        推送线性 pclv{T, DX, DV, 0};
        提交("linear_fwd", {X.取句柄(), Wv.取句柄(), 零偏DV.取句柄(), Vp.取句柄()}, &pclv, sizeof(pclv),
             (T * DV + 63) / 64, 1, 1);

        推送块和 pcbs{B, H, T, DK, NB, BLOCK, 0, 0};
        提交("ersa_block_sum", {Kp.取句柄(), BS.取句柄()}, &pcbs, sizeof(pcbs), NB, H, B);
        Qp.下载(qh.data(), qh.size() * 4);
        BS.下载(bsh.data(), bsh.size() * 4);

        // 主机路由（与内核同规则）与掩码
        std::fill(masks.begin(), masks.end(), 0.0f);
        std::vector<std::uint32_t> 选块(T * TOPB, 0u);
        for (std::uint32_t t = 0; t < T; ++t) {
            std::vector<float> sc(NB);
            for (std::uint32_t i = 0; i < NB; ++i) {
                float d = 0.0f;
                for (std::uint32_t k = 0; k < DK; ++k) d += qh[t * DK + k] * bsh[i * DK + k];
                sc[i] = d * scale / tau;
            }
            const std::uint32_t topn = std::min(TOPB, NB);
            for (std::uint32_t kk = 0; kk < topn; ++kk) {
                float best = -3.402823e38f; std::uint32_t bi = 0u;
                for (std::uint32_t i = 0; i < NB; ++i) if (sc[i] > best) { best = sc[i]; bi = i; }
                选块[t * TOPB + kk] = bi;
                sc[bi] = -3.402823e38f;
            }
            for (std::uint32_t kk = 0; kk < topn; ++kk) {
                const std::uint32_t bi = 选块[t * TOPB + kk];
                const std::uint32_t start = bi * BLOCK, end = std::min(start + BLOCK, T);
                for (std::uint32_t idx = start; idx < end; ++idx)
                    if (idx <= t) masks[t * T + idx] = 1.0f;
            }
            const std::uint32_t wstart = (t + 1u > WINDOW) ? (t + 1u - WINDOW) : 0u;
            for (std::uint32_t idx = wstart; idx <= t; ++idx) masks[t * T + idx] = 1.0f;
        }

        // ERSA 前向
        推送前向 pcf{B, H, T, T, DK, DV, NB, BLOCK, TOPB, WINDOW, 0, 0, tau, 初始sink};
        Sink.下载(&pcf.sink, 4);   // 当前 sink 参数
        提交("ersa_attn_fused", {Qp.取句柄(), Kp.取句柄(), Vp.取句柄(), BS.取句柄(), Out.取句柄()},
             &pcf, sizeof(pcf), T, H, B);
        Out.下载(outh.data(), outh.size() * 4);

        // loss = mean((out-target)^2)，dOut = 2(out-target)/N
        float 损 = 0.0f;
        std::vector<float> dyh(T * DV);
        for (std::uint32_t i = 0; i < T * DV; ++i) {
            const float d = outh[i] - 目标[i];
            损 += d * d;
            dyh[i] = 2.0f * d / float(T * DV);
        }
        损 /= float(T * DV);
        损耗[s] = 损;
        Dy.上传(dyh.data(), dyh.size() * 4);

        // 逐 query 反传并累加
        std::fill(dQh.begin(), dQh.end(), 0.f);
        std::fill(dKh.begin(), dKh.end(), 0.f);
        std::fill(dVh.begin(), dVh.end(), 0.f);
        float dsink总和 = 0.0f;
        std::vector<float> sinkh(1);
        Sink.下载(sinkh.data(), 4);
        sm.上传(sinkh.data(), 4);
        for (std::uint32_t t = 0; t < T; ++t) {
            std::vector<float> qa(DK);
            for (std::uint32_t d = 0; d < DK; ++d) qa[d] = qh[t * DK + d] / tau;
            q1.上传(qa.data(), qa.size() * 4);
            m1.上传(&masks[t * T], T * 4);
            dy1.上传(&dyh[t * DV], DV * 4);
            推送反向 pcb{B, H, T, DK, DV, 0};
            提交("ersa_attn_bwd",
                 {q1.取句柄(), Kp.取句柄(), Vp.取句柄(), m1.取句柄(), sm.取句柄(), dy1.取句柄(),
                  dq1.取句柄(), dk1.取句柄(), dv1.取句柄(), ds1.取句柄()},
                 &pcb, sizeof(pcb), B * H, 1, 1);
            缓冲.assign(DK, 0.0f);
            dq1.下载(缓冲.data(), DK * 4);
            for (std::uint32_t d = 0; d < DK; ++d) dQh[t * DK + d] = 缓冲[d] / tau;
            缓冲.assign(T * DK, 0.0f);
            dk1.下载(缓冲.data(), T * DK * 4);
            for (std::uint32_t i = 0; i < T * DK; ++i) dKh[i] += 缓冲[i];
            缓冲.assign(T * DV, 0.0f);
            dv1.下载(缓冲.data(), T * DV * 4);
            for (std::uint32_t i = 0; i < T * DV; ++i) dVh[i] += 缓冲[i];
            std::vector<float> dsbuf(1, 0.0f);
            ds1.下载(dsbuf.data(), 4);
            dsink总和 += dsbuf[0];
        }
        dQd.上传(dQh.data(), dQh.size() * 4);
        dKd.上传(dKh.data(), dKh.size() * 4);
        dVd.上传(dVh.data(), dVh.size() * 4);
        dS.上传(&dsink总和, 4);

        // 投影反向 dW = dY^T @ X
        提交("linear_bwd_dwdb", {X.取句柄(), dQd.取句柄(), dWq.取句柄(), dbk.取句柄()}, &pcl, sizeof(pcl),
             (DK * DX + 63) / 64, 1, 1);
        提交("linear_bwd_dwdb", {X.取句柄(), dKd.取句柄(), dWk.取句柄(), dbk.取句柄()}, &pcl, sizeof(pcl),
             (DK * DX + 63) / 64, 1, 1);
        提交("linear_bwd_dwdb", {X.取句柄(), dVd.取句柄(), dWv.取句柄(), dbv.取句柄()}, &pclv, sizeof(pclv),
             (DV * DX + 63) / 64, 1, 1);

        // AdamW（内核，等同 优化器/AdamW 的数学）
        const float m_corr = 1.0f / (1.0f - std::pow(beta1, float(s + 1)));
        const float v_corr = 1.0f / (1.0f - std::pow(beta2, float(s + 1)));
        auto AdamW步 = [&](vk张量& P, vk张量& G, vk张量& M, vk张量& V,
                          std::uint32_t 个数, float 衰减, std::uint32_t gx) {
            推送AdamW pc{lr, beta1, beta2, eps, m_corr, v_corr, 个数, 衰减, 1u};
            提交("adamw_update", {G.取句柄(), P.取句柄(), M.取句柄(), V.取句柄()}, &pc, sizeof(pc),
                 gx, 1, 1);
        };
        AdamW步(Wq, dWq, mWq, vWq, DK * DX, wd, 1);
        AdamW步(Wk, dWk, mWk, vWk, DK * DX, wd, 1);
        AdamW步(Wv, dWv, mWv, vWv, DV * DX, wd, 1);
        AdamW步(Sink, dS, mS, vS, 1, 0.0f, 1);

        if (s == 0 || s == STEPS - 1)
            std::printf("step %u loss = %.6f | sink = %.6f\n", s, 损, sinkh[0]);
    }

    // 导出
    std::ofstream f(argv[2], std::ios::binary);
    if (!f.is_open()) { std::fprintf(stderr, "无法写导出文件\n"); return 1; }
    std::uint32_t hdr[9] = {T, DX, DK, DV, BLOCK, NB, TOPB, WINDOW, STEPS};
    f.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
    auto 写 = [&](const void* p, std::size_t 个) { f.write(static_cast<const char*>(p), 个 * 4); };
    写(&tau, 1); 写(&初始sink, 1); 写(&lr, 1); 写(&wd, 1);
    写(&beta1, 1); 写(&beta2, 1); 写(&eps, 1);
    写(x.data(), x.size()); 写(目标.data(), 目标.size());
    写(Wq0.data(), Wq0.size()); 写(Wk0.data(), Wk0.size()); 写(Wv0.data(), Wv0.size());
    写(损耗.data(), 损耗.size());
    std::vector<float> WqF(DK * DX), WkF(DK * DX), WvF(DV * DX), sinkF(1);
    Wq.下载(WqF.data(), WqF.size() * 4);
    Wk.下载(WkF.data(), WkF.size() * 4);
    Wv.下载(WvF.data(), WvF.size() * 4);
    Sink.下载(sinkF.data(), 4);
    写(WqF.data(), WqF.size()); 写(WkF.data(), WkF.size()); 写(WvF.data(), WvF.size());
    写(sinkF.data(), 1);
    f.close();

    调度器.停止();
    std::printf("M3c 对拍数据已写入 %s\n", argv[2]);
    return 0;
}
