// P4-M3 第 3 步：ERSA 层逐 token 训练环（KV cache 逐步增长，BPTT 式梯度累计）
// 每步：cache 追加 k_t/v_t → ersa_block_sum 全量重算（并与主机增量式和对照）
//      → ersa_attn_fused（T=1，仅新 query）→ ersa_attn_bwd（B=1）逐 query 反传
// dq_t 逐 token 保存；dK/dV 在 cache 槽位跨步累加；dsink 求和
// 用法：train_m3b.exe <spv目录> <导出.bin>
#include "流水线调度器.hpp"
#include "vk张量.hpp"
#include "ersa_block_sum.spv.hpp"
#include "ersa_attn_fused.spv.hpp"
#include "ersa_attn_bwd.spv.hpp"

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
std::uint32_t g_lcg = 9001u;
float 随机数() {
    g_lcg = g_lcg * 1664525u + 1013904223u;
    return (static_cast<float>((g_lcg >> 8) & 0xFFFFFFu) / 16777216.0f) * 2.0f - 1.0f;
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
} // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "用法: %s <spv目录> <导出.bin>\n", argv[0]); return 2; }
    流水线调度器& 调度器 = 流水线调度器::获取();
    设备注册信息 注册; 注册.应用名 = "train_m3b";
    阶段配置 图形; 阶段配置 计算{阶段配置::自动队列族, 0, 1}; 阶段配置 传输;
    try { 调度器.启动(注册, {图形, 计算, 传输}, argv[1]); }
    catch (const std::exception& e) { std::fprintf(stderr, "启动失败: %s\n", e.what()); return 1; }

    const std::uint32_t T = 8, B = 1, H = 1, DK = 4, DV = 4;
    const std::uint32_t BLOCK = 2, NBMAX = (T + BLOCK - 1) / BLOCK, TOPB = 2, WINDOW = 2;
    const float tau = 0.8f, sink = -1.3f;

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

    std::vector<float> qh(T * DK), kh(T * DK), vh(T * DV), dOh(T * DV);
    for (auto& x : qh) x = 随机数();
    for (auto& x : kh) x = 随机数();
    for (auto& x : vh) x = 随机数();
    for (auto& x : dOh) x = 随机数();

    vk张量 Kc = 建({B, H, T, DK}, "Kcache");
    vk张量 Vc = 建({B, H, T, DV}, "Vcache");
    vk张量 BSt = 建({B, H, NBMAX, DK}, "bsum");
    vk张量 Qt = 建({B, H, T, DK}, "Qprefix");   // 前向按前缀派发，取第 t 行
    vk张量 Ot = 建({B, H, T, DV}, "outprefix");
    vk张量 q1 = 建({B, H, DK}, "q1");            // 反向的单个 query
    vk张量 m1 = 建({B, H, T}, "mask1");
    vk张量 s1 = 建({B, H}, "sink1");
    vk张量 dy1 = 建({B, H, DV}, "dy1");
    vk张量 dq1 = 建({B, H, DK}, "dq1");
    vk张量 dk1 = 建({T, H, DK}, "dk1");
    vk张量 dv1 = 建({T, H, DV}, "dv1");
    vk张量 ds1 = 建({B, H}, "ds1");
    s1.上传(&sink, 4);
    Qt.上传(qh.data(), qh.size() * 4);

    const float scale = 1.0f / std::sqrt(float(DK));
    std::vector<float> 缓存K(T * DK, 0.f), 缓存V(T * DV, 0.f);
    std::vector<float> dqh(T * DK, 0.f), dKh(T * DK, 0.f), dVh(T * DV, 0.f), dsinkh(T, 0.f);
    std::vector<float> outh(T * DV, 0.f);
    std::vector<std::uint32_t> selh(T * TOPB, 0u);
    std::vector<float> masks(T * T, 0.f);
    std::vector<float> bs_gpu(T * NBMAX * DK, 0.f), bs_host(T * NBMAX * DK, 0.f);
    // 主机增量块和（仅用于和 GPU 全量重算对照）
    std::vector<double> 块和(NBMAX * DK, 0.0), 块数(NBMAX, 0.0);

    float bs_max_err = 0.0f;
    std::vector<float> 缓冲;
    for (std::uint32_t t = 0; t < T; ++t) {
        const std::uint32_t Nt = t + 1, NBt = (Nt + BLOCK - 1) / BLOCK;
        for (std::uint32_t d = 0; d < DK; ++d) 缓存K[t * DK + d] = kh[t * DK + d];
        for (std::uint32_t d = 0; d < DV; ++d) 缓存V[t * DV + d] = vh[t * DV + d];
        Kc.上传(缓存K.data(), 缓存K.size() * 4);
        Vc.上传(缓存V.data(), 缓存V.size() * 4);

        // 块摘要：GPU 全量重算 + 主机增量式
        {
            const std::uint32_t bi = t / BLOCK;
            for (std::uint32_t d = 0; d < DK; ++d) 块和[bi * DK + d] += double(kh[t * DK + d]);
            块数[bi] += 1.0;
            for (std::uint32_t b = 0; b < NBt; ++b)
                for (std::uint32_t d = 0; d < DK; ++d)
                    bs_host[(t * NBMAX + b) * DK + d] =
                        float(块和[b * DK + d] / 块数[b]);
        }
        推送块和 pcbs{B, H, Nt, DK, NBt, BLOCK, 0, 0};
        提交("ersa_block_sum", {Kc.取句柄(), BSt.取句柄()}, &pcbs, sizeof(pcbs), NBt, H, B);
        BSt.下载(&bs_gpu[t * NBMAX * DK], NBMAX * DK * 4);
        for (std::uint32_t i = 0; i < NBt * DK; ++i)
            bs_max_err = std::max(bs_max_err,
                std::fabs(bs_gpu[t * NBMAX * DK + i] - bs_host[t * NBMAX * DK + i]));

        // 路由 + 掩码（主机，与着色器同规则）
        std::vector<float> s(NBt);
        for (std::uint32_t i = 0; i < NBt; ++i) {
            float d = 0.0f;
            for (std::uint32_t k = 0; k < DK; ++k) d += qh[t * DK + k] * bs_gpu[(t * NBMAX + i) * DK + k];
            s[i] = d * scale / tau;
        }
        const std::uint32_t topn = std::min(TOPB, NBt);
        for (std::uint32_t kk = 0; kk < topn; ++kk) {
            float best = -3.402823e38f; std::uint32_t bi = 0u;
            for (std::uint32_t i = 0; i < NBt; ++i) if (s[i] > best) { best = s[i]; bi = i; }
            selh[t * TOPB + kk] = bi;
            s[bi] = -3.402823e38f;
        }
        for (std::uint32_t kk = 0; kk < topn; ++kk) {
            const std::uint32_t bi = selh[t * TOPB + kk];
            const std::uint32_t start = bi * BLOCK, end = std::min(start + BLOCK, Nt);
            for (std::uint32_t idx = start; idx < end; ++idx) masks[t * T + idx] = 1.0f;
        }
        const std::uint32_t wstart = (t + 1u > WINDOW) ? (t + 1u - WINDOW) : 0u;
        for (std::uint32_t idx = wstart; idx <= t; ++idx) masks[t * T + idx] = 1.0f;

        // 前向：按前缀长度派发（内核的 t = 序列位置），取第 t 行
        推送前向 pcf{B, H, Nt, Nt, DK, DV, NBt, BLOCK, TOPB, WINDOW, 0, 0, tau, sink};
        提交("ersa_attn_fused", {Qt.取句柄(), Kc.取句柄(), Vc.取句柄(), BSt.取句柄(), Ot.取句柄()},
             &pcf, sizeof(pcf), Nt, H, B);
        缓冲.assign(T * DV, 0.0f);
        Ot.下载(缓冲.data(), T * DV * 4);
        for (std::uint32_t d = 0; d < DV; ++d) outh[t * DV + d] = 缓冲[t * DV + d];

        // 反向（T=1，cache 长度 Nt）
        std::vector<float> qa(DK);
        for (std::uint32_t d = 0; d < DK; ++d) qa[d] = qh[t * DK + d] / tau;
        q1.上传(qa.data(), qa.size() * 4);
        m1.上传(&masks[t * T], T * 4);
        dy1.上传(&dOh[t * DV], DV * 4);
        推送反向 pcb{B, H, Nt, DK, DV, 0};
        提交("ersa_attn_bwd",
             {q1.取句柄(), Kc.取句柄(), Vc.取句柄(), m1.取句柄(), s1.取句柄(), dy1.取句柄(),
              dq1.取句柄(), dk1.取句柄(), dv1.取句柄(), ds1.取句柄()},
             &pcb, sizeof(pcb), B * H, 1, 1);
        缓冲.assign(DK, 0.0f);
        dq1.下载(缓冲.data(), DK * 4);
        for (std::uint32_t d = 0; d < DK; ++d) dqh[t * DK + d] = 缓冲[d] / tau;
        缓冲.assign(T * DK, 0.0f);
        dk1.下载(缓冲.data(), T * DK * 4);
        for (std::uint32_t i = 0; i < Nt * DK; ++i) dKh[i] += 缓冲[i];
        缓冲.assign(T * DV, 0.0f);
        dv1.下载(缓冲.data(), T * DV * 4);
        for (std::uint32_t i = 0; i < Nt * DV; ++i) dVh[i] += 缓冲[i];
        缓冲.assign(1, 0.0f);
        ds1.下载(缓冲.data(), 4);
        dsinkh[t] = 缓冲[0];

        std::uint32_t cnt = 0;
        for (std::uint32_t idx = 0; idx < Nt; ++idx) cnt += (masks[t * T + idx] > 0.5f) ? 1u : 0u;
        std::printf("t=%u N=%u NB=%u sel=[%u,%u] cand=%u out0=%.6f dsink=%.6f\n",
                    t, Nt, NBt, selh[t * TOPB], topn > 1 ? selh[t * TOPB + 1] : selh[t * TOPB],
                    cnt, outh[t * DV], dsinkh[t]);
    }
    std::printf("block_sum 增量 vs GPU 全量: max|err| = %.3e\n", bs_max_err);

    std::ofstream f(argv[2], std::ios::binary);
    if (!f.is_open()) { std::fprintf(stderr, "无法写导出文件\n"); return 1; }
    std::uint32_t hdr[10] = {T, B, H, T, BLOCK, TOPB, WINDOW, DK, DV, NBMAX};
    f.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
    auto 写 = [&](const void* p, std::size_t 个) { f.write(static_cast<const char*>(p), 个 * 4); };
    写(&tau, 1); 写(&sink, 1);
    写(qh.data(), qh.size());
    写(kh.data(), kh.size());
    写(vh.data(), vh.size());
    写(dOh.data(), dOh.size());
    for (std::uint32_t t = 0; t < T; ++t) {
        写(&outh[t * DV], DV);
        写(&selh[t * TOPB], TOPB);
        写(&masks[t * T], T);
        写(&bs_gpu[t * NBMAX * DK], NBMAX * DK);
        写(&bs_host[t * NBMAX * DK], NBMAX * DK);
    }
    写(dqh.data(), dqh.size());
    写(dKh.data(), dKh.size());
    写(dVh.data(), dVh.size());
    写(dsinkh.data(), dsinkh.size());
    float ds_tot = 0.0f;
    for (float x : dsinkh) ds_tot += x;
    写(&ds_tot, 1);
    f.close();

    调度器.停止();
    std::printf("M3b 对拍数据已写入 %s (dsink_total=%.6f)\n", argv[2], ds_tot);
    return 0;
}
