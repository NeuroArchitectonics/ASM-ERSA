// P4-M3 第 1/2 步：ERSA 层前向 + 块摘要 + 逐 query 反向（与 PyTorch 对拍）
// 前向：ersa_block_sum → ersa_attn_fused（T 个 query 一次派发）
// 反向：ersa_attn_bwd 逐 query 调用（B=1，dK/dV 主机侧跨 query 累加）
// tau 折进反向输入：q_arg = q/tau，dq_actual = dq_arg/tau
// 用法：train_m3.exe <spv目录> <导出.bin>
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
std::uint32_t g_lcg = 4242u;
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
static_assert(sizeof(推送块和) == 32, "block_sum push");
static_assert(sizeof(推送前向) == 56, "ersa fwd push");
static_assert(sizeof(推送反向) == 24, "ersa bwd push");
} // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "用法: %s <spv目录> <导出.bin>\n", argv[0]); return 2; }
    流水线调度器& 调度器 = 流水线调度器::获取();
    设备注册信息 注册; 注册.应用名 = "train_m3";
    阶段配置 图形; 阶段配置 计算{阶段配置::自动队列族, 0, 1}; 阶段配置 传输;
    try { 调度器.启动(注册, {图形, 计算, 传输}, argv[1]); }
    catch (const std::exception& e) { std::fprintf(stderr, "启动失败: %s\n", e.what()); return 1; }
    for (const char* 名 : {"ersa_block_sum", "ersa_attn_fused", "ersa_attn_bwd"}) {
        if (!调度器.存在算子(名)) { std::fprintf(stderr, "缺少算子: %s\n", 名); return 1; }
    }

    const std::uint32_t T = 8, B = 1, H = 1, N = 8, DK = 4, DV = 4;
    const std::uint32_t BLOCK = 2, NB = (N + BLOCK - 1) / BLOCK, TOPB = 2, WINDOW = 2;
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

    // ── 输入 ──
    std::vector<float> qh(T * DK), kh(N * DK), vh(N * DV), dOh(T * DV);
    for (auto& x : qh) x = 随机数();
    for (auto& x : kh) x = 随机数();
    for (auto& x : vh) x = 随机数();
    for (auto& x : dOh) x = 随机数();

    // ── 块摘要（GPU）与主机复算对照 ──
    vk张量 Kt = 建({B, H, N, DK}, "K"); Kt.上传(kh.data(), kh.size() * 4);
    vk张量 Vt = 建({B, H, N, DV}, "V"); Vt.上传(vh.data(), vh.size() * 4);
    vk张量 BSt = 建({B, H, NB, DK}, "bsum");
    推送块和 pcbs{B, H, N, DK, NB, BLOCK, 0, 0};
    提交("ersa_block_sum", {Kt.取句柄(), BSt.取句柄()}, &pcbs, sizeof(pcbs), NB, H, B);
    std::vector<float> bs_gpu(NB * DK), bs_host(NB * DK);
    BSt.下载(bs_gpu.data(), bs_gpu.size() * 4);
    for (std::uint32_t bi = 0; bi < NB; ++bi) {
        const std::uint32_t start = bi * BLOCK;
        const std::uint32_t end = std::min(start + BLOCK, N);
        const float inv = 1.0f / float(std::max<std::uint32_t>(1, end - start));
        for (std::uint32_t d = 0; d < DK; ++d) {
            float s = 0.0f;
            for (std::uint32_t idx = start; idx < end; ++idx) s += kh[idx * DK + d];
            bs_host[bi * DK + d] = s * inv;
        }
    }
    float bs_err = 0.0f;
    for (std::size_t i = 0; i < bs_gpu.size(); ++i)
        bs_err = std::max(bs_err, std::fabs(bs_gpu[i] - bs_host[i]));
    std::printf("block_sum GPU vs host: max|err| = %.3e\n", bs_err);

    // ── 前向 ──
    vk张量 Qt = 建({B, H, T, DK}, "q"); Qt.上传(qh.data(), qh.size() * 4);
    vk张量 Ot = 建({B, H, T, DV}, "out");
    推送前向 pcf{B, H, T, N, DK, DV, NB, BLOCK, TOPB, WINDOW, 0, 0, tau, sink};
    提交("ersa_attn_fused", {Qt.取句柄(), Kt.取句柄(), Vt.取句柄(), BSt.取句柄(), Ot.取句柄()},
         &pcf, sizeof(pcf), T, H, B);
    std::vector<float> outh(T * DV);
    Ot.下载(outh.data(), outh.size() * 4);

    // ── 主机复算候选集（与着色器同规则）──
    const float scale = 1.0f / std::sqrt(float(DK));
    std::vector<float> masks(T * N, 0.0f);
    std::vector<std::uint32_t> selh(T * TOPB, 0u);
    std::vector<std::uint32_t> cand_count(T, 0u);
    for (std::uint32_t t = 0; t < T; ++t) {
        std::vector<float> s(NB);
        for (std::uint32_t i = 0; i < NB; ++i) {
            float d = 0.0f;
            for (std::uint32_t k = 0; k < DK; ++k) d += qh[t * DK + k] * bs_gpu[i * DK + k];
            s[i] = d * scale / tau;
        }
        const std::uint32_t topn = std::min(TOPB, NB);
        for (std::uint32_t kk = 0; kk < topn; ++kk) {
            float best = -3.402823e38f; std::uint32_t bi = 0u;
            for (std::uint32_t i = 0; i < NB; ++i) if (s[i] > best) { best = s[i]; bi = i; }
            selh[t * TOPB + kk] = bi;
            s[bi] = -3.402823e38f;
        }
        for (std::uint32_t kk = 0; kk < topn; ++kk) {
            const std::uint32_t bi = selh[t * TOPB + kk];
            const std::uint32_t start = bi * BLOCK, end = std::min(start + BLOCK, N);
            for (std::uint32_t idx = start; idx < end; ++idx)
                if (idx <= t) masks[t * N + idx] = 1.0f;
        }
        const std::uint32_t wstart = (t + 1u > WINDOW) ? (t + 1u - WINDOW) : 0u;
        for (std::uint32_t idx = wstart; idx <= t; ++idx) masks[t * N + idx] = 1.0f;
        std::uint32_t cnt = 0;
        for (std::uint32_t idx = 0; idx < N; ++idx) cnt += (masks[t * N + idx] > 0.5f) ? 1u : 0u;
        cand_count[t] = cnt;
        std::printf("t=%u sel=[%u,%u] cand=%u\n", t, selh[t * TOPB], selh[t * TOPB + 1], cnt);
    }

    // ── 反向：逐 query 调用（pc.T = N），dK/dV 跨 query 累加 ──
    vk张量 q1 = 建({B, H, DK}, "q1");
    vk张量 m1 = 建({B, H, N}, "mask1");
    vk张量 s1 = 建({B, H}, "sink1");
    vk张量 dy1 = 建({B, H, DV}, "dy1");
    vk张量 dq1 = 建({B, H, DK}, "dq1");
    vk张量 dk1 = 建({N, H, DK}, "dk1");
    vk张量 dv1 = 建({N, H, DV}, "dv1");
    vk张量 ds1 = 建({B, H}, "ds1");
    s1.上传(&sink, 4);

    std::vector<float> dqh(T * DK, 0.0f), dKh(N * DK, 0.0f), dVh(N * DV, 0.0f), dsinkh(T, 0.0f);
    std::vector<float> 缓冲;
    for (std::uint32_t t = 0; t < T; ++t) {
        std::vector<float> qa(DK);
        for (std::uint32_t d = 0; d < DK; ++d) qa[d] = qh[t * DK + d] / tau;
        q1.上传(qa.data(), qa.size() * 4);
        m1.上传(&masks[t * N], N * 4);
        dy1.上传(&dOh[t * DV], DV * 4);

        推送反向 pcb{B, H, N, DK, DV, 0};
        提交("ersa_attn_bwd",
             {q1.取句柄(), Kt.取句柄(), Vt.取句柄(), m1.取句柄(), s1.取句柄(), dy1.取句柄(),
              dq1.取句柄(), dk1.取句柄(), dv1.取句柄(), ds1.取句柄()},
             &pcb, sizeof(pcb), B * H, 1, 1);

        缓冲.assign(DK, 0.0f);
        dq1.下载(缓冲.data(), DK * 4);
        for (std::uint32_t d = 0; d < DK; ++d) dqh[t * DK + d] = 缓冲[d] / tau;
        缓冲.assign(N * DK, 0.0f);
        dk1.下载(缓冲.data(), N * DK * 4);
        for (std::uint32_t i = 0; i < N * DK; ++i) dKh[i] += 缓冲[i];
        缓冲.assign(N * DV, 0.0f);
        dv1.下载(缓冲.data(), N * DV * 4);
        for (std::uint32_t i = 0; i < N * DV; ++i) dVh[i] += 缓冲[i];
        缓冲.assign(1, 0.0f);
        ds1.下载(缓冲.data(), 4);
        dsinkh[t] = 缓冲[0];
    }

    std::printf("out[0]= %.6f %.6f %.6f %.6f\n", outh[0], outh[1], outh[2], outh[3]);
    std::printf("dq[0]= %.6f %.6f %.6f %.6f | dsink[0]=%.6f\n",
                dqh[0], dqh[1], dqh[2], dqh[3], dsinkh[0]);
    std::printf("dK[0]= %.6f %.6f %.6f %.6f | dV[0]= %.6f %.6f %.6f %.6f\n",
                dKh[0], dKh[1], dKh[2], dKh[3], dVh[0], dVh[1], dVh[2], dVh[3]);

    // ── 导出（顺序即读序）──
    std::ofstream f(argv[2], std::ios::binary);
    if (!f.is_open()) { std::fprintf(stderr, "无法写导出文件\n"); return 1; }
    std::uint32_t hdr[10] = {T, B, H, N, BLOCK, TOPB, WINDOW, DK, DV, NB};
    f.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
    auto 写 = [&](const void* p, std::size_t 个) { f.write(static_cast<const char*>(p), 个 * 4); };
    写(&tau, 1); 写(&sink, 1);
    写(qh.data(), qh.size());
    写(kh.data(), kh.size());
    写(vh.data(), vh.size());
    写(dOh.data(), dOh.size());
    写(bs_gpu.data(), bs_gpu.size());
    写(bs_host.data(), bs_host.size());
    写(outh.data(), outh.size());
    写(masks.data(), masks.size());
    写(selh.data(), selh.size());
    写(dqh.data(), dqh.size());
    写(dKh.data(), dKh.size());
    写(dVh.data(), dVh.size());
    写(dsinkh.data(), dsinkh.size());
    f.close();

    调度器.停止();
    std::printf("M3 对拍数据已写入 %s\n", argv[2]);
    return 0;
}
