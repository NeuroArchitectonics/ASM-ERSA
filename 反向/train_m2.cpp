// P4-M2：单层 ASM 的 BPTT 训练器（前向 asm_step_fused，反向 asm_step_bwd 串联）
// B=1,H=1；每步前快照状态供反向使用；idx 在主机按 top-r 计算
// 用法：train_m2.exe <spv目录> <导出.bin>
#include "流水线调度器.hpp"
#include "vk张量.hpp"
#include "asm_step_fused.spv.hpp"
#include "asm_step_bwd.spv.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace vk张量系统;

namespace {
std::uint32_t g_lcg = 777u;
float 随机数() {
    g_lcg = g_lcg * 1664525u + 1013904223u;
    return (static_cast<float>((g_lcg >> 8) & 0xFFFFFFu) / 16777216.0f) * 2.0f - 1.0f;
}
struct 推送前向 {
    std::uint32_t B, H, M, DK, DV, R, pad0, pad1;
    float tau, tau_w, eta, rho, lam, null_beta;
};
struct 推送反向 {
    std::uint32_t B, H, M, R, DK, DV;
    float tau, tau_w, rho, lam, beta_null;
};
} // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "用法: %s <spv目录> <导出.bin>\n", argv[0]); return 2; }
    流水线调度器& 调度器 = 流水线调度器::获取();
    设备注册信息 注册; 注册.应用名 = "train_m2";
    阶段配置 图形; 阶段配置 计算{阶段配置::自动队列族, 0, 1}; 阶段配置 传输;
    try { 调度器.启动(注册, {图形, 计算, 传输}, argv[1]); }
    catch (const std::exception& e) { std::fprintf(stderr, "启动失败: %s\n", e.what()); return 1; }

    const uint32_t B = 1, H = 1, M = 8, R = 3, DK = 4, DV = 4, T = 8;
    const float tau = 1.0f, tau_w = 1.0f, rho = 0.9f, lam = 0.01f, bnull = -2.0f;

    auto 建 = [](std::vector<uint32_t> 形状, 张量类型 类型, const char* 名) {
        return vk张量(张量布局::连续(std::move(形状)), 类型, 名);
    };
    auto 提交 = [&](const char* op, std::vector<张量句柄> 句柄, const void* pc,
                   std::uint32_t pcB, std::uint32_t threads) {
        uint64_t id = 调度器.提交自定义算子(op, std::move(句柄), pc, pcB, {(threads + 255u) / 256u, 1u, 1u});
        调度器.等待(id);
        if (!调度器.已完成(id))
            throw std::runtime_error(std::string("算子失败: ") + op + " | " + 调度器.查询失败原因(id));
    };

    // 输入（固定随机）：q,k,v,dO,eta,gamma；初始状态 K,V,beta
    std::vector<std::vector<float>> qh(T, std::vector<float>(DK)), kh(T, std::vector<float>(DK)),
        vh(T, std::vector<float>(DV)), dOh(T, std::vector<float>(DV));
    std::vector<float> etah(T), gh(M), K0(M * DK), V0(M * DV), B0(M);
    for (uint32_t t = 0; t < T; ++t) {
        for (auto& x : qh[t]) x = 随机数();
        for (auto& x : kh[t]) x = 随机数();
        for (auto& x : vh[t]) x = 随机数();
        for (auto& x : dOh[t]) x = 随机数();
        etah[t] = 0.1f + 0.01f * float(t);
    }
    for (auto& x : gh) x = 0.9f + 0.01f * 随机数();
    for (uint32_t m = 0; m < M; ++m) {
        float n = 0.0f;
        for (uint32_t i = 0; i < DK; ++i) { K0[m*DK+i] = 随机数(); n += K0[m*DK+i]*K0[m*DK+i]; }
        n = std::sqrt(n);
        for (uint32_t i = 0; i < DK; ++i) K0[m*DK+i] /= n;
    }
    for (auto& x : V0) x = 随机数();
    for (auto& x : B0) x = 0.05f * 随机数();

    vk张量 gamma = 建({M}, 张量类型::浮点32, "gamma"); gamma.上传(gh.data(), gh.size()*4);
    vk张量 Kt = 建({B,H,M,DK}, 张量类型::浮点32, "K"); Kt.上传(K0.data(), K0.size()*4);
    vk张量 Vt = 建({B,H,M,DV}, 张量类型::浮点32, "V"); Vt.上传(V0.data(), V0.size()*4);
    vk张量 Bt = 建({B,H,M}, 张量类型::浮点32, "beta"); Bt.上传(B0.data(), B0.size()*4);

    std::vector<vk张量> qs, ks, vs, dOs, outs, idxt, snapK, snapV, snapB, etas;
    std::vector<std::vector<std::int32_t>> idxh(T);
    vk张量 dbg = 建({B,H,3u*R}, 张量类型::浮点32, "dbg");
    std::vector<float> dbg0(3u*R*B*H, 0.0f);
    std::vector<std::vector<float>> outh(T, std::vector<float>(DV)),
        Kafter(T, std::vector<float>(M * DK)), Vafter(T, std::vector<float>(M * DV)),
        Bafter(T, std::vector<float>(M));
    for (uint32_t t = 0; t < T; ++t) {
        vk张量 q = 建({B,H,DK}, 张量类型::浮点32, "q"); q.上传(qh[t].data(), DK*4);
        vk张量 k = 建({B,H,DK}, 张量类型::浮点32, "k"); k.上传(kh[t].data(), DK*4);
        vk张量 v = 建({B,H,DV}, 张量类型::浮点32, "v"); v.上传(vh[t].data(), DV*4);
        vk张量 dO = 建({B,H,DV}, 张量类型::浮点32, "dO"); dO.上传(dOh[t].data(), DV*4);
        vk张量 eta = 建({B}, 张量类型::浮点32, "eta"); eta.上传(&etah[t], 4);
        qs.push_back(std::move(q)); ks.push_back(std::move(k)); vs.push_back(std::move(v));
        dOs.push_back(std::move(dO)); etas.push_back(std::move(eta));
        outs.push_back(建({B,H,DV}, 张量类型::浮点32, "out"));
        snapK.push_back(建({B,H,M,DK}, 张量类型::浮点32, "Kold"));
        snapV.push_back(建({B,H,M,DV}, 张量类型::浮点32, "Vold"));
        snapB.push_back(建({B,H,M}, 张量类型::浮点32, "Bold"));
        idxt.push_back(vk张量(张量布局::连续({B,H,R}), 张量类型::整数32, "idx"));
        idxh[t].assign(R, 0);
    }

    // ── 前向：每步快照 → asm_step_fused（状态原地更新）──
    for (uint32_t t = 0; t < T; ++t) {
        Kt.拷贝到(snapK[t]); Vt.拷贝到(snapV[t]); Bt.拷贝到(snapB[t]);
        // 主机算 idx（top-r，按 q·K/τ + β）
        std::vector<float> Kh(M * DK), Bh(M);
        snapK[t].下载(Kh.data(), Kh.size()*4); snapB[t].下载(Bh.data(), Bh.size()*4);
        std::vector<std::pair<float,int>> sc(M);
        for (uint32_t m = 0; m < M; ++m) {
            float s = 0.0f;
            for (uint32_t i = 0; i < DK; ++i) s += qh[t][i] * Kh[m*DK+i];
            sc[m] = {s / tau + Bh[m], int(m)};
        }
        std::partial_sort(sc.begin(), sc.begin() + R, sc.end(), std::greater<>());
        for (uint32_t j = 0; j < R; ++j) idxh[t][j] = sc[j].second;
        idxt[t].上传(idxh[t].data(), R*4);

        推送前向 pc{B,H,M,DK,DV,R,0,0, tau,tau_w,etah[t],rho,lam,bnull};
        提交("asm_step_fused",
             {qs[t].取句柄(), ks[t].取句柄(), vs[t].取句柄(), Kt.取句柄(), Vt.取句柄(),
              Bt.取句柄(), gamma.取句柄(), outs[t].取句柄(), dbg.取句柄()},
             &pc, sizeof(pc), B*H);
        if (t == 0) dbg.下载(dbg0.data(), dbg0.size()*4);
        outs[t].下载(outh[t].data(), DV*4);
        Kt.下载(Kafter[t].data(), Kafter[t].size()*4);
        Vt.下载(Vafter[t].data(), Vafter[t].size()*4);
        Bt.下载(Bafter[t].data(), Bafter[t].size()*4);
    }

    // ── 反向：t=T-1..0，梯度累计（dKnew/dVnew/dβnew ← 下一步的 dKold/dVold/dBold）──
    vk张量 dKn = 建({B,H,M,DK}, 张量类型::浮点32, "dKn");
    vk张量 dVn = 建({B,H,M,DV}, 张量类型::浮点32, "dVn");
    vk张量 dBn = 建({B,H,M}, 张量类型::浮点32, "dBn");
    std::vector<float> 零K(M*DK, 0.f), 零V(M*DV, 0.f), 零B(M, 0.f);
    dKn.上传(零K.data(), 零K.size()*4); dVn.上传(零V.data(), 零V.size()*4); dBn.上传(零B.data(), 零B.size()*4);

    std::vector<std::vector<float>> dqh(T, std::vector<float>(DK)), dkh(T, std::vector<float>(DK)),
        dvh(T, std::vector<float>(DV));
    vk张量 dq = 建({B,H,DK}, 张量类型::浮点32, "dq");
    vk张量 dk = 建({B,H,DK}, 张量类型::浮点32, "dk");
    vk张量 dv = 建({B,H,DV}, 张量类型::浮点32, "dv");
    vk张量 deta = 建({B}, 张量类型::浮点32, "deta");
    vk张量 dKo = 建({B,H,M,DK}, 张量类型::浮点32, "dKo");
    vk张量 dVo = 建({B,H,M,DV}, 张量类型::浮点32, "dVo");
    vk张量 dBo = 建({B,H,M}, 张量类型::浮点32, "dBo");
    vk张量 dgamma = 建({M}, 张量类型::浮点32, "dgamma");
    vk张量 dbn = 建({B,H}, 张量类型::浮点32, "dbn");

    vk张量 dbg2 = 建({R + DK + DV}, 张量类型::浮点32, "dbg2");
    std::vector<float> dbg2h(R + DK + DV, 0.0f);
    for (int t = int(T) - 1; t >= 0; --t) {
        if (t == int(T) - 1) {
            std::vector<float> 检查(4, 0.0f);
            dKn.下载(检查.data(), 4 * sizeof(float));
            std::printf("t=%d 调用前 dKn[0..3] = %.6f %.6f %.6f %.6f\n",
                        t, 检查[0], 检查[1], 检查[2], 检查[3]);
        }
        推送反向 pc{B,H,M,R,DK,DV, tau,tau_w,rho,lam,bnull};
        提交("asm_step_bwd",
             {qs[t].取句柄(), ks[t].取句柄(), vs[t].取句柄(), etas[t].取句柄(),
              snapK[t].取句柄(), snapV[t].取句柄(), snapB[t].取句柄(), gamma.取句柄(),
              dOs[t].取句柄(), dKn.取句柄(), dVn.取句柄(), dBn.取句柄(), idxt[t].取句柄(),
              dq.取句柄(), dk.取句柄(), dv.取句柄(), deta.取句柄(),
              dKo.取句柄(), dVo.取句柄(), dBo.取句柄(), dgamma.取句柄(), dbn.取句柄(),
              dbg2.取句柄()},
             &pc, sizeof(pc), B*H);
        if (t == int(T) - 1) {   // 最后一步：dKn=0，ds_p/dk_acc/du 应全 0
            dbg2.下载(dbg2h.data(), dbg2h.size()*4);
            std::printf("t=%d bwd ds_p= %.6f %.6f %.6f | dk_acc= %.6f %.6f %.6f %.6f | du0= %.6f %.6f %.6f %.6f\n",
                        t, dbg2h[0], dbg2h[1], dbg2h[2],
                        dbg2h[R], dbg2h[R+1], dbg2h[R+2], dbg2h[R+3],
                        dbg2h[R+DK], dbg2h[R+DK+1], dbg2h[R+DK+2], dbg2h[R+DK+3]);
            std::vector<float> 出q(DK), 出k(DK), 出v(DV), 出eta(1);
            dq.下载(出q.data(), DK*4); dk.下载(出k.data(), DK*4);
            dv.下载(出v.data(), DV*4); deta.下载(出eta.data(), 4);
            std::printf("t=%d 输出 dq= %.6f %.6f %.6f %.6f | dk= %.6f %.6f %.6f %.6f | dv= %.6f %.6f %.6f %.6f | deta=%.6f\n",
                        t, 出q[0], 出q[1], 出q[2], 出q[3],
                        出k[0], 出k[1], 出k[2], 出k[3],
                        出v[0], 出v[1], 出v[2], 出v[3], 出eta[0]);
        }
        dq.下载(dqh[t].data(), DK*4); dk.下载(dkh[t].data(), DK*4); dv.下载(dvh[t].data(), DV*4);
        if (t > 0) {   // 累计到上一步：dKn←dKo 等
            std::vector<float> bK(M*DK), bV(M*DV), bB(M);
            dKo.下载(bK.data(), bK.size()*4); dVo.下载(bV.data(), bV.size()*4); dBo.下载(bB.data(), bB.size()*4);
            dKn.上传(bK.data(), bK.size()*4); dVn.上传(bV.data(), bV.size()*4); dBn.上传(bB.data(), bB.size()*4);
        }
    }

    // ── 导出 ──
    std::ofstream f(argv[2], std::ios::binary);
    if (!f.is_open()) { std::fprintf(stderr, "无法写导出文件\n"); return 1; }
    uint32_t hdr[7] = {T, B, H, M, R, DK, DV};
    f.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
    auto 写I = [&](const std::vector<int32_t>& v) { f.write(reinterpret_cast<const char*>(v.data()), v.size()*4); };
    (void)写I;
    auto 写 = [&](const std::vector<float>& v) { f.write(reinterpret_cast<const char*>(v.data()), v.size()*4); };
    写(K0); 写(V0); 写(B0); 写(gh);
    for (uint32_t t = 0; t < T; ++t) { 写(qh[t]); 写(kh[t]); 写(vh[t]); 写(dOh[t]); }
    写(etah);
    for (uint32_t t = 0; t < T; ++t) { 写(dqh[t]); 写(dkh[t]); 写(dvh[t]); }
    std::vector<float> fK(M*DK), fV(M*DV), fB(M);
    dKo.下载(fK.data(), fK.size()*4); dVo.下载(fV.data(), fV.size()*4); dBo.下载(fB.data(), fB.size()*4);
    std::printf("dq = %.4f %.4f %.4f %.4f\n", dqh[0][0], dqh[0][1], dqh[0][2], dqh[0][3]);
    // 主机闭式（T=1，dKn=dVn=dBn=0 → 只剩读路径）
    {
        std::vector<float> s(R), a(R + 1), da(R), ds(R);
        float mx = bnull;
        for (uint32_t j = 0; j < R; ++j) {
            float d = 0.0f;
            for (uint32_t i = 0; i < DK; ++i) d += qh[0][i] * K0[idxh[0][j] * DK + i];
            s[j] = d / tau + B0[idxh[0][j]];
            mx = std::max(mx, s[j]);
        }
        float den = std::exp(bnull - mx);
        for (uint32_t j = 0; j < R; ++j) den += std::exp(s[j] - mx);
        for (uint32_t j = 0; j < R; ++j) a[j] = std::exp(s[j] - mx) / den;
        float dotada = 0.0f;
        for (uint32_t j = 0; j < R; ++j) {
            float d = 0.0f;
            for (uint32_t i = 0; i < DV; ++i) d += dOh[0][i] * V0[idxh[0][j] * DV + i];
            da[j] = d; dotada += a[j] * d;
        }
        for (uint32_t j = 0; j < R; ++j) ds[j] = a[j] * (da[j] - dotada);
        float dq_exp[4] = {0, 0, 0, 0};
        for (uint32_t j = 0; j < R; ++j)
            for (uint32_t i = 0; i < DK; ++i)
                dq_exp[i] += ds[j] * K0[idxh[0][j] * DK + i] / tau;
        std::printf("dq_exp= %.4f %.4f %.4f %.4f  (主机闭式)\n", dq_exp[0], dq_exp[1], dq_exp[2], dq_exp[3]);
        uint32_t m0 = uint32_t(idxh[0][0]);
        std::printf("slot%u: a=%.4f ds=%.4f | dV_exp=%.4f dV_vk=%.4f | dK_exp=%.4f dK_vk=%.4f | dB_exp=%.4f dB_vk=%.4f\n",
                    m0, a[0], ds[0],
                    a[0] * dOh[0][0], fV[m0 * DV],
                    ds[0] * qh[0][0] / tau, fK[m0 * DK],
                    ds[0], fB[m0]);
    }
    std::printf("dK0= %.4f %.4f %.4f %.4f\n", fK[0], fK[1], fK[2], fK[3]);
    std::printf("dV0= %.4f %.4f %.4f %.4f\n", fV[0], fV[1], fV[2], fV[3]);
    std::printf("dB0= %.4f %.4f %.4f %.4f\n", fB[0], fB[1], fB[2], fB[3]);
    std::printf("期望: dq~0, dK0~0, dV0(选中)=0.3333, dB0~0, dk=dv=0\n");
    写(fK); 写(fV); 写(fB);
    for (uint32_t t = 0; t < T; ++t) {   // 前向逐步：out / K_after / V_after / β_after
        写(outh[t]); 写(Kafter[t]); 写(Vafter[t]); 写(Bafter[t]);
    }
    写(dbg0);   // [B,H,3R] = sel_q / sel_k / p（t=0）
    f.close();

    调度器.停止();
    std::printf("M2 对拍数据已写入 %s\n", argv[2]);
    return 0;
}
