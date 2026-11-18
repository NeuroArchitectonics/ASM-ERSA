// 主模型回召 smoke：直接测 cpp/glsl 的 asm_step_eta 槽位读和 ersa_attn_step
// 的块路由 + 精确 K/V 回召。VK-only，不接 PyTorch。
#include "流水线调度器.hpp"
#include "vk张量.hpp"
#include "cpp/spv/asm_step_eta.spv.hpp"
#include "cpp/spv/ersa_attn_step.spv.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vk张量系统;

namespace {
std::uint32_t R = 20261118u;
float rnd() {
    R = R * 1664525u + 1013904223u;
    return (float((R >> 8) & 0xFFFFFFu) / 16777216.0f) * 2.0f - 1.0f;
}
void norm(float* x, std::uint32_t D) {
    double s = 0.0;
    for (std::uint32_t d = 0; d < D; ++d) s += double(x[d]) * x[d];
    const float inv = float(1.0 / std::sqrt(s + 1e-12));
    for (std::uint32_t d = 0; d < D; ++d) x[d] *= inv;
}
vk张量 mk(std::vector<std::uint32_t> s, 张量类型 t, const char* n) {
    return vk张量(张量布局::连续(std::move(s)), t, n);
}
struct AsmPC {
    std::uint32_t B, H, M, DK, DV, R, pad0, pad1;
    float tau, tau_w, rho, lam, null_beta, pad2;
};
struct ErsaPC {
    std::uint32_t B, H, T, N, DK, DV, NB, BLOCK, TOPB, WINDOW, QOFF, NBMAX;
    float tau, sink;
};
} // namespace

int main(int argc, char** argv) {
    const std::string dir = (argc >= 2) ? argv[1] : "cpp/spv";
    流水线调度器& sched = 流水线调度器::获取();
    设备注册信息 reg; reg.应用名 = "recall_main";
    阶段配置 g; 阶段配置 c{阶段配置::自动队列族, 0, 1}; 阶段配置 t;
    try { sched.启动(reg, {g, c, t}, dir); }
    catch (const std::exception& e) { std::fprintf(stderr, "启动失败: %s\n", e.what()); return 1; }
    std::printf("设备: %s\n", sched.获取设备().名称());
    const float tau_arg = (argc >= 3) ? std::atof(argv[2]) : 0.0f;

    auto run = [&](const char* name, std::vector<张量句柄> hs, const void* pc,
                   std::uint32_t pcb, std::uint32_t gx, std::uint32_t gy) {
        const std::uint64_t id = sched.提交自定义算子(name, std::move(hs), pc, pcb,
                                                      派发规格{gx, gy, 1});
        sched.等待(id);
        if (!sched.已完成(id))
            throw std::runtime_error(std::string(name) + " 失败: " + sched.查询失败原因(id));
    };

    // ── ASM：写入目标槽位后，用 q=K[target] 精确读回 V[target] ──
    {
        const std::uint32_t B = 1, H = 4, M = 16, Rr = 4, DK = 8, DV = 8;
        std::vector<float> K(B * H * M * DK), V(B * H * M * DV), beta(B * H * M, 0.0f);
        std::vector<float> gamma(M, 1.0f);
        std::vector<float> q(B * H * DK), k(B * H * DK), v(B * H * DV), eta(B, 0.5f);
        std::vector<std::uint32_t> targets(H);
        for (std::uint32_t h = 0; h < H; ++h) {
            std::uint32_t target = 3u + h * 2u;
            targets[h] = target;
            std::vector<float> needle(DK);
            for (auto& x : needle) x = rnd();
            norm(needle.data(), DK);
            for (std::uint32_t m = 0; m < M; ++m) {
                for (std::uint32_t d = 0; d < DK; ++d)
                    K[(h * M + m) * DK + d] = (m == target) ? needle[d] : rnd();
                norm(&K[(h * M + m) * DK], DK);
                for (std::uint32_t d = 0; d < DV; ++d)
                    V[(h * M + m) * DV + d] = (m == target && d == 0) ? 1.0f : 0.0f;
            }
            for (std::uint32_t d = 0; d < DK; ++d) {
                q[h * DK + d] = needle[d];
                k[h * DK + d] = rnd();
                norm(&k[h * DK], DK);
            }
            for (std::uint32_t d = 0; d < DV; ++d) v[h * DV + d] = rnd();
        }
        vk张量 qd = mk({B, H, DK}, 张量类型::浮点32, "asm.q");
        vk张量 kd = mk({B, H, DK}, 张量类型::浮点32, "asm.k");
        vk张量 vd = mk({B, H, DV}, 张量类型::浮点32, "asm.v");
        vk张量 etad = mk({B}, 张量类型::浮点32, "asm.eta");
        vk张量 Kd = mk({B, H, M, DK}, 张量类型::浮点32, "asm.K");
        vk张量 Vd = mk({B, H, M, DV}, 张量类型::浮点32, "asm.V");
        vk张量 Bd = mk({B, H, M}, 张量类型::浮点32, "asm.beta");
        vk张量 Gd = mk({M}, 张量类型::浮点32, "asm.gamma");
        vk张量 od = mk({B, H * DV}, 张量类型::浮点32, "asm.o");
        vk张量 dbg = mk({B, H, 3 * Rr}, 张量类型::浮点32, "asm.dbg");
        qd.上传(q.data(), q.size() * 4); kd.上传(k.data(), k.size() * 4);
        vd.上传(v.data(), v.size() * 4); etad.上传(eta.data(), 4);
        Kd.上传(K.data(), K.size() * 4); Vd.上传(V.data(), V.size() * 4);
        Bd.上传(beta.data(), beta.size() * 4); Gd.上传(gamma.data(), gamma.size() * 4);
        const float tau = (tau_arg > 0.0f) ? tau_arg : 0.005f;
        const AsmPC pc{B, H, M, DK, DV, Rr, 0, 0, tau, tau, 0.9f, 0.01f, -2.0f, 0.0f};
        run("asm_step_eta", {qd.取句柄(), kd.取句柄(), vd.取句柄(), Kd.取句柄(),
                            Vd.取句柄(), Bd.取句柄(), Gd.取句柄(), od.取句柄(),
                            etad.取句柄(), dbg.取句柄()},
            &pc, sizeof(pc), H, 1);
        std::vector<float> oh(B * H * DV);
        od.下载(oh.data(), oh.size() * 4);
        std::printf("[ASM recall tau=%.3g] target slots:", tau);
        for (std::uint32_t h = 0; h < H; ++h) {
            std::printf(" %u", targets[h]);
            std::printf("(mass=%.3f)", oh[h * DV]);
        }
        std::printf("\n");
    }

    // ── ERSA：目标块外置于窗口，q 对齐目标 K；检查候选块是否回召目标 V ──
    {
        const std::uint32_t B = 1, H = 4, T = 64, N = 64, DK = 8, DV = 8;
        const std::uint32_t BLOCK = 8, NB = 8, TOPB = 2, WINDOW = 8, QOFF = 63, NBMAX = 8;
        std::vector<float> q(H * T * DK), k(H * T * DK), v(H * T * DV);
        std::vector<float> bs(H * NBMAX * DK, 0.0f);
        std::vector<std::uint32_t> target(H);
        for (std::uint32_t h = 0; h < H; ++h) {
            const std::uint32_t tg = 10u + h;
            target[h] = tg;
            std::vector<float> needle(DK);
            for (auto& x : needle) x = rnd();
            norm(needle.data(), DK);
            for (std::uint32_t t = 0; t < T; ++t) {
                for (std::uint32_t d = 0; d < DK; ++d) {
                    float kk = (t == tg) ? needle[d] : rnd();
                    k[(h * T + t) * DK + d] = kk;
                }
                norm(&k[(h * T + t) * DK], DK);
                for (std::uint32_t d = 0; d < DK; ++d)
                    q[(h * T + t) * DK + d] = rnd();
                norm(&q[(h * T + t) * DK], DK);
                for (std::uint32_t d = 0; d < DV; ++d)
                    v[(h * T + t) * DV + d] = 0.0f;
            }
            for (std::uint32_t d = 0; d < DK; ++d) q[(h * T + QOFF) * DK + d] = needle[d];
            v[(h * T + tg) * DV] = 1.0f;
            for (std::uint32_t bi = 0; bi < NB; ++bi) {
                for (std::uint32_t d = 0; d < DK; ++d) {
                    bs[(h * NBMAX + bi) * DK + d] =
                        (bi == (tg / BLOCK)) ? needle[d] : 0.05f * rnd();
                }
            }
        }
        vk张量 qd = mk({H, T, DK}, 张量类型::浮点32, "ersa.q");
        vk张量 kd = mk({H, T, DK}, 张量类型::浮点32, "ersa.k");
        vk张量 vd = mk({H, T, DV}, 张量类型::浮点32, "ersa.v");
        vk张量 bd = mk({H, NBMAX, DK}, 张量类型::浮点32, "ersa.bs");
        vk张量 od = mk({H * DV}, 张量类型::浮点32, "ersa.o");
        qd.上传(q.data(), q.size() * 4); kd.上传(k.data(), k.size() * 4);
        vd.上传(v.data(), v.size() * 4); bd.上传(bs.data(), bs.size() * 4);
        const ErsaPC pc{B, H, T, N, DK, DV, NB, BLOCK, TOPB, WINDOW, QOFF, NBMAX,
                           (tau_arg > 0.0f) ? tau_arg : 0.005f, -30.0f};
        run("ersa_attn_step", {qd.取句柄(), kd.取句柄(), vd.取句柄(),
                              bd.取句柄(), od.取句柄()},
            &pc, sizeof(pc), H, 1);
        std::vector<float> oh(H * DV);
        od.下载(oh.data(), oh.size() * 4);
        std::printf("[ERSA recall tau=%.3g] target tokens:", (tau_arg > 0.0f) ? tau_arg : 0.005f);
        for (std::uint32_t h = 0; h < H; ++h)
            std::printf(" %u(mass=%.3f)", target[h], oh[h * DV]);
        std::printf("\n");
    }

    sched.停止();
    return 0;
}
