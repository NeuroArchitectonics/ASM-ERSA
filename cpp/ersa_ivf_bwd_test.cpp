// VK-only IVF backward check: rebuild the candidate set with centroid/top-B/
// member/window and compare against the existing block-routing backward on
// candidate sets engineered to be identical. No PyTorch/CPU math.
#include "流水线调度器.hpp"
#include "vk张量.hpp"
#include "cpp/spv/ersa_attn_bwd_query.spv.hpp"
#include "cpp/spv/ersa_ivf_attn_bwd_query.spv.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vk张量系统;

namespace {
struct BlockBwdPC {
    std::uint32_t B, H, T, N, DK, DV, NB, BLOCK, TOPB, WINDOW, QOFF, NBMAX;
    float tau, sink, pad0, pad1;
};
struct IvfBwdPC {
    std::uint32_t B, H, T, N, DK, DV, C, LMAX, TOPB, WINDOW, QPOS, pad1;
    float tau, sink;
};

std::uint32_t seed = 20261118u;
float rnd() {
    seed = seed * 1664525u + 1013904223u;
    return (float((seed >> 8) & 0xFFFFFFu) / 16777216.0f) * 2.0f - 1.0f;
}
void norm(float* x, std::uint32_t d) {
    double s = 0.0;
    for (std::uint32_t i = 0; i < d; ++i) s += double(x[i]) * x[i];
    const float inv = float(1.0 / std::sqrt(s + 1e-12));
    for (std::uint32_t i = 0; i < d; ++i) x[i] *= inv;
}
vk张量 mk(std::vector<std::uint32_t> s, 张量类型 t, const char* n) {
    return vk张量(张量布局::连续(std::move(s)), t, n);
}
void fill0(vk张量& t) {
    std::vector<std::uint8_t> z(t.字节大小(), 0);
    t.上传(z.data(), z.size());
}
float max_abs(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 1e30f;
    float m = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i)
        m = std::max(m, static_cast<float>(std::fabs(a[i] - b[i])));
    return m;
}
} // namespace

int main(int argc, char** argv) {
    const std::string spv = (argc >= 2) ? argv[1] : "cpp/spv";
    流水线调度器& sched = 流水线调度器::获取();
    设备注册信息 reg;
    reg.应用名 = "ersa_ivf_bwd_test";
    阶段配置 g;
    阶段配置 c{阶段配置::自动队列族, 0, 1};
    阶段配置 t;
    try {
        sched.启动(reg, {g, c, t}, spv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "启动失败: %s\n", e.what());
        return 1;
    }
    std::printf("设备: %s | 算子: %zu\n", sched.获取设备().名称(), sched.算子数量());

    auto run = [&](const char* name, std::vector<张量句柄> hs, const void* pc,
                   std::uint32_t bytes, std::uint32_t gx, std::uint32_t gy,
                   std::uint32_t gz) {
        const std::uint64_t id = sched.提交自定义算子(
            name, std::move(hs), pc, bytes, 派发规格{gx, gy, gz});
        sched.等待(id);
        if (!sched.已完成(id))
            throw std::runtime_error(std::string(name) + " 失败: " +
                                     sched.查询失败原因(id));
    };

    const std::uint32_t H = 2, N = 64, DK = 64, DV = 64, C = 2, LMAX = 32;
    const std::uint32_t QPOS = N - 1, TOPB = 1;
    const float tau = 1.0f, sink = -4.0f;

    std::vector<float> q(H * DK), k(H * N * DK), v(H * N * DV), dy(H * DV);
    std::vector<float> cent(H * C * DK), bs(H * C * DK);
    std::vector<float> qblk(H * N * DK, 0.0f);
    for (auto& x : k) x = rnd();
    for (auto& x : v) x = rnd();
    for (auto& x : dy) x = rnd();
    for (std::uint32_t h = 0; h < H; ++h) {
        for (std::uint32_t n = 0; n < N; ++n) norm(&k[(h * N + n) * DK], DK);
        for (std::uint32_t d = 0; d < DK; ++d) q[h * DK + d] = rnd();
        norm(&q[h * DK], DK);
        for (std::uint32_t d = 0; d < DK; ++d) qblk[(h * N + QPOS) * DK + d] = q[h * DK + d];
    }

    std::vector<std::int32_t> member(H * C * LMAX, 0), counts(H * C, 0);
    for (std::uint32_t h = 0; h < H; ++h) {
        counts[h * C + 0] = LMAX;
        counts[h * C + 1] = LMAX;
        for (std::uint32_t j = 0; j < LMAX; ++j) {
            member[(h * C + 0) * LMAX + j] = std::int32_t(j);
            member[(h * C + 1) * LMAX + j] = std::int32_t(LMAX + j);
        }
    }

    vk张量 qb = mk({H, N, DK}, 张量类型::浮点32, "bwd.qblk");
    vk张量 qi = mk({H, 1, DK}, 张量类型::浮点32, "bwd.qivf");
    vk张量 kd = mk({H, N, DK}, 张量类型::浮点32, "bwd.k");
    vk张量 vd = mk({H, N, DV}, 张量类型::浮点32, "bwd.v");
    vk张量 dyb = mk({H, DV}, 张量类型::浮点32, "bwd.dyblk");
    vk张量 dyi = mk({H, 1, DV}, 张量类型::浮点32, "bwd.dyivf");
    vk张量 bsd = mk({H, C, DK}, 张量类型::浮点32, "bwd.bs");
    vk张量 ctd = mk({H, C, DK}, 张量类型::浮点32, "bwd.cent");
    vk张量 mid = mk({H, C, LMAX}, 张量类型::整数32, "bwd.member");
    vk张量 cnd = mk({H, C}, 张量类型::整数32, "bwd.counts");
    vk张量 dqb = mk({H, N, DK}, 张量类型::浮点32, "bwd.dqblk");
    vk张量 dqi = mk({H, 1, DK}, 张量类型::浮点32, "bwd.dqivf");
    vk张量 dkb = mk({H, N, DK}, 张量类型::浮点32, "bwd.dkblk");
    vk张量 dki = mk({H, N, DK}, 张量类型::浮点32, "bwd.dkivf");
    vk张量 dvb = mk({H, N, DV}, 张量类型::浮点32, "bwd.dvblk");
    vk张量 dvi = mk({H, N, DV}, 张量类型::浮点32, "bwd.dvivf");
    vk张量 dsb = mk({H}, 张量类型::浮点32, "bwd.dsblk");
    vk张量 dsi = mk({H}, 张量类型::浮点32, "bwd.dsivf");

    qb.上传(qblk.data(), qblk.size() * 4);
    qi.上传(q.data(), q.size() * 4);
    kd.上传(k.data(), k.size() * 4);
    vd.上传(v.data(), v.size() * 4);
    dyb.上传(dy.data(), dy.size() * 4);
    dyi.上传(dy.data(), dy.size() * 4);
    mid.上传(member.data(), member.size() * 4);
    cnd.上传(counts.data(), counts.size() * 4);

    bool ok = true;
    for (int cse = 0; cse < 2; ++cse) {
        const bool use_window = (cse == 0);
        for (std::uint32_t h = 0; h < H; ++h) {
            for (std::uint32_t d = 0; d < DK; ++d) {
                const float qv = q[h * DK + d];
                const bool hi_cluster = !use_window;
                cent[(h * C + 0) * DK + d] = hi_cluster ? -qv : qv;
                cent[(h * C + 1) * DK + d] = hi_cluster ? qv : -qv;
                bs[(h * C + 0) * DK + d] = hi_cluster ? -qv : qv;
                bs[(h * C + 1) * DK + d] = hi_cluster ? qv : -qv;
            }
        }
        ctd.上传(cent.data(), cent.size() * 4);
        bsd.上传(bs.data(), bs.size() * 4);
        fill0(dqb); fill0(dqi); fill0(dkb); fill0(dki);
        fill0(dvb); fill0(dvi); fill0(dsb); fill0(dsi);

        const BlockBwdPC bp{1, H, N, N, DK, DV, C, LMAX, TOPB,
                           use_window ? 8u : 0u, QPOS, C, tau, sink, 0.0f, 0.0f};
        run("ersa_attn_bwd_query",
            {qb.取句柄(), kd.取句柄(), vd.取句柄(), bsd.取句柄(),
             dyb.取句柄(), dqb.取句柄(), dkb.取句柄(), dvb.取句柄(),
             dsb.取句柄()},
            &bp, sizeof(bp), H, 1, 1);

        const IvfBwdPC ip{1, H, 1, N, DK, DV, C, LMAX, TOPB,
                          use_window ? 8u : 0u, QPOS, 0, tau, sink};
        run("ersa_ivf_attn_bwd_query",
            {qi.取句柄(), kd.取句柄(), vd.取句柄(), ctd.取句柄(),
             mid.取句柄(), cnd.取句柄(), dyi.取句柄(), dqi.取句柄(),
             dki.取句柄(), dvi.取句柄(), dsi.取句柄()},
            &ip, sizeof(ip), 1, H, 1);

        std::vector<float> hdqb(H * N * DK), hdqi(H * DK);
        std::vector<float> hdkb(H * N * DK), hdki(H * N * DK);
        std::vector<float> hdvb(H * N * DV), hdvi(H * N * DV);
        std::vector<float> hdsb(H), hdsi(H);
        dqb.下载(hdqb.data(), hdqb.size() * 4);
        dqi.下载(hdqi.data(), hdqi.size() * 4);
        dkb.下载(hdkb.data(), hdkb.size() * 4);
        dki.下载(hdki.data(), hdki.size() * 4);
        dvb.下载(hdvb.data(), hdvb.size() * 4);
        dvi.下载(hdvi.data(), hdvi.size() * 4);
        dsb.下载(hdsb.data(), hdsb.size() * 4);
        dsi.下载(hdsi.data(), hdsi.size() * 4);

        std::vector<float> dq_block(H * DK), dq_ivf(H * DK);
        for (std::uint32_t h = 0; h < H; ++h) {
            for (std::uint32_t d = 0; d < DK; ++d)
                dq_block[h * DK + d] = hdqb[(h * N + QPOS) * DK + d];
        }
        dq_ivf = hdqi;
        const float edq = max_abs(dq_block, dq_ivf);
        const float edk = max_abs(hdkb, hdki);
        const float edv = max_abs(hdvb, hdvi);
        const float eds = max_abs(hdsb, hdsi);
        std::printf("case %d (%s): dQ %.3g dK %.3g dV %.3g dsink %.3g\n",
                    cse, use_window ? "topB+window" : "topB only",
                    edq, edk, edv, eds);
        ok = ok && edq <= 1e-6f && edk <= 1e-6f && edv <= 1e-6f && eds <= 1e-6f;
    }

    std::printf("%s\n", ok ? "PASS" : "FAIL");
    sched.停止();
    return ok ? 0 : 1;
}
