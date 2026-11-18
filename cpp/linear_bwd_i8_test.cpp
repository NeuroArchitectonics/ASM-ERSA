// VK int8-direct linear dx vs CPU reference (weights stay int8 + scales).
#include "流水线调度器.hpp"
#include "vk张量.hpp"
#include "cpp/spv/linear_bwd_dx_i8.spv.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vk张量系统;

namespace {
struct PC { std::uint32_t N, K, M, G; };
std::uint32_t seed = 20261118u;
float rnd() {
    seed = seed * 1664525u + 1013904223u;
    return (float((seed >> 8) & 0xFFFFFFu) / 16777216.0f) * 2.0f - 1.0f;
}
} // namespace

int main(int argc, char** argv) {
    const std::string spv = (argc >= 2) ? argv[1] : "cpp/spv";
    流水线调度器& sched = 流水线调度器::获取();
    设备注册信息 reg; reg.应用名 = "linear_bwd_i8_test";
    阶段配置 g; 阶段配置 c{阶段配置::自动队列族, 0, 1}; 阶段配置 t;
    try { sched.启动(reg, {g, c, t}, spv); }
    catch (const std::exception& e) { std::fprintf(stderr, "启动失败: %s\n", e.what()); return 1; }
    std::printf("设备: %s\n", sched.获取设备().名称());

    bool ok = true;
    for (int cse = 0; cse < 2; ++cse) {
        const std::uint32_t N = cse == 0 ? 3u : 2u;
        const std::uint32_t M = cse == 0 ? 64u : 37u;
        const std::uint32_t K = cse == 0 ? 128u : 200u;
        const std::uint32_t G = (K + 127u) / 128u;
        std::vector<float> dy(N * M), ws(M * G), dx(N * K), ref(N * K);
        std::vector<std::int8_t> w(M * K);
        for (auto& x : dy) x = rnd();
        for (auto& x : ws) x = 0.001f + std::fabs(rnd()) * 0.02f;
        for (auto& x : w) x = std::int8_t(std::max(-127, std::min(127, int(rnd() * 127.0f))));
        for (std::uint32_t n = 0; n < N; ++n) {
            for (std::uint32_t k = 0; k < K; ++k) {
                double a = 0.0;
                for (std::uint32_t m = 0; m < M; ++m) {
                    a += double(dy[n * M + m]) * double(w[m * K + k]) *
                         double(ws[m * G + k / 128u]);
                }
                ref[n * K + k] = float(a);
            }
        }
        vk张量 dyd = vk张量(张量布局::连续({N, M}), 张量类型::浮点32, "dy");
        vk张量 wd = vk张量(张量布局::连续({M, K}), 张量类型::整数8, "w");
        vk张量 wsd = vk张量(张量布局::连续({M, G}), 张量类型::浮点32, "ws");
        vk张量 dxd = vk张量(张量布局::连续({N, K}), 张量类型::浮点32, "dx");
        dyd.上传(dy.data(), dy.size() * 4);
        wd.上传(w.data(), w.size());
        wsd.上传(ws.data(), ws.size() * 4);
        const PC pc{N, K, M, G};
        const std::uint64_t id = sched.提交自定义算子(
            "linear_bwd_dx_i8",
            {dyd.取句柄(), wd.取句柄(), wsd.取句柄(), dxd.取句柄()},
            &pc, sizeof(pc), 派发规格{(N * K + 63u) / 64u, 1, 1});
        sched.等待(id);
        if (!sched.已完成(id)) {
            std::fprintf(stderr, "kernel 失败: %s\n", sched.查询失败原因(id).c_str());
            ok = false;
            break;
        }
        dxd.下载(dx.data(), dx.size() * 4);
        float mx = 0.0f;
        for (std::size_t i = 0; i < dx.size(); ++i)
            mx = std::max(mx, std::fabs(dx[i] - ref[i]));
        std::printf("case %d N=%u M=%u K=%u G=%u: max abs %.3g\n",
                    cse, N, M, K, G, mx);
        ok = ok && mx <= 1e-5f;
    }
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    sched.停止();
    return ok ? 0 : 1;
}
