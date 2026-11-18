// 主模型 ASM_ERSAModel::ersa_ivf_forward 的独立 N 回召 smoke。
// Q: T=512 queries, K/V: N=4096 context; target outside the local window.
#include "asm_ersa_model.hpp"

#include "cpp/spv/kmeans_assign.spv.hpp"
#include "cpp/spv/kmeans_update.spv.hpp"
#include "cpp/spv/kmeans_fill_members.spv.hpp"
#include "cpp/spv/kmeans_inc_assign.spv.hpp"
#include "cpp/spv/kmeans_inc_append.spv.hpp"
#include "cpp/spv/kmeans_inc_update.spv.hpp"
#include "cpp/spv/ivf_scatter_members.spv.hpp"
#include "cpp/spv/ersa_ivf_attn_fused.spv.hpp"
#include "cpp/spv/dense_attn_fused.spv.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace asm_ersa_model;
using namespace vk张量系统;

int main(int argc, char** argv) {
    const std::string spv = (argc >= 2) ? argv[1] : "cpp/spv";
    流水线调度器& sched = 流水线调度器::获取();
    设备注册信息 reg; reg.应用名 = "ersa_ivf_cache_test";
    阶段配置 g; 阶段配置 c{阶段配置::自动队列族, 0, 1}; 阶段配置 t;
    try { sched.启动(reg, {g, c, t}, spv); }
    catch (const std::exception& e) { std::fprintf(stderr, "启动失败: %s\n", e.what()); return 1; }
    std::printf("设备: %s | 算子: %zu\n", sched.获取设备().名称(), sched.算子数量());

    ModelConfig cfg;
    cfg.d_model = 128; cfg.n_heads = 4; cfg.d_k = 64; cfg.d_v = 64;
    cfg.d_ff = 64; cfg.n_layers = 4; cfg.vocab_size = 64;
    cfg.n_slots = 16; cfg.top_r = 4; cfg.n_experts = 8; cfg.moe_top_k = 2;
    cfg.max_seq_len = 4096; cfg.block_size = 64; cfg.top_blocks = 2; cfg.window = 64;
    ASM_ERSAModel model(cfg, true);

    const std::uint32_t H = 4, T = 512, N = 4096, DK = 64, DV = 64;
    std::uint32_t seed = 20261118u;
    auto rnd = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return (float((seed >> 8) & 0xFFFFFFu) / 16777216.0f) * 2.0f - 1.0f;
    };
    auto normalize = [](float* x, std::uint32_t d) {
        double s = 0.0;
        for (std::uint32_t i = 0; i < d; ++i) s += double(x[i]) * x[i];
        const float inv = float(1.0 / std::sqrt(s + 1e-9));
        for (std::uint32_t i = 0; i < d; ++i) x[i] *= inv;
    };

    std::vector<float> K(std::size_t(H) * N * DK), V(std::size_t(H) * N * DV);
    std::vector<float> Q(std::size_t(H) * T * DK);
    for (auto& x : K) x = rnd();
    for (auto& x : V) x = 0.0f;
    for (auto& x : Q) x = rnd();
    for (std::uint32_t h = 0; h < H; ++h) {
        for (std::uint32_t n = 0; n < N; ++n) normalize(&K[(std::size_t(h) * N + n) * DK], DK);
        for (std::uint32_t q = 0; q < T; ++q) normalize(&Q[(std::size_t(h) * T + q) * DK], DK);
        const std::uint32_t target = 300u + h;
        std::vector<float> needle(DK);
        for (auto& x : needle) x = rnd();
        normalize(needle.data(), DK);
        for (std::uint32_t d = 0; d < DK; ++d) {
            K[(std::size_t(h) * N + target) * DK + d] = needle[d];
            Q[(std::size_t(h) * T + (T - 1)) * DK + d] = needle[d];
        }
        V[(std::size_t(h) * N + target) * DV] = 1.0f;
    }

    const std::vector<float> out = model.ersa_ivf_forward(
        Q, K, V, H, T, N, 64, 8, 8, 0.005f, -30.0f);
    bool ok = true;
    std::printf("主模型 ERSA IVF T=%u N=%u target masses:", T, N);
    for (std::uint32_t h = 0; h < H; ++h) {
        const float mass = out[(std::size_t(h) * T + (T - 1)) * DV];
        std::printf(" %.6f", mass);
        ok = ok && mass > 0.999f;
    }
    std::printf(" | %s\n", ok ? "PASS" : "FAIL");
    sched.停止();
    return ok ? 0 : 1;
}
