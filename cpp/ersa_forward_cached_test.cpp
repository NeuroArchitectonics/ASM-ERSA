// 全模型 prefill/query smoke: context 先写 ERSA K/V cache，query 段用 IVF。
#include "asm_ersa_model.hpp"
#include "流水线调度器.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace asm_ersa_model;
using namespace vk张量系统;

int main(int argc, char** argv) {
    const std::string spv = (argc >= 2) ? argv[1] : "cpp/spv";
    const std::uint32_t N = (argc >= 3) ? std::uint32_t(std::stoul(argv[2])) : 64;
    const std::uint32_t T = (argc >= 4) ? std::uint32_t(std::stoul(argv[3])) : 8;

    流水线调度器& sched = 流水线调度器::获取();
    设备注册信息 reg;
    reg.应用名 = "ersa_forward_cached_test";
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

    ModelConfig cfg;
    cfg.d_model = 128;
    cfg.n_heads = 4;
    cfg.d_k = 64;
    cfg.d_v = 64;
    cfg.d_ff = 64;
    cfg.n_layers = 4;
    cfg.vocab_size = 64;
    cfg.n_slots = 16;
    cfg.top_r = 4;
    cfg.n_experts = 8;
    cfg.moe_top_k = 2;
    cfg.max_seq_len = 4096;
    cfg.block_size = 64;
    cfg.top_blocks = 2;
    cfg.window = 64;

    std::vector<std::int32_t> context(N), query(T);
    for (std::uint32_t i = 0; i < N; ++i) context[i] = std::int32_t((i * 7u + 3u) % cfg.vocab_size);
    for (std::uint32_t i = 0; i < T; ++i) query[i] = std::int32_t((i * 11u + 5u) % cfg.vocab_size);

    bool ok = true;
    try {
        const ModelWeights weights = make_random_weights(cfg, 20261118u);
        ASM_ERSAModel model(cfg, weights);

        const auto t0 = std::chrono::steady_clock::now();
        std::vector<float> a = model.forward_cached(context, query);
        const auto t1 = std::chrono::steady_clock::now();
        std::vector<float> b = model.forward_cached(context, query);
        const auto t2 = std::chrono::steady_clock::now();

        const double ms1 = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double ms2 = std::chrono::duration<double, std::milli>(t2 - t1).count();
        std::printf("forward_cached N=%u T=%u out=%zu dispatch=%llu ivf_layers=%u "
                    "run1=%.2fms run2=%.2fms\n",
                    N, T, a.size(),
                    static_cast<unsigned long long>(model.last_dispatch_count()),
                    model.last_ivf_layer_count(), ms1, ms2);
        if (a.size() != std::size_t(T) * cfg.vocab_size) ok = false;
        double max_diff = 0.0;
        bool finite = true;
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (!std::isfinite(a[i]) || !std::isfinite(b[i])) finite = false;
            max_diff = std::max(max_diff, double(std::fabs(a[i] - b[i])));
        }
        std::printf("finite=%d repeat_max_diff=%.3g\n", finite ? 1 : 0, max_diff);
        ok = ok && finite && max_diff <= 1e-2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "forward_cached 失败: %s\n", e.what());
        ok = false;
    }
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    sched.停止();
    return ok ? 0 : 1;
}
