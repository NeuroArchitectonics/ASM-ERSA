#include "asm_ersa_model.hpp"

#include "vk张量.hpp"
#include "流水线调度器.hpp"
#include "cpp/spv/stream_read.spv.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#endif

using namespace vk张量系统;
using namespace asm_ersa_model;


int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string spv_dir = (argc >= 2) ? argv[1] : "cpp/spv";

    流水线调度器& sched = 流水线调度器::获取();
    设备注册信息 reg;
    reg.应用名 = "ASM-ERSA C++ main model";
    阶段配置 graphics;
    阶段配置 compute{阶段配置::自动队列族, 0, 2};
    阶段配置 transfer;
    try {
        sched.启动(reg, {graphics, compute, transfer}, spv_dir);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "启动失败: %s\n", e.what());
        return 1;
    }

    std::printf("设备: %s\n", sched.获取设备().名称());
    std::printf("已加载算子: %zu\n", sched.算子数量());

    const char* required[] = {
        "gr_qkv_int8", "asm_reset", "asm_step_eta", "ersa_attn_step",
        "wo_gr_route_int8", "moe_experts_int8",
        "moe_h_fused_int8", "moe_tail_w2_int8", "moe_grwrite_int8",
        "rmsnorm_quant", "linear_fwd_int8_f32"};
    for (const char* op : required) {
        if (!sched.存在算子(op)) {
            std::fprintf(stderr, "缺少算子: %s\n", op);
            sched.停止();
            return 1;
        }
    }

    const std::string mode = (argc >= 3) ? argv[2] : "";
    if (mode == "8b" || mode == "8bparams") {
        const ModelConfig c8 = ModelConfig8B();
        const std::uint64_t 总参 = count_params(c8, true);
        const std::uint64_t 激活 = count_active_params(c8, true);
        std::printf("[8B] layers=%u d=%u ff=%u experts=%u top_k=%u | 总参 %.3fB | 激活 %.3fB (%.1f%%)\n",
                    c8.n_layers, c8.d_model, c8.d_ff, c8.n_experts, c8.moe_top_k,
                    double(总参) / 1e9, double(激活) / 1e9,
                    100.0 * double(激活) / double(总参));
        return 0;
    }
    if (mode == "6b" || mode == "--6b" || mode == "6bparams") {
        const ModelConfig c6 = ModelConfig6BLite();
        const std::uint64_t tot = count_params(c6, true);
        const std::uint64_t act = count_active_params(c6, true);
        std::printf("6B-lite: layers=%u D=%u H=%u F=%u E=%u topk=%u tied_embed=1\n",
                    c6.n_layers, c6.d_model, c6.n_heads, c6.d_ff, c6.n_experts, c6.moe_top_k);
        std::printf("total=%.3fB active=%.3fB (%.1f%%)\n",
                    double(tot) / 1e9, double(act) / 1e9,
                    100.0 * double(act) / double(tot));
        std::printf("INT8 weights ~%.2f GB | FP16 ~%.2f GB | train ~%.1f GB (16B/param)\n",
                    double(tot) * 1.031 / 1e9, double(tot) * 2 / 1e9, double(tot) * 16 / 1e9);
        sched.停止();
        return 0;
    }
    if (mode == "6bmem" || mode == "--6bmem") {
        const ModelConfig c6 = ModelConfig6BLite();
        const std::uint64_t tot = count_params(c6, true);
        std::printf("6B-lite: params=%.3fB, active=%.3fB\n",
                    double(tot) / 1e9,
                    double(count_active_params(c6, true)) / 1e9);
        auto 报RSS = [](const char* 标) {
#ifdef _WIN32
            PROCESS_MEMORY_COUNTERS pmc{};
            GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
            std::printf("  [%s] RSS=%.2f GB peak=%.2f GB\n", 标,
                        double(pmc.WorkingSetSize) / 1e9, double(pmc.PeakWorkingSetSize) / 1e9);
#else
            (void)标;
#endif
        };
        auto 主机分配 = [&](std::uint64_t bytes, const char* 名) {
            // 单阶段分配：测完即释放，避免主机与 GTT（共享同一物理内存）叠加爆表
            std::vector<std::unique_ptr<char[]>> 块;
            块.reserve(64);
            const auto t0 = std::chrono::steady_clock::now();
            std::uint64_t done = 0;
            while (done < bytes) {
                const std::size_t n = static_cast<std::size_t>(
                    std::min<std::uint64_t>(1ull << 30, bytes - done));
                auto p = std::make_unique<char[]>(n);
                std::memset(p.get(), 1, n);   // 触页，计入 RSS
                块.push_back(std::move(p));
                done += n;
            }
            const double s = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            std::printf("%-14s %.2f GB  (%.2f s)\n", 名, double(bytes) / 1e9, s);
            报RSS(名);
        };
        主机分配(tot * 4, "fp32 params");
        主机分配(tot * 2, "bf16 params");
        主机分配(tot, "int8 weights");

        // GPU：INT8 权重实际分配（每块 ≤2e9 元素，避开 vk张量 的 uint32 形状上限）
        std::vector<std::unique_ptr<vk张量>> gpu块;
        std::uint64_t 剩余 = tot;
        const auto gt0 = std::chrono::steady_clock::now();
        try {
            while (剩余 > 0) {
                const std::uint32_t n = static_cast<std::uint32_t>(
                    std::min<std::uint64_t>(2000000000ull, 剩余));
                gpu块.push_back(std::make_unique<vk张量>(
                    张量布局::连续({n}), 张量类型::整数8, "mem_int8"));
                剩余 -= n;
            }
            const double s = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - gt0).count();
            std::printf("GPU INT8 weights: %.2f GB allocated (%.2f s)\n",
                        double(tot) / 1e9, s);
        } catch (const std::exception& e) {
            std::printf("GPU INT8 allocation failed after %.2f GB: %s\n",
                        double(tot - 剩余) / 1e9, e.what());
        }

        // 训练态：主机 fp32 梯度（APOLLO-Mini 状态另计 ~80 MB）
        主机分配(tot * 4, "fp32 grads");
        // GPU 训练态：再试 24 GB fp32 梯度（780M 的 GTT 上限读数）
        std::uint64_t 剩余2 = tot * 4;
        const auto gt1 = std::chrono::steady_clock::now();
        try {
            while (剩余2 > 0) {
                const std::uint32_t n = static_cast<std::uint32_t>(
                    std::min<std::uint64_t>(1000000000ull, 剩余2 / 4));
                gpu块.push_back(std::make_unique<vk张量>(
                    张量布局::连续({n}), 张量类型::浮点32, "mem_f32"));
                剩余2 -= std::uint64_t(n) * 4;
            }
            const double s = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - gt1).count();
            std::printf("GPU fp32 grads: %.2f GB allocated (%.2f s)\n",
                        double(tot * 4) / 1e9, s);
        } catch (const std::exception& e) {
            std::printf("GPU fp32 grads failed after %.2f GB: %s\n",
                        double(tot * 4 - 剩余2) / 1e9, e.what());
        }

        // 不探测 48GB：780M 的 GTT 与主机是同一块物理内存，主机已占 70GB+ 时再叠加会爆系统内存。
        sched.停止();
        return 0;
    }
    if (mode == "membw" || mode == "--membw") {
        const uint32_t words = 1u << 27;                 // 512 MB
        const uint32_t grid = 128;
        const uint32_t stride = words / grid;            // 4 MB / workgroup
        struct PC { uint32_t words, stride, pad0, pad1; };
        const PC pc{words, stride, 0, 0};

        bool ok = false;
        try {
            {
                vk张量 in(张量布局::连续({words}), 张量类型::整数32,
                          std::string("membw_in"), true);
                vk张量 out(张量布局::连续({grid}), 张量类型::无符号32,
                           std::string("membw_out"));
                void* mapped = in.映射();
                if (mapped == nullptr) throw std::runtime_error("membw 输入未映射");
                std::memset(mapped, 0, std::size_t(words) * sizeof(int32_t));

                auto run = [&]() {
                    const uint64_t id = sched.提交自定义算子(
                        "stream_read", {in.取句柄(), out.取句柄()}, &pc, sizeof(pc),
                        派发规格{grid, 1, 1});
                    sched.等待(id);
                    if (!sched.已完成(id)) throw std::runtime_error("stream_read 失败");
                };
                run();
                double best = 1e30;
                for (int it = 0; it < 3; ++it) {
                    const auto t0 = std::chrono::steady_clock::now();
                    run();
                    const auto t1 = std::chrono::steady_clock::now();
                    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    best = (ms < best) ? ms : best;
                }
                const double gb = double(words) * 4.0 / (1024.0 * 1024.0 * 1024.0);
                std::printf("membw: %.2f GB stream read | best %.3f ms | %.2f GB/s\n",
                            gb, best, gb / (best / 1000.0));
                ok = true;
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "membw 失败: %s\n", e.what());
        }
        sched.停止();
        return ok ? 0 : 1;
    }
    if (mode == "3b" || mode == "--3b" || mode == "3bprof" || mode == "--3bprof") {
        const bool prof = (mode == "3bprof" || mode == "--3bprof");
        const int arg_seq = (argc >= 4) ? std::atoi(argv[3]) : 1;
        if (arg_seq <= 0 || arg_seq > 16) {
            std::fprintf(stderr, "3b 模式的 seq 必须在 1..16\n");
            sched.停止();
            return 1;
        }

        ModelConfig cfg3;
        cfg3.d_model = 2048;
        cfg3.n_heads = 16;
        cfg3.d_k = 128;
        cfg3.d_v = 128;
        cfg3.d_ff = 384;
        cfg3.n_layers = 32;
        cfg3.vocab_size = 152064;
        cfg3.n_slots = 512;
        cfg3.top_r = 16;
        cfg3.n_scales = 4;
        cfg3.block_size = 64;
        cfg3.top_blocks = 4;
        cfg3.window = 64;
        cfg3.sketch_dim = 32;
        cfg3.n_experts = 32;
        cfg3.moe_top_k = 4;
        cfg3.max_seq_len = 16;

        std::vector<int32_t> ids(static_cast<std::size_t>(arg_seq));
        for (int i = 0; i < arg_seq; ++i) {
            ids[static_cast<std::size_t>(i)] = i % int(cfg3.vocab_size);
        }

        std::printf("3B timing: layers=%u D=%u H=%u F=%u E=%u topk=%u seq=%d\n",
                    cfg3.n_layers, cfg3.d_model, cfg3.n_heads, cfg3.d_ff,
                    cfg3.n_experts, cfg3.moe_top_k, arg_seq);

        bool ok = false;
        try {
            double build_ms = 0.0;
            uint64_t dc = 0, sc = 0;
            double best_ms = 1e30, avg_ms = 0.0;
            long long live_bytes = 0;
            size_t bad = 0;
            {
                const auto c0 = std::chrono::steady_clock::now();
                ASM_ERSAModel model(cfg3, true);
                const auto c1 = std::chrono::steady_clock::now();
                build_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();
                live_bytes = vk底层::缓冲::存活字节计().load();

                model.forward(ids);  // warmup
                if (prof) {
                    std::printf("per-stage (layer0/3, reps=20):\n");
                    model.profile_stages(0, 20);
                    ok = true;
                } else {
                const int iters = (arg_seq <= 4) ? 3 : 1;
                std::vector<float> logits;
                for (int it = 0; it < iters; ++it) {
                    const auto t0 = std::chrono::steady_clock::now();
                    logits = model.forward(ids);
                    const auto t1 = std::chrono::steady_clock::now();
                    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    best_ms = (ms < best_ms) ? ms : best_ms;
                    avg_ms += ms;
                    dc = model.last_dispatch_count();
                    sc = model.last_sync_count();
                }
                avg_ms /= double(iters);
                for (float v : logits) if (!std::isfinite(v)) ++bad;
                }
            }
            if (prof) {
                std::printf("3B build %.1f ms | live %.1f MB | profiled\n",
                            build_ms, double(live_bytes) / (1024.0 * 1024.0));
            } else {
                std::printf("3B build %.1f ms | live %.1f MB | forward best %.3f ms avg %.3f ms | "
                            "%.3f ms/token | dispatch %llu | sync %llu | non-finite %zu\n",
                            build_ms, double(live_bytes) / (1024.0 * 1024.0),
                            best_ms, avg_ms, avg_ms / double(arg_seq),
                            static_cast<unsigned long long>(dc),
                            static_cast<unsigned long long>(sc), bad);
            }
            ok = (bad == 0);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "3B 计时失败: %s\n", e.what());
        }

        sched.停止();
        return ok ? 0 : 1;
    }

    ModelConfig cfg;
    cfg.d_model = 32;
    cfg.n_heads = 4;
    cfg.d_k = 8;
    cfg.d_v = 8;
    cfg.d_ff = 64;
    cfg.vocab_size = 64;
    cfg.n_slots = 16;
    cfg.top_r = 4;
    cfg.n_scales = 4;
    cfg.block_size = 2;
    cfg.top_blocks = 2;
    cfg.window = 2;
    cfg.sketch_dim = 4;
    cfg.n_experts = 8;
    cfg.moe_top_k = 2;
    cfg.max_seq_len = 16;

    const int arg_layers = (argc >= 3) ? std::atoi(argv[2]) : 4;
    const int arg_seq = (argc >= 4) ? std::atoi(argv[3]) : 4;
    if (arg_layers <= 0 || arg_layers > 4096) {
        std::fprintf(stderr, "layers 参数非法\n");
        sched.停止();
        return 1;
    }
    if (arg_seq <= 0 || arg_seq > int(cfg.max_seq_len)) {
        std::fprintf(stderr, "seq 参数非法（当前 max_seq_len=16）\n");
        sched.停止();
        return 1;
    }
    cfg.n_layers = static_cast<uint32_t>(arg_layers);

    std::printf("config: layers=%d seq=%d d_model=%u heads=%u d_k=%u d_v=%u experts=%u\n",
                arg_layers, arg_seq, cfg.d_model, cfg.n_heads, cfg.d_k, cfg.d_v,
                cfg.n_experts);

    const ModelWeights weights = make_random_weights(cfg, 20261118u);
    std::vector<int32_t> ids;
    ids.resize(static_cast<std::size_t>(arg_seq));
    for (int i = 0; i < arg_seq; ++i) {
        ids[static_cast<std::size_t>(i)] = i % int(cfg.vocab_size);
    }

    std::vector<float> vk_logits;
    std::vector<float> cpu_logits;
    uint64_t dispatch_count = 0;
    uint64_t model_sync_count = 0;
    double vk_ms = 0.0;
    double cpu_ms = 0.0;

    try {
        {
            ASM_ERSAModel model(cfg, weights);
            model.forward(ids);  // warmup
            const auto t0 = std::chrono::steady_clock::now();
            vk_logits = model.forward(ids);
            const auto t1 = std::chrono::steady_clock::now();
            vk_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            dispatch_count = model.last_dispatch_count();
            model_sync_count = model.last_sync_count();
        }
        const auto c0 = std::chrono::steady_clock::now();
        cpu_logits = cpu_forward(cfg, weights, ids);
        const auto c1 = std::chrono::steady_clock::now();
        cpu_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "前向失败: %s\n", e.what());
        sched.停止();
        return 1;
    }

    if (vk_logits.size() != cpu_logits.size()) {
        std::fprintf(stderr, "输出大小不一致\n");
        sched.停止();
        return 1;
    }

    float max_abs = 0.0f;
    double max_rel = 0.0;
    size_t bad = 0;
    for (size_t i = 0; i < vk_logits.size(); ++i) {
        const float a = vk_logits[i];
        const float b = cpu_logits[i];
        if (!std::isfinite(a) || !std::isfinite(b)) {
            ++bad;
            continue;
        }
        const float diff = std::fabs(a - b);
        max_abs = std::fmax(max_abs, diff);
        max_rel = std::fmax(max_rel, double(diff) / (double(std::fabs(b)) + 1e-6));
    }

    std::printf("\n[VK] logits[0..7]:");
    for (size_t i = 0; i < 8 && i < vk_logits.size(); ++i) {
        std::printf(" %.7f", vk_logits[i]);
    }
    std::printf("\n[CPU] logits[0..7]:");
    for (size_t i = 0; i < 8 && i < cpu_logits.size(); ++i) {
        std::printf(" %.7f", cpu_logits[i]);
    }
    std::printf("\n");
    std::printf("INT8 vs FP32 oracle: max abs %.6g | max rel %.6g | non-finite %zu"
                "  (quantization error, not a parity check)\n",
                max_abs, max_rel, bad);
    std::printf("dispatch/forward: %llu | sync/forward: %llu | VK: %.3f ms | CPU ref: %.3f ms\n",
                static_cast<unsigned long long>(dispatch_count),
                static_cast<unsigned long long>(model_sync_count), vk_ms, cpu_ms);

    bool rt_ok = false;
    float rt_max = 0.0f;
    size_t rt_bad = 0;
    try {
        std::string exe = argv[0];
        const auto pos = exe.find_last_of("/\\");
        const std::string dir = (pos == std::string::npos) ? "." : exe.substr(0, pos);
        const std::string bin_path = dir + "/asm_ersa_int8.bin";
        save_int8_binary(bin_path, cfg, weights);

        ASM_ERSAModel loaded(cfg, bin_path);
        const std::vector<float> rt_logits = loaded.forward(ids);
        if (rt_logits.size() != vk_logits.size()) {
            throw std::runtime_error("round-trip 输出大小不一致");
        }
        for (size_t i = 0; i < rt_logits.size(); ++i) {
            const float a = rt_logits[i];
            const float b = vk_logits[i];
            if (!std::isfinite(a) || !std::isfinite(b)) {
                ++rt_bad;
                continue;
            }
            rt_max = std::fmax(rt_max, std::fabs(a - b));
        }
        rt_ok = (rt_bad == 0 && rt_max < 1e-5f);
        std::printf("INT8 binary round-trip: %s | max abs: %.6g | dispatch: %llu | %s\n",
                    bin_path.c_str(), rt_max,
                    static_cast<unsigned long long>(loaded.last_dispatch_count()),
                    rt_ok ? "PASS" : "FAIL");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "round-trip 失败: %s\n", e.what());
    }

    const bool ok = (bad == 0) && rt_ok && (max_abs < 0.5f);
    std::printf("%s\n", ok ? "PASS (INT8 path; binary round-trip exact)" : "FAIL");

    sched.停止();
    return ok ? 0 : 1;
}
