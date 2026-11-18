#include "asm_ersa_model.hpp"

#include "vk张量.hpp"

#include "cpp/spv/asm_reset.spv.hpp"
#include "cpp/spv/asm_step_eta.spv.hpp"
#include "cpp/spv/ersa_attn_step.spv.hpp"
#include "cpp/spv/gr_qkv_fused.spv.hpp"
#include "cpp/spv/moe_experts_fused.spv.hpp"
#include "cpp/spv/moe_reduce_grwrite_fused.spv.hpp"
#include "cpp/spv/gr_qkv_int8.spv.hpp"
#include "cpp/spv/wo_gr_route_int8.spv.hpp"
#include "cpp/spv/moe_experts_int8.spv.hpp"
#include "cpp/spv/moe_reduce_grwrite_int8.spv.hpp"
#include "cpp/spv/gr_read_quant_int8.spv.hpp"
#include "cpp/spv/qkv_int8_head.spv.hpp"
#include "cpp/spv/qkv_gemv_int8.spv.hpp"
#include "cpp/spv/wo_int8_tile.spv.hpp"
#include "cpp/spv/wo_gemv_int8.spv.hpp"
#include "cpp/spv/norm_gr_route_int8.spv.hpp"
#include "cpp/spv/moe_h_int8.spv.hpp"
#include "cpp/spv/moe_h_gemv_int8.spv.hpp"
#include "cpp/spv/moe_out_int8.spv.hpp"
#include "cpp/spv/moe_out_gemv_int8.spv.hpp"
#include "cpp/spv/lm_head_gemv_int8.spv.hpp"
#include "cpp/spv/rmsnorm_quant.spv.hpp"
#include "cpp/spv/gr_qkv_fused.spv.hpp"
#include "cpp/spv/wo_gr_route_fused.spv.hpp"
#include "cpp/spv/moe_experts_fused.spv.hpp"
#include "cpp/spv/moe_reduce_grwrite_fused.spv.hpp"
#include "spv/linear_fwd_int8_f32.spv.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace vk张量系统;

namespace asm_ersa_model {
namespace {

struct GRQKVPC {
    uint32_t B, D, H, DK, DV, T, t, mode, first, NBmax, BLOCK, pad;
};
static_assert(sizeof(GRQKVPC) == 48, "GRQKVPC must be 48 bytes");

struct AsmResetPC {
    uint32_t B, H, M, DK, DV, pad0, pad1, pad2;
};
static_assert(sizeof(AsmResetPC) == 32, "AsmResetPC must be 32 bytes");

struct AsmStepPC {
    uint32_t B, H, M, DK, DV, R, pad0, pad1;
    float tau, tau_w, rho, lam, null_beta, pad2;
};
static_assert(sizeof(AsmStepPC) == 56, "AsmStepPC must be 56 bytes");

struct ErsaStepPC {
    uint32_t B, H, T, N, DK, DV, NB, BLOCK, TOPB, WINDOW, QOFF, NBMAX;
    float tau, sink;
};
static_assert(sizeof(ErsaStepPC) == 56, "ErsaStepPC must be 56 bytes");

struct WoRoutePC {
    uint32_t B, D, HDV, E, TOPK, pad0, pad1, pad2;
};
static_assert(sizeof(WoRoutePC) == 32, "WoRoutePC must be 32 bytes");

struct MoeExpertsPC {
    uint32_t B, D, F, E, TOPK, CHUNK, pad0, pad1;
};
static_assert(sizeof(MoeExpertsPC) == 32, "MoeExpertsPC must be 32 bytes");

struct MoeReducePC {
    uint32_t B, D, F, TOPK, CHUNK, t, T, write_out;
};
static_assert(sizeof(MoeReducePC) == 32, "MoeReducePC must be 32 bytes");

struct GRReadPC {
    uint32_t B, D, T, t, first, pad0, pad1, pad2;
};
static_assert(sizeof(GRReadPC) == 32, "GRReadPC must be 32 bytes");

struct QkvGemvPC {
    uint32_t B, D, H, DK, DV, T, t, proj, store, NBmax, BLOCK, pad;
};
static_assert(sizeof(QkvGemvPC) == 48, "QkvGemvPC must be 48 bytes");

struct WoTilePC {
    uint32_t B, D, HDV, TILE, pad0, pad1, pad2, pad3;
};
static_assert(sizeof(WoTilePC) == 32, "WoTilePC must be 32 bytes");

struct RMSNormPC {
    uint32_t R, D;
    float eps;
    uint32_t pad;
};
static_assert(sizeof(RMSNormPC) == 16, "RMSNormPC must be 16 bytes");

struct Int8LinearPC {
    uint32_t N, K, M, group, has_bias;
};
static_assert(sizeof(Int8LinearPC) == 20, "Int8LinearPC must be 20 bytes");

struct LmHeadPC {
    uint32_t T, D, V, TILE, pad0, pad1, pad2, pad3;
};
static_assert(sizeof(LmHeadPC) == 32, "LmHeadPC must be 32 bytes");

void require_size(const std::vector<float>& v, std::size_t expected, const std::string& name) {
    if (v.size() != expected) {
        throw std::runtime_error(name + " size mismatch: got " + std::to_string(v.size()) +
                                 ", expected " + std::to_string(expected));
    }
}

vk张量 make_f32(std::vector<uint32_t> shape, const std::vector<float>& data,
                const std::string& name, bool 读多 = false) {
    vk张量 t(张量布局::连续(std::move(shape)), 张量类型::浮点32, name, 读多);
    if (!data.empty()) t.上传(data.data(), data.size() * sizeof(float));
    return t;
}

vk张量 make_i32(std::vector<uint32_t> shape, const std::vector<int32_t>& data,
                const std::string& name) {
    vk张量 t(张量布局::连续(std::move(shape)), 张量类型::整数32, name);
    if (!data.empty()) t.上传(data.data(), data.size() * sizeof(int32_t));
    return t;
}

vk张量 make_empty_f32(std::vector<uint32_t> shape, const std::string& name,
                      bool 读多 = false) {
    return vk张量(张量布局::连续(std::move(shape)), 张量类型::浮点32, name, 读多);
}

vk张量 make_i8(std::vector<uint32_t> shape, const std::vector<std::int8_t>& data,
               const std::string& name, bool 读多 = false) {
    vk张量 t(张量布局::连续(std::move(shape)), 张量类型::整数8, name, 读多);
    if (!data.empty()) t.上传(data.data(), data.size());
    return t;
}

vk张量 make_empty_i8(std::vector<uint32_t> shape, const std::string& name,
                      bool 读多 = false) {
    return vk张量(张量布局::连续(std::move(shape)), 张量类型::整数8, name, 读多);
}

struct QuantRows {
    std::vector<std::int8_t> q;
    std::vector<float> s;
    uint32_t G = 0;
};

QuantRows quantize_rows(const std::vector<float>& w, uint32_t rows, uint32_t cols,
                        uint32_t group = 128) {
    QuantRows r;
    r.G = (cols + group - 1) / group;
    r.q.resize(std::size_t(rows) * cols);
    r.s.resize(std::size_t(rows) * r.G);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t g = 0; g < r.G; ++g) {
            const uint32_t k0 = g * group;
            const uint32_t k1 = std::min(k0 + group, cols);
            float mx = 0.0f;
            for (uint32_t k = k0; k < k1; ++k) {
                mx = std::max(mx, std::fabs(w[std::size_t(row) * cols + k]));
            }
            const float s = std::max(mx / 127.0f, 1e-10f);
            r.s[std::size_t(row) * r.G + g] = s;
            for (uint32_t k = k0; k < k1; ++k) {
                const int q = int(std::lround(w[std::size_t(row) * cols + k] / s));
                r.q[std::size_t(row) * cols + k] =
                    static_cast<std::int8_t>(std::clamp(q, -127, 127));
            }
        }
    }
    return r;
}

QuantRows quantize_w1(const std::vector<float>& w, uint32_t E, uint32_t D, uint32_t F) {
    QuantRows r;
    r.G = (D + 127) / 128;
    r.q.resize(std::size_t(E) * F * D);
    r.s.resize(std::size_t(E) * F * r.G);
    for (uint32_t e = 0; e < E; ++e) {
        for (uint32_t f = 0; f < F; ++f) {
            const uint32_t row = e * F + f;
            for (uint32_t g = 0; g < r.G; ++g) {
                const uint32_t d0 = g * 128;
                const uint32_t d1 = std::min(d0 + 128, D);
                float mx = 0.0f;
                for (uint32_t d = d0; d < d1; ++d) {
                    mx = std::max(mx, std::fabs(w[(std::size_t(e) * D + d) * F + f]));
                }
                const float s = std::max(mx / 127.0f, 1e-10f);
                r.s[std::size_t(row) * r.G + g] = s;
                for (uint32_t d = d0; d < d1; ++d) {
                    const int q = int(std::lround(w[(std::size_t(e) * D + d) * F + f] / s));
                    r.q[std::size_t(row) * D + d] =
                        static_cast<std::int8_t>(std::clamp(q, -127, 127));
                }
            }
        }
    }
    return r;
}

QuantRows quantize_w2(const std::vector<float>& w, uint32_t E, uint32_t D, uint32_t F) {
    QuantRows r;
    r.G = (F + 127) / 128;
    r.q.resize(std::size_t(E) * D * F);
    r.s.resize(std::size_t(E) * D * r.G);
    for (uint32_t e = 0; e < E; ++e) {
        for (uint32_t d = 0; d < D; ++d) {
            const uint32_t row = e * D + d;
            for (uint32_t g = 0; g < r.G; ++g) {
                const uint32_t f0 = g * 128;
                const uint32_t f1 = std::min(f0 + 128, F);
                float mx = 0.0f;
                for (uint32_t f = f0; f < f1; ++f) {
                    mx = std::max(mx, std::fabs(w[(std::size_t(e) * F + f) * D + d]));
                }
                const float s = std::max(mx / 127.0f, 1e-10f);
                r.s[std::size_t(row) * r.G + g] = s;
                for (uint32_t f = f0; f < f1; ++f) {
                    const int q = int(std::lround(w[(std::size_t(e) * F + f) * D + d] / s));
                    r.q[std::size_t(row) * F + f] =
                        static_cast<std::int8_t>(std::clamp(q, -127, 127));
                }
            }
        }
    }
    return r;
}

struct GRDev {
    vk张量 norm_g, read_w, read_b, write_w, write_b;
};

struct ASMDev {
    vk张量 Wq_i8, Wq_s, Wk_i8, Wk_s, Wv_i8, Wv_s, Wo_i8, Wo_s;
    vk张量 w_eta_w, w_eta_b;
    vk张量 slot_init_k, gamma, out_norm;
    vk张量 K, V, beta;
    float beta_null = -2.0f;
};

struct ERSADev {
    vk张量 Wq_i8, Wq_s, Wk_i8, Wk_s, Wv_i8, Wv_s, Wo_i8, Wo_s;
    vk张量 out_norm;
    vk张量 q_hist, k_hist, v_hist, block_sum;
    float sink = -4.0f;
};

struct MoEDev {
    vk张量 gate;
    vk张量 W1_i8, W1_s, W2_i8, W2_s;
    vk张量 sharedW1_i8, sharedW1_s, sharedWup_i8, sharedWup_s, sharedW2_i8, sharedW2_s;
};

struct LayerDev {
    bool is_ersa = false;
    GRDev gr1, gr2;
    ASMDev asm_dev;
    ERSADev ersa_dev;
    MoEDev moe;
};

} // namespace

struct Impl {
    ModelConfig cfg;
    uint32_t Tmax = 0;
    uint32_t NBmax = 0;
    uint32_t HDK = 0;
    uint32_t HDV = 0;
    static constexpr uint32_t CHUNK = 1024;

    std::vector<std::unique_ptr<LayerDev>> layers;

    vk张量 embed_i8, embed_s, lm_head_i8, lm_head_s, final_norm_w, zero_bias_v;
    vk张量 branches, out_tokens, normed_q, normed_s, logits, ids;
    vk张量 qkv_norm, moe_y;
    vk张量 x2, moe_idx, moe_weight, expert_out;
    vk张量 xq, sx, wo_y, wo_deq, h_q, h_s;
    vk张量 qkv_q, qkv_k, qkv_v, o, attn_out, eta, dbg, block_sum_dummy;
    vk张量 eta_w_dummy, eta_b_dummy;

    uint64_t dispatch_count = 0;
    uint64_t sync_count = 0;

    Impl(const ModelConfig& cfg_, const ModelWeights* weights, const std::string* path,
         bool synthetic = false);
    std::vector<float> forward(const std::vector<int32_t>& input_ids);

    void run(const char* name, std::vector<张量句柄> ins, const void* push,
             uint32_t push_bytes, uint32_t gx, uint32_t gy = 1, uint32_t gz = 1);
    void submit_chain(std::vector<算子步骤> steps);

    GRDev upload_gr(const GRWeights& w, const std::string& prefix);
    ASMDev upload_asm(const ASMWeights& w, const std::string& prefix);
    ERSADev upload_ersa(const ERSAWeights& w, const std::string& prefix);
    MoEDev upload_moe(const MoEWeights& w, const std::string& prefix);

    GRDev make_gr_dev(const std::string& prefix);
    ASMDev make_asm_dev(const std::string& prefix);
    ERSADev make_ersa_dev(const std::string& prefix);
    MoEDev make_moe_dev(const std::string& prefix);
    void load_int8_binary(const std::string& path);

    void reset_states();
    void append_gr_read(LayerDev& L, uint32_t layer_index, uint32_t t,
                        std::vector<算子步骤>& steps);
    void append_qkv_head(LayerDev& L, uint32_t t, std::vector<算子步骤>& steps);
    void append_asm_step(LayerDev& L, std::vector<算子步骤>& steps);
    void append_ersa_step(LayerDev& L, uint32_t t, std::vector<算子步骤>& steps);
    void append_wo_tile(LayerDev& L, std::vector<算子步骤>& steps);
    void append_norm_route(LayerDev& L, std::vector<算子步骤>& steps);
    void append_moe_h(LayerDev& L, std::vector<算子步骤>& steps);
    void append_moe_out(LayerDev& L, std::vector<算子步骤>& steps);
    void append_moe_reduce(LayerDev& L, uint32_t t, bool write_out,
                           std::vector<算子步骤>& steps);
    void fill_synthetic();
    void profile_stages(uint32_t t = 0, uint32_t reps = 20);
};

template <class PC>
void append_step(std::vector<算子步骤>& steps, const char* name,
                 std::vector<张量句柄> ins, const PC& pc,
                 uint32_t gx, uint32_t gy = 1, uint32_t gz = 1) {
    steps.push_back(构造算子步骤(name, std::move(ins), pc, 派发规格{gx, gy, gz}));
}

void Impl::run(const char* name, std::vector<张量句柄> ins, const void* push,
               uint32_t push_bytes, uint32_t gx, uint32_t gy, uint32_t gz) {
    流水线调度器& sched = 流水线调度器::获取();
    const uint64_t id = sched.提交自定义算子(
        name, std::move(ins), push, push_bytes, 派发规格{gx, gy, gz});
    sched.等待(id);
    if (!sched.已完成(id)) {
        throw std::runtime_error(std::string("算子失败: ") + name + " | " +
                                 sched.查询失败原因(id));
    }
    ++dispatch_count;
    ++sync_count;
}

void Impl::submit_chain(std::vector<算子步骤> steps) {
    if (steps.empty()) return;
    const uint64_t n = steps.size();
    流水线调度器& sched = 流水线调度器::获取();
    const uint64_t id = sched.提交算子串(std::move(steps));
    sched.等待(id);
    if (!sched.已完成(id)) {
        throw std::runtime_error("算子串失败: " + sched.查询失败原因(id));
    }
    dispatch_count += n;
    ++sync_count;
}

GRDev Impl::upload_gr(const GRWeights& w, const std::string& prefix) {
    const uint32_t D = cfg.d_model;
    GRDev d;
    for (uint32_t i = 0; i < 4; ++i) {
        require_size(w.norm_g[i], D, prefix + ".norm_g" + std::to_string(i));
    }
    std::vector<float> norm(4 * std::size_t(D));
    for (uint32_t i = 0; i < 4; ++i) {
        std::copy(w.norm_g[i].begin(), w.norm_g[i].end(),
                  norm.begin() + std::size_t(i) * D);
    }
    require_size(w.read_w, std::size_t(4) * 4 * D, prefix + ".read_w");
    require_size(w.read_b, 4, prefix + ".read_b");
    require_size(w.write_w, std::size_t(4) * D, prefix + ".write_w");
    require_size(w.write_b, 4, prefix + ".write_b");

    d.norm_g = make_f32({4, D}, norm, prefix + ".norm_g");
    d.read_w = make_f32({4, 4 * D}, w.read_w, prefix + ".read_w");
    d.read_b = make_f32({4}, w.read_b, prefix + ".read_b");
    d.write_w = make_f32({4, D}, w.write_w, prefix + ".write_w");
    d.write_b = make_f32({4}, w.write_b, prefix + ".write_b");
    return d;
}

ASMDev Impl::upload_asm(const ASMWeights& w, const std::string& prefix) {
    const uint32_t D = cfg.d_model, H = cfg.n_heads, DK = cfg.d_k, DV = cfg.d_v;
    const uint32_t M = cfg.n_slots;
    ASMDev d;
    require_size(w.Wq, std::size_t(H) * DK * D, prefix + ".Wq");
    require_size(w.Wk, std::size_t(H) * DK * D, prefix + ".Wk");
    require_size(w.Wv, std::size_t(H) * DV * D, prefix + ".Wv");
    require_size(w.Wo, std::size_t(D) * H * DV, prefix + ".Wo");
    require_size(w.w_eta_w, D, prefix + ".w_eta_w");
    require_size(w.w_eta_b, 1, prefix + ".w_eta_b");
    require_size(w.slot_init_k, std::size_t(M) * DK, prefix + ".slot_init_k");
    require_size(w.gamma, M, prefix + ".gamma");
    require_size(w.out_norm, D, prefix + ".out_norm");

    const QuantRows qwq = quantize_rows(w.Wq, H * DK, D);
    const QuantRows qwk = quantize_rows(w.Wk, H * DK, D);
    const QuantRows qwv = quantize_rows(w.Wv, H * DV, D);
    const QuantRows qwo = quantize_rows(w.Wo, D, H * DV);
    d.Wq_i8 = make_i8({H * DK, D}, qwq.q, prefix + ".Wq_i8", true);
    d.Wq_s = make_f32({H * DK, qwq.G}, qwq.s, prefix + ".Wq_s", true);
    d.Wk_i8 = make_i8({H * DK, D}, qwk.q, prefix + ".Wk_i8", true);
    d.Wk_s = make_f32({H * DK, qwk.G}, qwk.s, prefix + ".Wk_s", true);
    d.Wv_i8 = make_i8({H * DV, D}, qwv.q, prefix + ".Wv_i8", true);
    d.Wv_s = make_f32({H * DV, qwv.G}, qwv.s, prefix + ".Wv_s", true);
    d.Wo_i8 = make_i8({D, H * DV}, qwo.q, prefix + ".Wo_i8", true);
    d.Wo_s = make_f32({D, qwo.G}, qwo.s, prefix + ".Wo_s", true);
    d.w_eta_w = make_f32({D}, w.w_eta_w, prefix + ".w_eta_w");
    d.w_eta_b = make_f32({1}, w.w_eta_b, prefix + ".w_eta_b");
    d.slot_init_k = make_f32({M, DK}, w.slot_init_k, prefix + ".slot_init_k");
    d.gamma = make_f32({M}, w.gamma, prefix + ".gamma");
    d.out_norm = make_f32({D}, w.out_norm, prefix + ".out_norm");
    d.beta_null = w.beta_null;

    d.K = make_empty_f32({1, H, M, DK}, prefix + ".K");
    d.V = make_empty_f32({1, H, M, DV}, prefix + ".V");
    d.beta = make_empty_f32({1, H, M}, prefix + ".beta");
    return d;
}

ERSADev Impl::upload_ersa(const ERSAWeights& w, const std::string& prefix) {
    const uint32_t D = cfg.d_model, H = cfg.n_heads, DK = cfg.d_k, DV = cfg.d_v;
    ERSADev d;
    require_size(w.Wq, std::size_t(H) * DK * D, prefix + ".Wq");
    require_size(w.Wk, std::size_t(H) * DK * D, prefix + ".Wk");
    require_size(w.Wv, std::size_t(H) * DV * D, prefix + ".Wv");
    require_size(w.Wo, std::size_t(D) * H * DV, prefix + ".Wo");
    require_size(w.out_norm, D, prefix + ".out_norm");

    const QuantRows qwq = quantize_rows(w.Wq, H * DK, D);
    const QuantRows qwk = quantize_rows(w.Wk, H * DK, D);
    const QuantRows qwv = quantize_rows(w.Wv, H * DV, D);
    const QuantRows qwo = quantize_rows(w.Wo, D, H * DV);
    d.Wq_i8 = make_i8({H * DK, D}, qwq.q, prefix + ".Wq_i8", true);
    d.Wq_s = make_f32({H * DK, qwq.G}, qwq.s, prefix + ".Wq_s", true);
    d.Wk_i8 = make_i8({H * DK, D}, qwk.q, prefix + ".Wk_i8", true);
    d.Wk_s = make_f32({H * DK, qwk.G}, qwk.s, prefix + ".Wk_s", true);
    d.Wv_i8 = make_i8({H * DV, D}, qwv.q, prefix + ".Wv_i8", true);
    d.Wv_s = make_f32({H * DV, qwv.G}, qwv.s, prefix + ".Wv_s", true);
    d.Wo_i8 = make_i8({D, H * DV}, qwo.q, prefix + ".Wo_i8", true);
    d.Wo_s = make_f32({D, qwo.G}, qwo.s, prefix + ".Wo_s", true);
    d.out_norm = make_f32({D}, w.out_norm, prefix + ".out_norm");
    d.sink = w.sink;

    d.q_hist = make_empty_f32({H, Tmax, DK}, prefix + ".q_hist");
    d.k_hist = make_empty_f32({H, Tmax, DK}, prefix + ".k_hist");
    d.v_hist = make_empty_f32({H, Tmax, DV}, prefix + ".v_hist");
    d.block_sum = make_empty_f32({H, NBmax, DK}, prefix + ".block_sum");
    return d;
}

MoEDev Impl::upload_moe(const MoEWeights& w, const std::string& prefix) {
    const uint32_t D = cfg.d_model, F = cfg.d_ff, E = cfg.n_experts;
    MoEDev d;
    require_size(w.gate, std::size_t(E) * D, prefix + ".gate");
    require_size(w.W1, std::size_t(E) * D * F, prefix + ".W1");
    require_size(w.W2, std::size_t(E) * F * D, prefix + ".W2");
    require_size(w.sharedW1, std::size_t(F) * D, prefix + ".sharedW1");
    require_size(w.sharedWup, std::size_t(F) * D, prefix + ".sharedWup");
    require_size(w.sharedW2, std::size_t(D) * F, prefix + ".sharedW2");

    const QuantRows q1 = quantize_w1(w.W1, E, D, F);
    const QuantRows q2 = quantize_w2(w.W2, E, D, F);
    const QuantRows qs1 = quantize_rows(w.sharedW1, F, D);
    const QuantRows qsu = quantize_rows(w.sharedWup, F, D);
    const QuantRows qs2 = quantize_rows(w.sharedW2, D, F);
    d.gate = make_f32({E, D}, w.gate, prefix + ".gate");
    d.W1_i8 = make_i8({E * F, D}, q1.q, prefix + ".W1_i8", true);
    d.W1_s = make_f32({E * F, q1.G}, q1.s, prefix + ".W1_s", true);
    d.W2_i8 = make_i8({E * D, F}, q2.q, prefix + ".W2_i8", true);
    d.W2_s = make_f32({E * D, q2.G}, q2.s, prefix + ".W2_s", true);
    d.sharedW1_i8 = make_i8({F, D}, qs1.q, prefix + ".sharedW1_i8", true);
    d.sharedW1_s = make_f32({F, qs1.G}, qs1.s, prefix + ".sharedW1_s", true);
    d.sharedWup_i8 = make_i8({F, D}, qsu.q, prefix + ".sharedWup_i8", true);
    d.sharedWup_s = make_f32({F, qsu.G}, qsu.s, prefix + ".sharedWup_s", true);
    d.sharedW2_i8 = make_i8({D, F}, qs2.q, prefix + ".sharedW2_i8", true);
    d.sharedW2_s = make_f32({D, qs2.G}, qs2.s, prefix + ".sharedW2_s", true);
    return d;
}

GRDev Impl::make_gr_dev(const std::string& prefix) {
    const uint32_t D = cfg.d_model;
    GRDev d;
    d.norm_g = make_empty_f32({4, D}, prefix + ".norm_g");
    d.read_w = make_empty_f32({4, 4 * D}, prefix + ".read_w");
    d.read_b = make_empty_f32({4}, prefix + ".read_b");
    d.write_w = make_empty_f32({4, D}, prefix + ".write_w");
    d.write_b = make_empty_f32({4}, prefix + ".write_b");
    return d;
}

ASMDev Impl::make_asm_dev(const std::string& prefix) {
    const uint32_t D = cfg.d_model, H = cfg.n_heads, DK = cfg.d_k, DV = cfg.d_v;
    const uint32_t M = cfg.n_slots;
    ASMDev d;
    const uint32_t Gd = (D + 127) / 128;
    const uint32_t Ghdv = (H * DV + 127) / 128;
    d.Wq_i8 = make_empty_i8({H * DK, D}, prefix + ".Wq_i8", true);
    d.Wq_s = make_empty_f32({H * DK, Gd}, prefix + ".Wq_s", true);
    d.Wk_i8 = make_empty_i8({H * DK, D}, prefix + ".Wk_i8", true);
    d.Wk_s = make_empty_f32({H * DK, Gd}, prefix + ".Wk_s", true);
    d.Wv_i8 = make_empty_i8({H * DV, D}, prefix + ".Wv_i8", true);
    d.Wv_s = make_empty_f32({H * DV, Gd}, prefix + ".Wv_s", true);
    d.Wo_i8 = make_empty_i8({D, H * DV}, prefix + ".Wo_i8", true);
    d.Wo_s = make_empty_f32({D, Ghdv}, prefix + ".Wo_s", true);
    d.w_eta_w = make_empty_f32({D}, prefix + ".w_eta_w");
    d.w_eta_b = make_empty_f32({1}, prefix + ".w_eta_b");
    d.slot_init_k = make_empty_f32({M, DK}, prefix + ".slot_init_k");
    d.gamma = make_empty_f32({M}, prefix + ".gamma");
    d.out_norm = make_empty_f32({D}, prefix + ".out_norm");
    d.K = make_empty_f32({1, H, M, DK}, prefix + ".K");
    d.V = make_empty_f32({1, H, M, DV}, prefix + ".V");
    d.beta = make_empty_f32({1, H, M}, prefix + ".beta");
    d.beta_null = -2.0f;
    return d;
}

ERSADev Impl::make_ersa_dev(const std::string& prefix) {
    const uint32_t D = cfg.d_model, H = cfg.n_heads, DK = cfg.d_k, DV = cfg.d_v;
    ERSADev d;
    const uint32_t Gd = (D + 127) / 128;
    const uint32_t Ghdv = (H * DV + 127) / 128;
    d.Wq_i8 = make_empty_i8({H * DK, D}, prefix + ".Wq_i8", true);
    d.Wq_s = make_empty_f32({H * DK, Gd}, prefix + ".Wq_s", true);
    d.Wk_i8 = make_empty_i8({H * DK, D}, prefix + ".Wk_i8", true);
    d.Wk_s = make_empty_f32({H * DK, Gd}, prefix + ".Wk_s", true);
    d.Wv_i8 = make_empty_i8({H * DV, D}, prefix + ".Wv_i8", true);
    d.Wv_s = make_empty_f32({H * DV, Gd}, prefix + ".Wv_s", true);
    d.Wo_i8 = make_empty_i8({D, H * DV}, prefix + ".Wo_i8", true);
    d.Wo_s = make_empty_f32({D, Ghdv}, prefix + ".Wo_s", true);
    d.out_norm = make_empty_f32({D}, prefix + ".out_norm");
    d.q_hist = make_empty_f32({H, Tmax, DK}, prefix + ".q_hist");
    d.k_hist = make_empty_f32({H, Tmax, DK}, prefix + ".k_hist");
    d.v_hist = make_empty_f32({H, Tmax, DV}, prefix + ".v_hist");
    d.block_sum = make_empty_f32({H, NBmax, DK}, prefix + ".block_sum");
    d.sink = -4.0f;
    return d;
}

MoEDev Impl::make_moe_dev(const std::string& prefix) {
    const uint32_t D = cfg.d_model, F = cfg.d_ff, E = cfg.n_experts;
    MoEDev d;
    const uint32_t Gd = (D + 127) / 128;
    const uint32_t Gf = (F + 127) / 128;
    d.gate = make_empty_f32({E, D}, prefix + ".gate");
    d.W1_i8 = make_empty_i8({E * F, D}, prefix + ".W1_i8", true);
    d.W1_s = make_empty_f32({E * F, Gd}, prefix + ".W1_s", true);
    d.W2_i8 = make_empty_i8({E * D, F}, prefix + ".W2_i8", true);
    d.W2_s = make_empty_f32({E * D, Gf}, prefix + ".W2_s", true);
    d.sharedW1_i8 = make_empty_i8({F, D}, prefix + ".sharedW1_i8", true);
    d.sharedW1_s = make_empty_f32({F, Gd}, prefix + ".sharedW1_s", true);
    d.sharedWup_i8 = make_empty_i8({F, D}, prefix + ".sharedWup_i8", true);
    d.sharedWup_s = make_empty_f32({F, Gd}, prefix + ".sharedWup_s", true);
    d.sharedW2_i8 = make_empty_i8({D, F}, prefix + ".sharedW2_i8", true);
    d.sharedW2_s = make_empty_f32({D, Gf}, prefix + ".sharedW2_s", true);
    return d;
}

Impl::Impl(const ModelConfig& cfg_, const ModelWeights* weights, const std::string* path,
           bool synthetic) : cfg(cfg_) {
    const uint32_t D = cfg.d_model, H = cfg.n_heads, DK = cfg.d_k, DV = cfg.d_v;
    const uint32_t F = cfg.d_ff, V = cfg.vocab_size, E = cfg.n_experts;

    if (D == 0 || H == 0 || DK == 0 || DV == 0 || F == 0 || V == 0 || E == 0) {
        throw std::runtime_error("模型维度不能为 0");
    }
    if (D > 4096) throw std::runtime_error("融合 kernel 限制 d_model <= 4096");
    if (D % 4 != 0) throw std::runtime_error("tiled INT8 GEMV 要求 d_model % 4 == 0");
    if (F % 4 != 0) throw std::runtime_error("tiled INT8 GEMV 要求 d_ff % 4 == 0");
    if (DK > 512 || DV > 512) throw std::runtime_error("融合 kernel 限制 d_k/d_v <= 512");
    if (cfg.n_slots == 0 || cfg.n_slots > 1024 || cfg.top_r == 0 || cfg.top_r > 32 ||
        cfg.top_r > cfg.n_slots) {
        throw std::runtime_error("ASM kernel 限制 0 < top_r <= n_slots <= 1024, top_r <= 32");
    }
    if (cfg.n_experts > 1024) throw std::runtime_error("MoE 专家数 <= 1024");
    if (cfg.moe_top_k == 0 || cfg.moe_top_k > 32 || cfg.moe_top_k > cfg.n_experts) {
        throw std::runtime_error("MoE 要求 0 < top_k <= min(32, n_experts)");
    }
    if (cfg.top_blocks == 0 || cfg.top_blocks > 16) {
        throw std::runtime_error("ERSA top_blocks 必须在 1..16");
    }
    if (cfg.max_seq_len == 0 || cfg.max_seq_len > 4096) {
        throw std::runtime_error("ERSA kernel 限制 max_seq_len <= 4096");
    }
    if (cfg.block_size == 0) throw std::runtime_error("block_size 必须 > 0");
    Tmax = cfg.max_seq_len;
    NBmax = (Tmax + cfg.block_size - 1) / cfg.block_size;
    if (NBmax > 128) throw std::runtime_error("ERSA kernel 限制 block 数 <= 128");
    if (cfg.n_layers == 0) throw std::runtime_error("n_layers 必须 > 0");
    if (!synthetic && weights == nullptr && path == nullptr) {
        throw std::runtime_error("权重来源为空");
    }
    if (weights != nullptr && weights->layers.size() != cfg.n_layers) {
        throw std::runtime_error("layers 数量与 n_layers 不一致");
    }

    HDK = H * DK;
    HDV = H * DV;
    if (HDV % 4 != 0) throw std::runtime_error("tiled INT8 Wo 要求 H*d_v % 4 == 0");

    const uint32_t Gd = (D + 127) / 128;
    if (weights != nullptr) {
        require_size(weights->embed, std::size_t(V) * D, "embed");
        require_size(weights->lm_head, std::size_t(V) * D, "lm_head");
        require_size(weights->out_norm, D, "out_norm");
        const QuantRows qe = quantize_rows(weights->embed, V, D);
        embed_i8 = make_i8({V, D}, qe.q, "embed_i8", true);
        embed_s = make_f32({V, qe.G}, qe.s, "embed_s", true);
        const QuantRows ql = quantize_rows(weights->lm_head, V, D);
        lm_head_i8 = make_i8({V, D}, ql.q, "lm_head_i8", true);
        lm_head_s = make_f32({V, ql.G}, ql.s, "lm_head_s", true);
        final_norm_w = make_f32({D}, weights->out_norm, "out_norm");
    } else {
        embed_i8 = make_empty_i8({V, D}, "embed_i8", true);
        embed_s = make_empty_f32({V}, "embed_s", true);
        lm_head_i8 = make_empty_i8({V, D}, "lm_head_i8", true);
        lm_head_s = make_empty_f32({V, Gd}, "lm_head_s", true);
        final_norm_w = make_empty_f32({D}, "out_norm");
    }
    std::vector<float> zero_v(V, 0.0f);
    zero_bias_v = make_f32({V}, zero_v, "zero_bias_v");

    branches = make_empty_f32({1, 4, D}, "branches");
    out_tokens = make_empty_f32({Tmax, D}, "out_tokens");
    normed_q = make_empty_i8({Tmax, D}, "normed_q");
    normed_s = make_empty_f32({Tmax}, "normed_s");
    logits = make_empty_f32({Tmax, V}, "logits");
    ids = make_i32({Tmax}, std::vector<int32_t>{}, "ids");
    qkv_norm = make_empty_f32({2, H, Tmax}, "qkv_norm");
    moe_y = make_empty_f32({1, D}, "moe_y");

    x2 = make_empty_f32({1, D}, "x2");
    moe_idx = make_i32({1, cfg.moe_top_k}, std::vector<int32_t>{}, "moe_idx");
    moe_weight = make_empty_f32({1, cfg.moe_top_k}, "moe_weight");
    expert_out = make_empty_f32({1, cfg.moe_top_k, D}, "expert_out");

    const uint32_t Gf = (F + 127) / 128;
    xq = make_empty_i8({1, D}, "xq");
    sx = make_empty_f32({1}, "sx");
    wo_y = make_empty_f32({1, D}, "wo_y");
    wo_deq = make_empty_f32({1, HDV}, "wo_deq");
    h_q = make_empty_i8({1, cfg.moe_top_k, F}, "h_q");
    h_s = make_empty_f32({1, cfg.moe_top_k, Gf}, "h_s");

    qkv_q = make_empty_f32({1, HDK}, "qkv_q");
    qkv_k = make_empty_f32({1, HDK}, "qkv_k");
    qkv_v = make_empty_f32({1, HDV}, "qkv_v");
    o = make_empty_f32({1, HDV}, "o");
    attn_out = make_empty_f32({1, HDV}, "attn_out");
    eta = make_empty_f32({1}, "eta");
    dbg = make_empty_f32({1, H, 3 * cfg.top_r}, "dbg");
    block_sum_dummy = make_empty_f32({1}, "block_sum_dummy");
    eta_w_dummy = make_f32({D}, std::vector<float>(D, 0.0f), "eta_w_dummy");
    eta_b_dummy = make_f32({1}, std::vector<float>(1, 0.0f), "eta_b_dummy");

    layers.reserve(cfg.n_layers);
    for (uint32_t i = 0; i < cfg.n_layers; ++i) {
        const bool expected_ersa = (i % 4 == 3);
        bool is_ersa = expected_ersa;
        if (weights != nullptr) {
            const LayerWeights& lw = weights->layers[i];
            if (lw.is_ersa != expected_ersa) {
                throw std::runtime_error("layer " + std::to_string(i) +
                                         " 的 is_ersa 与每 4 层一个 ERSA 的布局不一致");
            }
            is_ersa = lw.is_ersa;
        }
        auto ld = std::make_unique<LayerDev>();
        ld->is_ersa = is_ersa;
        if (weights != nullptr) {
            const LayerWeights& lw = weights->layers[i];
            ld->gr1 = upload_gr(lw.gr1, "L" + std::to_string(i) + ".gr1");
            ld->gr2 = upload_gr(lw.gr2, "L" + std::to_string(i) + ".gr2");
            ld->moe = upload_moe(lw.moe, "L" + std::to_string(i) + ".moe");
            if (ld->is_ersa) {
                ld->ersa_dev = upload_ersa(lw.ersa_w, "L" + std::to_string(i) + ".ersa");
            } else {
                ld->asm_dev = upload_asm(lw.asm_w, "L" + std::to_string(i) + ".asm");
            }
        } else {
            ld->gr1 = make_gr_dev("L" + std::to_string(i) + ".gr1");
            ld->gr2 = make_gr_dev("L" + std::to_string(i) + ".gr2");
            ld->moe = make_moe_dev("L" + std::to_string(i) + ".moe");
            if (ld->is_ersa) {
                ld->ersa_dev = make_ersa_dev("L" + std::to_string(i) + ".ersa");
            } else {
                ld->asm_dev = make_asm_dev("L" + std::to_string(i) + ".asm");
            }
        }
        layers.push_back(std::move(ld));
    }

    if (synthetic) {
        fill_synthetic();
    } else if (path != nullptr) {
        load_int8_binary(*path);
    }
    reset_states();
}

void Impl::fill_synthetic() {
    auto fill_i8 = [](vk张量& t, std::int8_t v) {
        std::vector<std::int8_t> data(static_cast<std::size_t>(t.布局().元素总数), v);
        t.上传(data.data(), data.size());
    };
    auto fill_f32 = [](vk张量& t, float v) {
        std::vector<float> data(static_cast<std::size_t>(t.布局().元素总数), v);
        t.上传(data.data(), data.size() * sizeof(float));
    };

    fill_i8(embed_i8, 1);
    fill_f32(embed_s, 1.0f);
    fill_i8(lm_head_i8, 0);
    fill_f32(lm_head_s, 1.0f);
    fill_f32(final_norm_w, 1.0f);
    fill_f32(zero_bias_v, 0.0f);

    for (auto& lp : layers) {
        LayerDev& L = *lp;
        fill_f32(L.gr1.norm_g, 1.0f);
        fill_f32(L.gr1.read_w, 0.0f);
        fill_f32(L.gr1.read_b, 0.0f);
        fill_f32(L.gr1.write_w, 0.0f);
        fill_f32(L.gr1.write_b, 0.0f);
        fill_f32(L.gr2.norm_g, 1.0f);
        fill_f32(L.gr2.read_w, 0.0f);
        fill_f32(L.gr2.read_b, 0.0f);
        fill_f32(L.gr2.write_w, 0.0f);
        fill_f32(L.gr2.write_b, 0.0f);

        if (L.is_ersa) {
            fill_i8(L.ersa_dev.Wq_i8, 0); fill_f32(L.ersa_dev.Wq_s, 1.0f);
            fill_i8(L.ersa_dev.Wk_i8, 0); fill_f32(L.ersa_dev.Wk_s, 1.0f);
            fill_i8(L.ersa_dev.Wv_i8, 0); fill_f32(L.ersa_dev.Wv_s, 1.0f);
            fill_i8(L.ersa_dev.Wo_i8, 0); fill_f32(L.ersa_dev.Wo_s, 1.0f);
            L.ersa_dev.sink = -4.0f;
            fill_f32(L.ersa_dev.out_norm, 1.0f);
        } else {
            fill_i8(L.asm_dev.Wq_i8, 0); fill_f32(L.asm_dev.Wq_s, 1.0f);
            fill_i8(L.asm_dev.Wk_i8, 0); fill_f32(L.asm_dev.Wk_s, 1.0f);
            fill_i8(L.asm_dev.Wv_i8, 0); fill_f32(L.asm_dev.Wv_s, 1.0f);
            fill_i8(L.asm_dev.Wo_i8, 0); fill_f32(L.asm_dev.Wo_s, 1.0f);
            fill_f32(L.asm_dev.w_eta_w, 0.0f);
            fill_f32(L.asm_dev.w_eta_b, -2.197f);
            fill_f32(L.asm_dev.slot_init_k, 0.0f);
            fill_f32(L.asm_dev.gamma, 0.999f);
            L.asm_dev.beta_null = -2.0f;
            fill_f32(L.asm_dev.out_norm, 1.0f);
        }

        fill_f32(L.moe.gate, 0.0f);
        fill_i8(L.moe.W1_i8, 0); fill_f32(L.moe.W1_s, 1.0f);
        fill_i8(L.moe.W2_i8, 0); fill_f32(L.moe.W2_s, 1.0f);
        fill_i8(L.moe.sharedW1_i8, 0); fill_f32(L.moe.sharedW1_s, 1.0f);
        fill_i8(L.moe.sharedWup_i8, 0); fill_f32(L.moe.sharedWup_s, 1.0f);
        fill_i8(L.moe.sharedW2_i8, 0); fill_f32(L.moe.sharedW2_s, 1.0f);
    }
}

void Impl::reset_states() {
    const uint32_t H = cfg.n_heads;
    const AsmResetPC pc{1, H, cfg.n_slots, cfg.d_k, cfg.d_v, 0, 0, 0};
    std::vector<算子步骤> steps;
    steps.reserve(layers.size());
    for (auto& lp : layers) {
        if (lp->is_ersa) continue;
        append_step(steps, "asm_reset",
                    {lp->asm_dev.slot_init_k.取句柄(), lp->asm_dev.K.取句柄(),
                     lp->asm_dev.V.取句柄(), lp->asm_dev.beta.取句柄()},
                    pc, H, 1, 1);
    }
    submit_chain(std::move(steps));
}

void Impl::load_int8_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("无法打开 INT8 权重: " + path);

    struct BinHeader {
        char magic[8];
        uint32_t version;
        uint32_t D, H, DK, DV, F, L, V, E, M, TOPR, GROUP, BLOCK, TOPB, WINDOW, TOPK, TMAX;
    };
    BinHeader h{};
    f.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!f || std::memcmp(h.magic, "ASM8BIN", 7) != 0) {
        throw std::runtime_error("不是 ASM8BIN 文件: " + path);
    }
    if (h.version != 1) throw std::runtime_error("ASM8BIN 版本不支持");
    if (h.D != cfg.d_model || h.H != cfg.n_heads || h.DK != cfg.d_k ||
        h.DV != cfg.d_v || h.F != cfg.d_ff || h.L != cfg.n_layers ||
        h.V != cfg.vocab_size || h.E != cfg.n_experts || h.M != cfg.n_slots) {
        throw std::runtime_error("ASM8BIN 头与 ModelConfig 不匹配");
    }

    const uint32_t D = cfg.d_model, H = cfg.n_heads, DK = cfg.d_k, DV = cfg.d_v;
    const uint32_t F = cfg.d_ff, V = cfg.vocab_size, E = cfg.n_experts;
    const uint32_t M = cfg.n_slots;
    const uint32_t Gd = (D + 127) / 128;
    const uint32_t Ghdv = (H * DV + 127) / 128;
    const uint32_t Gf = (F + 127) / 128;

    auto read_raw = [&](void* dst, std::size_t bytes) {
        f.read(reinterpret_cast<char*>(dst), std::streamsize(bytes));
        if (!f) throw std::runtime_error("ASM8BIN 读取截断: " + path);
    };
    auto read_i8 = [&](std::size_t n) {
        std::vector<std::int8_t> v(n);
        read_raw(v.data(), n);
        return v;
    };
    auto read_f32 = [&](std::size_t n) {
        std::vector<float> v(n);
        read_raw(v.data(), n * sizeof(float));
        return v;
    };
    auto up_i8 = [&](vk张量& t, std::size_t n) {
        std::vector<std::int8_t> v = read_i8(n);
        t.上传(v.data(), n);
    };
    auto up_f32 = [&](vk张量& t, std::size_t n) {
        std::vector<float> v = read_f32(n);
        t.上传(v.data(), n * sizeof(float));
    };
    auto read_scalar = [&]() {
        float v = 0.0f;
        read_raw(&v, sizeof(float));
        return v;
    };

    up_i8(embed_i8, std::size_t(V) * D);
    up_f32(embed_s, V);
    up_i8(lm_head_i8, std::size_t(V) * D);
    up_f32(lm_head_s, std::size_t(V) * Gd);
    up_f32(final_norm_w, D);

    for (uint32_t i = 0; i < cfg.n_layers; ++i) {
        LayerDev& L = *layers[i];
        for (int gi = 0; gi < 2; ++gi) {
            GRDev& g = (gi == 0) ? L.gr1 : L.gr2;
            up_f32(g.norm_g, std::size_t(4) * D);
            up_f32(g.read_w, std::size_t(4) * 4 * D);
            up_f32(g.read_b, 4);
            up_f32(g.write_w, std::size_t(4) * D);
            up_f32(g.write_b, 4);
        }

        if (L.is_ersa) {
            up_i8(L.ersa_dev.Wq_i8, std::size_t(H) * DK * D);
            up_f32(L.ersa_dev.Wq_s, std::size_t(H) * DK * Gd);
            up_i8(L.ersa_dev.Wk_i8, std::size_t(H) * DK * D);
            up_f32(L.ersa_dev.Wk_s, std::size_t(H) * DK * Gd);
            up_i8(L.ersa_dev.Wv_i8, std::size_t(H) * DV * D);
            up_f32(L.ersa_dev.Wv_s, std::size_t(H) * DV * Gd);
            up_i8(L.ersa_dev.Wo_i8, std::size_t(D) * H * DV);
            up_f32(L.ersa_dev.Wo_s, std::size_t(D) * Ghdv);
            L.ersa_dev.sink = read_scalar();
            up_f32(L.ersa_dev.out_norm, D);
        } else {
            up_i8(L.asm_dev.Wq_i8, std::size_t(H) * DK * D);
            up_f32(L.asm_dev.Wq_s, std::size_t(H) * DK * Gd);
            up_i8(L.asm_dev.Wk_i8, std::size_t(H) * DK * D);
            up_f32(L.asm_dev.Wk_s, std::size_t(H) * DK * Gd);
            up_i8(L.asm_dev.Wv_i8, std::size_t(H) * DV * D);
            up_f32(L.asm_dev.Wv_s, std::size_t(H) * DV * Gd);
            up_i8(L.asm_dev.Wo_i8, std::size_t(D) * H * DV);
            up_f32(L.asm_dev.Wo_s, std::size_t(D) * Ghdv);
            up_f32(L.asm_dev.w_eta_w, D);
            up_f32(L.asm_dev.w_eta_b, 1);
            up_f32(L.asm_dev.slot_init_k, std::size_t(M) * DK);
            up_f32(L.asm_dev.gamma, M);
            L.asm_dev.beta_null = read_scalar();
            up_f32(L.asm_dev.out_norm, D);
        }

        up_f32(L.moe.gate, std::size_t(E) * D);
        up_i8(L.moe.W1_i8, std::size_t(E) * F * D);
        up_f32(L.moe.W1_s, std::size_t(E) * F * Gd);
        up_i8(L.moe.W2_i8, std::size_t(E) * D * F);
        up_f32(L.moe.W2_s, std::size_t(E) * D * Gf);
        up_i8(L.moe.sharedW1_i8, std::size_t(F) * D);
        up_f32(L.moe.sharedW1_s, std::size_t(F) * Gd);
        up_i8(L.moe.sharedWup_i8, std::size_t(F) * D);
        up_f32(L.moe.sharedWup_s, std::size_t(F) * Gd);
        up_i8(L.moe.sharedW2_i8, std::size_t(D) * F);
        up_f32(L.moe.sharedW2_s, std::size_t(D) * Gf);
    }

    char extra = 0;
    if (f.read(&extra, 1)) throw std::runtime_error("ASM8BIN 尾部有多余数据");
}

void Impl::append_gr_read(LayerDev& L, uint32_t layer_index, uint32_t t,
                          std::vector<算子步骤>& steps) {
    const uint32_t D = cfg.d_model;
    const uint32_t first = (layer_index == 0) ? 1u : 0u;
    const GRReadPC pc{1, D, Tmax, t, first, 0, 0, 0};
    const vk张量& eww = L.is_ersa ? eta_w_dummy : L.asm_dev.w_eta_w;
    const vk张量& ebb = L.is_ersa ? eta_b_dummy : L.asm_dev.w_eta_b;
    append_step(steps, "gr_read_quant_int8",
                {embed_i8.取句柄(), embed_s.取句柄(), ids.取句柄(), branches.取句柄(),
                 L.gr1.norm_g.取句柄(), L.gr1.read_w.取句柄(), L.gr1.read_b.取句柄(),
                 eww.取句柄(), ebb.取句柄(),
                 xq.取句柄(), sx.取句柄(), eta.取句柄()},
                pc, 1, 1, 1);
}

void Impl::append_qkv_head(LayerDev& L, uint32_t t, std::vector<算子步骤>& steps) {
    const uint32_t H = cfg.n_heads, DK = cfg.d_k, DV = cfg.d_v;
    const uint32_t D = cfg.d_model;
    const uint32_t store = L.is_ersa ? 1u : 0u;

    const vk张量& Wq_i8 = L.is_ersa ? L.ersa_dev.Wq_i8 : L.asm_dev.Wq_i8;
    const vk张量& Wq_s = L.is_ersa ? L.ersa_dev.Wq_s : L.asm_dev.Wq_s;
    const vk张量& Wk_i8 = L.is_ersa ? L.ersa_dev.Wk_i8 : L.asm_dev.Wk_i8;
    const vk张量& Wk_s = L.is_ersa ? L.ersa_dev.Wk_s : L.asm_dev.Wk_s;
    const vk张量& Wv_i8 = L.is_ersa ? L.ersa_dev.Wv_i8 : L.asm_dev.Wv_i8;
    const vk张量& Wv_s = L.is_ersa ? L.ersa_dev.Wv_s : L.asm_dev.Wv_s;
    const vk张量& q_out = L.is_ersa ? L.ersa_dev.q_hist : qkv_q;
    const vk张量& k_out = L.is_ersa ? L.ersa_dev.k_hist : qkv_k;
    const vk张量& v_out = L.is_ersa ? L.ersa_dev.v_hist : qkv_v;
    const vk张量& bs = L.is_ersa ? L.ersa_dev.block_sum : block_sum_dummy;

    const QkvGemvPC pq{1, D, H, DK, DV, Tmax, t, 0, store, NBmax, cfg.block_size, 0};
    const QkvGemvPC pk{1, D, H, DK, DV, Tmax, t, 1, store, NBmax, cfg.block_size, 0};
    const QkvGemvPC pv{1, D, H, DK, DV, Tmax, t, 2, store, NBmax, cfg.block_size, 0};

    append_step(steps, "qkv_gemv_int8",
                {xq.取句柄(), sx.取句柄(), Wq_i8.取句柄(), Wq_s.取句柄(),
                 q_out.取句柄(), block_sum_dummy.取句柄(), qkv_norm.取句柄()},
                pq, H, 1, 1);
    append_step(steps, "qkv_gemv_int8",
                {xq.取句柄(), sx.取句柄(), Wk_i8.取句柄(), Wk_s.取句柄(),
                 k_out.取句柄(), bs.取句柄(), qkv_norm.取句柄()},
                pk, H, 1, 1);
    append_step(steps, "qkv_gemv_int8",
                {xq.取句柄(), sx.取句柄(), Wv_i8.取句柄(), Wv_s.取句柄(),
                 v_out.取句柄(), block_sum_dummy.取句柄(), qkv_norm.取句柄()},
                pv, H, 1, 1);
}

void Impl::append_asm_step(LayerDev& L, std::vector<算子步骤>& steps) {
    const uint32_t H = cfg.n_heads, DK = cfg.d_k, DV = cfg.d_v;
    const float tau = 1.0f / std::sqrt(static_cast<float>(DK));
    const AsmStepPC pc{1, H, cfg.n_slots, DK, DV, cfg.top_r, 0, 0,
                       tau, tau, 0.9f, 0.01f, L.asm_dev.beta_null, 0.0f};
    append_step(steps, "asm_step_eta",
                {qkv_q.取句柄(), qkv_k.取句柄(), qkv_v.取句柄(),
                 L.asm_dev.K.取句柄(), L.asm_dev.V.取句柄(), L.asm_dev.beta.取句柄(),
                 L.asm_dev.gamma.取句柄(), o.取句柄(), eta.取句柄(), dbg.取句柄()},
                pc, H, 1, 1);
}

void Impl::append_ersa_step(LayerDev& L, uint32_t t, std::vector<算子步骤>& steps) {
    const uint32_t H = cfg.n_heads, DK = cfg.d_k, DV = cfg.d_v;
    const uint32_t N = t + 1;
    const uint32_t NB = (N + cfg.block_size - 1) / cfg.block_size;
    const ErsaStepPC pc{1, H, Tmax, N, DK, DV, NB, cfg.block_size,
                        cfg.top_blocks, cfg.window, t, NBmax, 1.0f, L.ersa_dev.sink};
    append_step(steps, "ersa_attn_step",
                {L.ersa_dev.q_hist.取句柄(), L.ersa_dev.k_hist.取句柄(),
                 L.ersa_dev.v_hist.取句柄(), L.ersa_dev.block_sum.取句柄(),
                 attn_out.取句柄()},
                pc, H, 1, 1);
}

void Impl::append_wo_tile(LayerDev& L, std::vector<算子步骤>& steps) {
    const uint32_t D = cfg.d_model;
    const uint32_t TILE = 64;
    const WoTilePC pc{1, D, HDV, TILE, 0, 0, 0, 0};
    const vk张量& o_in = L.is_ersa ? attn_out : o;
    const vk张量& Wo_i8 = L.is_ersa ? L.ersa_dev.Wo_i8 : L.asm_dev.Wo_i8;
    const vk张量& Wo_s = L.is_ersa ? L.ersa_dev.Wo_s : L.asm_dev.Wo_s;
    const uint32_t tiles = (D + TILE - 1) / TILE;
    append_step(steps, "wo_gemv_int8",
                {o_in.取句柄(), Wo_i8.取句柄(), Wo_s.取句柄(), wo_y.取句柄(), wo_deq.取句柄()},
                pc, tiles, 1, 1);
}

void Impl::append_norm_route(LayerDev& L, std::vector<算子步骤>& steps) {
    const uint32_t D = cfg.d_model;
    const WoRoutePC pc{1, D, HDV, cfg.n_experts, cfg.moe_top_k, 0, 0, 0};
    const vk张量& out_norm = L.is_ersa ? L.ersa_dev.out_norm : L.asm_dev.out_norm;
    append_step(steps, "norm_gr_route_int8",
                {wo_y.取句柄(), out_norm.取句柄(), branches.取句柄(),
                 L.gr1.write_w.取句柄(), L.gr1.write_b.取句柄(),
                 L.gr2.norm_g.取句柄(), L.gr2.read_w.取句柄(), L.gr2.read_b.取句柄(),
                 L.moe.gate.取句柄(), x2.取句柄(), moe_idx.取句柄(), moe_weight.取句柄()},
                pc, 1, 1, 1);
}

void Impl::append_moe_h(LayerDev& L, std::vector<算子步骤>& steps) {
    const uint32_t D = cfg.d_model, F = cfg.d_ff;
    const uint32_t TILE_F = 128;
    const MoeExpertsPC pc{1, D, F, cfg.n_experts, cfg.moe_top_k, TILE_F, 0, 0};
    const uint32_t fgroups = (F + TILE_F - 1) / TILE_F;
    append_step(steps, "moe_h_gemv_int8",
                {x2.取句柄(), L.moe.W1_i8.取句柄(), L.moe.W1_s.取句柄(),
                 moe_idx.取句柄(), h_q.取句柄(), h_s.取句柄()},
                pc, cfg.moe_top_k, fgroups, 1);
}

void Impl::append_moe_out(LayerDev& L, std::vector<算子步骤>& steps) {
    const uint32_t D = cfg.d_model, F = cfg.d_ff;
    const uint32_t TILE_D = 64;
    const MoeExpertsPC pc{1, D, F, cfg.n_experts, cfg.moe_top_k, TILE_D, 0, 0};
    const uint32_t dtiles = (D + TILE_D - 1) / TILE_D;
    append_step(steps, "moe_out_gemv_int8",
                {h_q.取句柄(), h_s.取句柄(),
                 L.moe.W2_i8.取句柄(), L.moe.W2_s.取句柄(),
                 moe_idx.取句柄(), moe_weight.取句柄(), expert_out.取句柄()},
                pc, cfg.moe_top_k, dtiles, 1);
}

void Impl::append_moe_reduce(LayerDev& L, uint32_t t, bool write_out,
                             std::vector<算子步骤>& steps) {
    const uint32_t D = cfg.d_model, F = cfg.d_ff;
    const MoeReducePC rp{1, D, F, cfg.moe_top_k, CHUNK, t, Tmax, write_out ? 1u : 0u};
    append_step(steps, "moe_reduce_grwrite_int8",
                {x2.取句柄(), L.moe.sharedW1_i8.取句柄(), L.moe.sharedW1_s.取句柄(),
                 L.moe.sharedWup_i8.取句柄(), L.moe.sharedWup_s.取句柄(),
                 L.moe.sharedW2_i8.取句柄(), L.moe.sharedW2_s.取句柄(),
                 expert_out.取句柄(), branches.取句柄(),
                 L.gr2.write_w.取句柄(), L.gr2.write_b.取句柄(), out_tokens.取句柄(),
                 moe_y.取句柄()},
                rp, 1, 1, 1);
}

std::vector<float> Impl::forward(const std::vector<int32_t>& input_ids) {
    const uint32_t T = static_cast<uint32_t>(input_ids.size());
    if (T == 0) throw std::runtime_error("input_ids 不能为空");
    if (T > Tmax) throw std::runtime_error("input_ids 长度超过 max_seq_len");
    for (int32_t id : input_ids) {
        if (id < 0 || static_cast<uint32_t>(id) >= cfg.vocab_size) {
            throw std::runtime_error("input id 超出 vocab_size");
        }
    }

    dispatch_count = 0;
    sync_count = 0;
    reset_states();
    ids.上传(input_ids.data(), input_ids.size() * sizeof(int32_t));

    for (uint32_t t = 0; t < T; ++t) {
        std::vector<算子步骤> steps;
        steps.reserve(std::size_t(cfg.n_layers) * 8u);
        for (uint32_t li = 0; li < cfg.n_layers; ++li) {
            LayerDev& L = *layers[li];
            append_gr_read(L, li, t, steps);
            append_qkv_head(L, t, steps);
            if (L.is_ersa) {
                append_ersa_step(L, t, steps);
            } else {
                append_asm_step(L, steps);
            }
            append_wo_tile(L, steps);
            append_norm_route(L, steps);
            append_moe_h(L, steps);
            append_moe_out(L, steps);
            append_moe_reduce(L, t, li + 1 == cfg.n_layers, steps);
        }
        submit_chain(std::move(steps));
    }

    const uint32_t D = cfg.d_model, V = cfg.vocab_size;
    const RMSNormPC np{T, D, 1e-6f, 0};
    const LmHeadPC lp{T, D, V, 64, 0, 0, 0, 0};
    const uint32_t vtiles = (V + 63) / 64;

    std::vector<算子步骤> final_steps;
    final_steps.reserve(2);
    append_step(final_steps, "rmsnorm_quant",
                {out_tokens.取句柄(), final_norm_w.取句柄(),
                 normed_q.取句柄(), normed_s.取句柄()},
                np, T, 1, 1);
    append_step(final_steps, "lm_head_gemv_int8",
                {normed_q.取句柄(), normed_s.取句柄(), lm_head_i8.取句柄(),
                 lm_head_s.取句柄(), logits.取句柄()},
                lp, vtiles, T, 1);
    submit_chain(std::move(final_steps));

    std::vector<float> result(static_cast<std::size_t>(T) * V);
    logits.下载(result.data(), result.size() * sizeof(float));
    return result;
}

void Impl::profile_stages(uint32_t t, uint32_t reps) {
    if (layers.empty()) return;
    LayerDev& L0 = *layers[0];
    LayerDev& L1 = *layers[std::min<std::size_t>(1, layers.size() - 1)];
    LayerDev* LE = (layers.size() >= 4) ? layers[3].get() : nullptr;

    auto 计时 = [&](const char* 名, auto&& 构造) {
        std::vector<算子步骤> steps;
        for (uint32_t r = 0; r < reps; ++r) 构造(steps);
        const auto t0 = std::chrono::steady_clock::now();
        submit_chain(std::move(steps));
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / double(reps);
        std::printf("  %-22s %8.3f ms/step\n", 名, ms);
    };

    计时("gr_read", [&](std::vector<算子步骤>& s) { append_gr_read(L1, 1, t, s); });
    计时("qkv(3)", [&](std::vector<算子步骤>& s) { append_qkv_head(L0, t, s); });
    计时("asm_step", [&](std::vector<算子步骤>& s) { append_asm_step(L0, s); });
    if (LE != nullptr) {
        计时("ersa_step", [&](std::vector<算子步骤>& s) { append_ersa_step(*LE, t, s); });
    }
    计时("wo_tile", [&](std::vector<算子步骤>& s) { append_wo_tile(L0, s); });
    计时("norm_route", [&](std::vector<算子步骤>& s) { append_norm_route(L0, s); });
    计时("moe_h", [&](std::vector<算子步骤>& s) { append_moe_h(L0, s); });
    计时("moe_out", [&](std::vector<算子步骤>& s) { append_moe_out(L0, s); });
    计时("moe_reduce(shared)", [&](std::vector<算子步骤>& s) { append_moe_reduce(L0, t, false, s); });
    计时("head", [&](std::vector<算子步骤>& s) {
        const uint32_t D = cfg.d_model, V = cfg.vocab_size;
        const RMSNormPC np{1, D, 1e-6f, 0};
        const LmHeadPC lp{1, D, V, 64, 0, 0, 0, 0};
        append_step(s, "rmsnorm_quant",
                    {out_tokens.取句柄(), final_norm_w.取句柄(),
                     normed_q.取句柄(), normed_s.取句柄()},
                    np, 1, 1, 1);
        append_step(s, "lm_head_gemv_int8",
                    {normed_q.取句柄(), normed_s.取句柄(), lm_head_i8.取句柄(),
                     lm_head_s.取句柄(), logits.取句柄()},
                    lp, (V + 63) / 64, 1, 1);
    });
}

void save_int8_binary(const std::string& path, const ModelConfig& cfg,
                      const ModelWeights& weights) {
    if (weights.layers.size() != cfg.n_layers) {
        throw std::runtime_error("save_int8_binary: layers 数量不匹配");
    }
    struct BinHeader {
        char magic[8];
        uint32_t version;
        uint32_t D, H, DK, DV, F, L, V, E, M, TOPR, GROUP, BLOCK, TOPB, WINDOW, TOPK, TMAX;
    };
    BinHeader h{};
    std::memcpy(h.magic, "ASM8BIN", 7);
    h.magic[7] = '\0';
    h.version = 1;
    h.D = cfg.d_model; h.H = cfg.n_heads; h.DK = cfg.d_k; h.DV = cfg.d_v;
    h.F = cfg.d_ff; h.L = cfg.n_layers; h.V = cfg.vocab_size; h.E = cfg.n_experts;
    h.M = cfg.n_slots; h.TOPR = cfg.top_r; h.GROUP = 128; h.BLOCK = cfg.block_size;
    h.TOPB = cfg.top_blocks; h.WINDOW = cfg.window; h.TOPK = cfg.moe_top_k;
    h.TMAX = cfg.max_seq_len;

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("无法写入 " + path);
    auto wr_raw = [&](const void* p, std::size_t bytes) {
        f.write(reinterpret_cast<const char*>(p), std::streamsize(bytes));
        if (!f) throw std::runtime_error("写 ASM8BIN 失败: " + path);
    };
    auto wr_i8 = [&](const std::vector<std::int8_t>& v) { wr_raw(v.data(), v.size()); };
    auto wr_f32 = [&](const std::vector<float>& v) { wr_raw(v.data(), v.size() * sizeof(float)); };
    auto wr_scalar = [&](float v) { wr_raw(&v, sizeof(float)); };
    auto wr_qrows = [&](const std::vector<float>& w, uint32_t rows, uint32_t cols) {
        const QuantRows q = quantize_rows(w, rows, cols);
        wr_i8(q.q);
        wr_f32(q.s);
    };
    auto wr_qw1 = [&](const std::vector<float>& w, uint32_t E, uint32_t D, uint32_t F) {
        const QuantRows q = quantize_w1(w, E, D, F);
        wr_i8(q.q);
        wr_f32(q.s);
    };
    auto wr_qw2 = [&](const std::vector<float>& w, uint32_t E, uint32_t D, uint32_t F) {
        const QuantRows q = quantize_w2(w, E, D, F);
        wr_i8(q.q);
        wr_f32(q.s);
    };

    wr_raw(&h, sizeof(h));

    const uint32_t D = cfg.d_model, H = cfg.n_heads, DK = cfg.d_k, DV = cfg.d_v;
    const uint32_t F = cfg.d_ff, V = cfg.vocab_size, E = cfg.n_experts;

    wr_qrows(weights.embed, V, D);
    wr_qrows(weights.lm_head, V, D);
    wr_f32(weights.out_norm);

    for (uint32_t i = 0; i < cfg.n_layers; ++i) {
        const LayerWeights& lw = weights.layers[i];
        if (lw.is_ersa != (i % 4 == 3)) {
            throw std::runtime_error("save_int8_binary: layer ERSA 布局不一致");
        }
        for (int gi = 0; gi < 2; ++gi) {
            const GRWeights& g = (gi == 0) ? lw.gr1 : lw.gr2;
            std::vector<float> norm(4 * std::size_t(D));
            for (uint32_t j = 0; j < 4; ++j) {
                std::copy(g.norm_g[j].begin(), g.norm_g[j].end(),
                          norm.begin() + std::size_t(j) * D);
            }
            wr_f32(norm);
            wr_f32(g.read_w);
            wr_f32(g.read_b);
            wr_f32(g.write_w);
            wr_f32(g.write_b);
        }

        if (lw.is_ersa) {
            wr_qrows(lw.ersa_w.Wq, H * DK, D);
            wr_qrows(lw.ersa_w.Wk, H * DK, D);
            wr_qrows(lw.ersa_w.Wv, H * DV, D);
            wr_qrows(lw.ersa_w.Wo, D, H * DV);
            wr_scalar(lw.ersa_w.sink);
            wr_f32(lw.ersa_w.out_norm);
        } else {
            wr_qrows(lw.asm_w.Wq, H * DK, D);
            wr_qrows(lw.asm_w.Wk, H * DK, D);
            wr_qrows(lw.asm_w.Wv, H * DV, D);
            wr_qrows(lw.asm_w.Wo, D, H * DV);
            wr_f32(lw.asm_w.w_eta_w);
            wr_f32(lw.asm_w.w_eta_b);
            wr_f32(lw.asm_w.slot_init_k);
            wr_f32(lw.asm_w.gamma);
            wr_scalar(lw.asm_w.beta_null);
            wr_f32(lw.asm_w.out_norm);
        }

        wr_f32(lw.moe.gate);
        wr_qw1(lw.moe.W1, E, D, F);
        wr_qw2(lw.moe.W2, E, D, F);
        wr_qrows(lw.moe.sharedW1, F, D);
        wr_qrows(lw.moe.sharedWup, F, D);
        wr_qrows(lw.moe.sharedW2, D, F);
    }
    if (!f) throw std::runtime_error("写 ASM8BIN 失败: " + path);
}

ASM_ERSAModel::ASM_ERSAModel(const ModelConfig& cfg, const ModelWeights& weights)
    : impl_(std::make_unique<Impl>(cfg, &weights, nullptr, false)) {}

ASM_ERSAModel::ASM_ERSAModel(const ModelConfig& cfg, const std::string& int8_binary_path)
    : impl_(std::make_unique<Impl>(cfg, nullptr, &int8_binary_path, false)) {}

ASM_ERSAModel::ASM_ERSAModel(const ModelConfig& cfg, bool synthetic_int8_weights)
    : impl_(std::make_unique<Impl>(cfg, nullptr, nullptr, synthetic_int8_weights)) {}

ASM_ERSAModel::~ASM_ERSAModel() = default;

ASM_ERSAModel::ASM_ERSAModel(ASM_ERSAModel&&) noexcept = default;

ASM_ERSAModel& ASM_ERSAModel::operator=(ASM_ERSAModel&&) noexcept = default;

std::vector<float> ASM_ERSAModel::forward(const std::vector<int32_t>& input_ids) {
    return impl_->forward(input_ids);
}

void ASM_ERSAModel::profile_stages(uint32_t t, uint32_t reps) {
    impl_->profile_stages(t, reps);
}

const ModelConfig& ASM_ERSAModel::config() const {
    return impl_->cfg;
}

uint64_t ASM_ERSAModel::last_dispatch_count() const {
    return impl_->dispatch_count;
}

uint64_t ASM_ERSAModel::last_sync_count() const {
    return impl_->sync_count;
}

} // namespace asm_ersa_model
