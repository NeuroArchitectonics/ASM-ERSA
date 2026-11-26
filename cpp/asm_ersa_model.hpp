#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace asm_ersa_model {

using std::uint32_t;
using std::int32_t;

struct ModelConfig {
    uint32_t d_model = 32;
    uint32_t n_heads = 4;
    uint32_t d_k = 8;
    uint32_t d_v = 8;
    uint32_t d_ff = 64;
    uint32_t n_layers = 4;
    uint32_t vocab_size = 64;
    uint32_t n_slots = 16;
    uint32_t top_r = 4;
    uint32_t n_scales = 4;
    uint32_t block_size = 2;
    uint32_t top_blocks = 2;
    uint32_t window = 2;
    uint32_t sketch_dim = 4;
    uint32_t n_experts = 8;
    uint32_t moe_top_k = 2;
    uint32_t max_seq_len = 16;
};

// 6B-lite 预设（与 Python 侧 TrainConfig6BLite 同构）：
// D=2048/F=384 不变，24 层、138 专家 top-4、tied embedding。
// 目标：总参 ~5.99B、每 token 激活 ~0.93B（VK 主干口径，不含多模态头）。
inline ModelConfig ModelConfig6BLite() {
    ModelConfig c;
    c.d_model = 2048;
    c.n_heads = 16;
    c.d_k = 128;
    c.d_v = 128;
    c.d_ff = 384;
    c.n_layers = 24;
    c.vocab_size = 152064;
    c.n_slots = 512;
    c.top_r = 16;
    c.n_scales = 4;
    c.block_size = 64;
    c.top_blocks = 4;
    c.window = 64;
    c.sketch_dim = 32;
    c.n_experts = 138;
    c.moe_top_k = 4;
    c.max_seq_len = 16;
    return c;
}

// 8B：与 6B-lite 同骨架（24 层 / d=2048 / d_ff=384 / top-4 / tied），专家数 138→192。
// 只增专家数不改 top-k，激活参数几乎不变（约 0.93B），总参 ≈ 8.0B。
inline ModelConfig ModelConfig8B() {
    ModelConfig c = ModelConfig6BLite();
    c.n_experts = 192;
    return c;
}

// 参数统计：与 make_random_weights 的 shape 一致；tied_embed=true 时 lm_head 与词嵌入共用一份。
inline std::uint64_t count_params(const ModelConfig& c, bool tied_embed = true) {
    const std::uint64_t D = c.d_model, H = c.n_heads, DK = c.d_k, DV = c.d_v;
    const std::uint64_t F = c.d_ff, V = c.vocab_size, E = c.n_experts, M = c.n_slots, L = c.n_layers;
    const std::uint64_t gr = 4 * D + 4 * 4 * D + 4 + 4 * D + 4;
    const std::uint64_t asm_w = 4 * (H * DK * D) + D + 1 + M * DK + M + D;
    const std::uint64_t ersa_w = 4 * (H * DK * D) + D + 1;
    const std::uint64_t moe = E * D + 2 * (E * D * F) + 3 * (F * D);
    const std::uint64_t n_ersa = L / 4, n_asm = L - n_ersa;
    return L * 2 * gr + n_asm * asm_w + n_ersa * ersa_w + L * moe
           + (tied_embed ? 1u : 2u) * V * D + D;
}

// 每 token 激活参数：全部非专家 + moe_top_k/E 的专家
inline std::uint64_t count_active_params(const ModelConfig& c, bool tied_embed = true) {
    const std::uint64_t D = c.d_model, F = c.d_ff, E = c.n_experts, L = c.n_layers;
    const std::uint64_t expert = L * 2 * E * D * F;
    const std::uint64_t tot = count_params(c, tied_embed);
    return tot - expert + expert * c.moe_top_k / E;
}

struct GRWeights {
    std::vector<float> norm_g[4];
    std::vector<float> read_w;
    std::vector<float> read_b;
    std::vector<float> write_w;
    std::vector<float> write_b;
};

struct ASMWeights {
    std::vector<float> Wq;
    std::vector<float> Wk;
    std::vector<float> Wv;
    std::vector<float> Wo;
    std::vector<float> w_eta_w;
    std::vector<float> w_eta_b;
    std::vector<float> slot_init_k;
    std::vector<float> gamma;
    float beta_null = -2.0f;
    std::vector<float> out_norm;
};

struct ERSAWeights {
    std::vector<float> Wq;
    std::vector<float> Wk;
    std::vector<float> Wv;
    std::vector<float> Wo;
    std::vector<float> out_norm;
    float sink = -4.0f;
};

struct MoEWeights {
    std::vector<float> gate;
    std::vector<float> W1;
    std::vector<float> W2;
    std::vector<float> sharedW1;
    std::vector<float> sharedWup;
    std::vector<float> sharedW2;
};

struct LayerWeights {
    bool is_ersa = false;
    GRWeights gr1;
    GRWeights gr2;
    ASMWeights asm_w;
    ERSAWeights ersa_w;
    MoEWeights moe;
};

struct ModelWeights {
    std::vector<float> embed;
    std::vector<LayerWeights> layers;
    std::vector<float> out_norm;
    std::vector<float> lm_head;
};

ModelWeights make_random_weights(const ModelConfig& cfg, uint32_t seed);
std::vector<float> cpu_forward(const ModelConfig& cfg, const ModelWeights& weights,
                               const std::vector<int32_t>& input_ids);
void save_int8_binary(const std::string& path, const ModelConfig& cfg,
                      const ModelWeights& weights);

struct Impl;

class ASM_ERSAModel {
public:
    ASM_ERSAModel(const ModelConfig& cfg, const ModelWeights& weights);
    ASM_ERSAModel(const ModelConfig& cfg, const std::string& int8_binary_path);
    ASM_ERSAModel(const ModelConfig& cfg, bool synthetic_int8_weights);
    ~ASM_ERSAModel();

    ASM_ERSAModel(const ASM_ERSAModel&) = delete;
    ASM_ERSAModel& operator=(const ASM_ERSAModel&) = delete;
    ASM_ERSAModel(ASM_ERSAModel&&) noexcept;
    ASM_ERSAModel& operator=(ASM_ERSAModel&&) noexcept;

    std::vector<float> forward(const std::vector<int32_t>& input_ids);
    // 全模型 prefill/query：context 先逐 token 走完全部层并写入 ERSA K/V
    // cache，随后在 cache 上建 IVF，query token 的 ERSA 层走 IVF 路由。
    // 返回 query 段的 logits [Tq,V]。
    std::vector<float> forward_cached(const std::vector<int32_t>& context_ids,
                                      const std::vector<int32_t>& query_ids);
    // 独立 N 的 ERSA KV-cache forward：Q [H,T,DK]，K [H,N,DK]，V [H,N,DV]，
    // 返回 [H,T,DV]；索引与 attention 都在 VK 上。
    std::vector<float> ersa_ivf_forward(const std::vector<float>& q,
                                        const std::vector<float>& k,
                                        const std::vector<float>& v,
                                        uint32_t heads, uint32_t T, uint32_t N,
                                        uint32_t C = 64, uint32_t topb = 8,
                                        uint32_t window = 8, float tau = 0.005f,
                                        float sink = -30.0f);
    void profile_stages(uint32_t t = 0, uint32_t reps = 20);
    const ModelConfig& config() const;
    uint64_t last_dispatch_count() const;
    uint64_t last_sync_count() const;
    uint32_t last_ivf_layer_count() const;

private:
    std::unique_ptr<Impl> impl_;
};

} // namespace asm_ersa_model
