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

private:
    std::unique_ptr<Impl> impl_;
};

} // namespace asm_ersa_model
