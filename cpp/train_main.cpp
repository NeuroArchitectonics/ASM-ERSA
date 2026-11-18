// M-B2：tiny 主模型 VK 训练器（INT8 前向 + FP32 母版反向）
// 先实现前向/记录/参数注册，随后逐段接反向与 AdamW。
// 用法：train_main.exe <spv目录> [steps]
#include "asm_ersa_model.hpp"
#include "vk张量.hpp"
#include "流水线调度器.hpp"

// ── 前向融合 kernel ──
#include "cpp/spv/asm_reset.spv.hpp"
#include "cpp/spv/asm_step_eta.spv.hpp"
#include "cpp/spv/ersa_attn_step.spv.hpp"
#include "cpp/spv/gr_read_quant_int8.spv.hpp"
#include "cpp/spv/qkv_fused_gemv_int8.spv.hpp"
#include "cpp/spv/wo_gemv_int8.spv.hpp"
#include "cpp/spv/norm_gr_route_int8.spv.hpp"
#include "cpp/spv/moe_h_fused_int8.spv.hpp"
#include "cpp/spv/moe_out_gemv_int8.spv.hpp"
#include "cpp/spv/moe_tail_w2_int8.spv.hpp"
#include "cpp/spv/moe_grwrite_int8.spv.hpp"
#include "cpp/spv/rmsnorm_quant.spv.hpp"
#include "cpp/spv/lm_head_gemv_int8.spv.hpp"

// ── 反向 kernel ──
#include "spv/linear_bwd_dx.spv.hpp"
#include "cpp/spv/linear_bwd_dx_i8.spv.hpp"
#include "spv/linear_bwd_dwdb.spv.hpp"
#include "cpp/spv/linear_bwd_dwdb5.spv.hpp"
#include "spv/rmsnorm_bwd_dx.spv.hpp"
#include "spv/rmsnorm_bwd_dw.spv.hpp"
#include "spv/fused_ce_bwd.spv.hpp"
#include "spv/embedding_bwd.spv.hpp"
#include "cpp/spv/embedding_bwd_scatter.spv.hpp"
#include "spv/moe_bwd_dx.spv.hpp"
#include "cpp/spv/moe_bwd_dx_q.spv.hpp"
#include "cpp/spv/moe_bwd_dx_i8.spv.hpp"
#include "cpp/spv/moe_bwd_dx_shared_i8.spv.hpp"
#include "spv/moe_bwd_dw_expert.spv.hpp"
#include "cpp/spv/moe_bwd_dw_expert_q.spv.hpp"
#include "cpp/spv/moe_bwd_dw_expert_i8.spv.hpp"
#include "cpp/spv/moe_bwd_dw_all_i8.spv.hpp"
#include "cpp/spv/moe_build_route_list.spv.hpp"
#include "spv/moe_bwd_dw_shared.spv.hpp"
#include "cpp/spv/moe_bwd_dw_shared_q.spv.hpp"
#include "cpp/spv/moe_bwd_dw_shared_i8.spv.hpp"
#include "spv/adamw_update.spv.hpp"
#include "cpp/spv/gr_read_bwd_int8.spv.hpp"
#include "cpp/spv/gr_read_post_seq.spv.hpp"
#include "cpp/spv/gr_route_bwd_int8.spv.hpp"
#include "cpp/spv/gr2_write_bwd.spv.hpp"
#include "cpp/spv/l2norm_bwd.spv.hpp"
#include "cpp/spv/asm_step_bwd_multi.spv.hpp"
#include "cpp/spv/asm_step_bwd_seq.spv.hpp"
#include "cpp/spv/asm_step_bwd_seq_wg.spv.hpp"
#include "cpp/spv/ersa_attn_bwd_query.spv.hpp"
#include "cpp/spv/ersa_ivf_attn_query.spv.hpp"
#include "cpp/spv/ersa_ivf_attn_bwd_query.spv.hpp"
#include "cpp/spv/ivf_init_centroids.spv.hpp"
#include "cpp/spv/kmeans_assign.spv.hpp"
#include "cpp/spv/kmeans_update.spv.hpp"
#include "cpp/spv/kmeans_fill_members.spv.hpp"
#include "cpp/spv/copy_f32_slice.spv.hpp"
#include "cpp/spv/copy_i32_slice.spv.hpp"
#include "cpp/spv/copy_i8_slice.spv.hpp"
#include "cpp/spv/kmeans_inc_assign.spv.hpp"
#include "cpp/spv/kmeans_inc_append.spv.hpp"
#include "cpp/spv/kmeans_inc_update.spv.hpp"
#include "cpp/spv/ersa_ivf_attn_bwd_cand.spv.hpp"
#include "cpp/spv/add_inplace.spv.hpp"
#include "cpp/spv/dequant_i8.spv.hpp"
#include "cpp/spv/dequant_row_i8.spv.hpp"
#include "cpp/spv/quantize_rows_i8.spv.hpp"
#include "cpp/spv/transpose_f32.spv.hpp"
#include "cpp/spv/transpose_heads.spv.hpp"
#include "cpp/spv/sum_f32.spv.hpp"
#include "cpp/spv/sum_heads.spv.hpp"
#include "cpp/spv/sum_heads_multi.spv.hpp"
#include "cpp/spv/quantize_dequant_f32.spv.hpp"
#include "cpp/spv/moe_w1_unpack.spv.hpp"
#include "cpp/spv/moe_w2_unpack.spv.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace vk张量系统;

namespace {

struct 规格2 { std::uint32_t a, b; };
struct 规格3 { std::uint32_t a, b, c, pad; };
struct 规格4 { std::uint32_t a, b, c, d; };
struct LinI8PC {
    std::uint32_t N, K, M, G, accum;
    std::uint32_t pad0, pad1, pad2;
};
struct 规格F { std::uint32_t a, b; float eps; float pad; };
struct 规格Adam {
    float lr, beta1, beta2, eps, m_corr, v_corr;
    std::uint32_t size;
    float wd;
    std::uint32_t zero_grad;
};
struct GRReadPC { std::uint32_t B, D, T, t, first, pad0, pad1, pad2; };
struct GrPostPC { std::uint32_t T, D, first, pad0; };
struct QkvFusedPC {
    std::uint32_t B, D, H, DK, DV, Tq, Tk, t, qt, store, NBmax, BLOCK, pad;
};
struct AsmResetPC { std::uint32_t B, H, M, DK, DV, pad0, pad1, pad2; };
struct AsmStepPC {
    std::uint32_t B, H, M, DK, DV, R, pad0, pad1;
    float tau, tau_w, rho, lam, null_beta, pad2;
};
struct ErsaStepPC {
    std::uint32_t B, H, T, N, DK, DV, NB, BLOCK, TOPB, WINDOW, QOFF, NBMAX;
    float tau, sink;
};
struct WoTilePC { std::uint32_t B, D, HDV, TILE, pad0, pad1, pad2, pad3; };
struct WoRoutePC { std::uint32_t B, D, HDV, E, TOPK, pad0, pad1, pad2; };
struct MoeExpertsPC { std::uint32_t B, D, F, E, TOPK, CHUNK, pad0, pad1; };
struct MoeReducePC {
    std::uint32_t B, D, F, TOPK, CHUNK, t, T, write_out;
};
struct MoeTailW2PC { std::uint32_t B, D, F, TOPK, pad0, pad1, pad2, pad3; };
struct RMSNormPC { std::uint32_t R, D; float eps; std::uint32_t pad; };
struct LmHeadPC { std::uint32_t T, D, V, TILE, pad0, pad1, pad2, pad3; };
struct ASMBwdPC {
    std::uint32_t B, H, M, R, DK, DV;
    float tau, tau_w, rho, lam, beta_null;
};
struct ASMSeqPC {
    std::uint32_t B, H, M, R, DK, DV, T;
    float tau, tau_w, rho, lam, beta_null;
};
struct ErsaBwdPC {
    std::uint32_t B, H, T, N, DK, DV, NB, BLOCK, TOPB, WINDOW, QOFF, NBMAX;
    float tau, sink, pad0, pad1;
};
struct IvfPc {
    std::uint32_t B, H, T, N, DK, DV, C, LMAX, TOPB, WINDOW, QPOS, QIDX;
    float tau, sink;
};
static_assert(sizeof(IvfPc) == 56, "IvfPc must be 56 bytes");
struct IvfInitPC {
    std::uint32_t H, N, DK, C, pad0, pad1, pad2, pad3;
};
struct KMeansPC {
    std::uint32_t B, H, N, DK, C, pad0, pad1, pad2;
};
struct FillMemberPC {
    std::uint32_t B, H, N, C, LMAX, pad0, pad1, pad2;
};
struct IncAssignPC {
    std::uint32_t B, H, M, DK, C, pad0, pad1, pad2;
};
struct IncAppendPC {
    std::uint32_t B, H, M, C, LMAX, OFFSET, pad0, pad1;
};
struct IncUpdatePC {
    std::uint32_t B, H, M, DK, C, pad0, pad1, pad2;
    float lr, pad3, pad4, pad5;
};
struct CopySlicePC {
    std::uint32_t H, SRC_N, DST_N, D, ROWS, SRC_ROW0, DST_ROW0, pad0;
};
struct Gr2BwdPC { std::uint32_t B, D, write_out, token, out_token; };
struct L2BwdPC { std::uint32_t N, D, norm_offset, norm_stride; };
struct TransposePC { std::uint32_t R, C, pad0, pad1; };
struct HeadsPC { std::uint32_t H, T, D, pad0; };
struct SumPC { std::uint32_t N, offset, pad0, pad1; };
struct QuantPC { std::uint32_t R, C, G, pad0; };
struct MoeDxPC { std::uint32_t N, E, K, D, F, Fs, pad; };
struct MoeDxI8PC { std::uint32_t E, K, D, F, Fs, Gd, Gf, t, T; };
struct MoeDwI8PC { std::uint32_t T, E, K, D, F, mode; };
struct MoeDwSI8PC { std::uint32_t T, D, F, mode, pad0, pad1; };
struct MoeDwAllPC { std::uint32_t T, E, K, D, F; };
struct MoeDxSharedPC { std::uint32_t T, D, F, Gd, Gf, TILE, pad0; };
struct SumMultiPC { std::uint32_t count; std::uint32_t H[15]; std::uint32_t N[15]; };
struct MoeDwPC { std::uint32_t N, E, K, D, F, mode; };
struct MoeDwSharedPC { std::uint32_t N, D, Fs, mode, pad0, pad1; };
struct SumHeadsPC { std::uint32_t H, N, pad0, pad1; };
struct GrRouteBwdPC { std::uint32_t B, D, row, pad1; };
struct LinDwdb5PC {
    std::uint32_t N;
    std::uint32_t M0, K0, M1, K1, M2, K2, M3, K3, M4, K4;
    std::uint32_t pad;
};

std::uint32_t g_rng = 20261118u;
float 随机() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return (static_cast<float>((g_rng >> 8) & 0xFFFFFFu) / 16777216.0f) * 2.0f - 1.0f;
}

vk张量 建(std::vector<std::uint32_t> 形状, 张量类型 类型, const std::string& 名,
          bool 读多 = false) {
    return vk张量(张量布局::连续(std::move(形状)), 类型, 名, 读多);
}
vk张量 建零(std::uint32_t n, const char* 名) {
    vk张量 t = 建({n}, 张量类型::浮点32, 名);
    std::vector<float> 零(n, 0.0f);
    t.上传(零.data(), 零.size() * 4);
    return t;
}
void 填(vk张量& t, float v) {
    std::vector<float> d(static_cast<std::size_t>(t.布局().元素总数), v);
    t.上传(d.data(), d.size() * 4);
}
void 填零(vk张量& t) { 填(t, 0.0f); }
std::vector<vk张量> 建双份(std::vector<std::uint32_t> 形状, 张量类型 类型,
                           const std::string& 名) {
    std::vector<vk张量> v;
    v.reserve(2);
    for (int i = 0; i < 2; ++i)
        v.push_back(建(形状, 类型, 名 + "." + std::to_string(i)));
    return v;
}

struct 参数 {
    vk张量 p, g, m, v, i8, s;
    std::uint32_t n = 0;
    std::uint32_t rows = 0, cols = 0, G = 0;
    bool 量化 = false;
    std::string 名;
};

struct 记录 {
    vk张量 xq, sx, branch_in, branch_out;
    vk张量 o;
};

struct 层 {
    bool ersa = false;
    参数 *g1ng = nullptr, *g1rw = nullptr, *g1rb = nullptr, *g1ww = nullptr, *g1wb = nullptr;
    参数 *g2ng = nullptr, *g2rw = nullptr, *g2rb = nullptr, *g2ww = nullptr, *g2wb = nullptr;
    参数 *Wq = nullptr, *Wk = nullptr, *Wv = nullptr, *Wo = nullptr, *outNorm = nullptr;
    参数 *etaW = nullptr, *etaB = nullptr, *slotK = nullptr, *gamma = nullptr;
    参数 *betaNull = nullptr, *sink = nullptr;
    参数 *gate = nullptr, *W1 = nullptr, *W2 = nullptr, *shW1 = nullptr, *shWup = nullptr, *shW2 = nullptr;
    vk张量 K, V, beta, q_hist, k_hist, v_hist, block_sum, qkv_norm;
    vk张量 k_cache, v_cache, ctx_k;
    vk张量 cent, counts, member, assign;
    std::vector<vk张量> candSnap, candCnt;
    vk张量 candSnapSeq, candCntSeq;
    vk张量 moe_W1_i8, moe_W1_s, moe_W2_i8, moe_W2_s;
    float beta_null = -2.0f;
    float sink_value = -4.0f;
};

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "用法: %s <spv目录> [steps]\n", argv[0]);
        return 2;
    }
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string spv目录 = argv[1];
    const std::string 模式 = (argc >= 3) ? argv[2] : "";
    const bool smoke = (模式 == "smoke");
    const bool big3b = (模式 == "3b_cache" || 模式 == "3b_cache_smoke");
    const int 步数 = smoke ? 0 : ((argc >= 3) ? std::atoi(argv[2]) : 4);

    asm_ersa_model::ModelConfig cfg;
    cfg.d_model = 32;
    cfg.n_heads = 4;
    cfg.d_k = 8;
    cfg.d_v = 8;
    cfg.d_ff = 64;
    cfg.n_layers = 4;
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
    if (big3b) {
        cfg.d_model = 2048;
        cfg.n_heads = 16;
        cfg.d_k = 128;
        cfg.d_v = 128;
        cfg.d_ff = 384;
        cfg.n_layers = 32;
        cfg.vocab_size = 152064;
        cfg.n_slots = 512;
        cfg.top_r = 16;
        cfg.n_scales = 4;
        cfg.block_size = 64;
        cfg.top_blocks = 4;
        cfg.window = 64;
        cfg.sketch_dim = 32;
        cfg.n_experts = 32;
        cfg.moe_top_k = 4;
        cfg.max_seq_len = 16;
    }

    // cache0/cachefull：无 context / 全候选，用同一候选集对拍 block/cache。
    const bool parity0 = (模式 == "cache0" || 模式 == "cache0_smoke");
    const bool parityfull = (模式 == "cachefull" || 模式 == "cachefull_smoke");
    const bool parity_mode = parity0 || parityfull;
    const bool cache_mode = (模式 == "cache" || 模式 == "cache_smoke" ||
                             parity_mode || big3b);
    const std::uint32_t Tq = 4;
    const std::uint32_t Nctx = (cache_mode && !parity0) ? 8u : 0u;
    const std::uint32_t Ntot = cache_mode ? Nctx + Tq : 0u;
    const std::uint32_t T = cache_mode ? Ntot : Tq;
    const std::uint32_t IVFC = cache_mode
        ? std::max<std::uint32_t>(1u, std::min<std::uint32_t>(64u, Nctx)) : 0u;
    const std::uint32_t IVFLMAX = cache_mode
        ? std::max<std::uint32_t>(64u, std::min<std::uint32_t>(1024u, Ntot)) : 0u;
    const std::uint32_t NBtot = cache_mode
        ? (Ntot + cfg.block_size - 1) / cfg.block_size : 0u;
    const std::uint32_t IVFWIN = parityfull ? T : cfg.window;
    const std::uint32_t IVFTOPB = parity0 ? 0u
        : (parityfull ? IVFC : std::min<std::uint32_t>(8u, IVFC));
    const bool ingest_queries = cache_mode && !parity_mode;
    const std::uint32_t MAXCAND = 1024u;
    bool fwd_cache = cache_mode;
    bool bwd_cache = cache_mode;
    const std::uint32_t B = 1;
    const std::uint32_t D = cfg.d_model, H = cfg.n_heads, DK = cfg.d_k, DV = cfg.d_v;
    const std::uint32_t F = cfg.d_ff, E = cfg.n_experts, V = cfg.vocab_size;
    const std::uint32_t M = cfg.n_slots, R = cfg.top_r, K = cfg.moe_top_k;
    const std::uint32_t Tmax = cfg.max_seq_len;
    const std::uint32_t HDK = H * DK, HDV = H * DV;
    const std::uint32_t NBmax = (Tmax + cfg.block_size - 1) / cfg.block_size;
    const std::uint32_t NBmaxT = (T + cfg.block_size - 1) / cfg.block_size;
    if (parity0) cfg.top_blocks = 0;
    if (parityfull) cfg.top_blocks = NBmaxT;

    流水线调度器& 调度器 = 流水线调度器::获取();
    设备注册信息 注册; 注册.应用名 = "train_main_mb2";
    阶段配置 图形; 阶段配置 计算{阶段配置::自动队列族, 0, 2}; 阶段配置 传输;
    try {
        调度器.启动(注册, {图形, 计算, 传输}, spv目录);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "启动失败: %s\n", e.what());
        return 1;
    }
    std::printf("设备: %s | 算子: %zu\n", 调度器.获取设备().名称(), 调度器.算子数量());
    for (const char* 名 : {"gr_read_bwd_int8", "gr_route_bwd_int8", "gr2_write_bwd",
                           "l2norm_bwd", "asm_step_bwd_multi", "ersa_attn_bwd_query",
                           "ersa_ivf_attn_query", "ersa_ivf_attn_bwd_query",
                           "ersa_ivf_attn_bwd_cand",
                           "ivf_init_centroids", "kmeans_assign", "kmeans_update",
                           "kmeans_fill_members", "kmeans_inc_assign",
                           "kmeans_inc_append", "kmeans_inc_update",
                           "copy_f32_slice", "copy_i32_slice", "copy_i8_slice",
                           "linear_bwd_dx", "linear_bwd_dx_i8", "linear_bwd_dwdb",
                           "moe_bwd_dx_q",
                           "moe_bwd_dw_expert_q", "moe_bwd_dw_shared_q", "adamw_update"}) {
        if (!调度器.存在算子(名)) {
            std::fprintf(stderr, "缺少算子: %s\n", 名);
            调度器.停止();
            return 1;
        }
    }

    asm_ersa_model::ModelWeights 主 = asm_ersa_model::make_random_weights(cfg, 20261118u);

    std::vector<std::unique_ptr<参数>> 全部参数;
    auto 加参数 = [&](std::vector<std::uint32_t> 形状, const std::vector<float>& 初值,
                     const std::string& 名, bool 量化 = false,
                     std::uint32_t rows = 0, std::uint32_t cols = 0) -> 参数* {
        auto q = std::make_unique<参数>();
        q->名 = 名;
        q->n = 1;
        for (auto v : 形状) q->n *= v;
        if (初值.size() != q->n) throw std::runtime_error("初值大小不匹配: " + 名);
        q->p = 建(形状, 张量类型::浮点32, (名 + ".p").c_str());
        q->p.上传(初值.data(), 初值.size() * 4);
        q->g = 建零(q->n, (名 + ".g").c_str());
        if (!big3b) {
            q->m = 建零(q->n, (名 + ".m").c_str());
            q->v = 建零(q->n, (名 + ".v").c_str());
        }
        if (量化) {
            q->量化 = true;
            q->rows = rows;
            q->cols = cols;
            q->G = (cols + 127u) / 128u;
            q->i8 = 建({rows, cols}, 张量类型::整数8, (名 + ".i8").c_str(), true);
            q->s = 建({rows, q->G}, 张量类型::浮点32, (名 + ".s").c_str(), true);
        }
        参数* 结果 = q.get();
        全部参数.push_back(std::move(q));
        return 结果;
    };

    // PIC：host 侧按主模型前向布局量化（仅构造/刷新参数，不进推理计算）。
    auto 量化行 = [](const std::vector<float>& w, std::uint32_t rows, std::uint32_t cols,
                    std::uint32_t group, std::vector<std::int8_t>& q,
                    std::vector<float>& s) {
        const std::uint32_t G = (cols + group - 1) / group;
        q.resize(std::size_t(rows) * cols);
        s.resize(std::size_t(rows) * G);
        for (std::uint32_t r = 0; r < rows; ++r) {
            for (std::uint32_t g = 0; g < G; ++g) {
                const std::uint32_t c0 = g * group, c1 = std::min(c0 + group, cols);
                float mx = 0.0f;
                for (std::uint32_t c = c0; c < c1; ++c)
                    mx = std::max(mx, std::fabs(w[std::size_t(r) * cols + c]));
                const float sc = std::max(mx / 127.0f, 1e-10f);
                s[std::size_t(r) * G + g] = sc;
                for (std::uint32_t c = c0; c < c1; ++c) {
                    int v = int(std::lround(w[std::size_t(r) * cols + c] / sc));
                    q[std::size_t(r) * cols + c] = static_cast<std::int8_t>(std::clamp(v, -127, 127));
                }
            }
        }
    };
    auto 刷新量化 = [&](参数* p, const std::vector<float>& host) {
        if (!p->量化) return;
        std::vector<std::int8_t> q;
        std::vector<float> s;
        量化行(host, p->rows, p->cols, 128, q, s);
        p->i8.上传(q.data(), q.size());
        p->s.上传(s.data(), s.size() * 4);
    };

    // MoE 的 W1/W2 主模型前向是转置布局：W1 [E,F,D]，W2 [E,D,F]。
    auto 量化W1 = [](const std::vector<float>& w, std::uint32_t E_, std::uint32_t D_,
                     std::uint32_t F_, std::vector<std::int8_t>& q,
                     std::vector<float>& s) {
        const std::uint32_t G = (D_ + 127u) / 128u;
        q.resize(std::size_t(E_) * F_ * D_);
        s.resize(std::size_t(E_) * F_ * G);
        for (std::uint32_t e = 0; e < E_; ++e) {
            for (std::uint32_t f = 0; f < F_; ++f) {
                const std::uint32_t row = e * F_ + f;
                for (std::uint32_t g = 0; g < G; ++g) {
                    const std::uint32_t d0 = g * 128, d1 = std::min(d0 + 128u, D_);
                    float mx = 0.0f;
                    for (std::uint32_t d = d0; d < d1; ++d)
                        mx = std::max(mx, std::fabs(w[(std::size_t(e) * D_ + d) * F_ + f]));
                    const float sc = std::max(mx / 127.0f, 1e-10f);
                    s[std::size_t(row) * G + g] = sc;
                    for (std::uint32_t d = d0; d < d1; ++d) {
                        int v = int(std::lround(w[(std::size_t(e) * D_ + d) * F_ + f] / sc));
                        q[std::size_t(row) * D_ + d] = static_cast<std::int8_t>(std::clamp(v, -127, 127));
                    }
                }
            }
        }
    };
    auto 量化W2 = [](const std::vector<float>& w, std::uint32_t E_, std::uint32_t D_,
                     std::uint32_t F_, std::vector<std::int8_t>& q,
                     std::vector<float>& s) {
        const std::uint32_t G = (F_ + 127u) / 128u;
        q.resize(std::size_t(E_) * D_ * F_);
        s.resize(std::size_t(E_) * D_ * G);
        for (std::uint32_t e = 0; e < E_; ++e) {
            for (std::uint32_t d = 0; d < D_; ++d) {
                const std::uint32_t row = e * D_ + d;
                for (std::uint32_t g = 0; g < G; ++g) {
                    const std::uint32_t f0 = g * 128, f1 = std::min(f0 + 128u, F_);
                    float mx = 0.0f;
                    for (std::uint32_t f = f0; f < f1; ++f)
                        mx = std::max(mx, std::fabs(w[(std::size_t(e) * F_ + f) * D_ + d]));
                    const float sc = std::max(mx / 127.0f, 1e-10f);
                    s[std::size_t(row) * G + g] = sc;
                    for (std::uint32_t f = f0; f < f1; ++f) {
                        int v = int(std::lround(w[(std::size_t(e) * F_ + f) * D_ + d] / sc));
                        q[std::size_t(row) * F_ + f] = static_cast<std::int8_t>(std::clamp(v, -127, 127));
                    }
                }
            }
        }
    };

    const std::uint32_t Gd = (D + 127u) / 128u;
    const std::uint32_t Gf = (F + 127u) / 128u;
    const std::uint32_t Ghdv = (HDV + 127u) / 128u;

    参数* embed = 加参数({V, D}, 主.embed, "embed", true, V, D);
    参数* lmhead = 加参数({V, D}, 主.lm_head, "lm_head", true, V, D);
    参数* finalNorm = 加参数({D}, 主.out_norm, "final_norm");
    // embed/lmhead 的 scale 是每行一个（当前 kernel 的 embed 路径只读一个 scale）
    刷新量化(embed, 主.embed);
    刷新量化(lmhead, 主.lm_head);
    if (big3b) {
        主.embed.clear(); 主.embed.shrink_to_fit();
        主.lm_head.clear(); 主.lm_head.shrink_to_fit();
        主.out_norm.clear(); 主.out_norm.shrink_to_fit();
    }

    auto 合并4 = [](const std::vector<float>* v) {
        std::vector<float> 出(4 * std::size_t(v[0].size()), 1.0f);
        for (int i = 0; i < 4; ++i)
            std::copy(v[i].begin(), v[i].end(), 出.begin() + i * v[i].size());
        return 出;
    };

    std::vector<层> 层表;
    层表.resize(cfg.n_layers);
    std::vector<vk张量> asmQAll(cfg.n_layers), asmKAll(cfg.n_layers), asmVAll(cfg.n_layers);
    std::vector<vk张量> asmEtaAll(cfg.n_layers), asmKOldAll(cfg.n_layers);
    std::vector<vk张量> asmVOldAll(cfg.n_layers), asmBoldAll(cfg.n_layers);
    std::vector<vk张量> asmIdxAll(cfg.n_layers), asmXqAll(cfg.n_layers);
    std::vector<vk张量> asmSxAll(cfg.n_layers);
    for (std::uint32_t li = 0; li < cfg.n_layers; ++li) {
        const asm_ersa_model::LayerWeights& lw = 主.layers[li];
        层& L = 层表[li];
        L.ersa = lw.is_ersa;
        if (!L.ersa) {
            const std::string pre = "L" + std::to_string(li) + ".seq.";
            asmQAll[li] = 建({T, HDK}, 张量类型::浮点32, pre + "q");
            asmKAll[li] = 建({T, HDK}, 张量类型::浮点32, pre + "k");
            asmVAll[li] = 建({T, HDV}, 张量类型::浮点32, pre + "v");
            asmEtaAll[li] = 建({T, B}, 张量类型::浮点32, pre + "eta");
            asmKOldAll[li] = 建({T, B * H * M * DK}, 张量类型::浮点32, pre + "Kold");
            asmVOldAll[li] = 建({T, B * H * M * DV}, 张量类型::浮点32, pre + "Vold");
            asmBoldAll[li] = 建({T, B * H * M}, 张量类型::浮点32, pre + "Bold");
            asmIdxAll[li] = 建({T, B * H * 3 * R}, 张量类型::浮点32, pre + "idx");
            asmXqAll[li] = 建({T, D}, 张量类型::整数8, pre + "xq");
            asmSxAll[li] = 建({T}, 张量类型::浮点32, pre + "sx");
        }
        L.beta_null = lw.asm_w.beta_null;
        L.sink_value = lw.ersa_w.sink;
        L.g1ng = 加参数({4, D}, 合并4(lw.gr1.norm_g), "L" + std::to_string(li) + ".g1ng");
        L.g1rw = 加参数({4, 4 * D}, lw.gr1.read_w, "L" + std::to_string(li) + ".g1rw");
        L.g1rb = 加参数({4}, lw.gr1.read_b, "L" + std::to_string(li) + ".g1rb");
        L.g1ww = 加参数({4, D}, lw.gr1.write_w, "L" + std::to_string(li) + ".g1ww");
        L.g1wb = 加参数({4}, lw.gr1.write_b, "L" + std::to_string(li) + ".g1wb");
        L.g2ng = 加参数({4, D}, 合并4(lw.gr2.norm_g), "L" + std::to_string(li) + ".g2ng");
        L.g2rw = 加参数({4, 4 * D}, lw.gr2.read_w, "L" + std::to_string(li) + ".g2rw");
        L.g2rb = 加参数({4}, lw.gr2.read_b, "L" + std::to_string(li) + ".g2rb");
        L.g2ww = 加参数({4, D}, lw.gr2.write_w, "L" + std::to_string(li) + ".g2ww");
        L.g2wb = 加参数({4}, lw.gr2.write_b, "L" + std::to_string(li) + ".g2wb");

        const asm_ersa_model::ASMWeights* aw = lw.is_ersa ? nullptr : &lw.asm_w;
        const asm_ersa_model::ERSAWeights* ew = lw.is_ersa ? &lw.ersa_w : nullptr;
        const std::vector<float>& Wq = lw.is_ersa ? lw.ersa_w.Wq : lw.asm_w.Wq;
        const std::vector<float>& Wk = lw.is_ersa ? lw.ersa_w.Wk : lw.asm_w.Wk;
        const std::vector<float>& Wv = lw.is_ersa ? lw.ersa_w.Wv : lw.asm_w.Wv;
        const std::vector<float>& Wo = lw.is_ersa ? lw.ersa_w.Wo : lw.asm_w.Wo;
        const std::vector<float>& on = lw.is_ersa ? lw.ersa_w.out_norm : lw.asm_w.out_norm;
        L.Wq = 加参数({HDK, D}, Wq, "L" + std::to_string(li) + ".Wq", true, HDK, D);
        L.Wk = 加参数({HDK, D}, Wk, "L" + std::to_string(li) + ".Wk", true, HDK, D);
        L.Wv = 加参数({HDV, D}, Wv, "L" + std::to_string(li) + ".Wv", true, HDV, D);
        L.Wo = 加参数({D, HDV}, Wo, "L" + std::to_string(li) + ".Wo", true, D, HDV);
        L.outNorm = 加参数({D}, on, "L" + std::to_string(li) + ".out_norm");
        if (lw.is_ersa) {
            L.sink = 加参数({1}, {lw.ersa_w.sink}, "L" + std::to_string(li) + ".sink");
        } else {
            L.etaW = 加参数({D}, lw.asm_w.w_eta_w, "L" + std::to_string(li) + ".etaW");
            L.etaB = 加参数({1}, lw.asm_w.w_eta_b, "L" + std::to_string(li) + ".etaB");
            L.slotK = 加参数({M, DK}, lw.asm_w.slot_init_k, "L" + std::to_string(li) + ".slotK");
            L.gamma = 加参数({M}, lw.asm_w.gamma, "L" + std::to_string(li) + ".gamma");
            L.betaNull = 加参数({1}, {lw.asm_w.beta_null}, "L" + std::to_string(li) + ".beta_null");
        }

        L.gate = 加参数({E, D}, lw.moe.gate, "L" + std::to_string(li) + ".gate");
        L.W1 = 加参数({E * D, F}, lw.moe.W1, "L" + std::to_string(li) + ".W1");
        L.W2 = 加参数({E * F, D}, lw.moe.W2, "L" + std::to_string(li) + ".W2");
        if (big3b) {
            // 3B 只测 cache 前向/反向，不更新权重；int8 直读路径不再需要
            // routed MoE 的 FP32 master，释放两份 6.4GB 级缓冲。
            L.W1->p = vk张量();
            L.W2->p = vk张量();
        }
        L.shW1 = 加参数({F, D}, lw.moe.sharedW1, "L" + std::to_string(li) + ".shW1", true, F, D);
        L.shWup = 加参数({F, D}, lw.moe.sharedWup, "L" + std::to_string(li) + ".shWup", true, F, D);
        L.shW2 = 加参数({D, F}, lw.moe.sharedW2, "L" + std::to_string(li) + ".shW2", true, D, F);

        // 前向量化（复用 host 值；后续 AdamW 后接 VK 重量化）
        刷新量化(L.Wq, Wq); 刷新量化(L.Wk, Wk); 刷新量化(L.Wv, Wv); 刷新量化(L.Wo, Wo);
        刷新量化(L.shW1, lw.moe.sharedW1); 刷新量化(L.shWup, lw.moe.sharedWup);
        刷新量化(L.shW2, lw.moe.sharedW2);
        {
            std::vector<std::int8_t> q1, q2;
            std::vector<float> s1, s2;
            量化W1(lw.moe.W1, E, D, F, q1, s1);
            量化W2(lw.moe.W2, E, D, F, q2, s2);
            L.moe_W1_i8 = 建({E * F, D}, 张量类型::整数8, "L" + std::to_string(li) + ".W1f_i8", true);
            L.moe_W1_s = 建({E * F, (D + 127) / 128}, 张量类型::浮点32, "L" + std::to_string(li) + ".W1f_s", true);
            L.moe_W2_i8 = 建({E * D, F}, 张量类型::整数8, "L" + std::to_string(li) + ".W2f_i8", true);
            L.moe_W2_s = 建({E * D, (F + 127) / 128}, 张量类型::浮点32, "L" + std::to_string(li) + ".W2f_s", true);
            L.moe_W1_i8.上传(q1.data(), q1.size());
            L.moe_W1_s.上传(s1.data(), s1.size() * 4);
            L.moe_W2_i8.上传(q2.data(), q2.size());
            L.moe_W2_s.上传(s2.data(), s2.size() * 4);
        }

        // 层状态/历史
        if (lw.is_ersa) {
            L.q_hist = 建({H, T, DK}, 张量类型::浮点32, "ersa.q_hist");
            L.k_hist = 建({H, T, DK}, 张量类型::浮点32, "ersa.k_hist");
            L.v_hist = 建({H, T, DV}, 张量类型::浮点32, "ersa.v_hist");
            L.block_sum = 建({H, NBmaxT, DK}, 张量类型::浮点32, "ersa.block_sum");
            if (cache_mode) {
                const std::string p = "L" + std::to_string(li) + ".cache";
                L.k_cache = 建({H, Ntot, DK}, 张量类型::浮点32, p + ".k");
                L.v_cache = 建({H, Ntot, DV}, 张量类型::浮点32, p + ".v");
                L.cent = 建({H, IVFC, DK}, 张量类型::浮点32, p + ".cent");
                L.counts = 建({H, IVFC}, 张量类型::整数32, p + ".counts");
                L.member = 建({H, IVFC, IVFLMAX}, 张量类型::整数32, p + ".member");
                L.candSnap.reserve(Tq);
                L.candCnt.reserve(Tq);
                for (std::uint32_t q = 0; q < Tq; ++q) {
                    L.candSnap.push_back(建({H, MAXCAND}, 张量类型::整数32,
                                            p + ".cand" + std::to_string(q)));
                    L.candCnt.push_back(建({H}, 张量类型::整数32,
                                           p + ".candn" + std::to_string(q)));
                }
                L.candSnapSeq = 建({H, Tq, MAXCAND}, 张量类型::整数32, p + ".candseq");
                L.candCntSeq = 建({H, Tq}, 张量类型::整数32, p + ".candcntseq");
                if (Nctx > 0) {
                    L.ctx_k = 建({H, Nctx, DK}, 张量类型::浮点32, p + ".ctxk");
                    L.assign = 建({H, Nctx}, 张量类型::整数32, p + ".assign");
                } else {
                    std::vector<float> zc(std::size_t(H) * IVFC * DK, 0.0f);
                    std::vector<std::int32_t> zi(std::size_t(H) * IVFC, 0);
                    std::vector<std::int32_t> zm(std::size_t(H) * IVFC * IVFLMAX, 0);
                    L.cent.上传(zc.data(), zc.size() * 4);
                    L.counts.上传(zi.data(), zi.size() * 4);
                    L.member.上传(zm.data(), zm.size() * 4);
                }
            }
        } else {
            L.K = 建({B, H, M, DK}, 张量类型::浮点32, "asm.K");
            L.V = 建({B, H, M, DV}, 张量类型::浮点32, "asm.V");
            L.beta = 建({B, H, M}, 张量类型::浮点32, "asm.beta");
        }
        L.qkv_norm = 建({2, H, T}, 张量类型::浮点32, "L" + std::to_string(li) + ".qkv_norm");
        if (big3b) 主.layers[li] = asm_ersa_model::LayerWeights{};
    }

    if (big3b) 主 = asm_ersa_model::ModelWeights{};

    // 全局前向缓冲
    vk张量 branches = 建({B, 4, D}, 张量类型::浮点32, "branches");
    vk张量 out_tokens = 建({Tmax, D}, 张量类型::浮点32, "out_tokens");
    vk张量 normed_q = 建({Tmax, D}, 张量类型::整数8, "normed_q");
    vk张量 normed_s = 建({Tmax}, 张量类型::浮点32, "normed_s");
    vk张量 logits = 建({Tmax, V}, 张量类型::浮点32, "logits");
    vk张量 ids = 建({Tmax}, 张量类型::整数32, "ids");
    vk张量 targets = 建({Tmax}, 张量类型::整数32, "targets");
    vk张量 xq = 建({1, D}, 张量类型::整数8, "xq");
    vk张量 sx = 建({1}, 张量类型::浮点32, "sx");
    vk张量 eta = 建({1}, 张量类型::浮点32, "eta");
    vk张量 o = 建({1, HDV}, 张量类型::浮点32, "o");
    vk张量 attn_out = 建({1, HDV}, 张量类型::浮点32, "attn_out");
    vk张量 wo_y = 建({1, D}, 张量类型::浮点32, "wo_y");
    vk张量 wo_deq = 建({1, HDV}, 张量类型::浮点32, "wo_deq");
    vk张量 x2 = 建({1, D}, 张量类型::浮点32, "x2");
    vk张量 moe_idx = 建({1, K}, 张量类型::整数32, "moe_idx");
    vk张量 moe_w = 建({1, K}, 张量类型::浮点32, "moe_w");
    vk张量 h_q = 建({1, K, F}, 张量类型::整数8, "h_q");
    vk张量 h_s = 建({1, K, Gf}, 张量类型::浮点32, "h_s");
    vk张量 expert_out = 建({1, K, D}, 张量类型::浮点32, "expert_out");
    vk张量 y_moe = 建({1, D}, 张量类型::浮点32, "y_moe");
    vk张量 sh_hq = 建({1, F}, 张量类型::整数8, "sh_hq");
    vk张量 sh_hs = 建({1, Gf}, 张量类型::浮点32, "sh_hs");
    auto moePreV = 建双份({T, K, F}, 张量类型::浮点32, "moe.pre");
    auto moeHqV = 建双份({T, K, F}, 张量类型::浮点32, "moe.hq");
    auto moeDhV = 建双份({T, K, F}, 张量类型::浮点32, "moe.dh");
    auto moeShPreV = 建双份({T, F}, 张量类型::浮点32, "moe.shpre");
    auto moeShUpV = 建双份({T, F}, 张量类型::浮点32, "moe.shup");
    auto moeShHqV = 建双份({T, F}, 张量类型::浮点32, "moe.shhq");
    auto moeShDhV = 建双份({T, F}, 张量类型::浮点32, "moe.shdh");
    auto moeShSiluV = 建双份({T, F}, 张量类型::浮点32, "moe.shsilu");
    auto moeXsV = 建双份({T, D}, 张量类型::浮点32, "moe.xs");
    auto moeDysV = 建双份({T, D}, 张量类型::浮点32, "moe.dys");
    auto moeIdxSV = 建双份({T, K}, 张量类型::整数32, "moe.idxs");
    auto moeKwSV = 建双份({T, K}, 张量类型::浮点32, "moe.kws");
    vk张量 moeRouteList = 建({E * T * K}, 张量类型::整数32, "moe.routelist");
    vk张量 moeRouteCnt = 建({E}, 张量类型::整数32, "moe.routecnt");
    vk张量 qkv_q = 建({1, HDK}, 张量类型::浮点32, "qkv_q");
    vk张量 qkv_k = 建({1, HDK}, 张量类型::浮点32, "qkv_k");
    vk张量 qkv_v = 建({1, HDV}, 张量类型::浮点32, "qkv_v");
    vk张量 dbg = 建({1, H, 3 * R}, 张量类型::浮点32, "dbg");
    vk张量 block_dummy = 建({1}, 张量类型::浮点32, "block_dummy");
    vk张量 eta_w_dummy = 建零(D, "eta_w_dummy");
    vk张量 eta_b_dummy = 建零(1, "eta_b_dummy");

    std::vector<std::int32_t> idh(Tq), th(Tq), idsAll(T);
    for (std::uint32_t t = 0; t < Tq; ++t) {
        idh[t] = int((t * 7 + 3) % V);
        th[t] = int((idh[t] + 1) % V);
    }
    for (std::uint32_t t = 0; t < T; ++t) {
        if (cache_mode && t < Nctx) idsAll[t] = int((t * 5 + 1) % V);
        else idsAll[t] = idh[t - Nctx];
    }
    ids.上传(idsAll.data(), idsAll.size() * 4);
    targets.上传(th.data(), th.size() * 4);

    vk张量 ivfAssignNew, ivfOverflow;
    if (cache_mode) {
        ivfAssignNew = 建({H}, 张量类型::整数32, "cache.ivf_assign_new");
        ivfOverflow = 建({H, IVFC}, 张量类型::整数32, "cache.ivf_overflow");
        std::vector<std::int32_t> z(std::size_t(H) * IVFC, 0);
        ivfOverflow.上传(z.data(), z.size() * 4);
    }

    std::vector<vk张量> branchMidAll, yMoeAll, branchInAll, woYAll, x2All, oDeqAll;
    branchMidAll.reserve(cfg.n_layers);
    yMoeAll.reserve(cfg.n_layers);
    branchInAll.reserve(cfg.n_layers);
    woYAll.reserve(cfg.n_layers);
    x2All.reserve(cfg.n_layers);
    oDeqAll.reserve(cfg.n_layers);
    for (std::uint32_t li = 0; li < cfg.n_layers; ++li) {
        branchMidAll.push_back(建({T, 4, D}, 张量类型::浮点32,
                                  "seq.branch_mid." + std::to_string(li)));
        yMoeAll.push_back(建({T, D}, 张量类型::浮点32,
                             "seq.y_moe." + std::to_string(li)));
        branchInAll.push_back(建({T, 4, D}, 张量类型::浮点32,
                                 "seq.branch_in." + std::to_string(li)));
        woYAll.push_back(建({T, D}, 张量类型::浮点32,
                            "seq.wo_y." + std::to_string(li)));
        x2All.push_back(建({T, D}, 张量类型::浮点32,
                           "seq.x2." + std::to_string(li)));
        oDeqAll.push_back(建({T, HDV}, 张量类型::浮点32,
                             "seq.o_deq." + std::to_string(li)));
    }

    // MoE 反向所需的逐层记录：前向按行直写，bwd 不再逐 token 收集。
    std::vector<vk张量> x2DeqAll, moeIdxRecAll, moeKwRecAll;
    x2DeqAll.reserve(cfg.n_layers);
    moeIdxRecAll.reserve(cfg.n_layers);
    moeKwRecAll.reserve(cfg.n_layers);
    for (std::uint32_t li = 0; li < cfg.n_layers; ++li) {
        const std::string 名 = std::to_string(li);
        x2DeqAll.push_back(建({T, D}, 张量类型::浮点32, "seq.x2_deq." + 名));
        moeIdxRecAll.push_back(建({T, K}, 张量类型::整数32, "seq.moe_idx." + 名));
        moeKwRecAll.push_back(建({T, K}, 张量类型::浮点32, "seq.moe_w." + 名));
    }

    std::vector<std::vector<std::unique_ptr<记录>>> 记录表(T);
    for (std::uint32_t t = 0; t < T; ++t) {
        记录表[t].resize(cfg.n_layers);
        for (std::uint32_t li = 0; li < cfg.n_layers; ++li) {
            auto r = std::make_unique<记录>();
            const std::string 前缀 = "rec." + std::to_string(t) + "." + std::to_string(li);
            r->xq = 建({1, D}, 张量类型::整数8, 前缀 + ".xq");
            r->sx = 建({1}, 张量类型::浮点32, 前缀 + ".sx");
            r->branch_in = 建({B, 4, D}, 张量类型::浮点32, 前缀 + ".bin");
            r->branch_out = 建({B, 4, D}, 张量类型::浮点32, 前缀 + ".bout");
            r->o = 建({1, HDV}, 张量类型::浮点32, 前缀 + ".o");
            记录表[t][li] = std::move(r);
        }
    }

    bool 性能采样 = false;
    std::map<std::string, std::pair<double, std::uint64_t>> 性能表;

    // 非采样时把层内算子攒成一条算子串：一次提交、一次等待。
    // 任何主机侧读写（下载/上传/拷贝）前必须先冲刷，保证顺序。
    std::vector<算子步骤> 批步骤;
    批步骤.reserve(128);
    auto 冲刷批 = [&]() {
        if (批步骤.empty()) return;
        const std::uint64_t id = 调度器.提交算子串(std::move(批步骤));
        调度器.等待(id);
        if (!调度器.已完成(id))
            throw std::runtime_error(std::string("算子串失败: ") +
                                     调度器.查询失败原因(id));
        批步骤.clear();
    };

    auto 提交2 = [&](const char* 名, std::vector<张量句柄> 句柄, const void* pc,
                    std::uint32_t pcB, std::uint32_t gx, std::uint32_t gy,
                    std::uint32_t gz = 1) {
        if (!性能采样) {
            算子步骤 步骤;
            步骤.算子名 = 名;
            步骤.张量列表 = std::move(句柄);
            步骤.推送常量.resize(pcB);
            if (pcB > 0 && pc != nullptr)
                std::memcpy(步骤.推送常量.data(), pc, pcB);
            步骤.派发 = 派发规格{gx, gy, gz};
            批步骤.push_back(std::move(步骤));
            return;
        }
        std::chrono::steady_clock::time_point 起;
        起 = std::chrono::steady_clock::now();
        const std::uint64_t id = 调度器.提交自定义算子(
            名, std::move(句柄), pc, pcB, 派发规格{gx, gy, gz});
        调度器.等待(id);
        if (!调度器.已完成(id))
            throw std::runtime_error(std::string("算子失败: ") + 名 + " | " +
                                     调度器.查询失败原因(id));
        {
            const auto 止 = std::chrono::steady_clock::now();
            auto& e = 性能表[名];
            e.first += std::chrono::duration<double, std::milli>(止 - 起).count();
            e.second += 1;
        }
    };
    auto 提交 = [&](const char* 名, std::vector<张量句柄> 句柄, const void* pc,
                   std::uint32_t pcB, std::uint32_t 线程) {
        提交2(名, std::move(句柄), pc, pcB, (线程 + 31u) / 32u, 1, 1);
    };


    vk张量 dlogits = 建({Tq, V}, 张量类型::浮点32, "dlogits");
    vk张量 ce_loss = 建({Tq}, 张量类型::浮点32, "ce_loss");
    vk张量 dNormed = 建({Tq, D}, 张量类型::浮点32, "dNormed");
    vk张量 dOutQ = 建({Tq, D}, 张量类型::浮点32, "dOutQ");
    vk张量 out_q = 建({Tq, D}, 张量类型::浮点32, "out_q");
    vk张量 dOutTokens = 建({T, D}, 张量类型::浮点32, "dOutTokens");
    vk张量 dFinalNorm = 建({D}, 张量类型::浮点32, "dFinalNorm");
    vk张量 dLmHead = 建({V, D}, 张量类型::浮点32, "dLmHead");
    vk张量 dLmHeadDb = 建({V}, 张量类型::浮点32, "dLmHeadDb");
    vk张量 normed_f32 = 建({Tq, D}, 张量类型::浮点32, "normed_f32");
    vk张量 zero_TD = 建({T, D}, 张量类型::浮点32, "zero_TD");
    填零(zero_TD);
    vk张量 dBranchUpperAll = 建({T, 4, D}, 张量类型::浮点32, "dBranchUpperAll");
    填零(dBranchUpperAll);

    auto 累加 = [&](参数* p, const vk张量& tmp) {
        if (p == nullptr) return;
        if (tmp.字节大小() != p->g.字节大小())
            throw std::runtime_error("梯度累加大小不匹配: " + p->名);
        const SumPC pc{static_cast<std::uint32_t>(p->n), 0, 0, 0};
        提交2("add_inplace", {p->g.取句柄(), tmp.取句柄()}, &pc, sizeof(pc),
              (p->n + 255u) / 256u, 1, 1);
    };
    auto 清零梯度 = [&]() {
        冲刷批();
        for (auto& p : 全部参数) 填零(p->g);
    };
    auto 求标量 = [&](const vk张量& in, std::uint32_t n, std::uint32_t offset = 0) {
        vk张量 出 = 建({1}, 张量类型::浮点32, "sum_out");
        const SumPC pc{n, offset, 0, 0};
        提交2("sum_f32", {in.取句柄(), 出.取句柄()}, &pc, sizeof(pc), 1, 1, 1);
        return 出;
    };
    auto 范数 = [&](const vk张量& t) {
        std::vector<float> v(static_cast<std::size_t>(t.布局().元素总数));
        t.下载(v.data(), v.size() * 4);
        double s = 0.0;
        for (float x : v) s += double(x) * x;
        return float(std::sqrt(s));
    };

    auto 反向头 = [&]() {
        // CE：同一 kernel 输出 dlogits 与逐行 loss。
        const 规格4 ce{Tq, V, 0, 0};
        提交2("fused_ce_bwd",
              {logits.取句柄(), targets.取句柄(), dlogits.取句柄(), ce_loss.取句柄()},
              &ce, sizeof(ce), 1, 1, 1);
        std::vector<float> lossh(Tq);
        冲刷批();
        ce_loss.下载(lossh.data(), lossh.size() * 4);
        double 平均 = 0.0;
        for (float x : lossh) 平均 += x;
        平均 /= double(Tq);

        const QuantPC dq{Tq * D, D, 0, 0};
        提交2("dequant_row_i8",
              {normed_q.取句柄(), normed_s.取句柄(), normed_f32.取句柄()},
              &dq, sizeof(dq), (Tq * D + 255u) / 256u, 1, 1);

        const 规格4 lw{Tq, D, V, 0};
        提交2("linear_bwd_dwdb",
              {normed_f32.取句柄(), dlogits.取句柄(), lmhead->g.取句柄(), dLmHeadDb.取句柄()},
              &lw, sizeof(lw), (V * D + 63u) / 64u, 1, 1);
        const LinI8PC lhi{Tq, D, V, Gd, 0, 0, 0, 0};
        提交2("linear_bwd_dx_i8",
              {dlogits.取句柄(), lmhead->i8.取句柄(), lmhead->s.取句柄(),
               dNormed.取句柄()},
              &lhi, sizeof(lhi), (Tq * D + 63u) / 64u, 1, 1);
        const 规格F rn{Tq, D, 1e-6f, 0.0f};
        提交2("rmsnorm_bwd_dx",
              {out_q.取句柄(), dNormed.取句柄(), finalNorm->p.取句柄(),
               dOutQ.取句柄()},
              &rn, sizeof(rn), (Tq + 63u) / 64u, 1, 1);
        提交2("rmsnorm_bwd_dw",
              {out_q.取句柄(), dNormed.取句柄(), finalNorm->g.取句柄()},
              &rn, sizeof(rn), (D + 63u) / 64u, 1, 1);
        冲刷批();
        填零(dOutTokens);
        const CopySlicePC dcp{1, Tq, T, D, Tq, 0, Nctx, 0};
        提交2("copy_f32_slice", {dOutQ.取句柄(), dOutTokens.取句柄()},
              &dcp, sizeof(dcp), Tq, 1, 1);
        冲刷批();
        if (!big3b) {
            std::printf("头反向: loss=%.6f | ||dOutTokens||=%.6g | ||dLmHead||=%.6g | ||dFinalNorm||=%.6g\n",
                        平均, 范数(dOutTokens), 范数(lmhead->g), 范数(finalNorm->g));
        } else {
            std::printf("头反向: loss=%.6f\n", 平均);
        }
        return 平均;
    };

    // ── 反向工作张量（逐 token 复用；参数梯度累加进 参数::g）──
    auto dYAllV = 建双份({T, D}, 张量类型::浮点32, "bwd.dYAll");
    auto dBranchMidAllV = 建双份({T, 4, D}, 张量类型::浮点32, "bwd.dBmidAll");
    auto dW2wV = 建双份({4, D}, 张量类型::浮点32, "bwd.dW2w");
    auto db2wV = 建双份({4}, 张量类型::浮点32, "bwd.db2w");
    vk张量 dW2wAll = 建({T, 4, D}, 张量类型::浮点32, "bwd.dW2wAll");
    vk张量 db2wAll = 建({T, 4}, 张量类型::浮点32, "bwd.db2wAll");
    auto dX2AllV = 建双份({T, D}, 张量类型::浮点32, "bwd.dX2All");
    auto dScoreAllV = 建双份({T, E}, 张量类型::浮点32, "bwd.dScoreAll");
    vk张量 dGateW = 建({E, D}, 张量类型::浮点32, "bwd.dGateW");
    vk张量 dGateDb = 建({E}, 张量类型::浮点32, "bwd.dGateDb");
    auto dW1V = 建双份({E * D, F}, 张量类型::浮点32, "bwd.dW1");
    auto dW2V = 建双份({E * F, D}, 张量类型::浮点32, "bwd.dW2");
    auto dShW1TV = 建双份({D, F}, 张量类型::浮点32, "bwd.dShW1T");
    auto dShWupTV = 建双份({D, F}, 张量类型::浮点32, "bwd.dShWupT");
    auto dShW2TV = 建双份({F, D}, 张量类型::浮点32, "bwd.dShW2T");
    auto dShW1V = 建双份({F, D}, 张量类型::浮点32, "bwd.dShW1");
    auto dShWupV = 建双份({F, D}, 张量类型::浮点32, "bwd.dShWup");
    auto dShW2V = 建双份({D, F}, 张量类型::浮点32, "bwd.dShW2");

    vk张量 dOutNorm = 建({D}, 张量类型::浮点32, "bwd.dOutNorm");
    vk张量 dW1W = 建({4, D}, 张量类型::浮点32, "bwd.dW1W");
    vk张量 dW1B = 建({4}, 张量类型::浮点32, "bwd.dW1B");
    vk张量 dG2G = 建({4, D}, 张量类型::浮点32, "bwd.dG2G");
    vk张量 dG2W = 建({4, 4 * D}, 张量类型::浮点32, "bwd.dG2W");
    vk张量 dG2B = 建({4}, 张量类型::浮点32, "bwd.dG2B");
    vk张量 dWoYAll = 建({T, D}, 张量类型::浮点32, "bwd.dWoYAll");
    vk张量 dOAll = 建({T, HDV}, 张量类型::浮点32, "bwd.dOAll");
    vk张量 dBranchRAll = 建({T, 4, D}, 张量类型::浮点32, "bwd.dBrouteAll");
    vk张量 dOutNormAll = 建({T, D}, 张量类型::浮点32, "bwd.dOutNormAll");
    vk张量 dW1WAll = 建({T, 4, D}, 张量类型::浮点32, "bwd.dW1WAll");
    vk张量 dW1BAll = 建({T, 4}, 张量类型::浮点32, "bwd.dW1BAll");
    vk张量 dG2GAll = 建({T, 4, D}, 张量类型::浮点32, "bwd.dG2GAll");
    vk张量 dG2WAll = 建({T, 4, 4 * D}, 张量类型::浮点32, "bwd.dG2WAll");
    vk张量 dG2BAll = 建({T, 4}, 张量类型::浮点32, "bwd.dG2BAll");
    vk张量 dWo = 建({D, HDV}, 张量类型::浮点32, "bwd.dWo");
    vk张量 dWoDb = 建({D}, 张量类型::浮点32, "bwd.dWoDb");

    vk张量 dKnSeq = 建({B, H, M, DK}, 张量类型::浮点32, "bwd.dKnSeq");
    vk张量 dVnSeq = 建({B, H, M, DV}, 张量类型::浮点32, "bwd.dVnSeq");
    vk张量 dBnSeq = 建({B, H, M}, 张量类型::浮点32, "bwd.dBnSeq");
    vk张量 dbnH = 建({H}, 张量类型::浮点32, "bwd.dbnH");
    vk张量 dgammaH = 建({H, M}, 张量类型::浮点32, "bwd.dgammaH");
    vk张量 dGamma = 建({M}, 张量类型::浮点32, "bwd.dGamma");
    vk张量 dWq = 建({HDK, D}, 张量类型::浮点32, "bwd.dWq");
    vk张量 dWk = 建({HDK, D}, 张量类型::浮点32, "bwd.dWk");
    vk张量 dWv = 建({HDV, D}, 张量类型::浮点32, "bwd.dWv");
    vk张量 dWqDb = 建({HDK}, 张量类型::浮点32, "bwd.dWqDb");
    vk张量 dWkDb = 建({HDK}, 张量类型::浮点32, "bwd.dWkDb");
    vk张量 dWvDb = 建({HDV}, 张量类型::浮点32, "bwd.dWvDb");

    vk张量 dNormG = 建({4, D}, 张量类型::浮点32, "bwd.dNormG");
    vk张量 dReadW = 建({4, 4 * D}, 张量类型::浮点32, "bwd.dReadW");
    vk张量 dReadB = 建({4}, 张量类型::浮点32, "bwd.dReadB");
    vk张量 dEtaW = 建({D}, 张量类型::浮点32, "bwd.dEtaW");
    vk张量 dEtaB = 建({1}, 张量类型::浮点32, "bwd.dEtaB");
    vk张量 dInAll = 建({T, 4, D}, 张量类型::浮点32, "bwd.dInAll");
    vk张量 dNormGAll = 建({T, 4, D}, 张量类型::浮点32, "bwd.dNormGAll");
    vk张量 dReadWAll = 建({T, 4, 4 * D}, 张量类型::浮点32, "bwd.dReadWAll");
    vk张量 dReadBAll = 建({T, 4}, 张量类型::浮点32, "bwd.dReadBAll");
    vk张量 dEtaWAll = 建({T, D}, 张量类型::浮点32, "bwd.dEtaWAll");
    vk张量 dEtaBAll = 建({T}, 张量类型::浮点32, "bwd.dEtaBAll");
    vk张量 dEtaAll = 建({T, H}, 张量类型::浮点32, "bwd.dEtaAll");
    vk张量 dEtaZero = 建({T, H}, 张量类型::浮点32, "bwd.dEtaZero");
    填零(dEtaZero);
    vk张量 dEmbedAll = 建({T, D}, 张量类型::浮点32, "bwd.dEmbedAll");
    vk张量 dTable = 建({V, D}, 张量类型::浮点32, "bwd.dTable");

    // ERSA 历史梯度
    vk张量 dQH = 建({H, T, DK}, 张量类型::浮点32, "bwd.dQH");
    vk张量 dQ31, dK31, dV31, dXAll31;
    if (big3b) {
        dQ31 = 建({H, T, DK}, 张量类型::浮点32, "probe.dQ31");
        dK31 = 建({H, T, DK}, 张量类型::浮点32, "probe.dK31");
        dV31 = 建({H, T, DV}, 张量类型::浮点32, "probe.dV31");
        dXAll31 = 建({T, D}, 张量类型::浮点32, "probe.dX31");
    }
    vk张量 dKH = 建({H, T, DK}, 张量类型::浮点32, "bwd.dKH");
    vk张量 dVH = 建({H, T, DV}, 张量类型::浮点32, "bwd.dVH");
    vk张量 dSinkH = 建({H}, 张量类型::浮点32, "bwd.dsink");
    vk张量 dqRawH = 建({H, T, DK}, 张量类型::浮点32, "bwd.dqRawH");
    vk张量 dkRawH = 建({H, T, DK}, 张量类型::浮点32, "bwd.dkRawH");
    vk张量 dqRawAll = 建({T, HDK}, 张量类型::浮点32, "bwd.dqRawAll");
    vk张量 dkRawAll = 建({T, HDK}, 张量类型::浮点32, "bwd.dkRawAll");
    vk张量 dvAll = 建({T, HDV}, 张量类型::浮点32, "bwd.dvAll");
    vk张量 dqT = 建({T, HDK}, 张量类型::浮点32, "bwd.dqT");
    vk张量 dkT = 建({T, HDK}, 张量类型::浮点32, "bwd.dkT");
    vk张量 dvT = 建({T, HDV}, 张量类型::浮点32, "bwd.dvT");
    vk张量 dSlot = 建({M * DK}, 张量类型::浮点32, "bwd.dSlot");
    vk张量 xAll = 建({T, D}, 张量类型::整数8, "bwd.xAll");
    vk张量 sAll = 建({T}, 张量类型::浮点32, "bwd.sAll");
    vk张量 xDeq = 建({T, D}, 张量类型::浮点32, "bwd.xDeq");
    vk张量 dXAll = 建({T, D}, 张量类型::浮点32, "bwd.dXAll");

    auto 转置 = [&](const vk张量& in, vk张量& out, std::uint32_t R, std::uint32_t C) {
        const TransposePC pc{R, C, 0, 0};
        提交2("transpose_f32", {in.取句柄(), out.取句柄()}, &pc, sizeof(pc),
              (R * C + 255u) / 256u, 1, 1);
    };
    auto 求和头 = [&](const vk张量& in, vk张量& out, std::uint32_t h, std::uint32_t n) {
        const SumHeadsPC pc{h, n, 0, 0};
        提交2("sum_heads", {in.取句柄(), out.取句柄()}, &pc, sizeof(pc),
              (n + 255u) / 256u, 1, 1);
    };

    auto 反向层 = [&](std::uint32_t li) {
        层& L = 层表[li];
        const bool 最后 = (li + 1 == cfg.n_layers);
        const bool first = (li == 0);


        冲刷批();
        if (!L.ersa) {
            填零(dKnSeq); 填零(dVnSeq); 填零(dBnSeq);
        } else {
            填零(dQH); 填零(dKH); 填零(dVH); 填零(dSinkH);
        }

        const std::uint32_t buf = li & 1u;
        vk张量& dYAll = dYAllV[buf];
        vk张量& dBranchMidAll = dBranchMidAllV[buf];
        vk张量& dX2All = dX2AllV[buf];
        vk张量& dScoreAll = dScoreAllV[buf];
        vk张量& dW2w = dW2wV[buf];
        vk张量& db2w = db2wV[buf];
        vk张量& moePre = moePreV[buf];
        vk张量& moeHq = moeHqV[buf];
        vk张量& moeDh = moeDhV[buf];
        vk张量& moeShPre = moeShPreV[buf];
        vk张量& moeShUp = moeShUpV[buf];
        vk张量& moeShHq = moeShHqV[buf];
        vk张量& moeShDh = moeShDhV[buf];
        vk张量& moeShSilu = moeShSiluV[buf];
        vk张量& moeXs = moeXsV[buf];
        vk张量& moeDys = moeDysV[buf];
        vk张量& moeIdxS = moeIdxSV[buf];
        vk张量& moeKwS = moeKwSV[buf];
        vk张量& dW1 = dW1V[buf];
        vk张量& dW2 = dW2V[buf];
        vk张量& dShW1T = dShW1TV[buf];
        vk张量& dShWupT = dShWupTV[buf];
        vk张量& dShW2T = dShW2TV[buf];
        vk张量& dShW1 = dShW1V[buf];
        vk张量& dShWup = dShWupV[buf];
        vk张量& dShW2 = dShW2V[buf];

        // ── 阶段 A：一次序列并行 GR2 写回，连续写入 dYAll / dBranchMidAll ──
        const Gr2BwdPC g2{T, D, 最后 ? 1u : 0u, 0, 0};
        提交2("gr2_write_bwd",
              {branchMidAll[li].取句柄(), yMoeAll[li].取句柄(), L.g2ww->p.取句柄(),
               L.g2wb->p.取句柄(), dBranchUpperAll.取句柄(), dOutTokens.取句柄(),
               dYAll.取句柄(), dBranchMidAll.取句柄(), dW2wAll.取句柄(), db2wAll.取句柄()},
              &g2, sizeof(g2), T, 1, 1);

        // ── 阶段 B：一次批量 MoE dx，读写连续 [T,*] buffer ──
        const MoeDxI8PC mdx{E, K, D, F, F, Gd, Gf, 0, T};
        提交2("moe_bwd_dx_i8",
              {x2DeqAll[li].取句柄(), dYAll.取句柄(), L.gate->p.取句柄(),
               L.moe_W1_i8.取句柄(), L.moe_W1_s.取句柄(),
               L.moe_W2_i8.取句柄(), L.moe_W2_s.取句柄(),
               moeIdxRecAll[li].取句柄(), moeKwRecAll[li].取句柄(),
               L.shW1->i8.取句柄(), L.shW1->s.取句柄(),
               L.shWup->i8.取句柄(), L.shWup->s.取句柄(),
               L.shW2->i8.取句柄(), L.shW2->s.取句柄(),
               dX2All.取句柄(), dScoreAll.取句柄(),
               moePre.取句柄(), moeHq.取句柄(), moeDh.取句柄(),
               moeShPre.取句柄(), moeShUp.取句柄(), moeShHq.取句柄(),
               moeShDh.取句柄(), moeXs.取句柄(), moeDys.取句柄(),
               moeIdxS.取句柄(), moeKwS.取句柄(), moeShSilu.取句柄()},
              &mdx, sizeof(mdx), T, 1, 1);
        const std::uint32_t ftiles = (F + 127u) / 128u;
        const MoeDxSharedPC mds{T, D, F, Gd, Gf, 128u, 0};
        提交2("moe_bwd_dx_shared_i8",
              {moeXs.取句柄(), moeDys.取句柄(),
               L.shW1->i8.取句柄(), L.shW1->s.取句柄(),
               L.shWup->i8.取句柄(), L.shWup->s.取句柄(),
               L.shW2->i8.取句柄(), L.shW2->s.取句柄(),
               moeShPre.取句柄(), moeShUp.取句柄(), moeShHq.取句柄(),
               moeShDh.取句柄(), moeShSilu.取句柄(), dX2All.取句柄()},
              &mds, sizeof(mds), T, ftiles, 1);

        // gr_route 也按序列并行：读连续 [T,*] 记录，权重梯度按 token 输出后归约。
        const GrRouteBwdPC rr{T, D, 0, 0};
        提交2("gr_route_bwd_int8",
              {woYAll[li].取句柄(), branchInAll[li].取句柄(), L.outNorm->p.取句柄(),
               L.g1ww->p.取句柄(), L.g1wb->p.取句柄(), L.g2ng->p.取句柄(),
               L.g2rw->p.取句柄(), L.g2rb->p.取句柄(), dX2All.取句柄(),
               dBranchMidAll.取句柄(), dWoYAll.取句柄(), dBranchRAll.取句柄(),
               dOutNormAll.取句柄(), dW1WAll.取句柄(), dW1BAll.取句柄(),
               dG2GAll.取句柄(), dG2WAll.取句柄(), dG2BAll.取句柄()},
              &rr, sizeof(rr), T, 1, 1);

        const LinI8PC woiT{T, HDV, D, Ghdv, 0, 0, 0, 0};
        提交2("linear_bwd_dx_i8",
              {dWoYAll.取句柄(), L.Wo->i8.取句柄(), L.Wo->s.取句柄(),
               dOAll.取句柄()},
              &woiT, sizeof(woiT), (T * D + 63u) / 64u, 1, 1);

        // ── 阶段 C：逐 token 的 gate/route/Wo/attention/QKV/GR read ──
        if (!L.ersa) {
            const float tau = 0.005f;
            const ASMSeqPC ap{B, H, M, R, DK, DV, T,
                              tau, tau, 0.9f, 0.01f, L.beta_null};
            提交2("asm_step_bwd_seq_wg",
                  {asmQAll[li].取句柄(), asmKAll[li].取句柄(), asmVAll[li].取句柄(),
                   asmEtaAll[li].取句柄(), asmKOldAll[li].取句柄(),
                   asmVOldAll[li].取句柄(), asmBoldAll[li].取句柄(),
                   L.gamma->p.取句柄(), dOAll.取句柄(),
                   dKnSeq.取句柄(), dVnSeq.取句柄(), dBnSeq.取句柄(),
                   asmIdxAll[li].取句柄(), dqRawAll.取句柄(), dkRawAll.取句柄(),
                   dvAll.取句柄(), dEtaAll.取句柄(), dgammaH.取句柄(),
                   dbnH.取句柄(), L.qkv_norm.取句柄()},
                  &ap, sizeof(ap), B * H, 1, 1);
            const SumPC dnullpc{H, 0, 0, 0};
            提交2("sum_f32", {dbnH.取句柄(), L.betaNull->g.取句柄()},
                  &dnullpc, sizeof(dnullpc), 1, 1, 1);
        }

        // 先按 expert 建命中 token-slot 列表，dW 只遍历命中项。
        const 规格4 lbr{T, E, K, 0};
        提交2("moe_build_route_list",
              {moeIdxRecAll[li].取句柄(), moeRouteList.取句柄(), moeRouteCnt.取句柄()},
              &lbr, sizeof(lbr), 1, 1, 1);

        // MoE 全部 dW 合并：routed + shared 一次 dispatch，直接写参数梯度。
        const MoeDwAllPC dwall{T, E, K, D, F};
        const std::uint32_t dwN = 2u * E * D * F + 3u * D * F;
        提交2("moe_bwd_dw_all_i8",
              {moeXs.取句柄(), moeDys.取句柄(), moeIdxS.取句柄(),
               moeKwS.取句柄(), moePre.取句柄(), moeHq.取句柄(),
               moeDh.取句柄(), moeShPre.取句柄(), moeShUp.取句柄(),
               moeShHq.取句柄(), moeShDh.取句柄(), moeShSilu.取句柄(),
               L.W1->g.取句柄(), L.W2->g.取句柄(), L.shW1->g.取句柄(),
               L.shWup->g.取句柄(), L.shW2->g.取句柄(),
               moeRouteList.取句柄(), moeRouteCnt.取句柄()},
              &dwall, sizeof(dwall), (dwN + 63u) / 64u, 1, 1);

        if (!L.ersa) {
            const QuantPC dqall{T * D, D, 0, 0};
            提交2("dequant_row_i8",
                  {asmXqAll[li].取句柄(), asmSxAll[li].取句柄(), xDeq.取句柄()},
                  &dqall, sizeof(dqall), (T * D + 255u) / 256u, 1, 1);
            const LinI8PC lqi{T, D, HDK, Gd, 0, 0, 0, 0};
            const LinI8PC lkia{T, D, HDK, Gd, 1, 0, 0, 0};
            提交2("linear_bwd_dx_i8",
                  {dqRawAll.取句柄(), L.Wq->i8.取句柄(), L.Wq->s.取句柄(),
                   dXAll.取句柄()},
                  &lqi, sizeof(lqi), (T * D + 63u) / 64u, 1, 1);
            提交2("linear_bwd_dx_i8",
                  {dkRawAll.取句柄(), L.Wk->i8.取句柄(), L.Wk->s.取句柄(),
                   dXAll.取句柄()},
                  &lkia, sizeof(lkia), (T * D + 63u) / 64u, 1, 1);
            const LinI8PC lvia{T, D, HDV, Gd, 1, 0, 0, 0};
            提交2("linear_bwd_dx_i8",
                  {dvAll.取句柄(), L.Wv->i8.取句柄(), L.Wv->s.取句柄(),
                   dXAll.取句柄()},
                  &lvia, sizeof(lvia), (T * D + 63u) / 64u, 1, 1);
        }

        if (L.ersa) {
            // context 段块路由反向 + query 段 IVF 反向，都按序列并行；
            // dK/dV/dsink 用原子累加合并到同一 cache 梯度。
            if (bwd_cache && Nctx > 0) {
                const ErsaBwdPC ep{1, H, T, 0, DK, DV, 0, cfg.block_size,
                                   cfg.top_blocks, cfg.window, 0, NBmaxT, 0.005f,
                                   L.sink_value, 0.0f, 0.0f};
                提交2("ersa_attn_bwd_query",
                      {L.q_hist.取句柄(), L.k_cache.取句柄(), L.v_cache.取句柄(),
                       L.block_sum.取句柄(), dOAll.取句柄(), dQH.取句柄(),
                       dKH.取句柄(), dVH.取句柄(), dSinkH.取句柄()},
                      &ep, sizeof(ep), H * Nctx, 1, 1);
            }
            if (bwd_cache) {
                const IvfPc bp{B, H, T, T, DK, DV, IVFC, IVFLMAX,
                               IVFTOPB, IVFWIN, Nctx, 0, 0.005f, L.sink_value};
                提交2("ersa_ivf_attn_bwd_cand",
                      {L.q_hist.取句柄(), L.k_cache.取句柄(), L.v_cache.取句柄(),
                       L.candSnapSeq.取句柄(), L.candCntSeq.取句柄(),
                       dOAll.取句柄(), dQH.取句柄(), dKH.取句柄(),
                       dVH.取句柄(), dSinkH.取句柄()},
                      &bp, sizeof(bp), H * Tq, 1, 1);
            } else {
                const ErsaBwdPC ep{1, H, T, 0, DK, DV, 0, cfg.block_size,
                                   cfg.top_blocks, cfg.window, 0, NBmaxT, 0.005f,
                                   L.sink_value, 0.0f, 0.0f};
                提交2("ersa_attn_bwd_query",
                      {L.q_hist.取句柄(),
                       (cache_mode ? L.k_cache : L.k_hist).取句柄(),
                       (cache_mode ? L.v_cache : L.v_hist).取句柄(),
                       L.block_sum.取句柄(), dOAll.取句柄(), dQH.取句柄(),
                       dKH.取句柄(), dVH.取句柄(), dSinkH.取句柄()},
                      &ep, sizeof(ep), H * T, 1, 1);
            }

            const L2BwdPC l2q{H * T, DK, 0, 1};
            提交2("l2norm_bwd",
                  {L.q_hist.取句柄(), dQH.取句柄(), L.qkv_norm.取句柄(), dqRawH.取句柄()},
                  &l2q, sizeof(l2q), H * T, 1, 1);

            const HeadsPC th{H, T, DK, 0};
            const HeadsPC tv{H, T, DV, 0};
            提交2("transpose_heads", {dqRawH.取句柄(), dqT.取句柄()}, &th, sizeof(th),
                  (H * T * DK + 255u) / 256u, 1, 1);
            提交2("transpose_heads", {dKH.取句柄(), dkT.取句柄()}, &th, sizeof(th),
                  (H * T * DK + 255u) / 256u, 1, 1);
            提交2("transpose_heads", {dVH.取句柄(), dvT.取句柄()}, &tv, sizeof(tv),
                  (H * T * DV + 255u) / 256u, 1, 1);

            // 逐 token xq/sx 拼成 [T,D]，反量化后做一次 batched QKV 反向。
            {
                std::vector<std::int8_t> xh(static_cast<std::size_t>(T) * D);
                std::vector<float> sh(T);
                for (std::uint32_t t = 0; t < T; ++t) {
                    std::vector<std::int8_t> q(D);
                    冲刷批();
                    记录表[t][li]->xq.下载(q.data(), q.size());
                    std::copy(q.begin(), q.end(), xh.begin() + std::size_t(t) * D);
                    记录表[t][li]->sx.下载(&sh[t], 4);
                }
                xAll.上传(xh.data(), xh.size());
                sAll.上传(sh.data(), sh.size() * 4);
                const QuantPC dq{T * D, D, 0, 0};
                提交2("dequant_row_i8",
                      {xAll.取句柄(), sAll.取句柄(), xDeq.取句柄()},
                      &dq, sizeof(dq), (T * D + 255u) / 256u, 1, 1);
            }

            const LinI8PC lqbi{T, D, HDK, Gd, 0, 0, 0, 0};
            const LinI8PC lkbi{T, D, HDK, Gd, 1, 0, 0, 0};
            提交2("linear_bwd_dx_i8",
                  {dqT.取句柄(), L.Wq->i8.取句柄(), L.Wq->s.取句柄(),
                   dXAll.取句柄()},
                  &lqbi, sizeof(lqbi), (T * D + 63u) / 64u, 1, 1);
            提交2("linear_bwd_dx_i8",
                  {dkT.取句柄(), L.Wk->i8.取句柄(), L.Wk->s.取句柄(),
                   dXAll.取句柄()},
                  &lkbi, sizeof(lkbi), (T * D + 63u) / 64u, 1, 1);
            const LinI8PC lvbi{T, D, HDV, Gd, 1, 0, 0, 0};
            提交2("linear_bwd_dx_i8",
                  {dvT.取句柄(), L.Wv->i8.取句柄(), L.Wv->s.取句柄(),
                   dXAll.取句柄()},
                  &lvbi, sizeof(lvbi), (T * D + 63u) / 64u, 1, 1);
        }

        // 每层 5 组 dW/db 合并成一次 dispatch（gate/Wo/Wq/Wk/Wv）。
        {
            const vk张量& dyq = L.ersa ? dqT : dqRawAll;
            const vk张量& dyk = L.ersa ? dkT : dkRawAll;
            const vk张量& dyv = L.ersa ? dvT : dvAll;
            const LinDwdb5PC dpc{T,
                                 E, D,
                                 D, HDV,
                                 HDK, D,
                                 HDK, D,
                                 HDV, D, 0};
            const std::uint32_t dwdbN =
                E * D + D * HDV + 2u * HDK * D + HDV * D;
            提交2("linear_bwd_dwdb5",
                  {x2All[li].取句柄(), oDeqAll[li].取句柄(), xDeq.取句柄(),
                   dScoreAll.取句柄(), dWoYAll.取句柄(),
                   dyq.取句柄(), dyk.取句柄(), dyv.取句柄(),
                   L.gate->g.取句柄(), L.Wo->g.取句柄(),
                   L.Wq->g.取句柄(), L.Wk->g.取句柄(), L.Wv->g.取句柄(),
                   dGateDb.取句柄(), dWoDb.取句柄(),
                   dWqDb.取句柄(), dWkDb.取句柄(), dWvDb.取句柄()},
                  &dpc, sizeof(dpc), (dwdbN + 63u) / 64u, 1, 1);
        }

        if (L.ersa) {
            const SumPC sinkpc{H, 0, 0, 0};
            提交2("sum_f32", {dSinkH.取句柄(), L.sink->g.取句柄()},
                  &sinkpc, sizeof(sinkpc), 1, 1, 1);
        }

        // ── 序列并行 GR read + 融合 dIn/dBranchR 后处理 ──
        const vk张量& etaWsrc = L.ersa ? eta_w_dummy : L.etaW->p;
        const vk张量& etaBsrc = L.ersa ? eta_b_dummy : L.etaB->p;
        const vk张量& detaSrc = L.ersa ? dEtaZero : dEtaAll;
        const GRReadPC gr{T, D, T, 0, first ? 1u : 0u, H, 0, 0};
        提交2("gr_read_bwd_int8",
              {embed->i8.取句柄(), embed->s.取句柄(), ids.取句柄(),
               branchInAll[li].取句柄(), L.g1ng->p.取句柄(), L.g1rw->p.取句柄(),
               L.g1rb->p.取句柄(), etaWsrc.取句柄(), etaBsrc.取句柄(),
               dXAll.取句柄(), detaSrc.取句柄(), dInAll.取句柄(), dNormGAll.取句柄(),
               dReadWAll.取句柄(), dReadBAll.取句柄(), dEtaWAll.取句柄(),
               dEtaBAll.取句柄()},
              &gr, sizeof(gr), T, 1, 1);
        const GrPostPC post{T, D, first ? 1u : 0u, 0};
        提交2("gr_read_post_seq",
              {dInAll.取句柄(), dBranchRAll.取句柄(),
               dBranchUpperAll.取句柄(), dEmbedAll.取句柄()},
              &post, sizeof(post), T, 1, 1);
        SumMultiPC sm{};
        sm.count = 15;
        const std::uint32_t hs[15] = {T, T, T, T, T, T, T, T,
                                      L.ersa ? 0u : H, L.ersa ? 0u : H,
                                      T, T, T, L.ersa ? 0u : T, L.ersa ? 0u : T};
        const std::uint32_t ns[15] = {4 * D, 4, D, 4 * D, 4, 4 * D, 16 * D, 4,
                                      L.ersa ? 0u : M, L.ersa ? 0u : M * DK,
                                      4 * D, 16 * D, 4,
                                      L.ersa ? 0u : D, L.ersa ? 0u : 1};
        std::uint32_t sm_total = 0;
        for (int i = 0; i < 15; ++i) {
            sm.H[i] = hs[i];
            sm.N[i] = ns[i];
            sm_total += ns[i];
        }
        提交2("sum_heads_multi",
              {dW2wAll.取句柄(), db2wAll.取句柄(), dOutNormAll.取句柄(),
               dW1WAll.取句柄(), dW1BAll.取句柄(), dG2GAll.取句柄(),
               dG2WAll.取句柄(), dG2BAll.取句柄(), dgammaH.取句柄(),
               dKnSeq.取句柄(), dNormGAll.取句柄(), dReadWAll.取句柄(),
               dReadBAll.取句柄(), dEtaWAll.取句柄(), dEtaBAll.取句柄(),
               L.g2ww->g.取句柄(), L.g2wb->g.取句柄(), L.outNorm->g.取句柄(),
               L.g1ww->g.取句柄(), L.g1wb->g.取句柄(), L.g2ng->g.取句柄(),
               L.g2rw->g.取句柄(), L.g2rb->g.取句柄(),
               (L.ersa ? dGamma : L.gamma->g).取句柄(),
               (L.ersa ? dSlot : L.slotK->g).取句柄(),
               L.g1ng->g.取句柄(), L.g1rw->g.取句柄(),
               L.g1rb->g.取句柄(),
               (L.ersa ? dEtaW : L.etaW->g).取句柄(),
               (L.ersa ? dEtaB : L.etaB->g).取句柄()},
              &sm, sizeof(sm), (sm_total + 255u) / 256u, 1, 1);
        if (big3b && li + 1 == cfg.n_layers) {
            冲刷批();
            dQH.拷贝到(dQ31);
            dKH.拷贝到(dK31);
            dVH.拷贝到(dV31);
            dXAll.拷贝到(dXAll31);
        }
    };
    auto 反向全部 = [&]() {
        冲刷批();
        填零(dBranchUpperAll);
        const float loss = 反向头();
        for (int li = int(cfg.n_layers) - 1; li >= 0; --li) 反向层(std::uint32_t(li));

        // layer0 的融合后处理已经写好 dEmbedAll。
        {
            const 规格4 ep{T, V, D, 0};
            提交2("embedding_bwd_scatter",
                  {dEmbedAll.取句柄(), ids.取句柄(), embed->g.取句柄()},
                  &ep, sizeof(ep), (T * D + 255u) / 256u, 1, 1);
        }
        冲刷批();
        if (!big3b) {
            std::printf("反向: loss=%.6f | ||dEmbed||=%.6g | ||dWq(L0)||=%.6g | ||dWo(L0)||=%.6g | ||dW1(L0)||=%.6g\n",
                        loss, 范数(embed->g), 范数(层表[0].Wq->g), 范数(层表[0].Wo->g),
                        范数(层表[0].W1->g));
        } else {
            std::printf("反向: loss=%.6f (3B 省略梯度范数)\n", loss);
        }
        return loss;
    };

    auto 刷新前向量化 = [&]() {
        冲刷批();
        for (auto& p : 全部参数) {
            if (!p->量化) continue;
            const QuantPC pc{p->rows, p->cols, p->G, 0};
            提交2("quantize_rows_i8",
                  {p->p.取句柄(), p->i8.取句柄(), p->s.取句柄()},
                  &pc, sizeof(pc), (p->rows * p->G + 127u) / 128u, 1, 1);
        }
        冲刷批();
        for (std::uint32_t li = 0; li < cfg.n_layers; ++li) {
            层& L = 层表[li];
            std::vector<float> w1(static_cast<std::size_t>(E) * D * F), w2(static_cast<std::size_t>(E) * F * D);
            L.W1->p.下载(w1.data(), w1.size() * 4);
            L.W2->p.下载(w2.data(), w2.size() * 4);
            std::vector<std::int8_t> q1, q2;
            std::vector<float> s1, s2;
            量化W1(w1, E, D, F, q1, s1);
            量化W2(w2, E, D, F, q2, s2);
            L.moe_W1_i8.上传(q1.data(), q1.size());
            L.moe_W1_s.上传(s1.data(), s1.size() * 4);
            L.moe_W2_i8.上传(q2.data(), q2.size());
            L.moe_W2_s.上传(s2.data(), s2.size() * 4);
            if (L.ersa) {
                L.sink->p.下载(&L.sink_value, 4);
            } else {
                L.betaNull->p.下载(&L.beta_null, 4);
            }
        }
    };

    auto 更新 = [&](std::uint32_t step) {
        const float lr = 3e-4f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f, wd = 0.01f;
        const float b1t = std::pow(beta1, float(step));
        const float b2t = std::pow(beta2, float(step));
        for (auto& p : 全部参数) {
            const 规格Adam pc{lr, beta1, beta2, eps,
                              1.0f / (1.0f - b1t), 1.0f / (1.0f - b2t),
                              static_cast<std::uint32_t>(p->n), wd, 1u};
            提交2("adamw_update",
                  {p->g.取句柄(), p->p.取句柄(), p->m.取句柄(), p->v.取句柄()},
                  &pc, sizeof(pc), (p->n + 63u) / 64u, 1, 1);
        }
        刷新前向量化();
    };

    auto 重置ASM = [&]() {
        for (std::uint32_t li = 0; li < cfg.n_layers; ++li) {
            层& L = 层表[li];
            if (L.ersa) continue;
            const AsmResetPC pc{B, H, M, DK, DV, 0, 0, 0};
            提交2("asm_reset", {L.slotK->p.取句柄(), L.K.取句柄(), L.V.取句柄(),
                               L.beta.取句柄()}, &pc, sizeof(pc), H, 1, 1);
        }
    };


    auto 建缓存索引 = [&]() {
        if (!fwd_cache || Nctx == 0) return;
        for (std::uint32_t li = 0; li < cfg.n_layers; ++li) {
            层& L = 层表[li];
            if (!L.ersa) continue;
            const CopySlicePC cp{H, T, Nctx, DK, Nctx, 0, 0, 0};
            提交2("copy_f32_slice", {L.k_cache.取句柄(), L.ctx_k.取句柄()},
                  &cp, sizeof(cp), Nctx, H, 1);
            const IvfInitPC ip{H, Nctx, DK, IVFC, 0, 0, 0, 0};
            提交2("ivf_init_centroids", {L.ctx_k.取句柄(), L.cent.取句柄()},
                  &ip, sizeof(ip), IVFC, H, 1);
            const KMeansPC kp{1, H, Nctx, DK, IVFC, 0, 0, 0};
            for (int it = 0; it < 10; ++it) {
                提交2("kmeans_assign",
                      {L.ctx_k.取句柄(), L.cent.取句柄(), L.assign.取句柄()},
                      &kp, sizeof(kp), Nctx, H, 1);
                提交2("kmeans_update",
                      {L.ctx_k.取句柄(), L.assign.取句柄(), L.cent.取句柄(),
                       L.counts.取句柄()},
                      &kp, sizeof(kp), IVFC, H, 1);
            }
            const FillMemberPC fp{1, H, Nctx, IVFC, IVFLMAX, 0, 0, 0};
            提交2("kmeans_fill_members", {L.assign.取句柄(), L.member.取句柄()},
                  &fp, sizeof(fp), IVFC, H, 1);
        }
    };

    auto 前向 = [&]() {
        重置ASM();
        for (std::uint32_t t = 0; t < T; ++t) {
            for (std::uint32_t li = 0; li < cfg.n_layers; ++li) {
                层& L = 层表[li];
                记录& rec = *记录表[t][li];
                const bool first = (li == 0);
                const GRReadPC gr{ B, D, T, t, first ? 1u : 0u, 0,
                                   L.ersa ? 0u : 1u, 0 };
                const vk张量& eww = L.ersa ? eta_w_dummy : L.etaW->p;
                const vk张量& ebb = L.ersa ? eta_b_dummy : L.etaB->p;
                const vk张量& xqs = L.ersa ? xq : asmXqAll[li];
                const vk张量& sxs = L.ersa ? sx : asmSxAll[li];
                const vk张量& ets = L.ersa ? eta : asmEtaAll[li];
                提交("gr_read_quant_int8",
                     {embed->i8.取句柄(), embed->s.取句柄(), ids.取句柄(), branches.取句柄(),
                      L.g1ng->p.取句柄(), L.g1rw->p.取句柄(), L.g1rb->p.取句柄(),
                      eww.取句柄(), ebb.取句柄(), xq.取句柄(), sx.取句柄(), eta.取句柄(),
                      branchInAll[li].取句柄(), xqs.取句柄(), sxs.取句柄(), ets.取句柄()},
                     &gr, sizeof(gr), 1);
                const CopySlicePC bic{1, 1, T, 4 * D, 1, 0, t, 0};
                提交2("copy_f32_slice", {branches.取句柄(), branchInAll[li].取句柄()},
                      &bic, sizeof(bic), 1, 1, 1);
                if (L.ersa) {
                    冲刷批();
                    xq.拷贝到(rec.xq);
                    sx.拷贝到(rec.sx);
                }

                const std::uint32_t store = L.ersa ? 1u : 0u;
                const vk张量& qo = L.ersa ? L.q_hist : qkv_q;
                const vk张量& ko = L.ersa
                    ? (cache_mode ? L.k_cache : L.k_hist) : qkv_k;
                const vk张量& vo = L.ersa
                    ? (cache_mode ? L.v_cache : L.v_hist) : qkv_v;
                const vk张量& bs = L.ersa ? L.block_sum : block_dummy;
                const QkvFusedPC qk{1, D, H, DK, DV, T, T, t, t, store,
                                    NBmaxT, cfg.block_size, 1};
                提交2("qkv_fused_gemv_int8",
                      {xq.取句柄(), sx.取句柄(),
                       L.Wq->i8.取句柄(), L.Wq->s.取句柄(),
                       L.Wk->i8.取句柄(), L.Wk->s.取句柄(),
                       L.Wv->i8.取句柄(), L.Wv->s.取句柄(),
                       qo.取句柄(), ko.取句柄(), vo.取句柄(), bs.取句柄(),
                       L.qkv_norm.取句柄(),
                       (L.ersa ? block_dummy : asmQAll[li]).取句柄(),
                       (L.ersa ? block_dummy : asmKAll[li]).取句柄(),
                       (L.ersa ? block_dummy : asmVAll[li]).取句柄()},
                      &qk, sizeof(qk), H, 3, 1);

                if (L.ersa) {
                    if (fwd_cache && t >= Nctx) {
                        if (ingest_queries) {
                            const IncAssignPC ia{1, H, 1, DK, IVFC, T, t, 0};
                            提交2("kmeans_inc_assign",
                                  {L.k_cache.取句柄(), L.cent.取句柄(),
                                   ivfAssignNew.取句柄()},
                                  &ia, sizeof(ia), 1, H, 1);
                            const IncAppendPC ap{1, H, 1, IVFC, IVFLMAX, t, 0, 0};
                            提交2("kmeans_inc_append",
                                  {ivfAssignNew.取句柄(), L.counts.取句柄(),
                                   L.member.取句柄(), ivfOverflow.取句柄()},
                                  &ap, sizeof(ap), IVFC, H, 1);
                            const IncUpdatePC iu{1, H, 1, DK, IVFC, T, t, 0,
                                                 0.2f, 0, 0, 0};
                            提交2("kmeans_inc_update",
                                  {L.k_cache.取句柄(), ivfAssignNew.取句柄(),
                                   L.cent.取句柄(), L.counts.取句柄()},
                                  &iu, sizeof(iu), IVFC, H, 1);
                        }
                        const IvfPc qp{B, H, T, T, DK, DV, IVFC, IVFLMAX,
                                       IVFTOPB, IVFWIN, t, t,
                                       0.005f, L.sink_value};
                        提交2("ersa_ivf_attn_query",
                              {L.q_hist.取句柄(), L.k_cache.取句柄(),
                               L.v_cache.取句柄(), L.cent.取句柄(),
                               L.member.取句柄(), L.counts.取句柄(),
                               attn_out.取句柄(),
                               L.candSnap[t - Nctx].取句柄(),
                               L.candCnt[t - Nctx].取句柄()},
                              &qp, sizeof(qp), 1, H, 1);
                        const CopySlicePC csc{H, 1, Tq, MAXCAND, 1, 0, t - Nctx, 0};
                        提交2("copy_i32_slice",
                              {L.candSnap[t - Nctx].取句柄(), L.candSnapSeq.取句柄()},
                              &csc, sizeof(csc), 1, H, 1);
                        const CopySlicePC ccc{H, 1, Tq, 1, 1, 0, t - Nctx, 0};
                        提交2("copy_i32_slice",
                              {L.candCnt[t - Nctx].取句柄(), L.candCntSeq.取句柄()},
                              &ccc, sizeof(ccc), 1, H, 1);
                    } else {
                        const std::uint32_t N = t + 1;
                        const std::uint32_t NB = (N + cfg.block_size - 1) / cfg.block_size;
                        const ErsaStepPC pc{B, H, T, N, DK, DV, NB, cfg.block_size,
                                            cfg.top_blocks, cfg.window, t, NBmaxT, 0.005f,
                                            L.sink_value};
                        提交2("ersa_attn_step",
                              {L.q_hist.取句柄(),
                               (cache_mode ? L.k_cache : L.k_hist).取句柄(),
                               (cache_mode ? L.v_cache : L.v_hist).取句柄(),
                               L.block_sum.取句柄(), attn_out.取句柄()},
                              &pc, sizeof(pc), H, 1, 1);
                    }
                } else {
                    const float tau = 0.005f;
                    const AsmStepPC pc{B, H, M, DK, DV, R, t, T,
                                       tau, tau, 0.9f, 0.01f, L.beta_null, 0.0f};
                    提交2("asm_step_eta",
                          {qkv_q.取句柄(), qkv_k.取句柄(), qkv_v.取句柄(),
                           L.K.取句柄(), L.V.取句柄(), L.beta.取句柄(), L.gamma->p.取句柄(),
                           o.取句柄(), eta.取句柄(), dbg.取句柄(),
                           asmKOldAll[li].取句柄(), asmVOldAll[li].取句柄(),
                           asmBoldAll[li].取句柄(), asmIdxAll[li].取句柄()},
                          &pc, sizeof(pc), H, 1, 1);
                }
                const vk张量& attn_src = L.ersa ? attn_out : o;
                冲刷批();
                attn_src.拷贝到(rec.o);

                const WoTilePC wp{1, D, HDV, 64, 1, t, T, 0};
                const vk张量& oi = L.ersa ? attn_out : o;
                提交2("wo_gemv_int8",
                      {oi.取句柄(), L.Wo->i8.取句柄(), L.Wo->s.取句柄(), wo_y.取句柄(), wo_deq.取句柄(),
                       woYAll[li].取句柄(), oDeqAll[li].取句柄()},
                      &wp, sizeof(wp), (D + 63) / 64, 1, 1);

                const WoRoutePC np{1, D, HDV, E, K, 1, t, T};
                提交2("norm_gr_route_int8",
                      {wo_y.取句柄(), L.outNorm->p.取句柄(), branches.取句柄(),
                       L.g1ww->p.取句柄(), L.g1wb->p.取句柄(),
                       L.g2ng->p.取句柄(), L.g2rw->p.取句柄(), L.g2rb->p.取句柄(),
                       L.gate->p.取句柄(), x2.取句柄(), moe_idx.取句柄(), moe_w.取句柄(),
                       branchMidAll[li].取句柄(), x2All[li].取句柄(),
                       moeIdxRecAll[li].取句柄(), moeKwRecAll[li].取句柄()},
                      &np, sizeof(np), 1, 1, 1);
                {
                    const QuantPC qx{1, D, t, 0};
                    提交2("quantize_dequant_f32", {x2.取句柄(), x2DeqAll[li].取句柄()},
                          &qx, sizeof(qx), 1, 1, 1);
                }

                const MoeExpertsPC hp{1, D, F, E, K, 128, 0, 0};
                const std::uint32_t fg = (F + 127) / 128;
                提交2("moe_h_fused_int8",
                      {x2.取句柄(), L.moe_W1_i8.取句柄(), L.moe_W1_s.取句柄(),
                       moe_idx.取句柄(), h_q.取句柄(), h_s.取句柄(),
                       L.shW1->i8.取句柄(), L.shW1->s.取句柄(),
                       L.shWup->i8.取句柄(), L.shWup->s.取句柄(),
                       sh_hq.取句柄(), sh_hs.取句柄()},
                      &hp, sizeof(hp), K + 1u, fg, 1);
                const MoeExpertsPC op{1, D, F, E, K, 64, 0, 0};
                const std::uint32_t dt = (D + 63) / 64;
                提交2("moe_out_gemv_int8",
                      {h_q.取句柄(), h_s.取句柄(), L.moe_W2_i8.取句柄(), L.moe_W2_s.取句柄(),
                       moe_idx.取句柄(), moe_w.取句柄(), expert_out.取句柄()},
                      &op, sizeof(op), K, dt, 1);
                const MoeTailW2PC tw{1, D, F, K, t, 1u, 0, 0};
                提交2("moe_tail_w2_int8",
                      {expert_out.取句柄(), sh_hq.取句柄(), sh_hs.取句柄(),
                       L.shW2->i8.取句柄(), L.shW2->s.取句柄(), y_moe.取句柄(),
                       yMoeAll[li].取句柄()},
                      &tw, sizeof(tw), (D + 63) / 64, 1, 1);
                const MoeReducePC rp{1, D, F, K, 1024, t, T,
                                     (li + 1 == cfg.n_layers) ? 1u : 0u};
                提交2("moe_grwrite_int8",
                      {y_moe.取句柄(), L.g2ww->p.取句柄(), L.g2wb->p.取句柄(),
                       branches.取句柄(), out_tokens.取句柄(), rec.branch_out.取句柄()},
                      &rp, sizeof(rp), (D + 255) / 256, 1, 1);
            }
            if (cache_mode && fwd_cache && t + 1 == Nctx) 建缓存索引();
        }

        const CopySlicePC qcp{1, T, Tq, D, Tq, Nctx, 0, 0};
        提交2("copy_f32_slice", {out_tokens.取句柄(), out_q.取句柄()},
              &qcp, sizeof(qcp), Tq, 1, 1);
        const RMSNormPC fn{Tq, D, 1e-6f, 0};
        提交2("rmsnorm_quant",
              {out_q.取句柄(), finalNorm->p.取句柄(), normed_q.取句柄(), normed_s.取句柄()},
              &fn, sizeof(fn), Tq, 1, 1);
        const LmHeadPC lh{Tq, D, V, 64, 0, 0, 0, 0};
        提交2("lm_head_gemv_int8",
              {normed_q.取句柄(), normed_s.取句柄(), lmhead->i8.取句柄(), lmhead->s.取句柄(),
               logits.取句柄()},
              &lh, sizeof(lh), (V + 63) / 64, Tq, 1);
        冲刷批();
    };

    // 与主模型 INT8 前向做一次回归：训练器用自己的参数/缓冲，数值应一致。
    if (cache_mode) {
        try {
            if (big3b) {
                std::printf("3B cache: layers=%u D=%u H=%u Ttot=%u | 参数构建完成\n",
                            cfg.n_layers, D, H, T);
                const auto f0 = std::chrono::steady_clock::now();
                fwd_cache = true;
                性能表.clear();
                性能采样 = true;
                前向();
                性能采样 = false;
                const auto f1 = std::chrono::steady_clock::now();
                const auto 前向性能表 = 性能表;
                std::vector<float> lg(static_cast<std::size_t>(Tq) * V);
                logits.下载(lg.data(), lg.size() * 4);
                size_t bad = 0;
                for (float x : lg) if (!std::isfinite(x)) ++bad;

                清零梯度();
                bwd_cache = true;
                const auto b0 = std::chrono::steady_clock::now();
                性能表.clear();
                性能采样 = true;
                const float loss = 反向全部();
                性能采样 = false;
                const auto b1 = std::chrono::steady_clock::now();
                const double fms = std::chrono::duration<double, std::milli>(f1 - f0).count();
                const double bms = std::chrono::duration<double, std::milli>(b1 - b0).count();

                // 采样关闭后再整段跑一遍：层内算子串成一条命令缓冲，只等一次。
                const auto cf0 = std::chrono::steady_clock::now();
                fwd_cache = true;
                前向();
                const auto cf1 = std::chrono::steady_clock::now();
                清零梯度();
                bwd_cache = true;
                const auto cb0 = std::chrono::steady_clock::now();
                const float loss2 = 反向全部();
                const auto cb1 = std::chrono::steady_clock::now();
                const double cfms =
                    std::chrono::duration<double, std::milli>(cf1 - cf0).count();
                const double cbms =
                    std::chrono::duration<double, std::milli>(cb1 - cb0).count();
                std::printf("3B 串提交实测: fwd %.2f ms | bwd %.2f ms | loss %.6f\n",
                            cfms, cbms, loss2);
                size_t grad_elems = 0;
                for (auto& p : 全部参数) grad_elems += p->n;
                const auto live = vk底层::缓冲::存活字节计().load();
                std::printf("3B cache 实测: fwd %.2f ms | bwd %.2f ms | loss %.6f | "
                            "params %zu | grad_elems %zu | nonfinite_logits %zu | "
                            "VK live %.2f GiB\n",
                            fms, bms, loss, 全部参数.size(), grad_elems, bad,
                            double(live) / (1024.0 * 1024.0 * 1024.0));
                auto 打表 = [&](const char* 标签, double 总ms,
                                const std::map<std::string,
                                               std::pair<double, std::uint64_t>>& 表) {
                    std::vector<std::pair<double, std::string>> v;
                    double 合计 = 0.0;
                    for (const auto& kv : 表) {
                        v.push_back({kv.second.first, kv.first});
                        合计 += kv.second.first;
                    }
                    std::sort(v.begin(), v.end(),
                              [](const auto& a, const auto& b) { return a.first > b.first; });
                    std::printf("  %s top: total=%.1fms dispatched=%.1fms host/other=%.1fms\n",
                                标签, 总ms, 合计, 总ms - 合计);
                    const std::size_t n = std::min<std::size_t>(v.size(), 18);
                    for (std::size_t i = 0; i < n; ++i) {
                        const auto& kv = v[i];
                        const std::uint64_t cnt = 表.at(kv.second).second;
                        std::printf("    %-28s %9.1f ms  %6.1f%%  n=%llu\n",
                                    kv.second.c_str(), kv.first, 100.0 * kv.first / 总ms,
                                    static_cast<unsigned long long>(cnt));
                    }
                };
                打表("fwd", fms, 前向性能表);
                打表("bwd", bms, 性能表);
                std::vector<float> douth(std::size_t(T) * D);
                dOutTokens.下载(douth.data(), douth.size() * 4);
                auto rown = [&](const std::vector<float>& v, std::uint32_t row) {
                    double s2 = 0.0;
                    for (std::uint32_t d = 0; d < D; ++d)
                        s2 += double(v[std::size_t(row) * D + d]) * v[std::size_t(row) * D + d];
                    return float(std::sqrt(s2));
                };
                std::printf("  dOut rows t0/t4/t8/t11: %.4g %.4g %.4g %.4g\n",
                            rown(douth, 0), rown(douth, 4), rown(douth, 8), rown(douth, 11));
                std::printf("  L31 candidate counts (q0..q3):");
                for (std::uint32_t q = 0; q < Tq; ++q) {
                    std::vector<std::int32_t> cc(H);
                    层表[cfg.n_layers - 1].candCnt[q].下载(cc.data(), cc.size() * 4);
                    int sum = 0;
                    for (int z : cc) sum += z;
                    std::printf(" q%u=%d", q, sum);
                }
                std::printf("  L31 internals: dQ %.4g | dK %.4g | dV %.4g | dXAll %.4g\n",
                            范数(dQ31), 范数(dK31), 范数(dV31), 范数(dXAll31));
                std::printf("  internals: dQH %.4g | dKH %.4g | dVH %.4g | dXAll %.4g | dX2 %.4g\n",
                            范数(dQH), 范数(dKH), 范数(dVH), 范数(dXAll), 范数(dX2AllV[0]));
                std::printf("  gradients: dOutTokens %.6g | dY %.6g | moe.x %.6g | "
                            "moe.dh %.6g | moe.shpre %.6g\n",
                            范数(dOutTokens), 范数(dYAllV[0]), 范数(moeXsV[0]),
                            范数(moeDhV[0]), 范数(moeShPreV[0]));
                float nWq16 = 范数(层表[cfg.n_layers/2].Wq->g);
                float nSh16 = 范数(层表[cfg.n_layers/2].shW1->g);
                std::printf("  grad norms: finalNorm %.6g | Wq(L16) %.6g | Wq(L0) %.6g | "
                            "shW1(L16) %.6g | shW1(L0) %.6g | gate(L0) %.6g\n",
                            范数(finalNorm->g), nWq16,
                            范数(层表[0].Wq->g), nSh16,
                            范数(层表[0].shW1->g), 范数(层表[0].gate->g));
                const bool ok = (bad == 0 && nWq16 > 1e-12f && nSh16 > 1e-12f);
                std::printf("%s\n", ok ? "3B CACHE PASS" : "3B CACHE FAIL");
                调度器.停止();
                return ok ? 0 : 1;
            }
            auto 抓梯度 = [&]() {
                std::vector<float> out;
                for (auto& p : 全部参数) {
                    std::vector<float> g(static_cast<std::size_t>(p->n));
                    p->g.下载(g.data(), g.size() * 4);
                    out.insert(out.end(), g.begin(), g.end());
                }
                return out;
            };
            std::vector<float> lblk;
            if (parity_mode) {
                fwd_cache = false;
                前向();
                lblk.resize(static_cast<std::size_t>(Tq) * V);
                logits.下载(lblk.data(), lblk.size() * 4);
            }
            fwd_cache = true;
            前向();
            std::vector<float> l1(static_cast<std::size_t>(Tq) * V);
            logits.下载(l1.data(), l1.size() * 4);
            前向();
            std::vector<float> l2(static_cast<std::size_t>(Tq) * V);
            logits.下载(l2.data(), l2.size() * 4);
            float fdiff = 0.0f;
            size_t 非有限 = 0;
            for (std::size_t i = 0; i < l1.size(); ++i) {
                fdiff = std::max(fdiff, std::fabs(l1[i] - l2[i]));
                if (!std::isfinite(l1[i])) ++非有限;
            }
            float pfdiff = 0.0f;
            if (parity_mode) {
                for (std::size_t i = 0; i < l2.size(); ++i)
                    pfdiff = std::max(pfdiff, std::fabs(lblk[i] - l2[i]));
            }
            float model_diff = 0.0f;
            if (!parity_mode && !big3b) {
                std::vector<std::int32_t> ctxIds(idsAll.begin(),
                                                 idsAll.begin() + Nctx);
                asm_ersa_model::ASM_ERSAModel 参考(cfg, 主);
                const std::vector<float> ml = 参考.forward_cached(ctxIds, idh);
                for (std::size_t i = 0; i < l2.size() && i < ml.size(); ++i)
                    model_diff = std::max(model_diff, std::fabs(l2[i] - ml[i]));
            }

            清零梯度();
            bwd_cache = true;
            const float loss = 反向全部();
            const std::vector<float> g1 = 抓梯度();
            size_t 非有限梯度 = 0;
            double 最大梯度 = 0.0;
            for (float x : g1) {
                if (!std::isfinite(x)) ++非有限梯度;
                else 最大梯度 = std::max(最大梯度, double(std::fabs(x)));
            }
            清零梯度();
            bwd_cache = true;
            反向全部();
            const std::vector<float> g2 = 抓梯度();
            double gdiff = 0.0;
            for (std::size_t i = 0; i < g1.size() && i < g2.size(); ++i)
                gdiff = std::max(gdiff, double(std::fabs(g1[i] - g2[i])));

            double pgdiff = 0.0;
            if (parity_mode) {
                fwd_cache = false;
                前向();
                清零梯度();
                bwd_cache = false;
                反向全部();
                const std::vector<float> gb = 抓梯度();
                for (std::size_t i = 0; i < g1.size() && i < gb.size(); ++i)
                    pgdiff = std::max(pgdiff, double(std::fabs(g1[i] - gb[i])));
                bwd_cache = true;
                fwd_cache = true;
            }

            std::printf("cache 反向 smoke: Nctx=%u Tq=%u Ttot=%u | fwd diff %.3g | "
                        "parity fwd %.3g | model fwd %.3g | loss %.6f | 参数 %zu | "
                        "梯度元素 %zu | 非有限 %zu | max|g| %.6g | grad diff %.3g | "
                        "parity grad %.3g\n",
                        Nctx, Tq, T, fdiff, pfdiff, model_diff, loss, 全部参数.size(),
                        g1.size(), 非有限梯度, 最大梯度, gdiff, pgdiff);
            std::printf("  ||dEmbed||=%.6g ||dWq(L0)||=%.6g ||dWo(L0)||=%.6g ||dW1(L0)||=%.6g\n",
                        范数(embed->g), 范数(层表[0].Wq->g), 范数(层表[0].Wo->g),
                        范数(层表[0].W1->g));
            // dK/dV 走 float CAS 原子累加，累加顺序不确定；5e-4 对应 max|g|≈655 的
            // 相对误差 ~1e-6，仍能捕获真实的索引/归约错误。
            const bool ok = (非有限 == 0 && 非有限梯度 == 0 && fdiff <= 1e-6f &&
                             gdiff <= 5e-4 && model_diff <= 1e-5f && (!parity_mode ||
                             (pfdiff <= 1e-6f && pgdiff <= 1e-3)));
            std::printf("%s\n", ok ? "CACHE PASS" : "CACHE FAIL");
            调度器.停止();
            return ok ? 0 : 1;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "cache 反向失败: %s\n", e.what());
            调度器.停止();
            return 1;
        }
    }
    try {
        前向();
        std::vector<float> 训练logits(static_cast<std::size_t>(T) * V);
        logits.下载(训练logits.data(), 训练logits.size() * 4);

        asm_ersa_model::ASM_ERSAModel 参考(cfg, 主);
        const std::vector<float> 参考logits = 参考.forward({idh.begin(), idh.end()});
        float max_abs = 0.0f;
        for (std::size_t i = 0; i < 训练logits.size(); ++i)
            max_abs = std::max(max_abs, std::fabs(训练logits[i] - 参考logits[i]));
        std::printf("训练器前向 vs ASM_ERSAModel: max abs %.6g | dispatch %llu | sync %llu\n",
                    max_abs,
                    static_cast<unsigned long long>(参考.last_dispatch_count()),
                    static_cast<unsigned long long>(参考.last_sync_count()));
        for (std::uint32_t t = 0; t < T; ++t) {
            float tm = 0.0f;
            for (std::uint32_t v = 0; v < V; ++v) {
                const std::size_t i = std::size_t(t) * V + v;
                tm = std::max(tm, std::fabs(训练logits[i] - 参考logits[i]));
            }
            std::printf("  token %u max abs %.6g | 训练 %.6f 参考 %.6f\n", t, tm,
                        训练logits[std::size_t(t) * V], 参考logits[std::size_t(t) * V]);
        }
        std::printf("logits[0..7] 训练:");
        for (int i = 0; i < 8; ++i) std::printf(" %.7f", 训练logits[i]);
        std::printf("\nlogits[0..7] 参考:");
        for (int i = 0; i < 8; ++i) std::printf(" %.7f", 参考logits[i]);
        std::printf("\n");
        if (smoke) {
            auto 抓梯度 = [&]() {
                std::vector<float> out;
                for (auto& p : 全部参数) {
                    std::vector<float> g(static_cast<std::size_t>(p->n));
                    p->g.下载(g.data(), g.size() * 4);
                    out.insert(out.end(), g.begin(), g.end());
                }
                return out;
            };
            清零梯度();
            反向全部();
            const std::vector<float> g1 = 抓梯度();
            size_t 非有限 = 0;
            double 最大 = 0.0;
            for (float x : g1) {
                if (!std::isfinite(x)) ++非有限;
                else 最大 = std::max(最大, double(std::fabs(x)));
            }
            清零梯度();
            反向全部();
            const std::vector<float> g2 = 抓梯度();
            double 差 = 0.0;
            for (std::size_t i = 0; i < g1.size() && i < g2.size(); ++i)
                差 = std::max(差, double(std::fabs(g1[i] - g2[i])));
            std::printf("VK 反向 smoke: 参数 %zu | 梯度元素 %zu | 非有限 %zu | max|g| %.6g | 两次 max diff %.6g\n",
                        全部参数.size(), g1.size(), 非有限, 最大, 差);
            std::printf("  ||dEmbed||=%.6g ||dWq(L0)||=%.6g ||dWo(L0)||=%.6g ||dW1(L0)||=%.6g\n",
                        范数(embed->g), 范数(层表[0].Wq->g), 范数(层表[0].Wo->g), 范数(层表[0].W1->g));
            调度器.停止();
            return (非有限 == 0 && 差 <= 1e-6) ? 0 : 1;
        }
        if (argc >= 4) {
            清零梯度();
            反向全部();
            std::ofstream f(argv[3], std::ios::binary);
            if (!f) throw std::runtime_error("无法写梯度导出文件");
            const char magic[8] = {'M','B','2','G','R','A','D','1'};
            f.write(magic, 8);
            const std::uint32_t 头[] = {1, cfg.d_model, cfg.n_heads, cfg.d_k, cfg.d_v,
                                        cfg.d_ff, cfg.n_layers, cfg.vocab_size,
                                        cfg.n_experts, cfg.n_slots, cfg.top_r,
                                        cfg.moe_top_k, T, cfg.max_seq_len,
                                        cfg.block_size, cfg.top_blocks, cfg.window};
            f.write(reinterpret_cast<const char*>(头), sizeof(头));
            f.write(reinterpret_cast<const char*>(idh.data()), T * 4);
            f.write(reinterpret_cast<const char*>(th.data()), T * 4);
            std::vector<float> logits_h(static_cast<std::size_t>(T) * V);
            logits.下载(logits_h.data(), logits_h.size() * 4);
            f.write(reinterpret_cast<const char*>(logits_h.data()), logits_h.size() * 4);
            const std::uint32_t np = static_cast<std::uint32_t>(全部参数.size());
            f.write(reinterpret_cast<const char*>(&np), 4);
            for (auto& p : 全部参数) {
                const std::uint32_t nl = static_cast<std::uint32_t>(p->名.size());
                const std::uint32_t nd = static_cast<std::uint32_t>(p->p.布局().形状.size());
                f.write(reinterpret_cast<const char*>(&nl), 4);
                f.write(p->名.data(), nl);
                f.write(reinterpret_cast<const char*>(&nd), 4);
                f.write(reinterpret_cast<const char*>(p->p.布局().形状.data()), nd * 4);
                std::vector<float> ph(static_cast<std::size_t>(p->n));
                std::vector<float> gh(static_cast<std::size_t>(p->n));
                p->p.下载(ph.data(), ph.size() * 4);
                p->g.下载(gh.data(), gh.size() * 4);
                f.write(reinterpret_cast<const char*>(ph.data()), ph.size() * 4);
                f.write(reinterpret_cast<const char*>(gh.data()), gh.size() * 4);
            }
            f.close();
            std::printf("梯度导出: %s | 参数 %u 个\n", argv[3], np);
            调度器.停止();
            return 0;
        }
        清零梯度();
        反向全部();
        更新(1);
        for (int step = 2; step <= 步数; ++step) {
            前向();
            清零梯度();
            const float l = 反向全部();
            std::printf("step %3d loss=%.6f\n", step - 1, l);
            更新(std::uint32_t(step));
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "前向失败: %s\n", e.what());
        调度器.停止();
        return 1;
    }

    调度器.停止();
    return 0;
}
