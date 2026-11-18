#include "asm_ersa_model.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace asm_ersa_model {
namespace {

float sigmoidf(float z) {
    if (z >= 0.0f) {
        const float e = std::exp(-z);
        return 1.0f / (1.0f + e);
    }
    const float e = std::exp(z);
    return e / (1.0f + e);
}

float siluf(float z) {
    if (z >= 0.0f) {
        const float e = std::exp(-z);
        return z / (1.0f + e);
    }
    const float e = std::exp(z);
    return z * e / (1.0f + e);
}

void linear(const float* x, const float* w, const float* b, float* y,
            uint32_t N, uint32_t K, uint32_t M) {
    for (uint32_t n = 0; n < N; ++n) {
        const float* xr = x + std::size_t(n) * K;
        for (uint32_t m = 0; m < M; ++m) {
            float s = (b != nullptr) ? b[m] : 0.0f;
            const float* wr = w + std::size_t(m) * K;
            for (uint32_t k = 0; k < K; ++k) s += xr[k] * wr[k];
            y[std::size_t(n) * M + m] = s;
        }
    }
}

void rmsnorm(const float* x, const float* g, float* y, uint32_t R, uint32_t D, float eps) {
    for (uint32_t r = 0; r < R; ++r) {
        const float* xr = x + std::size_t(r) * D;
        float sq = 0.0f;
        for (uint32_t d = 0; d < D; ++d) sq += xr[d] * xr[d];
        const float inv = 1.0f / std::sqrt(sq / float(D) + eps);
        for (uint32_t d = 0; d < D; ++d) {
            y[std::size_t(r) * D + d] = xr[d] * inv * g[d];
        }
    }
}

void l2norm_rows(float* x, uint32_t R, uint32_t D) {
    for (uint32_t r = 0; r < R; ++r) {
        float* xr = x + std::size_t(r) * D;
        float sq = 0.0f;
        for (uint32_t d = 0; d < D; ++d) sq += xr[d] * xr[d];
        const float inv = 1.0f / std::sqrt(sq + 1e-12f);
        for (uint32_t d = 0; d < D; ++d) xr[d] *= inv;
    }
}

std::vector<uint32_t> top_r(const std::vector<float>& scores, uint32_t r) {
    std::vector<float> tmp = scores;
    r = std::min<uint32_t>(r, uint32_t(tmp.size()));
    std::vector<uint32_t> idx(r);
    for (uint32_t k = 0; k < r; ++k) {
        float best = -3.402823e38f;
        uint32_t besti = 0;
        for (uint32_t i = 0; i < tmp.size(); ++i) {
            if (tmp[i] > best) {
                best = tmp[i];
                besti = i;
            }
        }
        idx[k] = besti;
        tmp[besti] = -3.402823e38f;
    }
    return idx;
}

void gr_read(const float* branches, const GRWeights& gr, const float* embed_x,
             bool first, float* out, uint32_t D) {
    float inv[4];
    float gate[4];
    auto bv = [&](uint32_t i, uint32_t d) -> float {
        return first ? embed_x[d] : branches[i * D + d];
    };

    for (uint32_t i = 0; i < 4; ++i) {
        float sq = 0.0f;
        for (uint32_t d = 0; d < D; ++d) {
            const float v = bv(i, d);
            sq += v * v;
        }
        inv[i] = 1.0f / std::sqrt(sq / float(D) + 1e-6f);
    }

    for (uint32_t i = 0; i < 4; ++i) {
        float acc = 0.0f;
        for (uint32_t d = 0; d < D; ++d) {
            for (uint32_t j = 0; j < 4; ++j) {
                const float nv = bv(j, d) * inv[j] * gr.norm_g[j][d];
                acc += nv * gr.read_w[i * 4 * D + j * D + d];
            }
        }
        gate[i] = sigmoidf(acc + gr.read_b[i]);
    }

    for (uint32_t d = 0; d < D; ++d) {
        float acc = 0.0f;
        for (uint32_t i = 0; i < 4; ++i) {
            acc += gate[i] * bv(i, d) * inv[i] * gr.norm_g[i][d];
        }
        out[d] = acc * 0.25f;
    }
}

void gr_write(float* branches, const float* y, const GRWeights& gr, uint32_t D) {
    float scale[4];
    for (uint32_t i = 0; i < 4; ++i) {
        float acc = 0.0f;
        for (uint32_t d = 0; d < D; ++d) acc += y[d] * gr.write_w[i * D + d];
        scale[i] = 2.0f * sigmoidf(acc + gr.write_b[i]);
    }
    for (uint32_t d = 0; d < D; ++d) {
        for (uint32_t i = 0; i < 4; ++i) {
            branches[i * D + d] += scale[i] * y[d];
        }
    }
}

void asm_step(const float* q, const float* k, const float* v, const float* x,
              const ASMWeights& w, std::vector<float>& K, std::vector<float>& V,
              std::vector<float>& beta, uint32_t H, uint32_t M, uint32_t DK,
              uint32_t DV, uint32_t R, float* out) {
    const float tau = 0.005f;
    float eta_acc = w.w_eta_b[0];
    for (std::size_t d = 0; d < w.w_eta_w.size(); ++d) eta_acc += x[d] * w.w_eta_w[d];
    const float eta = sigmoidf(eta_acc);

    const uint32_t r = std::min(R, M);
    std::vector<float> sq(M), sw(M);
    std::vector<float> a(r), p(r);

    for (uint32_t h = 0; h < H; ++h) {
        const float* qh = q + std::size_t(h) * DK;
        const float* kh = k + std::size_t(h) * DK;
        const float* vh = v + std::size_t(h) * DV;
        float* Kh = K.data() + std::size_t(h) * M * DK;
        float* Vh = V.data() + std::size_t(h) * M * DV;
        float* bh = beta.data() + std::size_t(h) * M;

        for (uint32_t m = 0; m < M; ++m) {
            float dq = 0.0f, dk = 0.0f;
            for (uint32_t d = 0; d < DK; ++d) {
                dq += qh[d] * Kh[std::size_t(m) * DK + d];
                dk += kh[d] * Kh[std::size_t(m) * DK + d];
            }
            sq[m] = dq / tau + bh[m];
            sw[m] = dk / tau + bh[m];
        }

        const std::vector<uint32_t> idx = top_r(sq, r);

        float mx = w.beta_null;
        for (uint32_t kk = 0; kk < r; ++kk) mx = std::max(mx, sq[idx[kk]]);
        float sum = std::exp(w.beta_null - mx);
        for (uint32_t kk = 0; kk < r; ++kk) {
            a[kk] = std::exp(sq[idx[kk]] - mx);
            sum += a[kk];
        }
        const float invsum = 1.0f / sum;
        for (uint32_t d = 0; d < DV; ++d) {
            float acc = 0.0f;
            for (uint32_t kk = 0; kk < r; ++kk) {
                acc += (a[kk] * invsum) * Vh[std::size_t(idx[kk]) * DV + d];
            }
            out[std::size_t(h) * DV + d] = acc;
        }

        float mxp = sw[idx[0]];
        for (uint32_t kk = 1; kk < r; ++kk) mxp = std::max(mxp, sw[idx[kk]]);
        float sump = 0.0f;
        for (uint32_t kk = 0; kk < r; ++kk) {
            p[kk] = std::exp(sw[idx[kk]] - mxp);
            sump += p[kk];
        }
        const float invp = (sump > 0.0f) ? (1.0f / sump) : 0.0f;
        for (uint32_t kk = 0; kk < r; ++kk) p[kk] *= invp;

        for (uint32_t kk = 0; kk < r; ++kk) {
            const uint32_t m = idx[kk];
            const float pm = p[kk];
            const float gm = w.gamma[m];
            float norm2 = 0.0f;
            for (uint32_t d = 0; d < DK; ++d) {
                const float nk = gm * Kh[std::size_t(m) * DK + d] + eta * pm * kh[d];
                Kh[std::size_t(m) * DK + d] = nk;
                norm2 += nk * nk;
            }
            const float invk = 1.0f / std::sqrt(norm2 + 1e-6f);
            for (uint32_t d = 0; d < DK; ++d) Kh[std::size_t(m) * DK + d] *= invk;
            for (uint32_t d = 0; d < DV; ++d) {
                Vh[std::size_t(m) * DV + d] =
                    gm * Vh[std::size_t(m) * DV + d] + eta * pm * vh[d];
            }
            bh[m] = 0.9f * bh[m] + 0.01f * pm;
        }
    }
}

void ersa_step(const std::vector<float>& qh, const std::vector<float>& kh,
               const std::vector<float>& vh, const std::vector<float>& bs,
               uint32_t H, uint32_t Tmax, uint32_t N, uint32_t DK, uint32_t DV,
               uint32_t NB, uint32_t BLOCK, uint32_t TOPB, uint32_t WINDOW,
               uint32_t t, float sink, uint32_t NBmax, float* out) {
    const float scale = 1.0f / std::sqrt(float(DK));
    const float tau = 0.005f;
    std::vector<float> bscore(NB);

    for (uint32_t h = 0; h < H; ++h) {
        const float* q = qh.data() + (std::size_t(h) * Tmax + t) * DK;
        for (uint32_t bi = 0; bi < NB; ++bi) {
            float s = 0.0f;
            for (uint32_t d = 0; d < DK; ++d) {
                s += q[d] * bs[std::size_t(h) * NBmax * DK + std::size_t(bi) * DK + d];
            }
            bscore[bi] = s * scale / tau;
        }

        const std::vector<uint32_t> top = top_r(bscore, std::min(TOPB, NB));
        std::vector<uint8_t> seen(N, 0);
        for (uint32_t bi : top) {
            const uint32_t start = bi * BLOCK;
            const uint32_t end = std::min(start + BLOCK, N);
            for (uint32_t idx = start; idx < end; ++idx) {
                if (idx <= t) seen[idx] = 1;
            }
        }
        const uint32_t wstart = (t + 1u > WINDOW) ? (t + 1u - WINDOW) : 0u;
        for (uint32_t idx = wstart; idx <= t; ++idx) seen[idx] = 1;

        std::vector<uint32_t> cand;
        for (uint32_t idx = 0; idx < N; ++idx) if (seen[idx]) cand.push_back(idx);

        std::vector<float> sc(cand.size());
        float mx = sink;
        for (std::size_t i = 0; i < cand.size(); ++i) {
            const uint32_t idx = cand[i];
            float s = 0.0f;
            for (uint32_t d = 0; d < DK; ++d) {
                s += q[d] * kh[std::size_t(h) * Tmax * DK + std::size_t(idx) * DK + d];
            }
            sc[i] = s * scale / tau;
            mx = std::max(mx, sc[i]);
        }
        float sum = std::exp(sink - mx);
        for (std::size_t i = 0; i < cand.size(); ++i) {
            sc[i] = std::exp(sc[i] - mx);
            sum += sc[i];
        }
        const float inv = 1.0f / sum;
        for (uint32_t d = 0; d < DV; ++d) {
            float acc = 0.0f;
            for (std::size_t i = 0; i < cand.size(); ++i) {
                acc += (sc[i] * inv) *
                       vh[std::size_t(h) * Tmax * DV + std::size_t(cand[i]) * DV + d];
            }
            out[std::size_t(h) * DV + d] = acc;
        }
    }
}

void moe(const float* x, const MoEWeights& w, uint32_t D, uint32_t F, uint32_t E,
         uint32_t TOPK, float* out) {
    std::vector<float> scores(E);
    for (uint32_t e = 0; e < E; ++e) {
        float s = 0.0f;
        for (uint32_t d = 0; d < D; ++d) s += x[d] * w.gate[std::size_t(e) * D + d];
        scores[e] = s;
    }

    const std::vector<uint32_t> top = top_r(scores, TOPK);
    float mx = scores[top[0]];
    for (uint32_t k = 1; k < TOPK; ++k) mx = std::max(mx, scores[top[k]]);
    float sum = 0.0f;
    std::vector<float> wt(TOPK);
    for (uint32_t k = 0; k < TOPK; ++k) {
        wt[k] = std::exp(scores[top[k]] - mx);
        sum += wt[k];
    }
    const float inv = 1.0f / sum;
    for (uint32_t k = 0; k < TOPK; ++k) wt[k] *= inv;

    std::fill(out, out + D, 0.0f);
    std::vector<float> h(F);

    for (uint32_t k = 0; k < TOPK; ++k) {
        const uint32_t e = top[k];
        const float wk = wt[k];
        for (uint32_t f = 0; f < F; ++f) {
            float pre = 0.0f;
            for (uint32_t d = 0; d < D; ++d) {
                pre += x[d] * w.W1[(std::size_t(e) * D + d) * F + f];
            }
            h[f] = siluf(pre);
        }
        for (uint32_t d = 0; d < D; ++d) {
            float acc = 0.0f;
            for (uint32_t f = 0; f < F; ++f) {
                acc += h[f] * w.W2[(std::size_t(e) * F + f) * D + d];
            }
            out[d] += wk * acc;
        }
    }

    for (uint32_t f = 0; f < F; ++f) {
        float pre = 0.0f, up = 0.0f;
        for (uint32_t d = 0; d < D; ++d) {
            pre += x[d] * w.sharedW1[std::size_t(f) * D + d];
            up += x[d] * w.sharedWup[std::size_t(f) * D + d];
        }
        h[f] = siluf(pre) * up;
    }
    for (uint32_t d = 0; d < D; ++d) {
        float acc = 0.0f;
        for (uint32_t f = 0; f < F; ++f) {
            acc += h[f] * w.sharedW2[std::size_t(d) * F + f];
        }
        out[d] += acc;
    }
}

void fill_gr(GRWeights& gr, uint32_t D, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (uint32_t i = 0; i < 4; ++i) gr.norm_g[i].assign(D, 1.0f);
    gr.read_w.resize(std::size_t(4) * 4 * D);
    for (auto& v : gr.read_w) v = 0.02f * dist(rng);
    gr.read_b.resize(4);
    for (auto& v : gr.read_b) v = 0.01f * dist(rng);
    gr.write_w.resize(std::size_t(4) * D);
    for (auto& v : gr.write_w) v = 0.02f * dist(rng);
    gr.write_b.resize(4);
    for (auto& v : gr.write_b) v = 0.01f * dist(rng);
}

void fill_moe(MoEWeights& m, uint32_t D, uint32_t F, uint32_t E, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    m.gate.resize(std::size_t(E) * D);
    for (auto& v : m.gate) v = 0.02f * dist(rng);
    m.W1.resize(std::size_t(E) * D * F);
    for (auto& v : m.W1) v = 0.02f * dist(rng);
    m.W2.resize(std::size_t(E) * F * D);
    for (auto& v : m.W2) v = 0.02f * dist(rng);
    m.sharedW1.resize(std::size_t(F) * D);
    for (auto& v : m.sharedW1) v = 0.02f * dist(rng);
    m.sharedWup.resize(std::size_t(F) * D);
    for (auto& v : m.sharedWup) v = 0.02f * dist(rng);
    m.sharedW2.resize(std::size_t(D) * F);
    for (auto& v : m.sharedW2) v = 0.02f * dist(rng);
}

} // namespace

ModelWeights make_random_weights(const ModelConfig& cfg, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const uint32_t D = cfg.d_model, H = cfg.n_heads, DK = cfg.d_k, DV = cfg.d_v;
    const uint32_t F = cfg.d_ff, V = cfg.vocab_size, E = cfg.n_experts;
    const uint32_t M = cfg.n_slots;

    ModelWeights w;
    w.embed.resize(std::size_t(V) * D);
    for (auto& v : w.embed) v = 0.05f * dist(rng);
    w.lm_head.resize(std::size_t(V) * D);
    for (auto& v : w.lm_head) v = 0.05f * dist(rng);
    w.out_norm.assign(D, 1.0f);
    w.layers.resize(cfg.n_layers);

    const float decays[4] = {0.99f, 0.999f, 0.9999f, 0.99999f};
    const uint32_t groups = std::max(1u, cfg.n_scales);
    const uint32_t per = std::max(1u, M / groups);

    for (uint32_t li = 0; li < cfg.n_layers; ++li) {
        LayerWeights& lw = w.layers[li];
        lw.is_ersa = (li % 4 == 3);
        fill_gr(lw.gr1, D, rng);
        fill_gr(lw.gr2, D, rng);
        fill_moe(lw.moe, D, F, E, rng);

        if (lw.is_ersa) {
            lw.ersa_w.Wq.resize(std::size_t(H) * DK * D);
            lw.ersa_w.Wk.resize(std::size_t(H) * DK * D);
            lw.ersa_w.Wv.resize(std::size_t(H) * DV * D);
            lw.ersa_w.Wo.resize(std::size_t(D) * H * DV);
            for (auto& v : lw.ersa_w.Wq) v = 0.05f * dist(rng);
            for (auto& v : lw.ersa_w.Wk) v = 0.05f * dist(rng);
            for (auto& v : lw.ersa_w.Wv) v = 0.05f * dist(rng);
            for (auto& v : lw.ersa_w.Wo) v = 0.05f * dist(rng);
            lw.ersa_w.out_norm.assign(D, 1.0f);
            lw.ersa_w.sink = -4.0f;
        } else {
            ASMWeights& a = lw.asm_w;
            a.Wq.resize(std::size_t(H) * DK * D);
            a.Wk.resize(std::size_t(H) * DK * D);
            a.Wv.resize(std::size_t(H) * DV * D);
            a.Wo.resize(std::size_t(D) * H * DV);
            for (auto& v : a.Wq) v = 0.05f * dist(rng);
            for (auto& v : a.Wk) v = 0.05f * dist(rng);
            for (auto& v : a.Wv) v = 0.05f * dist(rng);
            for (auto& v : a.Wo) v = 0.05f * dist(rng);

            a.w_eta_w.assign(D, 0.0f);
            a.w_eta_b.assign(1, -2.197f);

            a.slot_init_k.resize(std::size_t(M) * DK);
            for (uint32_t m = 0; m < M; ++m) {
                float sq = 0.0f;
                for (uint32_t d = 0; d < DK; ++d) {
                    const float v = dist(rng);
                    a.slot_init_k[std::size_t(m) * DK + d] = v;
                    sq += v * v;
                }
                const float inv = 1.0f / std::sqrt(sq + 1e-9f);
                for (uint32_t d = 0; d < DK; ++d) {
                    a.slot_init_k[std::size_t(m) * DK + d] *= inv;
                }
            }

            a.gamma.resize(M);
            for (uint32_t m = 0; m < M; ++m) {
                const uint32_t gi = std::min(m / per, groups - 1);
                a.gamma[m] = decays[gi % 4];
            }
            a.beta_null = -2.0f;
            a.out_norm.assign(D, 1.0f);
        }
    }
    return w;
}

std::vector<float> cpu_forward(const ModelConfig& cfg, const ModelWeights& weights,
                               const std::vector<int32_t>& input_ids) {
    const uint32_t D = cfg.d_model, H = cfg.n_heads, DK = cfg.d_k, DV = cfg.d_v;
    const uint32_t F = cfg.d_ff, V = cfg.vocab_size, E = cfg.n_experts;
    const uint32_t M = cfg.n_slots, R = cfg.top_r, TOPK = cfg.moe_top_k;
    const uint32_t Tmax = cfg.max_seq_len;
    const uint32_t HDK = H * DK, HDV = H * DV;
    const uint32_t NBmax = (Tmax + cfg.block_size - 1) / cfg.block_size;
    const uint32_t T = uint32_t(input_ids.size());

    if (T == 0 || T > Tmax) throw std::runtime_error("cpu_forward: 输入长度非法");
    if (weights.layers.size() != cfg.n_layers) {
        throw std::runtime_error("cpu_forward: layers 数量不匹配");
    }

    struct State {
        std::vector<float> K, V, beta;
        std::vector<float> qh, kh, vh, bs;
    };
    std::vector<State> st(cfg.n_layers);

    for (uint32_t li = 0; li < cfg.n_layers; ++li) {
        const LayerWeights& lw = weights.layers[li];
        if (!lw.is_ersa) {
            st[li].K.assign(std::size_t(H) * M * DK, 0.0f);
            st[li].V.assign(std::size_t(H) * M * DV, 0.0f);
            st[li].beta.assign(std::size_t(H) * M, 0.0f);
            for (uint32_t h = 0; h < H; ++h) {
                for (uint32_t m = 0; m < M; ++m) {
                    for (uint32_t d = 0; d < DK; ++d) {
                        st[li].K[(std::size_t(h) * M + m) * DK + d] =
                            lw.asm_w.slot_init_k[std::size_t(m) * DK + d];
                    }
                }
            }
        } else {
            st[li].qh.assign(std::size_t(H) * Tmax * DK, 0.0f);
            st[li].kh.assign(std::size_t(H) * Tmax * DK, 0.0f);
            st[li].vh.assign(std::size_t(H) * Tmax * DV, 0.0f);
            st[li].bs.assign(std::size_t(H) * NBmax * DK, 0.0f);
        }
    }

    std::vector<float> branches(std::size_t(4) * D);
    std::vector<float> x(D), q_raw(HDK), k_raw(HDK), v_cur(HDV);
    std::vector<float> o(HDV), y(D), x2(D), moe_out(D);
    std::vector<float> out_tokens(std::size_t(T) * D);
    std::vector<float> normed(std::size_t(T) * D);
    std::vector<float> logits(std::size_t(T) * V);

    for (uint32_t t = 0; t < T; ++t) {
        const int32_t id = input_ids[t];
        if (id < 0 || uint32_t(id) >= V) throw std::runtime_error("cpu_forward: id 越界");
        const float* xt = weights.embed.data() + std::size_t(id) * D;
        for (uint32_t i = 0; i < 4; ++i) {
            std::copy(xt, xt + D, branches.begin() + std::size_t(i) * D);
        }

        for (uint32_t li = 0; li < cfg.n_layers; ++li) {
            const LayerWeights& lw = weights.layers[li];
            gr_read(branches.data(), lw.gr1, xt, li == 0, x.data(), D);

            const std::vector<float>& Wq = lw.is_ersa ? lw.ersa_w.Wq : lw.asm_w.Wq;
            const std::vector<float>& Wk = lw.is_ersa ? lw.ersa_w.Wk : lw.asm_w.Wk;
            const std::vector<float>& Wv = lw.is_ersa ? lw.ersa_w.Wv : lw.asm_w.Wv;
            linear(x.data(), Wq.data(), nullptr, q_raw.data(), 1, D, HDK);
            linear(x.data(), Wk.data(), nullptr, k_raw.data(), 1, D, HDK);
            linear(x.data(), Wv.data(), nullptr, v_cur.data(), 1, D, HDV);
            l2norm_rows(q_raw.data(), H, DK);
            if (!lw.is_ersa) l2norm_rows(k_raw.data(), H, DK);

            if (lw.is_ersa) {
                const uint32_t N = t + 1;
                const uint32_t NB = (N + cfg.block_size - 1) / cfg.block_size;
                for (uint32_t h = 0; h < H; ++h) {
                    for (uint32_t d = 0; d < DK; ++d) {
                        st[li].qh[(std::size_t(h) * Tmax + t) * DK + d] =
                            q_raw[std::size_t(h) * DK + d];
                        st[li].kh[(std::size_t(h) * Tmax + t) * DK + d] =
                            k_raw[std::size_t(h) * DK + d];
                    }
                    for (uint32_t d = 0; d < DV; ++d) {
                        st[li].vh[(std::size_t(h) * Tmax + t) * DV + d] =
                            v_cur[std::size_t(h) * DV + d];
                    }
                    const uint32_t bi = t / cfg.block_size;
                    const uint32_t start = bi * cfg.block_size;
                    const float count = float(t - start + 1);
                    for (uint32_t d = 0; d < DK; ++d) {
                        float sum = k_raw[std::size_t(h) * DK + d];
                        for (uint32_t idx = start; idx < t; ++idx) {
                            sum += st[li].kh[(std::size_t(h) * Tmax + idx) * DK + d];
                        }
                        st[li].bs[(std::size_t(h) * NBmax + bi) * DK + d] = sum / count;
                    }
                }
                ersa_step(st[li].qh, st[li].kh, st[li].vh, st[li].bs,
                          H, Tmax, N, DK, DV, NB, cfg.block_size,
                          cfg.top_blocks, cfg.window, t, lw.ersa_w.sink,
                          NBmax, o.data());
            } else {
                asm_step(q_raw.data(), k_raw.data(), v_cur.data(), x.data(),
                         lw.asm_w, st[li].K, st[li].V, st[li].beta,
                         H, M, DK, DV, R, o.data());
            }

            const std::vector<float>& Wo = lw.is_ersa ? lw.ersa_w.Wo : lw.asm_w.Wo;
            const std::vector<float>& on = lw.is_ersa ? lw.ersa_w.out_norm : lw.asm_w.out_norm;
            linear(o.data(), Wo.data(), nullptr, y.data(), 1, HDV, D);
            rmsnorm(y.data(), on.data(), y.data(), 1, D, 1e-6f);
            gr_write(branches.data(), y.data(), lw.gr1, D);

            gr_read(branches.data(), lw.gr2, nullptr, false, x2.data(), D);
            moe(x2.data(), lw.moe, D, F, E, TOPK, moe_out.data());
            gr_write(branches.data(), moe_out.data(), lw.gr2, D);
        }

        for (uint32_t d = 0; d < D; ++d) out_tokens[std::size_t(t) * D + d] = branches[d];
    }

    rmsnorm(out_tokens.data(), weights.out_norm.data(), normed.data(), T, D, 1e-6f);
    linear(normed.data(), weights.lm_head.data(), nullptr, logits.data(), T, D, V);
    return logits;
}

} // namespace asm_ersa_model
