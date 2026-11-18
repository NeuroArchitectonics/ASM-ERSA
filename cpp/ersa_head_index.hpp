#pragma once
// Per-head IVF index with dynamic capacity, async rebuild and dense fallback.
// A head is invalidated only by its own overflow; other HeadIndex instances keep
// running IVF independently. Rebuild tasks form a per-head dependency chain and
// are not dependencies of other heads' queries.
//
// Dynamic CAP: after k-means converges, counts give the exact largest cluster.
// The member buffer is allocated to that size (plus headroom) instead of the
// worst-case CAP. Rebuild is two-stage: the k-means chain runs in the background,
// then finish_rebuild() sizes the member buffer and fills it. If the largest
// cluster exceeds CAP even after rebuild, the head stays invalid and queries use
// dense fallback (or throw when the dense budget is exceeded).

#include "vk张量.hpp"
#include "流水线调度器.hpp"
#include "ersa_budget.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace asm_ersa {

using namespace vk张量系统;

struct CapPolicy {
    bool enabled = false;
    uint32_t min_cap = 64;
    uint32_t max_cap = 0;    // 0 => cfg.CAP
    float headroom = 1.125f; // margin over the observed largest cluster
    uint32_t align = 16;     // round the chosen CAP
};

struct HeadIndexConfig {
    uint32_t N = 0;
    uint32_t N0 = 0;
    uint32_t C = 0;
    uint32_t DK = 0;
    uint32_t DV = 0;
    uint32_t T = 0;
    uint32_t CAP = 0;
    uint32_t TOPB = 8;
    uint32_t WINDOW = 8;
    float tau = 1.0f;
    float sink = -1e30f;
    // Dense fallback is only allowed while T*N <= dense_budget. Above it,
    // the affected head blocks until its rebuild finishes, then uses IVF.
    uint64_t dense_budget = 1ull << 20;
    DenseBudgetPolicy budget_policy{};
    bool use_adaptive_budget = false;
    CapPolicy cap_policy{};
};

enum class FallbackMode { Ivf, Dense, Block };

struct IvfParams {
    uint32_t B, H, T, N, DK, DV, C, LMAX, TOPB, WINDOW, pad0, pad1;
    float tau, sink;
};
static_assert(sizeof(IvfParams) == 56, "IvfParams must be 56 bytes");

struct DenseParams {
    uint32_t B, H, T, N, DK, DV, pad0, pad1;
    float tau, sink, pad2, pad3;
};
static_assert(sizeof(DenseParams) == 48, "DenseParams must be 48 bytes");

struct KParams {
    uint32_t B, H, N, DK, C, pad0, pad1, pad2;
};
static_assert(sizeof(KParams) == 32, "KParams must be 32 bytes");

struct FillParams {
    uint32_t B, H, N, C, LMAX, pad0, pad1, pad2;
};
static_assert(sizeof(FillParams) == 32, "FillParams must be 32 bytes");

struct IncAssignParams {
    uint32_t B, H, M, DK, C, pad0, pad1, pad2;
};
static_assert(sizeof(IncAssignParams) == 32, "IncAssignParams must be 32 bytes");

struct IncAppendParams {
    uint32_t B, H, M, C, LMAX, OFFSET, pad0, pad1;
};
static_assert(sizeof(IncAppendParams) == 32, "IncAppendParams must be 32 bytes");

struct IncUpdateParams {
    uint32_t B, H, M, DK, C, pad0, pad1, pad2;
    float lr, pad3, pad4, pad5;
};
static_assert(sizeof(IncUpdateParams) == 48, "IncUpdateParams must be 48 bytes");

inline vk张量 make_f32(size_t n, const std::string& name) {
    return vk张量(张量布局::连续({static_cast<uint32_t>(n)}), 张量类型::浮点32, name);
}

inline vk张量 make_u32(size_t n, const std::string& name) {
    return vk张量(张量布局::连续({static_cast<uint32_t>(n)}), 张量类型::整数32, name);
}

inline void normalize_row(float* v, uint32_t d) {
    double n2 = 0.0;
    for (uint32_t i = 0; i < d; ++i) n2 += double(v[i]) * double(v[i]);
    const float inv = float(1.0 / std::sqrt(n2 + 1e-9));
    for (uint32_t i = 0; i < d; ++i) v[i] *= inv;
}

class HeadIndex {
public:
    HeadIndex(流水线调度器& sched, HeadIndexConfig cfg)
        : sched_(sched), cfg_(cfg),
          tK_(make_f32(size_t(cfg.N) * cfg.DK, "hi_K")),
          tV_(make_f32(size_t(cfg.N) * cfg.DV, "hi_V")),
          tQ_(make_f32(size_t(cfg.T) * cfg.DK, "hi_Q")),
          tOut_(make_f32(size_t(cfg.T) * cfg.DV, "hi_Out")),
          tCent_(make_f32(size_t(cfg.C) * cfg.DK, "hi_Cent")),
          tCounts_(make_u32(cfg.C, "hi_Counts")),
          tMember_(make_u32(size_t(cfg.C) * initial_cap(cfg), "hi_Member")),
          tAssign_(make_u32(cfg.N, "hi_Assign")),
          tKnew_(make_f32(cfg.DK, "hi_Knew")),
          tAssignNew_(make_u32(1, "hi_AssignNew")),
          tOverflow_(make_u32(cfg.C, "hi_Overflow")) {
        if (cfg_.N == 0 || cfg_.N0 == 0 || cfg_.C == 0 || cfg_.DK == 0 ||
            cfg_.DV == 0 || cfg_.T == 0 || cfg_.CAP == 0) {
            throw std::runtime_error("HeadIndex: N/N0/C/DK/DV/T/CAP must be > 0");
        }
        if (cfg_.N0 > cfg_.N) throw std::runtime_error("HeadIndex: N0 must be <= N");
        active_cap_ = initial_cap(cfg_);
        fixed_cap_ = cfg_.CAP;
    }

    void set_data(const std::vector<float>& K, const std::vector<float>& V,
                  const std::vector<float>& Q) {
        if (K.size() != size_t(cfg_.N) * cfg_.DK ||
            V.size() != size_t(cfg_.N) * cfg_.DV ||
            Q.size() != size_t(cfg_.T) * cfg_.DK) {
            throw std::runtime_error("HeadIndex::set_data size mismatch");
        }
        wait_idle();
        hK_ = K;
        hV_ = V;
        hQ_ = Q;
        tK_.上传(hK_.data(), hK_.size() * sizeof(float));
        tV_.上传(hV_.data(), hV_.size() * sizeof(float));
        tQ_.上传(hQ_.data(), hQ_.size() * sizeof(float));
        active_cap_ = initial_cap(cfg_);
        ensure_member_capacity(active_cap_);
        reset_centroids();
        zero_assign_counts();
        valid_ = false;
        capacity_overflow_ = false;
        overflow_count_ = 0;
        max_cluster_ = 0;
    }

    // cap only applies when CapPolicy is disabled; dynamic mode derives it from counts.
    void build_initial(int iters = 10, uint32_t cap = 0) {
        wait_idle();
        if (!cfg_.cap_policy.enabled) {
            fixed_cap_ = (cap == 0) ? cfg_.CAP : std::min(cap, cfg_.CAP);
        }
        reset_centroids();
        zero_assign_counts();
        valid_ = false;

        KParams kp{1, 1, cfg_.N0, cfg_.DK, cfg_.C, 0, 0, 0};
        uint64_t prev = 0;
        for (int it = 0; it < iters; ++it) {
            std::vector<张量句柄> insA = {tK_.取句柄(), tCent_.取句柄(), tAssign_.取句柄()};
            uint64_t ta = prev == 0
                ? sched_.提交自定义算子("kmeans_assign", insA, &kp, sizeof(kp), {cfg_.N0, 1, 1})
                : sched_.提交自定义算子("kmeans_assign", insA, &kp, sizeof(kp), {cfg_.N0, 1, 1}, {prev});
            std::vector<张量句柄> insU = {tK_.取句柄(), tAssign_.取句柄(), tCent_.取句柄(),
                                          tCounts_.取句柄()};
            prev = sched_.提交自定义算子("kmeans_update", insU, &kp, sizeof(kp), {cfg_.C, 1, 1}, {ta});
        }
        sched_.等待(prev);
        finalize_index();
    }

    void ingest(const std::vector<float>& Knew, const std::vector<float>& Vnew,
                uint32_t offset, float lr = 0.2f) {
        if (rebuild_pending()) finish_rebuild();
        wait_idle();
        const uint32_t M = uint32_t(Knew.size() / cfg_.DK);
        if (M == 0 || M > cfg_.N || Vnew.size() != size_t(M) * cfg_.DV ||
            offset + M > cfg_.N) {
            throw std::runtime_error("HeadIndex::ingest size mismatch");
        }
        if (tKnew_.字节大小() != size_t(M) * cfg_.DK * sizeof(float)) {
            tKnew_ = make_f32(size_t(M) * cfg_.DK, "hi_Knew");
        }
        if (tAssignNew_.字节大小() != size_t(M) * sizeof(uint32_t)) {
            tAssignNew_ = make_u32(M, "hi_AssignNew");
        }
        tKnew_.上传(Knew.data(), Knew.size() * sizeof(float));
        std::vector<uint32_t> zero(cfg_.C, 0u);
        tOverflow_.上传(zero.data(), zero.size() * sizeof(uint32_t));

        IncAssignParams iap{1, 1, M, cfg_.DK, cfg_.C, 0, 0, 0};
        IncAppendParams app{1, 1, M, cfg_.C, active_cap_, offset, 0, 0};
        IncUpdateParams iup{1, 1, M, cfg_.DK, cfg_.C, 0, 0, 0, lr, 0, 0, 0};

        std::vector<张量句柄> insA = {tKnew_.取句柄(), tCent_.取句柄(), tAssignNew_.取句柄()};
        uint64_t ta = sched_.提交自定义算子("kmeans_inc_assign", insA, &iap, sizeof(iap), {M, 1, 1});
        std::vector<张量句柄> insP = {tAssignNew_.取句柄(), tCounts_.取句柄(), tMember_.取句柄(),
                                      tOverflow_.取句柄()};
        uint64_t ap = sched_.提交自定义算子("kmeans_inc_append", insP, &app, sizeof(app),
                                            {cfg_.C, 1, 1}, {ta});
        std::vector<张量句柄> insU = {tKnew_.取句柄(), tAssignNew_.取句柄(), tCent_.取句柄(),
                                      tCounts_.取句柄()};
        last_task_ = sched_.提交自定义算子("kmeans_inc_update", insU, &iup, sizeof(iup),
                                           {cfg_.C, 1, 1}, {ap});
        sched_.等待(last_task_);

        std::vector<uint32_t> flags(cfg_.C, 0u);
        tOverflow_.下载(flags.data(), flags.size() * sizeof(uint32_t));
        overflow_count_ = 0;
        for (uint32_t v : flags) if (v != 0u) ++overflow_count_;
        if (overflow_count_ > 0) valid_ = false;

        std::vector<uint32_t> counts(cfg_.C, 0u);
        tCounts_.下载(counts.data(), counts.size() * sizeof(uint32_t));
        max_cluster_ = 0;
        for (uint32_t v : counts) max_cluster_ = std::max(max_cluster_, v);
        capacity_overflow_ = (overflow_count_ > 0);

        std::copy(Knew.begin(), Knew.end(), hK_.begin() + size_t(offset) * cfg_.DK);
        std::copy(Vnew.begin(), Vnew.end(), hV_.begin() + size_t(offset) * cfg_.DV);
        write_slice(tK_, Knew.data(), size_t(offset) * cfg_.DK, Knew.size());
        write_slice(tV_, Vnew.data(), size_t(offset) * cfg_.DV, Vnew.size());
    }

    // Submits attention for this head and returns its task id. The mode is
    // resolved here, so a head over the dense budget may block on its rebuild.
    uint64_t submit_attention(FallbackMode* mode_out = nullptr) {
        FallbackMode mode = fallback_mode();
        last_mode_ = mode;
        uint64_t task = 0;
        if (mode == FallbackMode::Ivf) {
            task = submit_ivf();
            last_ivf_task_ = task;
        } else if (mode == FallbackMode::Dense) {
            task = submit_dense();
        } else {
            if (rebuild_task_ == 0) {
                throw std::runtime_error(
                    "HeadIndex: index invalid over dense budget and no rebuild in flight");
            }
            finish_rebuild();
            if (fallback_mode() != FallbackMode::Ivf) {
                throw std::runtime_error(
                    "HeadIndex: CAP cannot hold the largest cluster; increase CAP");
            }
            task = submit_ivf();
            last_ivf_task_ = task;
            last_mode_ = FallbackMode::Ivf;
        }
        last_task_ = task;
        if (mode_out != nullptr) *mode_out = last_mode_;
        return task;
    }

    void wait_and_download(uint64_t task, float* out) {
        sched_.等待(task);
        tOut_.下载(out, size_t(cfg_.T) * cfg_.DV * sizeof(float));
    }

    void download_out(float* out) {
        if (last_task_ == 0) throw std::runtime_error("HeadIndex: no attention task submitted");
        wait_and_download(last_task_, out);
    }

    void attention(std::vector<float>& out, uint64_t* task_out = nullptr) {
        out.resize(size_t(cfg_.T) * cfg_.DV);
        const uint64_t task = submit_attention();
        if (task_out != nullptr) {
            *task_out = task;
            return;
        }
        sched_.等待(task);
        tOut_.下载(out.data(), out.size() * sizeof(float));
    }

    FallbackMode fallback_mode() const {
        if (valid_) return FallbackMode::Ivf;
        bool allow = false;
        if (cfg_.use_adaptive_budget) {
            allow = cfg_.budget_policy.allow_dense(cfg_.T, cfg_.N, cfg_.DK, cfg_.DV);
        } else {
            allow = (uint64_t(cfg_.T) * uint64_t(cfg_.N) <= cfg_.dense_budget);
        }
        return allow ? FallbackMode::Dense : FallbackMode::Block;
    }

    FallbackMode last_mode() const { return last_mode_; }

    static const char* mode_name(FallbackMode m) {
        switch (m) {
        case FallbackMode::Ivf: return "ivf";
        case FallbackMode::Dense: return "dense";
        case FallbackMode::Block: return "block";
        }
        return "?";
    }

    // Submits the k-means chain for this head and returns immediately. The head
    // is invalid during the rebuild; queries use dense attention. finish_rebuild()
    // sizes the member buffer from the new cluster counts and fills it.
    void start_async_rebuild(int iters = 12) {
        if (rebuild_pending()) finish_rebuild();
        wait_idle();
        if (!cfg_.cap_policy.enabled) fixed_cap_ = cfg_.CAP;
        reset_centroids();
        zero_assign_counts();
        valid_ = false;
        capacity_overflow_ = false;
        max_cluster_ = 0;

        KParams kp{1, 1, cfg_.N, cfg_.DK, cfg_.C, 0, 0, 0};
        uint64_t prev = 0;
        for (int it = 0; it < iters; ++it) {
            std::vector<张量句柄> insA = {tK_.取句柄(), tCent_.取句柄(), tAssign_.取句柄()};
            uint64_t ta = prev == 0
                ? sched_.提交自定义算子("kmeans_assign", insA, &kp, sizeof(kp), {cfg_.N, 1, 1})
                : sched_.提交自定义算子("kmeans_assign", insA, &kp, sizeof(kp), {cfg_.N, 1, 1}, {prev});
            std::vector<张量句柄> insU = {tK_.取句柄(), tAssign_.取句柄(), tCent_.取句柄(),
                                          tCounts_.取句柄()};
            prev = sched_.提交自定义算子("kmeans_update", insU, &kp, sizeof(kp),
                                         {cfg_.C, 1, 1}, {ta});
        }
        rebuild_task_ = prev;
        last_task_ = prev;
    }

    void finish_rebuild() {
        if (rebuild_task_ == 0) return;
        sched_.等待(rebuild_task_);
        rebuild_task_ = 0;
        finalize_index();
    }

    bool valid() const { return valid_; }
    bool rebuild_pending() const { return rebuild_task_ != 0; }
    bool capacity_overflow() const { return capacity_overflow_; }
    uint32_t overflow_count() const { return overflow_count_; }
    uint32_t active_cap() const { return active_cap_; }
    uint32_t max_cluster() const { return max_cluster_; }
    uint64_t member_bytes() const { return tMember_.字节大小(); }
    uint64_t rebuild_task() const { return rebuild_task_; }
    uint64_t last_task() const { return last_task_; }

private:
    static uint32_t initial_cap(const HeadIndexConfig& c) {
        uint32_t cap = c.CAP;
        if (c.cap_policy.enabled && c.cap_policy.min_cap > 0) {
            cap = std::min(c.CAP, c.cap_policy.min_cap);
        }
        return std::max(1u, cap);
    }

    uint32_t cap_limit() const {
        uint32_t limit = cfg_.CAP;
        if (cfg_.cap_policy.enabled && cfg_.cap_policy.max_cap > 0) {
            limit = std::min(limit, cfg_.cap_policy.max_cap);
        }
        return std::max(1u, limit);
    }

    uint32_t choose_cap(uint32_t max_count) const {
        if (!cfg_.cap_policy.enabled) return std::max(1u, fixed_cap_);
        const CapPolicy& p = cfg_.cap_policy;
        uint64_t want = uint64_t(std::ceil(double(max_count) * double(p.headroom)));
        want = std::max<uint64_t>(want, max_count);
        want = std::max<uint64_t>(want, p.min_cap == 0 ? 1u : p.min_cap);
        const uint32_t align = std::max(1u, p.align);
        if (align > 1) want = ((want + align - 1) / align) * align;
        return uint32_t(std::max<uint64_t>(1u, std::min<uint64_t>(want, cap_limit())));
    }

    void ensure_member_capacity(uint32_t cap) {
        const size_t need = size_t(cfg_.C) * cap;
        if (cap == active_cap_ && tMember_.字节大小() == need * sizeof(uint32_t)) return;
        active_cap_ = cap;
        tMember_ = make_u32(need, "hi_Member");
    }

    // Runs after the k-means chain has landed: pick CAP from exact counts, fill
    // member lists, and decide whether the head can serve IVF.
    void finalize_index() {
        std::vector<uint32_t> counts(cfg_.C, 0u);
        tCounts_.下载(counts.data(), counts.size() * sizeof(uint32_t));
        max_cluster_ = 0;
        for (uint32_t v : counts) max_cluster_ = std::max(max_cluster_, v);

        ensure_member_capacity(choose_cap(max_cluster_));
        FillParams fp{1, 1, cfg_.N, cfg_.C, active_cap_, 0, 0, 0};
        std::vector<张量句柄> insF = {tAssign_.取句柄(), tMember_.取句柄()};
        last_task_ = sched_.提交自定义算子("kmeans_fill_members", insF, &fp, sizeof(fp),
                                          {cfg_.C, 1, 1});
        sched_.等待(last_task_);

        overflow_count_ = 0;
        for (uint32_t v : counts) if (v > active_cap_) ++overflow_count_;
        capacity_overflow_ = (max_cluster_ > active_cap_);
        valid_ = !capacity_overflow_;
    }

    uint64_t submit_ivf() {
        IvfParams p{1, 1, cfg_.T, cfg_.N, cfg_.DK, cfg_.DV, cfg_.C, active_cap_,
                    cfg_.TOPB, cfg_.WINDOW, 0, 0, cfg_.tau, cfg_.sink};
        std::vector<张量句柄> ins = {tQ_.取句柄(), tK_.取句柄(), tV_.取句柄(),
                                      tCent_.取句柄(), tMember_.取句柄(), tCounts_.取句柄(),
                                      tOut_.取句柄()};
        return sched_.提交自定义算子("ersa_ivf_attn_fused", ins, &p, sizeof(p),
                                      {cfg_.T, 1, 1});
    }

    uint64_t submit_dense() {
        DenseParams p{1, 1, cfg_.T, cfg_.N, cfg_.DK, cfg_.DV, 0, 0,
                      cfg_.tau, cfg_.sink, 0, 0};
        std::vector<张量句柄> ins = {tQ_.取句柄(), tK_.取句柄(), tV_.取句柄(),
                                      tOut_.取句柄()};
        return sched_.提交自定义算子("dense_attn_fused", ins, &p, sizeof(p),
                                      {cfg_.T, 1, 1});
    }

    void wait_last_ivf() {
        if (last_ivf_task_ != 0) {
            sched_.等待(last_ivf_task_);
            last_ivf_task_ = 0;
        }
    }

    void wait_idle() {
        wait_last_ivf();
        if (rebuild_task_ != 0) {
            sched_.等待(rebuild_task_);
            rebuild_task_ = 0;
        }
        if (last_task_ != 0) {
            sched_.等待(last_task_);
            last_task_ = 0;
        }
    }

    // All buffers are persistently mapped and coherent in this runtime. Slice
    // writes avoid re-uploading the whole K/V on every incremental ingest.
    static void write_slice(vk张量& t, const float* src, size_t elem_offset, size_t elems) {
        t.同步();
        void* base = t.映射();
        if (base == nullptr) throw std::runtime_error("HeadIndex: tensor is not mapped");
        std::memcpy(static_cast<float*>(base) + elem_offset, src, elems * sizeof(float));
    }

    void reset_centroids() {
        if (hK_.empty()) throw std::runtime_error("HeadIndex: set_data required");
        std::vector<float> cent(size_t(cfg_.C) * cfg_.DK);
        for (uint32_t c = 0; c < cfg_.C; ++c) {
            const uint32_t src = c % cfg_.N;
            for (uint32_t d = 0; d < cfg_.DK; ++d) {
                cent[c * cfg_.DK + d] = hK_[size_t(src) * cfg_.DK + d];
            }
            normalize_row(&cent[c * cfg_.DK], cfg_.DK);
        }
        tCent_.上传(cent.data(), cent.size() * sizeof(float));
    }

    void zero_assign_counts() {
        std::vector<uint32_t> assign(cfg_.N, 0u);
        std::vector<uint32_t> counts(cfg_.C, 0u);
        tAssign_.上传(assign.data(), assign.size() * sizeof(uint32_t));
        tCounts_.上传(counts.data(), counts.size() * sizeof(uint32_t));
    }

    流水线调度器& sched_;
    HeadIndexConfig cfg_;
    vk张量 tK_, tV_, tQ_, tOut_, tCent_;
    vk张量 tCounts_, tMember_, tAssign_;
    vk张量 tKnew_, tAssignNew_, tOverflow_;
    std::vector<float> hK_, hV_, hQ_;
    bool valid_ = false;
    bool capacity_overflow_ = false;
    uint32_t overflow_count_ = 0;
    uint32_t active_cap_ = 0;
    uint32_t max_cluster_ = 0;
    uint32_t fixed_cap_ = 0;
    FallbackMode last_mode_ = FallbackMode::Dense;
    uint64_t last_task_ = 0;
    uint64_t last_ivf_task_ = 0;
    uint64_t rebuild_task_ = 0;
};

} // namespace asm_ersa
