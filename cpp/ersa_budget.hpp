#pragma once
// Adaptive dense-fallback budget. Work is measured in MAC-like units
// (T * N * (DK + DV)); gpu_work_per_ms is calibrated on the target GPU.
#include <cstdint>

namespace asm_ersa {

struct DenseBudgetPolicy {
    uint32_t max_dense_n = 4096;   // dense kernel shared-memory/scale limit
    double target_ms = 2.0;        // allowed dense latency budget
    double gpu_work_per_ms = 2.0e9; // MAC-like work per ms (calibrate per GPU)

    double dense_work(uint32_t T, uint32_t N, uint32_t DK, uint32_t DV) const {
        return double(T) * double(N) * double(uint64_t(DK) + uint64_t(DV));
    }

    bool allow_dense(uint32_t T, uint32_t N, uint32_t DK, uint32_t DV) const {
        if (N == 0 || N > max_dense_n) return false;
        return dense_work(T, N, DK, DV) <= target_ms * gpu_work_per_ms;
    }
};

} // namespace asm_ersa
