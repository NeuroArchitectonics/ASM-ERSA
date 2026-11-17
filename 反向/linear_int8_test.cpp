#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "流水线调度器.hpp"
#include "vk张量.hpp"
#include "linear_fwd_int8.spv.hpp"

using namespace vk张量系统;

namespace {
std::uint16_t 到半(float x) {
    std::uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    std::uint32_t sign = (u >> 16) & 0x8000u;
    std::int32_t exp = static_cast<std::int32_t>((u >> 23) & 0xFFu) - 127 + 15;
    std::uint32_t mant = u & 0x7FFFFFu;
    if (exp <= 0) return static_cast<std::uint16_t>(sign);
    if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7C00u);
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) | (mant >> 13));
}

float 到浮点(std::uint16_t h) {
    std::uint32_t sign = (h & 0x8000u) << 16;
    std::uint32_t exp = (h >> 10) & 0x1Fu;
    std::uint32_t mant = h & 0x3FFu;
    std::uint32_t u;
    if (exp == 0) u = sign;
    else if (exp == 31) u = sign | 0x7F800000u | (mant << 13);
    else u = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

std::int8_t 量化(float v, float s) {
    int q = static_cast<int>(std::lround(v / s));
    return static_cast<std::int8_t>(std::clamp(q, -127, 127));
}

struct 结果 {
    double 最大绝对 = 0.0;
    double 最大相对 = 0.0;
};

结果 测一层(std::uint32_t N, std::uint32_t K, std::uint32_t M, std::uint32_t GROUP,
            std::uint32_t 种子, bool 带bias) {
    const std::uint32_t G = (K + GROUP - 1) / GROUP;
    std::mt19937 随机(种子);
    std::normal_distribution<float> 分布(0.0f, 0.3f);
    std::vector<float> x(N * K), w(M * K), b(M, 0.0f);
    for (auto& v : x) v = 分布(随机);
    for (auto& v : w) v = 分布(随机);
    if (带bias) for (auto& v : b) v = 分布(随机);

    std::vector<std::int8_t> qx(N * K), qw(M * K);
    std::vector<float> sx(N), sw(M * G);
    for (std::uint32_t n = 0; n < N; ++n) {
        float 最大 = 0.0f;
        for (std::uint32_t k = 0; k < K; ++k) 最大 = std::max(最大, std::fabs(x[n * K + k]));
        const float s = std::max(最大 / 127.0f, 1e-10f);
        sx[n] = s;
        for (std::uint32_t k = 0; k < K; ++k) qx[n * K + k] = 量化(x[n * K + k], s);
    }
    for (std::uint32_t m = 0; m < M; ++m) {
        for (std::uint32_t g = 0; g < G; ++g) {
            float 最大 = 0.0f;
            for (std::uint32_t k = g * GROUP; k < std::min((g + 1) * GROUP, K); ++k)
                最大 = std::max(最大, std::fabs(w[m * K + k]));
            const float s = std::max(最大 / 127.0f, 1e-10f);
            sw[m * G + g] = s;
            for (std::uint32_t k = g * GROUP; k < std::min((g + 1) * GROUP, K); ++k)
                qw[m * K + k] = 量化(w[m * K + k], s);
        }
    }

    vk张量 Xq(张量布局::连续({N * K}), 张量类型::整数8, "x_i8");
    vk张量 SX(张量布局::连续({N}), 张量类型::浮点32, "x_scale");
    vk张量 Wq(张量布局::连续({M * K}), 张量类型::整数8, "w_i8");
    vk张量 SW(张量布局::连续({M * G}), 张量类型::浮点32, "w_scale");
    vk张量 Bias(张量布局::连续({M}), 张量类型::浮点32, "bias");
    vk张量 Y(张量布局::连续({N * M}), 张量类型::浮点16, "y_fp16");
    Xq.上传(qx.data(), qx.size());
    SX.上传(sx.data(), sx.size() * sizeof(float));
    Wq.上传(qw.data(), qw.size());
    SW.上传(sw.data(), sw.size() * sizeof(float));
    Bias.上传(b.data(), b.size() * sizeof(float));

    struct 推送 { std::uint32_t N, K, M, group, has_bias; } pc{N, K, M, GROUP, 带bias ? 1u : 0u};
    流水线调度器::获取().录制自定义派发算子(
        "linear_fwd_int8",
        {Xq.取句柄(), SX.取句柄(), Wq.取句柄(), SW.取句柄(), Bias.取句柄(), Y.取句柄()},
        &pc, sizeof(pc), 派发规格{(N * M + 255u) / 256u, 1u, 1u});

    std::vector<std::uint16_t> yh(N * M);
    Y.下载(yh.data(), yh.size() * sizeof(std::uint16_t));

    结果 r;
    for (std::uint32_t n = 0; n < N; ++n) {
        for (std::uint32_t m = 0; m < M; ++m) {
            double ref = b[m];
            for (std::uint32_t k = 0; k < K; ++k) ref += (double)x[n * K + k] * w[m * K + k];
            const double got = 到浮点(yh[n * M + m]);
            r.最大绝对 = std::max(r.最大绝对, std::fabs(got - ref));
            r.最大相对 = std::max(r.最大相对, std::fabs(got - ref) / (std::fabs(ref) + 1e-6));
        }
    }
    return r;
}
}  // namespace

int main(int argc, char** argv) {
    const std::string spv = argc > 1 ? argv[1] : "spv";
    流水线调度器& 调度器 = 流水线调度器::获取();
    设备注册信息 注册;
    注册.应用名 = "linear_int8_sensitivity";
    阶段配置 图形;
    阶段配置 计算{阶段配置::自动队列族, 0, 1};
    阶段配置 传输;
    调度器.启动(注册, {图形, 计算, 传输}, spv);
    std::printf("设备: %s\n", 调度器.获取设备().名称());

    const std::uint32_t T = 6, DX = 32, DK = 32, DV = 32;
    const auto wq = 测一层(T, DX, DK, 128, 1, true);
    const auto wk = 测一层(T, DX, DK, 128, 2, true);
    const auto wv = 测一层(T, DX, DV, 128, 3, true);
    std::printf("Wq  int8: max|Δ|=%.4e  max_rel=%.3f%%\n", wq.最大绝对, wq.最大相对 * 100.0);
    std::printf("Wk  int8: max|Δ|=%.4e  max_rel=%.3f%%\n", wk.最大绝对, wk.最大相对 * 100.0);
    std::printf("Wv  int8: max|Δ|=%.4e  max_rel=%.3f%%\n", wv.最大绝对, wv.最大相对 * 100.0);
    return 0;
}
