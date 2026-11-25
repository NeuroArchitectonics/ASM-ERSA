#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "流水线调度器.hpp"
#include "vk张量.hpp"
#include "apollo_mini_apply_moe_i8.spv.hpp"
#include "apollo_mini_apply_moe_i8_bf16.spv.hpp"

using namespace vk张量系统;

struct 推送 {
    std::uint32_t E, F, D, kind;
    float lr, s;
    std::uint32_t seed, mode;
};
static_assert(sizeof(推送) == 32, "moe apply push");

static std::uint16_t bf16_pack(float v) {
    std::uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    return static_cast<std::uint16_t>((u + 0x8000u) >> 16);
}

static float bf16_unpack(std::uint16_t h) {
    const std::uint32_t u = std::uint32_t(h) << 16;
    float v;
    std::memcpy(&v, &u, sizeof(v));
    return v;
}

static std::uint32_t hash32(std::uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static float rand01(std::uint32_t idx, std::uint32_t seed) {
    return float(hash32(idx ^ (seed * 0x9E3779B9u)) & 0xFFFFFFu) / 16777216.0f;
}

static void 映射(std::uint32_t i, std::uint32_t E, std::uint32_t F, std::uint32_t D,
                 std::uint32_t kind, std::uint32_t& gidx, std::uint32_t& row,
                 std::uint32_t& grp, std::uint32_t& groups) {
    if (kind == 1u) {
        const std::uint32_t e = i / (F * D);
        const std::uint32_t rem = i % (F * D);
        const std::uint32_t f = rem / D;
        const std::uint32_t d = rem % D;
        gidx = (e * D + d) * F + f;
        row = e * F + f;
        grp = d / 128u;
        groups = (D + 127u) / 128u;
    } else {
        const std::uint32_t e = i / (D * F);
        const std::uint32_t rem = i % (D * F);
        const std::uint32_t d = rem / F;
        const std::uint32_t f = rem % F;
        gidx = (e * F + f) * D + d;
        row = e * D + d;
        grp = f / 128u;
        groups = (F + 127u) / 128u;
    }
}

static std::vector<std::int8_t> 参考(const std::vector<std::int8_t>& q0,
                                     const std::vector<float>& scales,
                                     const std::vector<float>& grad,
                                     std::uint32_t E, std::uint32_t F, std::uint32_t D,
                                     std::uint32_t kind, float lr, float s,
                                     std::uint32_t seed, std::uint32_t mode) {
    const std::uint32_t n = E * F * D;
    std::vector<std::int8_t> out(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        std::uint32_t gidx = 0, row = 0, grp = 0, groups = 1;
        映射(i, E, F, D, kind, gidx, row, grp, groups);
        const float sc = scales[row * groups + grp];
        const float wf = float(q0[i]) * sc - lr * s * grad[gidx];
        float q = std::floor(wf / sc + (mode != 0u ? rand01(i, seed) : 0.5f));
        q = std::clamp(q, -127.0f, 127.0f);
        out[i] = static_cast<std::int8_t>(q);
    }
    return out;
}

int main(int argc, char** argv) {
    try {
        const std::string spv = argc > 1 ? argv[1] : "spv";
        流水线调度器& 调度器 = 流水线调度器::获取();
        设备注册信息 注册;
        注册.应用名 = "apollo_mini_moe_i8_test";
        阶段配置 图形;
        阶段配置 计算{阶段配置::自动队列族, 0, 1};
        阶段配置 传输;
        调度器.启动(注册, {图形, 计算, 传输}, spv);
        std::printf("设备: %s | 已加载算子: %zu\n", 调度器.获取设备().名称(), 调度器.算子数量());

        struct 形状 { std::uint32_t E, F, D; };
        const 形状 形状表[] = {{2u, 7u, 130u}, {2u, 130u, 7u}};
        const float lr = 0.03f, scale = 0.7f;
        int 总错 = 0;
        for (bool bf16 : {false, true}) {
            for (const auto& sh : 形状表) {
                const std::uint32_t n = sh.E * sh.F * sh.D;
                for (std::uint32_t kind = 1; kind <= 2; ++kind) {
                    const std::uint32_t rows = kind == 1 ? sh.E * sh.F : sh.E * sh.D;
                    const std::uint32_t groups = kind == 1 ? (sh.D + 127u) / 128u
                                                           : (sh.F + 127u) / 128u;
                    const std::uint32_t sn = rows * groups;
                    const std::uint32_t qcap = n + 256u, scap = sn + 16u, gcap = n + 256u;
                    std::mt19937 随机(20261125u + n + kind);
                    std::uniform_int_distribution<int> q分布(-127, 127);
                    std::uniform_real_distribution<float> s分布(0.01f, 0.1f);
                    std::uniform_real_distribution<float> g分布(-0.02f, 0.02f);
                    std::vector<std::int8_t> q0(qcap, 77);
                    for (std::uint32_t i = 0; i < n; ++i) q0[i] = static_cast<std::int8_t>(q分布(随机));
                    std::vector<float> scales(scap, 12345.0f), grad(gcap, 12345.0f);
                    for (std::uint32_t i = 0; i < sn; ++i) scales[i] = s分布(随机);
                    for (std::uint32_t i = 0; i < n; ++i) grad[i] = g分布(随机);
                    std::vector<std::uint16_t> grad16(gcap, 0x1234);
                    for (std::uint32_t i = 0; i < n; ++i) grad16[i] = bf16_pack(grad[i]);
                    if (bf16)
                        for (std::uint32_t i = 0; i < n; ++i) grad[i] = bf16_unpack(grad16[i]);

                    vk张量 W(张量布局::连续({qcap}), 张量类型::整数8, "moe_apply_W");
                    vk张量 S(张量布局::连续({scap}), 张量类型::浮点32, "moe_apply_S");
                    vk张量 G(张量布局::连续({gcap}), bf16 ? 张量类型::浮点bf16 : 张量类型::浮点32,
                             "moe_apply_G");
                    S.上传(scales.data(), scales.size() * sizeof(float));
                    if (bf16) G.上传(grad16.data(), grad16.size() * sizeof(std::uint16_t));
                    else G.上传(grad.data(), grad.size() * sizeof(float));
                    const char* 名 = bf16 ? "apollo_mini_apply_moe_i8_bf16"
                                          : "apollo_mini_apply_moe_i8";
                    int 本错 = 0;
                    for (std::uint32_t mode = 0; mode < 2; ++mode) {
                        W.上传(q0.data(), q0.size());
                        const 推送 pc{sh.E, sh.F, sh.D, kind, lr, scale, 12345u, mode};
                        调度器.录制自定义派发算子(名, {W.取句柄(), S.取句柄(), G.取句柄()},
                                                  &pc, sizeof(pc), 派发规格{(n + 255u) / 256u, 1u, 1u});
                        std::vector<std::int8_t> got(qcap, 0);
                        W.下载(got.data(), got.size());
                        const auto ref = 参考(q0, scales, grad, sh.E, sh.F, sh.D,
                                               kind, lr, scale, 12345u, mode);
                        for (std::uint32_t i = 0; i < n; ++i) 本错 += got[i] != ref[i];
                        for (std::uint32_t i = n; i < qcap; ++i) 本错 += got[i] != 77;
                    }
                    std::vector<float> 得S(scap, 0.0f);
                    S.下载(得S.data(), 得S.size() * sizeof(float));
                    for (std::uint32_t i = sn; i < scap; ++i) 本错 += 得S[i] != 12345.0f;
                    if (bf16) {
                        std::vector<std::uint16_t> 得G(gcap, 0);
                        G.下载(得G.data(), 得G.size() * sizeof(std::uint16_t));
                        for (std::uint32_t i = n; i < gcap; ++i) 本错 += 得G[i] != 0x1234;
                    } else {
                        std::vector<float> 得G(gcap, 0.0f);
                        G.下载(得G.data(), 得G.size() * sizeof(float));
                        for (std::uint32_t i = n; i < gcap; ++i) 本错 += 得G[i] != 12345.0f;
                    }
                    std::printf("bf16=%d kind=%u E=%u F=%u D=%u groups=%u: 不一致/越界 %d\n",
                                bf16 ? 1 : 0, kind, sh.E, sh.F, sh.D, groups, 本错);
                    总错 += 本错;
                }
            }
        }

        int 方向错 = 0;
        for (std::uint32_t kind = 1; kind <= 2; ++kind) {
            const 形状 sh{2u, 7u, 130u};
            const std::uint32_t n = sh.E * sh.F * sh.D;
            const std::uint32_t rows = kind == 1 ? sh.E * sh.F : sh.E * sh.D;
            const std::uint32_t groups = kind == 1 ? (sh.D + 127u) / 128u
                                                   : (sh.F + 127u) / 128u;
            std::vector<std::int8_t> q(n, 0);
            std::vector<float> s(rows * groups, 1.0f), g(n, 1.0f);
            std::vector<std::uint16_t> g16(n);
            for (auto& v : g16) v = bf16_pack(1.0f);
            vk张量 W(张量布局::连续({n}), 张量类型::整数8, "direction_W");
            vk张量 S(张量布局::连续({rows * groups}), 张量类型::浮点32, "direction_S");
            vk张量 G(张量布局::连续({n}), 张量类型::浮点bf16, "direction_G");
            W.上传(q.data(), q.size());
            S.上传(s.data(), s.size() * sizeof(float));
            G.上传(g16.data(), g16.size() * sizeof(std::uint16_t));
            const 推送 pc{sh.E, sh.F, sh.D, kind, 0.6f, 1.0f, 1u, 0u};
            调度器.录制自定义派发算子("apollo_mini_apply_moe_i8_bf16",
                                      {W.取句柄(), S.取句柄(), G.取句柄()},
                                      &pc, sizeof(pc), 派发规格{(n + 255u) / 256u, 1u, 1u});
            std::vector<std::int8_t> got(n, 0);
            W.下载(got.data(), got.size());
            for (auto v : got) 方向错 += v != -1;
        }
        std::printf("下降方向: 不一致 %d / %d\n", 方向错, 2 * 2 * 7 * 130);
        总错 += 方向错;
        std::printf("%s\n", 总错 == 0 ? "MOE_DIRECT_I8 PASS" : "MOE_DIRECT_I8 FAIL");
        return 总错 == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "异常: %s\n", e.what());
        return 1;
    }
}
