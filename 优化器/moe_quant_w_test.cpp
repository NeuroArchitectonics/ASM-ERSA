// moe_quant_w 对拍：VK 内核（fp32 master → 前向 int8 + 组 scale）vs 主机 量化W1/量化W2
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "流水线调度器.hpp"
#include "vk张量.hpp"
#include "moe_quant_w.spv.hpp"

using namespace vk张量系统;

struct 推送 { std::uint32_t E, F, D, mode, G, base; };
static_assert(sizeof(推送) == 24, "moe_quant_w push");

static void 量化W1主机(const std::vector<float>& w, std::uint32_t E_, std::uint32_t D_,
                       std::uint32_t F_, std::vector<std::int8_t>& q, std::vector<float>& s) {
    const std::uint32_t G = (D_ + 127u) / 128u;
    q.assign(std::size_t(E_) * F_ * D_, 0);
    s.assign(std::size_t(E_) * F_ * G, 0.0f);
    for (std::uint32_t e = 0; e < E_; ++e)
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

static void 量化W2主机(const std::vector<float>& w, std::uint32_t E_, std::uint32_t D_,
                       std::uint32_t F_, std::vector<std::int8_t>& q, std::vector<float>& s) {
    const std::uint32_t G = (F_ + 127u) / 128u;
    q.assign(std::size_t(E_) * D_ * F_, 0);
    s.assign(std::size_t(E_) * D_ * G, 0.0f);
    for (std::uint32_t e = 0; e < E_; ++e)
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

int main(int argc, char** argv) {
    try {
        const std::string spv = argc > 1 ? argv[1] : "spv";
        流水线调度器& 调度器 = 流水线调度器::获取();
        设备注册信息 注册;
        注册.应用名 = "moe_quant_w_test";
        阶段配置 图形;
        阶段配置 计算{阶段配置::自动队列族, 0, 1};
        阶段配置 传输;
        调度器.启动(注册, {图形, 计算, 传输}, spv);
        std::printf("设备: %s\n", 调度器.获取设备().名称());

        const std::uint32_t E = 2, D = 256, F = 40;
        const std::uint32_t G1 = (D + 127u) / 128u;  // W1 按 d 分组
        const std::uint32_t G2 = (F + 127u) / 128u;  // W2 按 f 分组
        std::mt19937 随机(20261125u);
        std::uniform_real_distribution<float> 分布(-0.5f, 0.5f);

        std::vector<float> p1(std::size_t(E) * D * F), p2(std::size_t(E) * F * D);
        for (auto& v : p1) v = 分布(随机);
        for (auto& v : p2) v = 分布(随机);

        int 错 = 0;
        // mode 0 : W1
        {
            std::vector<std::int8_t> q0(std::size_t(E) * F * D), qr;
            std::vector<float> s0(std::size_t(E) * F * G1), sr;
            量化W1主机(p1, E, D, F, qr, sr);
            vk张量 P(张量布局::连续({E * D, F}), 张量类型::浮点32, "p1");
            vk张量 Q(张量布局::连续({E * F * D}), 张量类型::整数8, "q1");
            vk张量 S(张量布局::连续({E * F * G1}), 张量类型::浮点32, "s1");
            P.上传(p1.data(), p1.size() * 4);
            const 推送 pc{E, F, D, 0u, G1, 0u};
            调度器.录制自定义派发算子("moe_quant_w", {P.取句柄(), Q.取句柄(), S.取句柄()},
                                      &pc, sizeof(pc), 派发规格{E * F * G1, 1u, 1u});
            Q.下载(q0.data(), q0.size());
            S.下载(s0.data(), s0.size() * 4);
            int q错 = 0, s错 = 0;
            for (std::size_t i = 0; i < q0.size(); ++i) q错 += q0[i] != qr[i];
            for (std::size_t i = 0; i < s0.size(); ++i)
                s错 += std::fabs(double(s0[i]) - double(sr[i])) > 1e-6 * (1.0 + std::fabs(double(sr[i])));
            std::printf("mode0(W1): q 不一致 %d/%zu | s 不一致 %d/%zu\n", q错, q0.size(), s错, s0.size());
            错 += q错 + s错;
        }
        // mode 1 : W2
        {
            std::vector<std::int8_t> q0(std::size_t(E) * D * F), qr;
            std::vector<float> s0(std::size_t(E) * D * G2), sr;
            量化W2主机(p2, E, D, F, qr, sr);
            vk张量 P(张量布局::连续({E * F, D}), 张量类型::浮点32, "p2");
            vk张量 Q(张量布局::连续({E * D * F}), 张量类型::整数8, "q2");
            vk张量 S(张量布局::连续({E * D * G2}), 张量类型::浮点32, "s2");
            P.上传(p2.data(), p2.size() * 4);
            const 推送 pc{E, F, D, 1u, G2, 0u};
            调度器.录制自定义派发算子("moe_quant_w", {P.取句柄(), Q.取句柄(), S.取句柄()},
                                      &pc, sizeof(pc), 派发规格{E * D * G2, 1u, 1u});
            Q.下载(q0.data(), q0.size());
            S.下载(s0.data(), s0.size() * 4);
            int q错 = 0, s错 = 0;
            for (std::size_t i = 0; i < q0.size(); ++i) q错 += q0[i] != qr[i];
            for (std::size_t i = 0; i < s0.size(); ++i)
                s错 += std::fabs(double(s0[i]) - double(sr[i])) > 1e-6 * (1.0 + std::fabs(double(sr[i])));
            std::printf("mode1(W2): q 不一致 %d/%zu | s 不一致 %d/%zu\n", q错, q0.size(), s错, s0.size());
            错 += q错 + s错;
        }
        std::printf("%s\n", 错 == 0 ? "MOE_QUANT_W PASS" : "MOE_QUANT_W FAIL");
        return 错 == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "异常: %s\n", e.what());
        return 1;
    }
}
