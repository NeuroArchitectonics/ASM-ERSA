// adamw_update_bf16 对拍：bf16 梯度输入 → fp32 master/m/v，与 CPU fp64 参考比较 5 步。
// 同时检查：(1) 越界写（N=4000，缓冲 4096，尾部哨兵必须不变）；(2) zero_grad 是否清零 g；(3) m/v 数值。
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
#include "adamw_update_bf16.spv.hpp"

using namespace vk张量系统;

struct 规格Adam {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float m_corr;
    float v_corr;
    std::uint32_t size;
    float wd;
    std::uint32_t zero_grad;
};
static_assert(sizeof(规格Adam) == 36, "adamw push");

static std::uint16_t bf16_pack(float v) {
    std::uint32_t u;
    std::memcpy(&u, &v, 4);
    u = (u + 0x8000u) >> 16;
    return static_cast<std::uint16_t>(u);
}
static float bf16_unpack(std::uint16_t h) {
    const std::uint32_t u = std::uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

int main(int argc, char** argv) {
    try {
        const std::string spv = argc > 1 ? argv[1] : "spv";
        流水线调度器& 调度器 = 流水线调度器::获取();
        设备注册信息 注册;
        注册.应用名 = "adamw_update_bf16_test";
        阶段配置 图形;
        阶段配置 计算{阶段配置::自动队列族, 0, 1};
        阶段配置 传输;
        调度器.启动(注册, {图形, 计算, 传输}, spv);
        std::printf("设备: %s | 已加载算子: %zu\n", 调度器.获取设备().名称(), 调度器.算子数量());

        const std::uint32_t N = 4000;      // 非 256 倍数，用来验证 size 上界
        const std::uint32_t 容量 = 4096;
        const int 步数 = 5;
        const float lr = 1e-3f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, wd = 0.01f;

        std::mt19937 随机(20261125u);
        std::uniform_real_distribution<float> 权重分布(-0.5f, 0.5f);
        std::normal_distribution<float> 梯度分布(0.0f, 1e-3f);

        std::vector<float> 主机P(容量, 0.0f), 主机M(容量, 0.0f), 主机V(容量, 0.0f);
        for (std::uint32_t i = 0; i < N; ++i) 主机P[i] = 权重分布(随机);
        for (std::uint32_t i = N; i < 容量; ++i) 主机P[i] = 12345.0f;  // 哨兵
        const std::vector<float> P初始 = 主机P;

        std::vector<std::uint16_t> 主机G(容量, 0x1234);  // 尾部保留 0x1234 作哨兵
        std::vector<std::vector<float>> 梯度表(步数, std::vector<float>(N));

        vk张量 dP(张量布局::连续({容量}), 张量类型::浮点32, "adamw_P");
        vk张量 dM(张量布局::连续({容量}), 张量类型::浮点32, "adamw_M");
        vk张量 dV(张量布局::连续({容量}), 张量类型::浮点32, "adamw_V");
        vk张量 dG(张量布局::连续({容量}), 张量类型::浮点bf16, "adamw_G");
        dP.上传(主机P.data(), 主机P.size() * 4);
        dM.上传(主机M.data(), 主机M.size() * 4);
        dV.上传(主机V.data(), 主机V.size() * 4);

        // CPU 参考（double）
        std::vector<double> 参考P(P初始.begin(), P初始.begin() + N), 参考M(N, 0.0), 参考V(N, 0.0);

        for (int s = 0; s < 步数; ++s) {
            for (std::uint32_t i = 0; i < N; ++i) {
                梯度表[s][i] = 梯度分布(随机);
                主机G[i] = bf16_pack(梯度表[s][i]);
            }
            dG.上传(主机G.data(), 主机G.size() * 2);

            const double b1t = std::pow(double(b1), double(s + 1));
            const double b2t = std::pow(double(b2), double(s + 1));
            const 规格Adam pc{lr, b1, b2, eps,
                              static_cast<float>(1.0 / (1.0 - b1t)),
                              static_cast<float>(1.0 / (1.0 - b2t)),
                              N, wd, 1u};
            const 派发规格 规格{(容量 + 255u) / 256u, 1u, 1u};
            std::vector<张量句柄> 绑{dG.取句柄(), dP.取句柄(), dM.取句柄(), dV.取句柄()};
            调度器.录制自定义派发算子("adamw_update_bf16", 绑, &pc, sizeof(pc), 规格);

            // 参考同步推进（用 bf16 解包后的值，公平对比）
            for (std::uint32_t i = 0; i < N; ++i) {
                const double g = double(bf16_unpack(主机G[i]));
                const double mc = 1.0 / (1.0 - b1t), vc = 1.0 / (1.0 - b2t);
                参考M[i] = double(b1) * 参考M[i] + (1.0 - double(b1)) * g;
                参考V[i] = double(b2) * 参考V[i] + (1.0 - double(b2)) * g * g;
                double w = 参考P[i];
                w -= double(lr) * (参考M[i] * mc) / (std::sqrt(参考V[i] * vc) + double(eps));
                w -= double(lr) * double(wd) * w;
                参考P[i] = w;
            }
        }

        std::vector<float> 设备P(容量, 0.0f), 设备M(容量, 0.0f), 设备V(容量, 0.0f);
        std::vector<std::uint16_t> 设备G(容量, 0xFFFF);
        dP.下载(设备P.data(), 设备P.size() * 4);
        dM.下载(设备M.data(), 设备M.size() * 4);
        dV.下载(设备V.data(), 设备V.size() * 4);
        dG.下载(设备G.data(), 设备G.size() * 2);

        double p差 = 0.0, m差 = 0.0, v差 = 0.0;
        int 非有限 = 0;
        for (std::uint32_t i = 0; i < N; ++i) {
            if (!std::isfinite(设备P[i]) || !std::isfinite(设备M[i]) || !std::isfinite(设备V[i])) ++非有限;
            p差 = std::max(p差, std::fabs(double(设备P[i]) - 参考P[i]));
            m差 = std::max(m差, std::fabs(double(设备M[i]) - 参考M[i]));
            v差 = std::max(v差, std::fabs(double(设备V[i]) - 参考V[i]));
        }
        int 越界 = 0, 未清零 = 0;
        for (std::uint32_t i = N; i < 容量; ++i)
            if (设备P[i] != P初始[i]) ++越界;
        for (std::uint32_t i = 0; i < N; ++i)
            if (设备G[i] != 0) ++未清零;
        int 梯度越界 = 0;
        for (std::uint32_t i = N; i < 容量; ++i)
            if (设备G[i] != 0x1234) ++梯度越界;

        std::printf("N=%u 步数=%d  lr=%.1e wd=%.2f\n", N, 步数, lr, wd);
        std::printf("  max|Δp|=%.3e  max|Δm|=%.3e  max|Δv|=%.3e  非有限=%d\n", p差, m差, v差, 非有限);
        std::printf("  越界写=%d  g 未清零=%d  g 越界写=%d\n", 越界, 未清零, 梯度越界);

        const bool 通过 = p差 < 1e-6 && m差 < 1e-7 && v差 < 1e-12 && 非有限 == 0 && 越界 == 0 && 未清零 == 0 && 梯度越界 == 0;
        std::printf("%s\n", 通过 ? "ADAMW_BF16 PASS" : "ADAMW_BF16 FAIL");
        return 通过 ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("异常: %s\n", e.what());
        return 2;
    }
}
