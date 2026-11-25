// MoE W1/W2 INT8 转置同步对拍：优化器布局 → 前向布局
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "流水线调度器.hpp"
#include "vk张量.hpp"
#include "moe_i8_transpose.spv.hpp"

using namespace vk张量系统;

struct 推送 { std::uint32_t E, F, D, dir, total; };
static_assert(sizeof(推送) == 20, "moe_i8_transpose push");

int main(int argc, char** argv) {
    try {
    const std::string spv = argc > 1 ? argv[1] : "spv";
    流水线调度器& 调度器 = 流水线调度器::获取();
    设备注册信息 注册;
    注册.应用名 = "moe_i8_transpose_test";
    阶段配置 图形;
    阶段配置 计算{阶段配置::自动队列族, 0, 1};
    阶段配置 传输;
    调度器.启动(注册, {图形, 计算, 传输}, spv);
    std::printf("设备: %s | 已加载算子: %zu\n", 调度器.获取设备().名称(), 调度器.算子数量());

    const std::uint32_t E = 2, F = 4, D = 8;
    const std::uint32_t n = E * F * D;
    std::mt19937 随机(2026);
    std::uniform_int_distribution<int> 分布(-127, 127);
    std::vector<std::int8_t> src(n), got(n), ref(n);
    for (auto& v : src) v = static_cast<std::int8_t>(分布(随机));

    vk张量 S(张量布局::连续({n}), 张量类型::整数8, "tr_src");
    vk张量 T(张量布局::连续({n}), 张量类型::整数8, "tr_dst");
    S.上传(src.data(), n);

    int 错 = 0;
    for (std::uint32_t dir = 0; dir < 2; ++dir) {
        const 推送 pc{E, F, D, dir, n};
        调度器.录制自定义派发算子("moe_i8_transpose", {S.取句柄(), T.取句柄()},
                                  &pc, sizeof(pc), 派发规格{(n + 255u) / 256u, 1u, 1u});
        T.下载(got.data(), n);
        for (std::uint32_t e = 0; e < E; ++e)
            for (std::uint32_t f = 0; f < F; ++f)
                for (std::uint32_t d = 0; d < D; ++d) {
                    if (dir == 0)
                        ref[e * F * D + f * D + d] = src[(e * D + d) * F + f];
                    else
                        ref[e * D * F + d * F + f] = src[(e * F + f) * D + d];
                }
        int 本错 = 0;
        for (std::uint32_t i = 0; i < n; ++i) 本错 += got[i] != ref[i];
        错 += 本错;
        std::printf("dir=%u: 不一致 %d / %u\n", dir, 本错, n);
    }
    std::printf("%s\n", 错 == 0 ? "MOE_I8_TRANSPOSE PASS" : "MOE_I8_TRANSPOSE FAIL");
    return 错 == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "异常: %s\n", e.what());
        return 1;
    }
}
