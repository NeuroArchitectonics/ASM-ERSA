// linear_bwd_dwdb5 对拍：fp32 版 vs bf16 版 vs CPU 参考，并检查 dw 缓冲越界（守卫哨兵）。
// 内核绑定固定为 [x2, xo, xq, dy0..dy4, dw0..dw4, db0..db4]，其中组 2/3/4 都读 xq（各自 K 作步长）。
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
#include "linear_bwd_dwdb5.spv.hpp"
#include "linear_bwd_dwdb5_bf16.spv.hpp"

using namespace vk张量系统;

struct 推送 {
    std::uint32_t N;
    std::uint32_t M0, K0;
    std::uint32_t M1, K1;
    std::uint32_t M2, K2;
    std::uint32_t M3, K3;
    std::uint32_t M4, K4;
    std::uint32_t pad;
};
static_assert(sizeof(推送) == 48, "linear_bwd_dwdb5 push");

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
        注册.应用名 = "linear_bwd_dwdb5_bf16_test";
        阶段配置 图形;
        阶段配置 计算{阶段配置::自动队列族, 0, 1};
        阶段配置 传输;
        调度器.启动(注册, {图形, 计算, 传输}, spv);
        std::printf("设备: %s | 已加载算子: %zu\n", 调度器.获取设备().名称(), 调度器.算子数量());

        struct 用例 { const char* 名; std::uint32_t N; std::uint32_t M[5], K[5]; };
        const 用例 用例表[] = {
            {"small", 8u, {3, 4, 6, 2, 5}, {5, 2, 3, 7, 4}},
            {"pow2", 32u, {512, 256, 1024, 64, 2048}, {256, 512, 1024, 1024, 1024}},
        };

        int 总错 = 0;
        for (const 用例& c : 用例表) {
            const std::uint32_t N = c.N;
            std::printf("\n== 用例 %s: N=%u M=[%u %u %u %u %u] K=[%u %u %u %u %u]\n", c.名, N,
                        c.M[0], c.M[1], c.M[2], c.M[3], c.M[4],
                        c.K[0], c.K[1], c.K[2], c.K[3], c.K[4]);

            std::mt19937 随机(20261125u);
            std::uniform_real_distribution<float> 分布(-1.0f, 1.0f);

            const std::uint32_t Kmax = std::max({c.K[2], c.K[3], c.K[4]});
            std::vector<std::vector<float>> dy(5);
            for (int g = 0; g < 5; ++g) {
                dy[g].resize(std::size_t(N) * c.M[g]);
                for (auto& v : dy[g]) v = 分布(随机);
            }
            std::vector<float> x2(std::size_t(N) * c.K[0]), xo(std::size_t(N) * c.K[1]),
                xq(std::size_t(N) * Kmax);
            for (auto& v : x2) v = 分布(随机);
            for (auto& v : xo) v = 分布(随机);
            for (auto& v : xq) v = 分布(随机);
            const std::vector<float>* xsrc[5] = {&x2, &xo, &xq, &xq, &xq};

            const std::uint32_t 哨兵数 = 256;
            const std::uint16_t 哨兵位 = 0x7E00;  // NaN 位模式：被写即越界

            std::vector<vk张量> ddy, dwF, dwB, dbF, dbB;
            for (int g = 0; g < 5; ++g) {
                vk张量 t(张量布局::连续({N, c.M[g]}), 张量类型::浮点32, "dy" + std::to_string(g));
                t.上传(dy[g].data(), dy[g].size() * 4);
                ddy.push_back(std::move(t));
            }
            vk张量 dx2(张量布局::连续({std::uint32_t(x2.size())}), 张量类型::浮点32, "x2");
            dx2.上传(x2.data(), x2.size() * 4);
            vk张量 dxo(张量布局::连续({std::uint32_t(xo.size())}), 张量类型::浮点32, "xo");
            dxo.上传(xo.data(), xo.size() * 4);
            vk张量 dxq(张量布局::连续({std::uint32_t(xq.size())}), 张量类型::浮点32, "xq");
            dxq.上传(xq.data(), xq.size() * 4);

            for (int g = 0; g < 5; ++g) {
                const std::size_t n = std::size_t(c.M[g]) * c.K[g];
                std::vector<float> 零(n, 0.0f);
                vk张量 f(张量布局::连续({std::uint32_t(n)}), 张量类型::浮点32, "dwF" + std::to_string(g));
                f.上传(零.data(), n * 4);
                dwF.push_back(std::move(f));
                std::vector<std::uint16_t> 零b(n + 哨兵数, 哨兵位);
                vk张量 b(张量布局::连续({std::uint32_t(n + 哨兵数)}), 张量类型::浮点bf16, "dwB" + std::to_string(g));
                b.上传(零b.data(), 零b.size() * 2);
                dwB.push_back(std::move(b));
                std::vector<float> 零r(c.M[g], 0.0f);
                vk张量 fb(张量布局::连续({c.M[g]}), 张量类型::浮点32, "dbF" + std::to_string(g));
                fb.上传(零r.data(), 零r.size() * 4);
                dbF.push_back(std::move(fb));
                vk张量 bb(张量布局::连续({c.M[g]}), 张量类型::浮点32, "dbB" + std::to_string(g));
                bb.上传(零r.data(), 零r.size() * 4);
                dbB.push_back(std::move(bb));
            }

            const 推送 pc{N, c.M[0], c.K[0], c.M[1], c.K[1], c.M[2], c.K[2],
                          c.M[3], c.K[3], c.M[4], c.K[4], 0};
            std::uint32_t 总元素 = 0;
            for (int g = 0; g < 5; ++g) 总元素 += c.M[g] * c.K[g];
            const 派发规格 规格{(总元素 + 63u) / 64u, 1u, 1u};

            std::vector<张量句柄> 绑;
            绑.push_back(dx2.取句柄());
            绑.push_back(dxo.取句柄());
            绑.push_back(dxq.取句柄());
            for (int g = 0; g < 5; ++g) 绑.push_back(ddy[g].取句柄());
            for (int g = 0; g < 5; ++g) 绑.push_back(dwF[g].取句柄());
            for (int g = 0; g < 5; ++g) 绑.push_back(dbF[g].取句柄());
            调度器.录制自定义派发算子("linear_bwd_dwdb5", 绑, &pc, sizeof(pc), 规格);

            绑.clear();
            绑.push_back(dx2.取句柄());
            绑.push_back(dxo.取句柄());
            绑.push_back(dxq.取句柄());
            for (int g = 0; g < 5; ++g) 绑.push_back(ddy[g].取句柄());
            for (int g = 0; g < 5; ++g) 绑.push_back(dwB[g].取句柄());
            for (int g = 0; g < 5; ++g) 绑.push_back(dbB[g].取句柄());
            调度器.录制自定义派发算子("linear_bwd_dwdb5_bf16", 绑, &pc, sizeof(pc), 规格);

            int 打包错 = 0, fp32错 = 0, db错 = 0, 越界 = 0;
            double fp32最大差 = 0.0;
            for (int g = 0; g < 5; ++g) {
                const std::uint32_t M = c.M[g], K = c.K[g];
                const std::size_t n = std::size_t(M) * K;
                const std::vector<float>& xs = *xsrc[g];
                std::vector<float> 参考(n, 0.0f), 参考b(M, 0.0f);
                for (std::uint32_t m = 0; m < M; ++m) {
                    for (std::uint32_t k = 0; k < K; ++k) {
                        float s = 0.0f;
                        for (std::uint32_t t = 0; t < N; ++t)
                            s += dy[g][std::size_t(t) * M + m] * xs[std::size_t(t) * K + k];
                        参考[std::size_t(m) * K + k] = s;
                    }
                    float bb = 0.0f;
                    for (std::uint32_t t = 0; t < N; ++t) bb += dy[g][std::size_t(t) * M + m];
                    参考b[m] = bb;
                }

                std::vector<float> 得F(n), 得Db(M);
                dwF[g].下载(得F.data(), n * 4);
                dbF[g].下载(得Db.data(), M * 4);
                for (std::size_t i = 0; i < n; ++i) {
                    const double d = std::fabs(double(得F[i]) - double(参考[i]));
                    if (d > fp32最大差) fp32最大差 = d;
                    if (d > 1e-5 * (1.0 + std::fabs(double(参考[i])))) ++fp32错;
                }
                for (std::uint32_t m = 0; m < M; ++m)
                    if (std::fabs(double(得Db[m]) - double(参考b[m])) > 1e-4 * (1.0 + std::fabs(double(参考b[m])))) ++db错;

                std::vector<std::uint16_t> 得B(n + 哨兵数);
                dwB[g].下载(得B.data(), 得B.size() * 2);
                for (std::size_t i = 0; i < n; ++i)
                    if (得B[i] != bf16_pack(参考[i])) ++打包错;
                // 判据：解包后相对误差 ≤ 1%（bf16 1 ulp ≈ 0.4%，FMA 顺序差异会造成 1 ulp 抖动）
                int 超差 = 0;
                for (std::size_t i = 0; i < n; ++i) {
                    const double d = std::fabs(double(bf16_unpack(得B[i])) - double(参考[i]));
                    if (d > 1e-2 * (1.0 + std::fabs(double(参考[i])))) ++超差;
                }
                打包错 = 超差;
                for (std::uint32_t i = 0; i < 哨兵数; ++i)
                    if (得B[n + i] != 哨兵位) ++越界;
            }
            std::printf("  fp32 版 vs CPU: 最大差 %.3g | 超差 %d\n", fp32最大差, fp32错);
            std::printf("  bf16 版 vs CPU: 超差(>1%%) %d | 守卫越界 %d\n", 打包错, 越界);
            std::printf("  db: 不一致 %d\n", db错);
            总错 += 打包错 + fp32错 + db错 + 越界;
        }
        std::printf("\n%s\n", 总错 == 0 ? "LINEAR_BF16 PASS" : "LINEAR_BF16 FAIL");
        return 总错 == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "异常: %s\n", e.what());
        return 1;
    }
}
