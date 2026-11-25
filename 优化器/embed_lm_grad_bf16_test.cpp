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
#include "linear_bwd_dwdb.spv.hpp"
#include "linear_bwd_dwdb_bf16.spv.hpp"
#include "embedding_bwd_scatter_partial.spv.hpp"
#include "embedding_bwd_partial_to_bf16.spv.hpp"

using namespace vk张量系统;

struct 规格4 { std::uint32_t N, K, M, pad; };
static_assert(sizeof(规格4) == 16, "grad push");

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

int main(int argc, char** argv) {
    try {
        const std::string spv = argc > 1 ? argv[1] : "spv";
        流水线调度器& 调度器 = 流水线调度器::获取();
        设备注册信息 注册;
        注册.应用名 = "embed_lm_grad_bf16_test";
        阶段配置 图形;
        阶段配置 计算{阶段配置::自动队列族, 0, 1};
        阶段配置 传输;
        调度器.启动(注册, {图形, 计算, 传输}, spv);
        std::printf("设备: %s | 已加载算子: %zu\n", 调度器.获取设备().名称(), 调度器.算子数量());

        int 总错 = 0;
        {
            const std::uint32_t N = 7, K = 5, M = 11, n = M * K, cap = n + 256;
            std::mt19937 随机(20261125u);
            std::uniform_real_distribution<float> 分布(-1.0f, 1.0f);
            std::vector<float> x(N * K), dy(N * M);
            for (auto& v : x) v = 分布(随机);
            for (auto& v : dy) v = 分布(随机);
            vk张量 X(张量布局::连续({N * K}), 张量类型::浮点32, "lm_x");
            vk张量 DY(张量布局::连续({N * M}), 张量类型::浮点32, "lm_dy");
            vk张量 WF(张量布局::连续({cap}), 张量类型::浮点32, "lm_wf");
            vk张量 WB(张量布局::连续({cap}), 张量类型::浮点bf16, "lm_wb");
            vk张量 DBF(张量布局::连续({M}), 张量类型::浮点32, "lm_dbf");
            vk张量 DBB(张量布局::连续({M}), 张量类型::浮点32, "lm_dbb");
            X.上传(x.data(), x.size() * sizeof(float));
            DY.上传(dy.data(), dy.size() * sizeof(float));
            std::vector<float> 零(cap, 0.0f);
            WF.上传(零.data(), 零.size() * sizeof(float));
            std::vector<std::uint16_t> 零b(cap, 0x7E00);
            WB.上传(零b.data(), 零b.size() * sizeof(std::uint16_t));
            const 规格4 pc{N, K, M, 0};
            const 派发规格 规格{(n + 63u) / 64u, 1u, 1u};
            调度器.录制自定义派发算子("linear_bwd_dwdb",
                                      {X.取句柄(), DY.取句柄(), WF.取句柄(), DBF.取句柄()},
                                      &pc, sizeof(pc), 规格);
            调度器.录制自定义派发算子("linear_bwd_dwdb_bf16",
                                      {X.取句柄(), DY.取句柄(), WB.取句柄(), DBB.取句柄()},
                                      &pc, sizeof(pc), 规格);
            std::vector<float> wf(cap), dbf(M), dbb(M);
            std::vector<std::uint16_t> wb(cap);
            WF.下载(wf.data(), wf.size() * sizeof(float));
            WB.下载(wb.data(), wb.size() * sizeof(std::uint16_t));
            DBF.下载(dbf.data(), dbf.size() * sizeof(float));
            DBB.下载(dbb.data(), dbb.size() * sizeof(float));
            int 错 = 0;
            double 最大差 = 0.0;
            for (std::uint32_t m = 0; m < M; ++m) {
                float b = 0.0f;
                for (std::uint32_t t = 0; t < N; ++t) b += dy[t * M + m];
                错 += std::fabs(dbf[m] - b) > 1e-5f * (1.0f + std::fabs(b));
                错 += std::fabs(dbb[m] - b) > 1e-5f * (1.0f + std::fabs(b));
                for (std::uint32_t k = 0; k < K; ++k) {
                    float s = 0.0f;
                    for (std::uint32_t t = 0; t < N; ++t) s += dy[t * M + m] * x[t * K + k];
                    const std::uint32_t i = m * K + k;
                    最大差 = std::max(最大差, double(std::fabs(wf[i] - s)));
                    错 += std::fabs(wf[i] - s) > 1e-5f * (1.0f + std::fabs(s));
                    错 += std::fabs(bf16_unpack(wb[i]) - s) > 1e-2f * (1.0f + std::fabs(s));
                }
            }
            for (std::uint32_t i = n; i < cap; ++i) 错 += wf[i] != 0.0f || wb[i] != 0x7E00;
            std::printf("lm_head: fp32最大差 %.3g | 不一致/越界 %d\n", 最大差, 错);
            总错 += 错;
        }

        {
            const std::uint32_t N = 6, D = 5, V = 11, U = 3;
            const std::vector<int> ids{2, 5, 2, 7, 5, 2};
            const std::vector<int> unique{2, 5, 7};
            const std::vector<int> map{0, 1, 0, 2, 1, 0};
            std::mt19937 随机(20261126u);
            std::uniform_real_distribution<float> 分布(-1.0f, 1.0f);
            std::vector<float> de(N * D), partial(U * D, 0.0f), ref(U * D, 0.0f);
            for (auto& v : de) v = 分布(随机);
            for (std::uint32_t n = 0; n < N; ++n)
                for (std::uint32_t d = 0; d < D; ++d)
                    ref[map[n] * D + d] += de[n * D + d];
            vk张量 DE(张量布局::连续({N * D}), 张量类型::浮点32, "emb_de");
            vk张量 IDS(张量布局::连续({N}), 张量类型::整数32, "emb_ids");
            vk张量 MAP(张量布局::连续({N}), 张量类型::整数32, "emb_map");
            vk张量 PART(张量布局::连续({U * D + 16}), 张量类型::浮点32, "emb_part");
            vk张量 UIDS(张量布局::连续({U}), 张量类型::整数32, "emb_unique");
            vk张量 TABLE(张量布局::连续({V * D + 32}), 张量类型::浮点bf16, "emb_table");
            DE.上传(de.data(), de.size() * sizeof(float));
            IDS.上传(ids.data(), ids.size() * sizeof(int));
            MAP.上传(map.data(), map.size() * sizeof(int));
            UIDS.上传(unique.data(), unique.size() * sizeof(int));
            std::vector<float> pzero(U * D + 16, 0.0f);
            PART.上传(pzero.data(), pzero.size() * sizeof(float));
            std::vector<std::uint16_t> table0(V * D + 32, 0x7E00);
            TABLE.上传(table0.data(), table0.size() * sizeof(std::uint16_t));
            const 规格4 spc{N, D, U, 0};
            调度器.录制自定义派发算子("embedding_bwd_scatter_partial",
                                      {DE.取句柄(), IDS.取句柄(), MAP.取句柄(), PART.取句柄()},
                                      &spc, sizeof(spc), 派发规格{(N * D + 255u) / 256u, 1u, 1u});
            const 规格4 cpc{U, D, V, 0};
            调度器.录制自定义派发算子("embedding_bwd_partial_to_bf16",
                                      {UIDS.取句柄(), PART.取句柄(), TABLE.取句柄()},
                                      &cpc, sizeof(cpc), 派发规格{(U * D + 255u) / 256u, 1u, 1u});
            std::vector<float> gotp(U * D + 16);
            std::vector<std::uint16_t> gott(V * D + 32);
            PART.下载(gotp.data(), gotp.size() * sizeof(float));
            TABLE.下载(gott.data(), gott.size() * sizeof(std::uint16_t));
            int 错 = 0;
            double 最大差 = 0.0;
            for (std::uint32_t i = 0; i < U * D; ++i) {
                最大差 = std::max(最大差, double(std::fabs(gotp[i] - ref[i])));
                错 += std::fabs(gotp[i] - ref[i]) > 1e-5f;
            }
            for (std::uint32_t u = 0; u < U; ++u)
                for (std::uint32_t d = 0; d < D; ++d)
                    错 += std::fabs(bf16_unpack(gott[unique[u] * D + d]) - ref[u * D + d]) >
                          1e-2f * (1.0f + std::fabs(ref[u * D + d]));
            for (std::uint32_t v = 0; v < V; ++v) {
                if (v == 2 || v == 5 || v == 7) continue;
                for (std::uint32_t d = 0; d < D; ++d) 错 += gott[v * D + d] != 0x7E00;
            }
            for (std::uint32_t i = U * D; i < U * D + 16; ++i) 错 += gotp[i] != 0.0f;
            for (std::uint32_t i = V * D; i < V * D + 32; ++i) 错 += gott[i] != 0x7E00;
            std::printf("embed: partial最大差 %.3g | 不一致/越界 %d\n", 最大差, 错);
            总错 += 错;
        }
        std::printf("%s\n", 总错 == 0 ? "EMBED_LM_GRAD_BF16 PASS" : "EMBED_LM_GRAD_BF16 FAIL");
        return 总错 == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "异常: %s\n", e.what());
        return 1;
    }
}
