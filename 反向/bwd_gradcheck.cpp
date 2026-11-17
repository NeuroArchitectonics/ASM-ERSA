// P1 融合反向的梯度对拍数据生成器
// 流程：跑 Vulkan 反向 kernel → 把输入/输出写成二进制 → gradcheck.py 用 PyTorch 复算比对
// 用法：bwd_gradcheck.exe <着色器目录> <输出文件>
#include "流水线调度器.hpp"
#include "vk张量.hpp"
// 元信息静态登记（每个算子一份，由 shader_tool 生成）
#include "linear_bwd_dx.spv.hpp"
#include "linear_bwd_dwdb.spv.hpp"
#include "rmsnorm_bwd_dx.spv.hpp"
#include "rmsnorm_bwd_dw.spv.hpp"
#include "masked_softmax_bwd.spv.hpp"
#include "fused_ce_bwd.spv.hpp"
#include "ersa_attn_bwd.spv.hpp"
#include "asm_step_bwd.spv.hpp"
#include "embedding_bwd.spv.hpp"
#include "moe_bwd_dx.spv.hpp"
#include "moe_bwd_dw_expert.spv.hpp"
#include "moe_bwd_dw_shared.spv.hpp"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vk张量系统;

namespace {

std::uint32_t g_lcg = 123456789u;

float 随机数() {
    g_lcg = g_lcg * 1664525u + 1013904223u;
    return (static_cast<float>((g_lcg >> 8) & 0xFFFFFFu) / 16777216.0f) * 2.0f - 1.0f;
}

struct 输出文件 {
    std::ofstream 流;
    std::uint32_t 个数 = 0;

    explicit 输出文件(const std::string& 路径) : 流(路径, std::ios::binary) {
        if (!流.is_open()) throw std::runtime_error("无法写输出文件: " + 路径);
        std::uint32_t 占位 = 0;
        流.write(reinterpret_cast<const char*>(&占位), sizeof(占位));
    }

    void 写F32(const std::string& 名, const std::vector<std::uint32_t>& 形状,
               const std::vector<float>& 数据) {
        写头(名, 形状, 0);
        流.write(reinterpret_cast<const char*>(数据.data()),
                 static_cast<std::streamsize>(数据.size() * sizeof(float)));
        个数++;
    }

    void 写I32(const std::string& 名, const std::vector<std::uint32_t>& 形状,
               const std::vector<std::int32_t>& 数据) {
        写头(名, 形状, 1);
        流.write(reinterpret_cast<const char*>(数据.data()),
                 static_cast<std::streamsize>(数据.size() * sizeof(std::int32_t)));
        个数++;
    }

    void 收尾() {
        流.seekp(0, std::ios::beg);
        流.write(reinterpret_cast<const char*>(&个数), sizeof(个数));
        流.close();
    }

private:
    void 写头(const std::string& 名, const std::vector<std::uint32_t>& 形状, std::uint32_t 类型) {
        std::uint32_t 名长 = static_cast<std::uint32_t>(名.size());
        std::uint32_t 维数 = static_cast<std::uint32_t>(形状.size());
        流.write(reinterpret_cast<const char*>(&名长), sizeof(名长));
        流.write(名.data(), 名长);
        流.write(reinterpret_cast<const char*>(&类型), sizeof(类型));
        流.write(reinterpret_cast<const char*>(&维数), sizeof(维数));
        流.write(reinterpret_cast<const char*>(形状.data()),
                 static_cast<std::streamsize>(形状.size() * sizeof(std::uint32_t)));
    }
};

struct 推送规格 {
    std::uint32_t a, b, c, d;
};

struct 推送规格F {
    std::uint32_t R, D;
    float eps;
    float pad;
};

struct 推送规格6 {
    std::uint32_t B, H, T, DK, DV, pad;
};

struct 推送规格ASM {
    std::uint32_t B, H, M, R, DK, DV;
    float tau, tauW, rho, lam, betaNull;
};

struct 推送规格7 {
    std::uint32_t N, E, K, D, F, Fs, pad;
};

struct 推送规格DW {
    std::uint32_t N, E, K, D, F, mode;
};

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "用法: %s <spv目录> <输出.bin>\n", argv[0]);
        return 2;
    }
    const std::string 着色器目录 = argv[1];
    const std::string 输出路径 = argv[2];

    流水线调度器& 调度器 = 流水线调度器::获取();
    设备注册信息 注册;
    注册.应用名 = "bwd_gradcheck";
    阶段配置 图形;
    阶段配置 计算{阶段配置::自动队列族, 0, 1};
    阶段配置 传输;
    try {
        调度器.启动(注册, {图形, 计算, 传输}, 着色器目录);
    } catch (const std::exception& 异常) {
        std::fprintf(stderr, "启动失败: %s\n", 异常.what());
        return 1;
    }
    std::printf("设备: %s | 已加载算子: %zu\n", 调度器.获取设备().名称(), 调度器.算子数量());

    for (const char* 名 : {"linear_bwd_dx", "linear_bwd_dwdb", "rmsnorm_bwd_dx",
                           "rmsnorm_bwd_dw", "masked_softmax_bwd", "fused_ce_bwd",
                           "ersa_attn_bwd", "asm_step_bwd", "embedding_bwd",
                           "moe_bwd_dx", "moe_bwd_dw_expert", "moe_bwd_dw_shared"}) {
        if (!调度器.存在算子(名)) {
            std::fprintf(stderr, "缺少算子: %s\n", 名);
            调度器.停止();
            return 1;
        }
    }

    auto 建张量 = [](std::vector<std::uint32_t> 形状, const char* 名) {
        return vk张量(张量布局::连续(std::move(形状)), 张量类型::浮点32, 名);
    };
    auto 提交并等待 = [&](const char* 算子, std::vector<张量句柄> 句柄,
                        const void* 推送, std::uint32_t 推送字节, std::uint32_t 线程数) {
        std::uint64_t 任务 = 调度器.提交自定义算子(
            算子, std::move(句柄), 推送, 推送字节, {(线程数 + 63u) / 64u, 1u, 1u});
        调度器.等待(任务);
        if (!调度器.已完成(任务)) {
            throw std::runtime_error(std::string("算子失败: ") + 算子 + " | " +
                                     调度器.查询失败原因(任务));
        }
    };

    输出文件 输出(输出路径);

    // ── 1. linear + bias 反向 ──
    {
        const std::uint32_t N = 8, K = 16, M = 24;
        vk张量 x = 建张量({N, K}, "x");
        vk张量 w = 建张量({M, K}, "w");
        vk张量 dy = 建张量({N, M}, "dy");
        vk张量 dx = 建张量({N, K}, "dx");
        vk张量 dw = 建张量({M, K}, "dw");
        vk张量 db = 建张量({M}, "db");

        std::vector<float> xh(N * K), wh(M * K), dyh(N * M);
        for (auto& v : xh) v = 随机数();
        for (auto& v : wh) v = 随机数();
        for (auto& v : dyh) v = 随机数();
        x.上传(xh.data(), xh.size() * 4);
        w.上传(wh.data(), wh.size() * 4);
        dy.上传(dyh.data(), dyh.size() * 4);

        推送规格 规格{N, K, M, 0};
        提交并等待("linear_bwd_dx", {dy.取句柄(), w.取句柄(), dx.取句柄()},
                 &规格, sizeof(规格), N * K);
        提交并等待("linear_bwd_dwdb", {x.取句柄(), dy.取句柄(), dw.取句柄(), db.取句柄()},
                 &规格, sizeof(规格), M * K);

        std::vector<float> dxh(N * K), dwh(M * K), dbh(M);
        dx.下载(dxh.data(), dxh.size() * 4);
        dw.下载(dwh.data(), dwh.size() * 4);
        db.下载(dbh.data(), dbh.size() * 4);

        输出.写F32("linear.x", {N, K}, xh);
        输出.写F32("linear.w", {M, K}, wh);
        输出.写F32("linear.dy", {N, M}, dyh);
        输出.写F32("linear.dx", {N, K}, dxh);
        输出.写F32("linear.dw", {M, K}, dwh);
        输出.写F32("linear.db", {M}, dbh);
    }

    // ── 2. rmsnorm 反向 ──
    {
        const std::uint32_t R = 8, D = 32;
        const float eps = 1e-6f;
        vk张量 x = 建张量({R, D}, "x");
        vk张量 dy = 建张量({R, D}, "dy");
        vk张量 w = 建张量({D}, "w");
        vk张量 dx = 建张量({R, D}, "dx");
        vk张量 dw = 建张量({D}, "dw");

        std::vector<float> xh(R * D), dyh(R * D), wh(D);
        for (auto& v : xh) v = 随机数();
        for (auto& v : dyh) v = 随机数();
        for (auto& v : wh) v = 随机数();
        x.上传(xh.data(), xh.size() * 4);
        dy.上传(dyh.data(), dyh.size() * 4);
        w.上传(wh.data(), wh.size() * 4);

        推送规格F 规格{R, D, eps, 0.0f};
        提交并等待("rmsnorm_bwd_dx", {x.取句柄(), dy.取句柄(), w.取句柄(), dx.取句柄()},
                 &规格, sizeof(规格), R);
        提交并等待("rmsnorm_bwd_dw", {x.取句柄(), dy.取句柄(), dw.取句柄()},
                 &规格, sizeof(规格), D);

        std::vector<float> dxh(R * D), dwh(D);
        dx.下载(dxh.data(), dxh.size() * 4);
        dw.下载(dwh.data(), dwh.size() * 4);

        输出.写F32("rmsnorm.x", {R, D}, xh);
        输出.写F32("rmsnorm.dy", {R, D}, dyh);
        输出.写F32("rmsnorm.w", {D}, wh);
        输出.写F32("rmsnorm.dx", {R, D}, dxh);
        输出.写F32("rmsnorm.dw", {D}, dwh);
        输出.写F32("rmsnorm.eps", {1},
                 {eps});
    }

    // ── 3. masked softmax 反向 ──
    {
        const std::uint32_t R = 8, C = 16;
        std::vector<float> ah(R * C), dah(R * C), mask(R * C);
        for (std::uint32_t r = 0; r < R; ++r) {
            float mx = -1e30f;
            std::vector<float> logits(C);
            for (std::uint32_t c = 0; c < C; ++c) {
                logits[c] = 随机数();
                mask[r * C + c] = ((c + r) % 7u == 0u) ? 0.0f : 1.0f;
                if (mask[r * C + c] > 0.5f) mx = std::max(mx, logits[c]);
            }
            float sum = 0.0f;
            for (std::uint32_t c = 0; c < C; ++c) {
                ah[r * C + c] = mask[r * C + c] > 0.5f ? std::exp(logits[c] - mx) : 0.0f;
                sum += ah[r * C + c];
            }
            for (std::uint32_t c = 0; c < C; ++c) ah[r * C + c] /= sum;
        }
        for (auto& v : dah) v = 随机数();

        vk张量 a = 建张量({R, C}, "a");
        vk张量 da = 建张量({R, C}, "da");
        vk张量 ds = 建张量({R, C}, "ds");
        a.上传(ah.data(), ah.size() * 4);
        da.上传(dah.data(), dah.size() * 4);

        推送规格 规格{R, C, 0, 0};
        提交并等待("masked_softmax_bwd", {a.取句柄(), da.取句柄(), ds.取句柄()},
                 &规格, sizeof(规格), R);

        std::vector<float> dsh(R * C);
        ds.下载(dsh.data(), dsh.size() * 4);

        输出.写F32("softmax.a", {R, C}, ah);
        输出.写F32("softmax.da", {R, C}, dah);
        输出.写F32("softmax.ds", {R, C}, dsh);
    }

    // ── 4. fused cross-entropy 反向 ──
    {
        const std::uint32_t R = 8, V = 32;
        vk张量 logits = 建张量({R, V}, "logits");
        vk张量 dlogits = 建张量({R, V}, "dlogits");
        vk张量 loss = 建张量({R}, "loss");
        vk张量 targets(张量布局::连续({R}), 张量类型::整数32, "targets");

        std::vector<float> lh(R * V);
        std::vector<std::int32_t> th(R);
        for (auto& v : lh) v = 随机数() * 2.0f;
        for (std::uint32_t r = 0; r < R; ++r) th[r] = static_cast<std::int32_t>(r % V);

        logits.上传(lh.data(), lh.size() * 4);
        targets.上传(th.data(), th.size() * 4);

        推送规格 规格{R, V, 0, 0};
        提交并等待("fused_ce_bwd",
                 {logits.取句柄(), targets.取句柄(), dlogits.取句柄(), loss.取句柄()},
                 &规格, sizeof(规格), R);

        std::vector<float> dlh(R * V), lossh(R);
        dlogits.下载(dlh.data(), dlh.size() * 4);
        loss.下载(lossh.data(), lossh.size() * 4);

        输出.写F32("ce.logits", {R, V}, lh);
        输出.写I32("ce.targets", {R}, th);
        输出.写F32("ce.dlogits", {R, V}, dlh);
        输出.写F32("ce.loss", {R}, lossh);
    }

    // ── 5. ERSA attention 反向（B=1；多 batch 写 dK 需原子加）──
    {
        const std::uint32_t B = 1, H = 2, T = 16, DK = 8, DV = 8;
        vk张量 q = 建张量({B, H, DK}, "q");
        vk张量 k = 建张量({T, H, DK}, "k");
        vk张量 v = 建张量({T, H, DV}, "v");
        vk张量 掩码 = 建张量({B, H, T}, "mask");
        vk张量 sink = 建张量({B, H}, "sink");
        vk张量 dy = 建张量({B, H, DV}, "dy");
        vk张量 dq = 建张量({B, H, DK}, "dq");
        vk张量 dk = 建张量({T, H, DK}, "dk");
        vk张量 dv = 建张量({T, H, DV}, "dv");
        vk张量 dsink = 建张量({B, H}, "dsink");

        std::vector<float> qh(B * H * DK), kh(T * H * DK), vh(T * H * DV);
        std::vector<float> mh(B * H * T), sh(B * H), dyh(B * H * DV);
        for (auto& x : qh) x = 随机数();
        for (auto& x : kh) x = 随机数();
        for (auto& x : vh) x = 随机数();
        for (auto& x : dyh) x = 随机数();
        for (std::uint32_t b = 0; b < B; ++b)
            for (std::uint32_t h = 0; h < H; ++h) {
                sh[b * H + h] = -0.5f + 0.1f * static_cast<float>(h);
                for (std::uint32_t t = 0; t < T; ++t)
                    mh[(b * H + h) * T + t] = ((t * 3 + h * 5 + 1) % 5 != 0) ? 1.0f : 0.0f;
            }

        q.上传(qh.data(), qh.size() * 4);
        k.上传(kh.data(), kh.size() * 4);
        v.上传(vh.data(), vh.size() * 4);
        掩码.上传(mh.data(), mh.size() * 4);
        sink.上传(sh.data(), sh.size() * 4);
        dy.上传(dyh.data(), dyh.size() * 4);

        推送规格6 规格{B, H, T, DK, DV, 0};
        提交并等待("ersa_attn_bwd",
                 {q.取句柄(), k.取句柄(), v.取句柄(), 掩码.取句柄(), sink.取句柄(),
                  dy.取句柄(), dq.取句柄(), dk.取句柄(), dv.取句柄(), dsink.取句柄()},
                 &规格, sizeof(规格), B * H);

        std::vector<float> dqh(B * H * DK), dkh(T * H * DK), dvh(T * H * DV), sh2(B * H);
        dq.下载(dqh.data(), dqh.size() * 4);
        dk.下载(dkh.data(), dkh.size() * 4);
        dv.下载(dvh.data(), dvh.size() * 4);
        dsink.下载(sh2.data(), sh2.size() * 4);

        输出.写F32("ersa.q", {B, H, DK}, qh);
        输出.写F32("ersa.k", {T, H, DK}, kh);
        输出.写F32("ersa.v", {T, H, DV}, vh);
        输出.写F32("ersa.mask", {B, H, T}, mh);
        输出.写F32("ersa.sink", {B, H}, sh);
        输出.写F32("ersa.dy", {B, H, DV}, dyh);
        输出.写F32("ersa.dq", {B, H, DK}, dqh);
        输出.写F32("ersa.dk", {T, H, DK}, dkh);
        输出.写F32("ersa.dv", {T, H, DV}, dvh);
        输出.写F32("ersa.dsink", {B, H}, sh2);
    }

    // ── 6. ASM 单步反向（B=1,H=1；多 batch/多头需原子加）──
    {
        const std::uint32_t B = 1, H = 1, M = 8, R = 3, DK = 4, DV = 4;
        vk张量 q = 建张量({B, H, DK}, "q");
        vk张量 k = 建张量({B, H, DK}, "k");
        vk张量 v = 建张量({B, H, DV}, "v");
        vk张量 eta = 建张量({B}, "eta");
        vk张量 K = 建张量({B, H, M, DK}, "K");
        vk张量 V = 建张量({B, H, M, DV}, "V");
        vk张量 beta = 建张量({B, H, M}, "beta");
        vk张量 gamma = 建张量({M}, "gamma");
        vk张量 dO = 建张量({B, H, DV}, "dO");
        vk张量 dKn = 建张量({B, H, M, DK}, "dKn");
        vk张量 dVn = 建张量({B, H, M, DV}, "dVn");
        vk张量 dBn = 建张量({B, H, M}, "dBn");
        vk张量 idx = vk张量(张量布局::连续({B, H, R}), 张量类型::整数32, "idx");
        vk张量 dq = 建张量({B, H, DK}, "dq");
        vk张量 dk = 建张量({B, H, DK}, "dk");
        vk张量 dv = 建张量({B, H, DV}, "dv");
        vk张量 deta = 建张量({B}, "deta");
        vk张量 dKo = 建张量({B, H, M, DK}, "dKo");
        vk张量 dVo = 建张量({B, H, M, DV}, "dVo");
        vk张量 dBo = 建张量({B, H, M}, "dBo");
        vk张量 dgamma = 建张量({M}, "dgamma");
        vk张量 dbn = 建张量({B, H}, "dbn");

        std::vector<float> qh(B * H * DK), kh(B * H * DK), vh(B * H * DV);
        std::vector<float> Kh(B * H * M * DK), Vh(B * H * M * DV);
        std::vector<float> bh_(B * H * M), gh(M), etah(B, 0.1f);
        std::vector<float> dOh(B * H * DV), dKnh(B * H * M * DK), dVnh(B * H * M * DV);
        std::vector<float> dBnh(B * H * M);
        std::vector<std::int32_t> idxh = {1, 3, 6};

        for (auto& x : qh) x = 随机数();
        for (auto& x : kh) x = 随机数();
        for (auto& x : vh) x = 随机数();
        for (auto& x : dOh) x = 随机数();
        for (auto& x : dKnh) x = 随机数();
        for (auto& x : dVnh) x = 0.0f;   // 最小复现：dVn=0，dVold 应只剩读路径 a·dO
        for (auto& x : dBnh) x = 随机数() * 0.1f;
        for (auto& x : bh_) x = 随机数() * 0.1f;
        for (uint32_t m = 0; m < M; ++m) gh[m] = 0.9f + 0.01f * float(m);
        for (uint32_t m = 0; m < M; ++m) {
            float n = 0.0f;
            for (uint32_t i = 0; i < DK; ++i) { Kh[m * DK + i] = 随机数(); n += Kh[m*DK+i]*Kh[m*DK+i]; }
            n = std::sqrt(n);
            for (uint32_t i = 0; i < DK; ++i) Kh[m * DK + i] /= n;
        }
        for (auto& x : Vh) x = 随机数();

        q.上传(qh.data(), qh.size() * 4);
        k.上传(kh.data(), kh.size() * 4);
        v.上传(vh.data(), vh.size() * 4);
        eta.上传(etah.data(), etah.size() * 4);
        K.上传(Kh.data(), Kh.size() * 4);
        V.上传(Vh.data(), Vh.size() * 4);
        beta.上传(bh_.data(), bh_.size() * 4);
        gamma.上传(gh.data(), gh.size() * 4);
        dO.上传(dOh.data(), dOh.size() * 4);
        dKn.上传(dKnh.data(), dKnh.size() * 4);
        dVn.上传(dVnh.data(), dVnh.size() * 4);
        dBn.上传(dBnh.data(), dBnh.size() * 4);
        idx.上传(idxh.data(), idxh.size() * 4);

        推送规格ASM 规格{B, H, M, R, DK, DV, 1.0f, 1.0f, 0.9f, 0.01f, -2.0f};
        提交并等待("asm_step_bwd",
                 {q.取句柄(), k.取句柄(), v.取句柄(), eta.取句柄(), K.取句柄(),
                  V.取句柄(), beta.取句柄(), gamma.取句柄(), dO.取句柄(),
                  dKn.取句柄(), dVn.取句柄(), dBn.取句柄(), idx.取句柄(),
                  dq.取句柄(), dk.取句柄(), dv.取句柄(), deta.取句柄(),
                  dKo.取句柄(), dVo.取句柄(), dBo.取句柄(), dgamma.取句柄(),
                  dbn.取句柄()},
                 &规格, sizeof(规格), B * H);

        std::vector<float> dqh(B * H * DK), dkh(B * H * DK), dvh(B * H * DV), etah2(B);
        std::vector<float> dKoh(B * H * M * DK), dVoh(B * H * M * DV), dBoh(B * H * M), gh2(M);
        std::vector<float> dbnh(B * H);
        dq.下载(dqh.data(), dqh.size() * 4);
        dk.下载(dkh.data(), dkh.size() * 4);
        dv.下载(dvh.data(), dvh.size() * 4);
        deta.下载(etah2.data(), etah2.size() * 4);
        dKo.下载(dKoh.data(), dKoh.size() * 4);
        dVo.下载(dVoh.data(), dVoh.size() * 4);
        dBo.下载(dBoh.data(), dBoh.size() * 4);
        dgamma.下载(gh2.data(), gh2.size() * 4);
        dbn.下载(dbnh.data(), dbnh.size() * 4);

        输出.写F32("asm.q", {B, H, DK}, qh);
        输出.写F32("asm.k", {B, H, DK}, kh);
        输出.写F32("asm.v", {B, H, DV}, vh);
        输出.写F32("asm.eta", {B}, etah);
        输出.写F32("asm.K", {B, H, M, DK}, Kh);
        输出.写F32("asm.V", {B, H, M, DV}, Vh);
        输出.写F32("asm.beta", {B, H, M}, bh_);
        输出.写F32("asm.gamma", {M}, gh);
        输出.写F32("asm.dO", {B, H, DV}, dOh);
        输出.写F32("asm.dKn", {B, H, M, DK}, dKnh);
        输出.写F32("asm.dVn", {B, H, M, DV}, dVnh);
        输出.写F32("asm.dBn", {B, H, M}, dBnh);
        输出.写I32("asm.idx", {B, H, R}, idxh);
        输出.写F32("asm.dq", {B, H, DK}, dqh);
        输出.写F32("asm.dk", {B, H, DK}, dkh);
        输出.写F32("asm.dv", {B, H, DV}, dvh);
        输出.写F32("asm.deta", {B}, etah2);
        输出.写F32("asm.dKo", {B, H, M, DK}, dKoh);
        输出.写F32("asm.dVo", {B, H, M, DV}, dVoh);
        输出.写F32("asm.dBo", {B, H, M}, dBoh);
        输出.写F32("asm.dgamma", {M}, gh2);
        输出.写F32("asm.dbn", {B, H}, dbnh);
    }

    // ── 7. embedding 反向（scatter-add）──
    {
        const std::uint32_t N = 16, V = 8, D = 4;
        vk张量 dEmbed = 建张量({N, D}, "dEmbed");
        vk张量 ids = vk张量(张量布局::连续({N}), 张量类型::整数32, "ids");
        vk张量 dTable = 建张量({V, D}, "dTable");

        std::vector<float> deh(N * D);
        std::vector<std::int32_t> idh(N);
        for (auto& x : deh) x = 随机数();
        for (uint32_t n = 0; n < N; ++n)
            idh[n] = static_cast<std::int32_t>((n * 5 + 3) % V);

        dEmbed.上传(deh.data(), deh.size() * 4);
        ids.上传(idh.data(), idh.size() * 4);

        推送规格 规格{N, V, D, 0};
        提交并等待("embedding_bwd",
                 {dEmbed.取句柄(), ids.取句柄(), dTable.取句柄()},
                 &规格, sizeof(规格), V);

        std::vector<float> dth(V * D);
        dTable.下载(dth.data(), dth.size() * 4);

        输出.写F32("emb.dEmbed", {N, D}, deh);
        输出.写I32("emb.ids", {N}, idh);
        输出.写F32("emb.dTable", {V, D}, dth);
    }

    // ── 8. MoE token 梯度（dScore + dX）──
    {
        const std::uint32_t N = 4, E = 3, K = 2, D = 4, F = 6, Fs = 6;
        vk张量 x = 建张量({N, D}, "x");
        vk张量 dy = 建张量({N, D}, "dy");
        vk张量 gateW = 建张量({E, D}, "gateW");
        vk张量 W1 = 建张量({E, D, F}, "W1");
        vk张量 W2 = 建张量({E, F, D}, "W2");
        vk张量 idx = vk张量(张量布局::连续({N, K}), 张量类型::整数32, "idx");
        vk张量 kw = 建张量({N, K}, "kw");
        vk张量 W1s = 建张量({D, Fs}, "W1s");
        vk张量 Wup = 建张量({D, Fs}, "Wup");
        vk张量 W2s = 建张量({Fs, D}, "W2s");
        vk张量 dx = 建张量({N, D}, "dx");
        vk张量 dscore = 建张量({N, E}, "dscore");

        std::vector<float> xh(N * D), dyh(N * D), gh(E * D), w1h(E * D * F), w2h(E * F * D);
        std::vector<float> kwh(N * K), w1sh(D * Fs), wuph(D * Fs), w2sh(Fs * D);
        std::vector<std::int32_t> idxh(N * K);
        for (auto& v : xh) v = 随机数();
        for (auto& v : dyh) v = 随机数();
        for (auto& v : gh) v = 随机数();
        for (auto& v : w1h) v = 随机数();
        for (auto& v : w2h) v = 随机数();
        for (auto& v : kwh) v = 0.5f + 0.1f * 随机数();
        for (auto& v : w1sh) v = 随机数();
        for (auto& v : wuph) v = 随机数();
        for (auto& v : w2sh) v = 随机数();
        for (std::uint32_t n = 0; n < N; ++n) {
            idxh[n * K + 0] = static_cast<std::int32_t>(n % E);
            idxh[n * K + 1] = static_cast<std::int32_t>((n + 1) % E);
        }

        x.上传(xh.data(), xh.size() * 4);
        dy.上传(dyh.data(), dyh.size() * 4);
        gateW.上传(gh.data(), gh.size() * 4);
        W1.上传(w1h.data(), w1h.size() * 4);
        W2.上传(w2h.data(), w2h.size() * 4);
        idx.上传(idxh.data(), idxh.size() * 4);
        kw.上传(kwh.data(), kwh.size() * 4);
        W1s.上传(w1sh.data(), w1sh.size() * 4);
        Wup.上传(wuph.data(), wuph.size() * 4);
        W2s.上传(w2sh.data(), w2sh.size() * 4);

        推送规格7 规格{N, E, K, D, F, Fs, 0};
        提交并等待("moe_bwd_dx",
                 {x.取句柄(), dy.取句柄(), gateW.取句柄(), W1.取句柄(), W2.取句柄(),
                  idx.取句柄(), kw.取句柄(), W1s.取句柄(), Wup.取句柄(), W2s.取句柄(),
                  dx.取句柄(), dscore.取句柄()},
                 &规格, sizeof(规格), N);

        std::vector<float> dxh(N * D), dsh(N * E);
        dx.下载(dxh.data(), dxh.size() * 4);
        dscore.下载(dsh.data(), dsh.size() * 4);

        // ── 权重梯度（在 W1/W2 置零之前）──
        vk张量 dW1 = 建张量({E, D, F}, "dW1");
        vk张量 dW2 = 建张量({E, F, D}, "dW2");
        推送规格DW dw0{N, E, K, D, F, 0};
        提交并等待("moe_bwd_dw_expert",
                 {x.取句柄(), dy.取句柄(), W1.取句柄(), W2.取句柄(), idx.取句柄(),
                  kw.取句柄(), dW1.取句柄()},
                 &dw0, sizeof(dw0), E * D * F);
        推送规格DW dw1{N, E, K, D, F, 1};
        提交并等待("moe_bwd_dw_expert",
                 {x.取句柄(), dy.取句柄(), W1.取句柄(), W2.取句柄(), idx.取句柄(),
                  kw.取句柄(), dW2.取句柄()},
                 &dw1, sizeof(dw1), E * F * D);
        vk张量 dW1s = 建张量({D, Fs}, "dW1s");
        vk张量 dWup = 建张量({D, Fs}, "dWup");
        vk张量 dW2s = 建张量({Fs, D}, "dW2s");
        推送规格DW sw0{N, D, Fs, 0, 0, 0};
        提交并等待("moe_bwd_dw_shared",
                 {x.取句柄(), dy.取句柄(), W1s.取句柄(), Wup.取句柄(), W2s.取句柄(),
                  dW1s.取句柄()}, &sw0, sizeof(sw0), D * Fs);
        推送规格DW sw1{N, D, Fs, 1, 0, 0};
        提交并等待("moe_bwd_dw_shared",
                 {x.取句柄(), dy.取句柄(), W1s.取句柄(), Wup.取句柄(), W2s.取句柄(),
                  dWup.取句柄()}, &sw1, sizeof(sw1), D * Fs);
        推送规格DW sw2{N, D, Fs, 2, 0, 0};
        提交并等待("moe_bwd_dw_shared",
                 {x.取句柄(), dy.取句柄(), W1s.取句柄(), Wup.取句柄(), W2s.取句柄(),
                  dW2s.取句柄()}, &sw2, sizeof(sw2), Fs * D);

        // 隔离：W1/W2 置零 → 只剩 shared + gate（gate 的 dtopk 此时为 0）
        std::vector<float> 零(w1h.size(), 0.0f);
        W1.上传(零.data(), 零.size() * 4);
        std::vector<float> 零2(w2h.size(), 0.0f);
        W2.上传(零2.data(), 零2.size() * 4);
        vk张量 dx2 = 建张量({N, D}, "dx2");
        vk张量 dscore2 = 建张量({N, E}, "dscore2");
        提交并等待("moe_bwd_dx",
                 {x.取句柄(), dy.取句柄(), gateW.取句柄(), W1.取句柄(), W2.取句柄(),
                  idx.取句柄(), kw.取句柄(), W1s.取句柄(), Wup.取句柄(), W2s.取句柄(),
                  dx2.取句柄(), dscore2.取句柄()},
                 &规格, sizeof(规格), N);
        std::vector<float> dx2h(N * D);
        dx2.下载(dx2h.data(), dx2h.size() * 4);

        输出.写F32("moe.x", {N, D}, xh);
        输出.写F32("moe.dy", {N, D}, dyh);
        输出.写F32("moe.gateW", {E, D}, gh);
        输出.写F32("moe.W1", {E, D, F}, w1h);
        输出.写F32("moe.W2", {E, F, D}, w2h);
        输出.写I32("moe.idx", {N, K}, idxh);
        输出.写F32("moe.kw", {N, K}, kwh);
        输出.写F32("moe.W1s", {D, Fs}, w1sh);
        输出.写F32("moe.Wup", {D, Fs}, wuph);
        输出.写F32("moe.W2s", {Fs, D}, w2sh);
        输出.写F32("moe.dx", {N, D}, dxh);
        输出.写F32("moe.dscore", {N, E}, dsh);
        输出.写F32("moe.dx_shared_only", {N, D}, dx2h);

        std::vector<float> dw1h(E * D * F), dw2h(E * F * D), dw1sh(D * Fs), dwuph(D * Fs), dw2sh(Fs * D);
        dW1.下载(dw1h.data(), dw1h.size() * 4);
        dW2.下载(dw2h.data(), dw2h.size() * 4);
        dW1s.下载(dw1sh.data(), dw1sh.size() * 4);
        dWup.下载(dwuph.data(), dwuph.size() * 4);
        dW2s.下载(dw2sh.data(), dw2sh.size() * 4);
        输出.写F32("moe.dW1", {E, D, F}, dw1h);
        输出.写F32("moe.dW2", {E, F, D}, dw2h);
        输出.写F32("moe.dW1s", {D, Fs}, dw1sh);
        输出.写F32("moe.dWup", {D, Fs}, dwuph);
        输出.写F32("moe.dW2s", {Fs, D}, dw2sh);
    }

    输出.收尾();
    调度器.停止();
    std::printf("对拍数据已写入: %s\n", 输出路径.c_str());
    return 0;
}
