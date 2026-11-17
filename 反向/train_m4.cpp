// P4-M4：融合 dispatch + 分块扫描（训练环基准）
// 目标：把 M3c 那种「每算子一次提交、一次围栏等待」的训练环，改成
//       前向 / 逐 query 反传 / 参数更新 各录成一条算子串，只提交一次、只等一个围栏。
// 三件事一次做完：
//   1) 正确性：融合路径与串行路径逐步 loss、最终参数必须逐位一致，再把融合结果导出给 m3c_check.py；
//   2) 分块扫描：逐 query 反传按「每串 N 个 query」分块录制，块大小可调；
//   3) 基准：测 dispatch 开销（单次提交+等待）、串行/融合每步耗时、tokens/s。
// 用法：
//   train_m4.exe <spv目录> check <导出.bin>             M3c 尺寸下串行 vs 融合对拍并导出
//   train_m4.exe <spv目录> bench [T] [步数] [每串query] 基准（默认 T=64，20 步，块=T）
// 说明：只动调度方式，不动任何 kernel 与数学；两条路径跑的是同一批已对拍的算子。
#include "流水线调度器.hpp"
#include "vk张量.hpp"
#include "ersa_block_sum.spv.hpp"
#include "ersa_attn_fused.spv.hpp"
#include "ersa_attn_bwd.spv.hpp"
#include "ersa_attn_bwd_batch.spv.hpp"
#include "linear_fwd.spv.hpp"
#include "linear_bwd_dwdb.spv.hpp"
#include "adamw_update.spv.hpp"
#include "apollo_reduce.spv.hpp"
#include "apollo_update.spv.hpp"
#include "linear_fwd_int8_f32.spv.hpp"
#include "ApolloSolver.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vk张量系统;

namespace {

std::uint32_t g_lcg = 31337u;
float 随机数(float 幅 = 1.0f) {
    g_lcg = g_lcg * 1664525u + 1013904223u;
    return 幅 * ((static_cast<float>((g_lcg >> 8) & 0xFFFFFFu) / 16777216.0f) * 2.0f - 1.0f);
}

struct 推送块和 {
    std::uint32_t B, H, N, DK, NB, BLOCK, pad0, pad1;
};
struct 推送前向 {
    std::uint32_t B, H, T, N, DK, DV, NB, BLOCK, TOPB, WINDOW, pad0, pad1;
    float tau, sink;
};
struct 推送反向 {
    std::uint32_t B, H, T, DK, DV, pad;
};
// 批式反传：同一位置换成 tau（q 用原始值，tau 折进内核 scale）
struct 推送反向批 {
    std::uint32_t B, H, T, DK, DV;
    float tau;
};
struct 推送线性 {
    std::uint32_t N, K, M, pad;
};
struct 推送AdamW {
    float lr, beta1, beta2, eps, m_corr, v_corr;
    std::uint32_t size;
    float weight_decay;
    std::uint32_t zero_grad;
};
static_assert(sizeof(推送块和) == 32, "块和 push");
static_assert(sizeof(推送前向) == 56, "前向 push");
static_assert(sizeof(推送反向) == 24, "反向 push");
static_assert(sizeof(推送反向批) == 24, "批式反向 push");
static_assert(sizeof(推送线性) == 16, "线性 push");
struct 推送Int8 {
    std::uint32_t N, K, M, group, has_bias;
};
static_assert(sizeof(推送Int8) == 20, "int8 线性 push");
static_assert(sizeof(推送AdamW) == 36, "AdamW push");

// ---------------------------------------------------------------------------
// 配置：M3c 尺寸用于对拍；基准尺寸可换 T / DK / DV
// ---------------------------------------------------------------------------
struct 配置 {
    std::uint32_t T = 6, DX = 3, DK = 4, DV = 4;
    std::uint32_t BLOCK = 2, NB = 3, TOPB = 2, WINDOW = 2;
    std::uint32_t STEPS = 20;
    std::uint32_t B = 1, H = 1;
    float tau = 0.8f;
};

配置 M3c配置() { return 配置{}; }

配置 基准配置(std::uint32_t T, std::uint32_t 步数) {
    配置 c;
    c.T = T;
    c.DX = 32;
    c.DK = 32;
    c.DV = 32;
    c.BLOCK = 8;
    c.NB = T / c.BLOCK;
    c.TOPB = 2;
    c.WINDOW = 8;
    c.STEPS = 步数;
    c.tau = std::sqrt(static_cast<float>(c.DK));   // 让 logits 落在 O(1)，避免基准里 softmax 过平
    return c;
}

vk张量 建(std::vector<std::uint32_t> 形状, const std::string& 名) {
    return vk张量(张量布局::连续(std::move(形状)), 张量类型::浮点32, 名);
}

struct 计时结果 {
    double 中位 = 0.0;
    double 最小 = 0.0;
    double 最大 = 0.0;
    std::size_t 样本数 = 0;
};

// 融合步的分段计时（挂在 训练器 上，bench 时置非空）
struct 分段耗时 {
    double 前向 = 0.0;   // 前向算子串 提交+等待
    double 主机 = 0.0;   // 下载 / 路由 / 损失 / 上传
    double 反传准备 = 0.0; // 逐 query 反传输入上传 + 构造步骤
    double 反传提交 = 0.0; // 反传算子串 提交+等待
    double 反传收集 = 0.0; // 反传输出下载 + 累加
    double 更新 = 0.0;   // 权重梯度 + AdamW 算子串
};

计时结果 汇总(std::vector<double> 样本) {
    计时结果 结果;
    if (样本.empty()) return 结果;
    结果.样本数 = 样本.size();
    std::sort(样本.begin(), 样本.end());
    结果.最小 = 样本.front();
    结果.最大 = 样本.back();
    结果.中位 = 样本[样本.size() / 2];
    return 结果;
}

// ---------------------------------------------------------------------------
// 训练器：同一张图，两种调度方式
// ---------------------------------------------------------------------------
class 训练器 {
public:
    训练器(流水线调度器& 调度器, const 配置& 参数)
        : 调度器_(调度器), 参数_(参数) {
        建图();
        建每query();
        重置();
    }

    void 重置();
    void 串行步(std::uint32_t 步序号);           // 基线：每算子提交 + 等待
    void 融合步(std::uint32_t 步序号, std::uint32_t 每串query数, bool 批式 = false);
    void 启用Apollo();
    void 启用Int8前向();
    void 刷新Int8权重();
    float 前向损失();
    void 导出(const char* 路径) const;
    void 取参数(std::vector<float>& Wq, std::vector<float>& Wk,
                std::vector<float>& Wv, std::vector<float>& sink) const {
        Wq.assign(参数_.DK * 参数_.DX, 0.f);
        Wk.assign(参数_.DK * 参数_.DX, 0.f);
        Wv.assign(参数_.DV * 参数_.DX, 0.f);
        sink.assign(1, 0.f);
        Wq_.下载(Wq.data(), Wq.size() * 4);
        Wk_.下载(Wk.data(), Wk.size() * 4);
        Wv_.下载(Wv.data(), Wv.size() * 4);
        Sink_.下载(sink.data(), 4);
    }
    const std::vector<float>& 损耗() const { return 损耗_; }
    const 配置& 参数() const { return 参数_; }
    分段耗时* 计时 = nullptr;   // 非空时 融合步 累加各段耗时

    static constexpr float 初始sink = -1.3f;
    static constexpr float lr = 0.02f;
    static constexpr float wd = 0.01f;
    static constexpr float beta1 = 0.9f;
    static constexpr float beta2 = 0.999f;
    static constexpr float eps = 1e-8f;

private:
    void 建图();
    void 建每query();
    void 主机路由();
    void 前向后主机(std::uint32_t 步序号);
    void 反传输入上传(std::uint32_t t);
    void 反传收集(std::uint32_t t);
    void 损失清零累积();
    void 反传上传梯度();

    std::vector<算子步骤> 前向步骤(float 当前sink) const;
    std::vector<算子步骤> 更新步骤(std::uint32_t 步序号) const;

    算子步骤 步线性(const vk张量& 输入, const vk张量& 权, const vk张量& 偏,
                    const vk张量& 出, std::uint32_t N, std::uint32_t K, std::uint32_t M) const;
    算子步骤 步块和() const;
    算子步骤 步注意力(float 当前sink) const;
    算子步骤 步反传(std::uint32_t t) const;
    算子步骤 步批式反传() const;
    算子步骤 步dwdb(const vk张量& X, const vk张量& dY, const vk张量& dW, const vk张量& db,
                    std::uint32_t N, std::uint32_t K, std::uint32_t M) const;
    算子步骤 步AdamW(const vk张量& G, const vk张量& P, const vk张量& M, const vk张量& V,
                     std::uint32_t 个数, float 衰减, std::uint32_t 步序号) const;

    bool 用Apollo_ = false;
    算子步骤 步线性int8(const vk张量& xq, const vk张量& sx,
                      const vk张量& wq, const vk张量& sw,
                      const vk张量& 偏, const vk张量& 出,
                      std::uint32_t N, std::uint32_t K, std::uint32_t M) const;
    void 提交单个(const 算子步骤& 步);
    void 提交串(const std::vector<算子步骤>& 步骤);
    void 等(std::uint64_t 编号, const char* 说明);
    float 读sink() const;

    流水线调度器& 调度器_;
    配置 参数_;

    // 图张量
    vk张量 X_, Qp_, Kp_, Vp_, BS_, Out_, Dy_;
    vk张量 dQd_, dKd_, dVd_;
    vk张量 Wq_, Wk_, Wv_, Sink_;
    vk张量 零偏DK_, 零偏DV_, dbk_, dbv_, dS_;
    vk张量 dWq_, dWk_, dWv_;
    vk张量 mWq_, vWq_, mWk_, vWk_, mWv_, vWv_, mS_, vS_;
    vk张量 sm_;
    vk张量 mFull_;   // [B,H,T,T] 全掩码，批式反传用

    // 逐 query 反传：每个 query 一套独立输出，块内互不依赖，才能并到一条算子串
    std::vector<vk张量> q1s_, m1s_, dy1s_, dq1s_, dk1s_, dv1s_, ds1s_;

    // Apollo solver 必须在所有它引用的张量之后声明：析构逆序时 solver 先销毁
    std::unique_ptr<ApolloSolver> apolloWq_, apolloWk_, apolloWv_, apolloSink_;
    bool 用Int8前向_ = false;
    std::unique_ptr<vk张量> 量化X_, 尺度X_;
    std::unique_ptr<vk张量> 量化Wq_, 尺度Wq_;
    std::unique_ptr<vk张量> 量化Wk_, 尺度Wk_;
    std::unique_ptr<vk张量> 量化Wv_, 尺度Wv_;

    // 主机侧数据
    std::vector<float> x_, 目标_, Wq0_, Wk0_, Wv0_;
    std::vector<float> qh_, bsh_, outh_, 掩码_, dyh_;
    std::vector<float> dQh_, dKh_, dVh_;
    float dsink_ = 0.0f;
    std::vector<float> 损耗_;
};

void 训练器::建图() {
    const std::uint32_t T = 参数_.T, DX = 参数_.DX, DK = 参数_.DK, DV = 参数_.DV;
    const std::uint32_t B = 参数_.B, H = 参数_.H, NB = 参数_.NB;

    X_   = 建({T, DX}, "x");
    Qp_  = 建({T, DK}, "q");
    Kp_  = 建({B, H, T, DK}, "K");
    Vp_  = 建({B, H, T, DV}, "V");
    BS_  = 建({B, H, NB, DK}, "bsum");
    Out_ = 建({B, H, T, DV}, "out");
    Dy_  = 建({B, H, T, DV}, "dy");
    dQd_ = 建({T, DK}, "dQ");
    dKd_ = 建({T, H, DK}, "dK");
    dVd_ = 建({T, H, DV}, "dV");

    Wq_ = 建({DK, DX}, "W_q");
    Wk_ = 建({DK, DX}, "W_k");
    Wv_ = 建({DV, DX}, "W_v");
    Sink_ = 建({1}, "sink_bias");

    零偏DK_ = 建({DK}, "b0_dk");
    零偏DV_ = 建({DV}, "b0_dv");
    dbk_ = 建({DK}, "db_dk");
    dbv_ = 建({DV}, "db_dv");
    dS_ = 建({1}, "dsink");

    dWq_ = 建({DK, DX}, "dWq");
    dWk_ = 建({DK, DX}, "dWk");
    dWv_ = 建({DV, DX}, "dWv");

    mWq_ = 建({DK, DX}, "mWq");
    vWq_ = 建({DK, DX}, "vWq");
    mWk_ = 建({DK, DX}, "mWk");
    vWk_ = 建({DK, DX}, "vWk");
    mWv_ = 建({DV, DX}, "mWv");
    vWv_ = 建({DV, DX}, "vWv");
    mS_ = 建({1}, "mS");
    vS_ = 建({1}, "vS");
    sm_ = 建({B, H}, "sinkbuf");
    mFull_ = 建({B, H, T, T}, "mask_full");

    g_lcg = 31337u;   // 固定随机源：每次构造训练器拿到同一份数据
    x_.assign(T * DX, 0.f);
    目标_.assign(T * DV, 0.f);
    Wq0_.assign(DK * DX, 0.f);
    Wk0_.assign(DK * DX, 0.f);
    Wv0_.assign(DV * DX, 0.f);
    for (auto& v : x_) v = 随机数();
    for (auto& v : 目标_) v = 随机数();
    for (auto& v : Wq0_) v = 随机数(0.5f);
    for (auto& v : Wk0_) v = 随机数(0.5f);
    for (auto& v : Wv0_) v = 随机数(0.5f);

    qh_.assign(T * DK, 0.f);
    bsh_.assign(NB * DK, 0.f);
    outh_.assign(T * DV, 0.f);
    掩码_.assign(T * T, 0.f);
    dyh_.assign(T * DV, 0.f);
    dQh_.assign(T * DK, 0.f);
    dKh_.assign(T * DK, 0.f);
    dVh_.assign(T * DV, 0.f);
    损耗_.assign(参数_.STEPS, 0.f);
}

void 训练器::建每query() {
    const std::uint32_t T = 参数_.T, DK = 参数_.DK, DV = 参数_.DV;
    const std::uint32_t B = 参数_.B, H = 参数_.H;
    q1s_.reserve(T); m1s_.reserve(T); dy1s_.reserve(T);
    dq1s_.reserve(T); dk1s_.reserve(T); dv1s_.reserve(T); ds1s_.reserve(T);
    for (std::uint32_t t = 0; t < T; ++t) {
        const std::string 号 = std::to_string(t);
        q1s_.push_back(建({B, H, DK}, "q1_" + 号));
        m1s_.push_back(建({B, H, T}, "mask1_" + 号));
        dy1s_.push_back(建({B, H, DV}, "dy1_" + 号));
        dq1s_.push_back(建({B, H, DK}, "dq1_" + 号));
        dk1s_.push_back(建({T, H, DK}, "dk1_" + 号));
        dv1s_.push_back(建({T, H, DV}, "dv1_" + 号));
        ds1s_.push_back(建({B, H}, "ds1_" + 号));
    }
}

void 训练器::重置() {
    X_.上传(x_.data(), x_.size() * 4);
    Dy_.上传(dyh_.data(), dyh_.size() * 4);
    Wq_.上传(Wq0_.data(), Wq0_.size() * 4);
    Wk_.上传(Wk0_.data(), Wk0_.size() * 4);
    Wv_.上传(Wv0_.data(), Wv0_.size() * 4);
    const float s = 初始sink;
    Sink_.上传(&s, 4);

    std::vector<float> 零(参数_.DK, 0.f);
    零偏DK_.上传(零.data(), 零.size() * 4);
    零.assign(参数_.DV, 0.f);
    零偏DV_.上传(零.data(), 零.size() * 4);
    dbk_.上传(零.data(), 参数_.DK * 4);
    dbv_.上传(零.data(), 参数_.DV * 4);
    std::fill(零.begin(), 零.end(), 0.f);

    零.assign(参数_.DK * 参数_.DX, 0.f);
    mWq_.上传(零.data(), 零.size() * 4);
    vWq_.上传(零.data(), 零.size() * 4);
    mWk_.上传(零.data(), 零.size() * 4);
    vWk_.上传(零.data(), 零.size() * 4);
    零.assign(参数_.DV * 参数_.DX, 0.f);
    mWv_.上传(零.data(), 零.size() * 4);
    vWv_.上传(零.data(), 零.size() * 4);
    零.assign(1, 0.f);
    mS_.上传(零.data(), 4);
    vS_.上传(零.data(), 4);
    dS_.上传(零.data(), 4);

    std::fill(dQh_.begin(), dQh_.end(), 0.f);
    std::fill(dKh_.begin(), dKh_.end(), 0.f);
    std::fill(dVh_.begin(), dVh_.end(), 0.f);
    dsink_ = 0.0f;
    std::fill(损耗_.begin(), 损耗_.end(), 0.f);
}

// ---- 同步 / 提交 ----
void 训练器::等(std::uint64_t 编号, const char* 说明) {
    调度器_.等待(编号);
    if (!调度器_.已完成(编号))
        throw std::runtime_error(std::string("算子失败: ") + 说明 + " | "
                                 + 调度器_.查询失败原因(编号));
}

void 训练器::提交单个(const 算子步骤& 步) {
    const std::uint64_t 编号 = 调度器_.提交自定义算子(
        步.算子名, 步.张量列表, 步.推送常量.data(),
        static_cast<std::uint32_t>(步.推送常量.size()), 步.派发);
    等(编号, 步.算子名.c_str());
}

void 训练器::提交串(const std::vector<算子步骤>& 步骤) {
    const std::uint64_t 编号 = 调度器_.提交算子串(步骤);
    等(编号, "算子串");
}

float 训练器::读sink() const {
    float s = 0.0f;
    Sink_.下载(&s, 4);
    return s;
}

// ---- 算子步骤 ----
算子步骤 训练器::步线性(const vk张量& 输入, const vk张量& 权, const vk张量& 偏,
                        const vk张量& 出, std::uint32_t N, std::uint32_t K, std::uint32_t M) const {
    推送线性 pc{N, K, M, 0};
    return 构造算子步骤("linear_fwd",
                        {输入.取句柄(), 权.取句柄(), 偏.取句柄(), 出.取句柄()}, pc,
                        派发规格{(N * M + 63u) / 64u, 1u, 1u});
}

算子步骤 训练器::步块和() const {
    推送块和 pc{参数_.B, 参数_.H, 参数_.T, 参数_.DK, 参数_.NB, 参数_.BLOCK, 0, 0};
    return 构造算子步骤("ersa_block_sum", {Kp_.取句柄(), BS_.取句柄()}, pc,
                        派发规格{参数_.NB, 参数_.H, 参数_.B});
}

算子步骤 训练器::步注意力(float 当前sink) const {
    推送前向 pc{参数_.B, 参数_.H, 参数_.T, 参数_.T, 参数_.DK, 参数_.DV,
                参数_.NB, 参数_.BLOCK, 参数_.TOPB, 参数_.WINDOW, 0, 0,
                参数_.tau, 当前sink};
    return 构造算子步骤("ersa_attn_fused",
                        {Qp_.取句柄(), Kp_.取句柄(), Vp_.取句柄(), BS_.取句柄(), Out_.取句柄()}, pc,
                        派发规格{参数_.T, 参数_.H, 参数_.B});
}

算子步骤 训练器::步反传(std::uint32_t t) const {
    推送反向 pc{参数_.B, 参数_.H, 参数_.T, 参数_.DK, 参数_.DV, 0};
    return 构造算子步骤("ersa_attn_bwd",
                        {q1s_[t].取句柄(), Kp_.取句柄(), Vp_.取句柄(), m1s_[t].取句柄(),
                         sm_.取句柄(), dy1s_[t].取句柄(), dq1s_[t].取句柄(),
                         dk1s_[t].取句柄(), dv1s_[t].取句柄(), ds1s_[t].取句柄()},
                        pc, 派发规格{参数_.B * 参数_.H, 1u, 1u});
}

算子步骤 训练器::步批式反传() const {
    推送反向批 pc{参数_.B, 参数_.H, 参数_.T, 参数_.DK, 参数_.DV, 参数_.tau};
    return 构造算子步骤("ersa_attn_bwd_batch",
                        {Qp_.取句柄(), Kp_.取句柄(), Vp_.取句柄(), mFull_.取句柄(),
                         sm_.取句柄(), Dy_.取句柄(), dQd_.取句柄(), dKd_.取句柄(),
                         dVd_.取句柄(), dS_.取句柄()},
                        pc, 派发规格{参数_.B * 参数_.H, 1u, 1u});
}

算子步骤 训练器::步dwdb(const vk张量& X, const vk张量& dY, const vk张量& dW, const vk张量& db,
                        std::uint32_t N, std::uint32_t K, std::uint32_t M) const {
    推送线性 pc{N, K, M, 0};
    return 构造算子步骤("linear_bwd_dwdb",
                        {X.取句柄(), dY.取句柄(), dW.取句柄(), db.取句柄()}, pc,
                        派发规格{(M * K + 63u) / 64u, 1u, 1u});
}

算子步骤 训练器::步AdamW(const vk张量& G, const vk张量& P, const vk张量& M, const vk张量& V,
                         std::uint32_t 个数, float 衰减, std::uint32_t 步序号) const {
    const float m_corr = 1.0f / (1.0f - std::pow(beta1, float(步序号 + 1)));
    const float v_corr = 1.0f / (1.0f - std::pow(beta2, float(步序号 + 1)));
    推送AdamW pc{lr, beta1, beta2, eps, m_corr, v_corr, 个数, 衰减, 1u};
    return 构造算子步骤("adamw_update",
                        {G.取句柄(), P.取句柄(), M.取句柄(), V.取句柄()}, pc,
                        派发规格{(个数 + 255u) / 256u, 1u, 1u});
}

std::vector<算子步骤> 训练器::前向步骤(float 当前sink) const {
    std::vector<算子步骤> 列表;
    列表.reserve(5);
    if (用Int8前向_) {
        列表.push_back(步线性int8(*量化X_, *尺度X_, *量化Wq_, *尺度Wq_,
                                  零偏DK_, Qp_, 参数_.T, 参数_.DX, 参数_.DK));
        列表.push_back(步线性int8(*量化X_, *尺度X_, *量化Wk_, *尺度Wk_,
                                  零偏DK_, Kp_, 参数_.T, 参数_.DX, 参数_.DK));
        列表.push_back(步线性int8(*量化X_, *尺度X_, *量化Wv_, *尺度Wv_,
                                  零偏DV_, Vp_, 参数_.T, 参数_.DX, 参数_.DV));
    } else {
        列表.push_back(步线性(X_, Wq_, 零偏DK_, Qp_, 参数_.T, 参数_.DX, 参数_.DK));
        列表.push_back(步线性(X_, Wk_, 零偏DK_, Kp_, 参数_.T, 参数_.DX, 参数_.DK));
        列表.push_back(步线性(X_, Wv_, 零偏DV_, Vp_, 参数_.T, 参数_.DX, 参数_.DV));
    }
    列表.push_back(步块和());
    列表.push_back(步注意力(当前sink));
    return 列表;
}

std::vector<算子步骤> 训练器::更新步骤(std::uint32_t 步序号) const {
    std::vector<算子步骤> 列表;
    列表.reserve(7);
    列表.push_back(步dwdb(X_, dQd_, dWq_, dbk_, 参数_.T, 参数_.DX, 参数_.DK));
    列表.push_back(步dwdb(X_, dKd_, dWk_, dbk_, 参数_.T, 参数_.DX, 参数_.DK));
    列表.push_back(步dwdb(X_, dVd_, dWv_, dbv_, 参数_.T, 参数_.DX, 参数_.DV));
    列表.push_back(步AdamW(dWq_, Wq_, mWq_, vWq_, 参数_.DK * 参数_.DX, wd, 步序号));
    列表.push_back(步AdamW(dWk_, Wk_, mWk_, vWk_, 参数_.DK * 参数_.DX, wd, 步序号));
    列表.push_back(步AdamW(dWv_, Wv_, mWv_, vWv_, 参数_.DV * 参数_.DX, wd, 步序号));
    列表.push_back(步AdamW(dS_, Sink_, mS_, vS_, 1u, 0.0f, 步序号));
    return 列表;
}

// ---- 主机侧：路由 / 损失 / 反传输入 ----
void 训练器::主机路由() {
    const std::uint32_t T = 参数_.T, NB = 参数_.NB, DK = 参数_.DK;
    const std::uint32_t TOPB = 参数_.TOPB, BLOCK = 参数_.BLOCK, WINDOW = 参数_.WINDOW;
    const float scale = 1.0f / std::sqrt(float(DK));

    std::fill(掩码_.begin(), 掩码_.end(), 0.0f);
    std::vector<std::uint32_t> 选块(T * TOPB, 0u);
    for (std::uint32_t t = 0; t < T; ++t) {
        std::vector<float> sc(NB, 0.f);
        for (std::uint32_t i = 0; i < NB; ++i) {
            float d = 0.0f;
            for (std::uint32_t k = 0; k < DK; ++k) d += qh_[t * DK + k] * bsh_[i * DK + k];
            sc[i] = d * scale / 参数_.tau;
        }
        const std::uint32_t topn = std::min(TOPB, NB);
        for (std::uint32_t kk = 0; kk < topn; ++kk) {
            float best = -3.402823e38f;
            std::uint32_t bi = 0u;
            for (std::uint32_t i = 0; i < NB; ++i)
                if (sc[i] > best) { best = sc[i]; bi = i; }
            选块[t * TOPB + kk] = bi;
            sc[bi] = -3.402823e38f;
        }
        for (std::uint32_t kk = 0; kk < topn; ++kk) {
            const std::uint32_t bi = 选块[t * TOPB + kk];
            const std::uint32_t start = bi * BLOCK;
            const std::uint32_t end = std::min(start + BLOCK, T);
            for (std::uint32_t idx = start; idx < end; ++idx)
                if (idx <= t) 掩码_[t * T + idx] = 1.0f;
        }
        const std::uint32_t wstart = (t + 1u > WINDOW) ? (t + 1u - WINDOW) : 0u;
        for (std::uint32_t idx = wstart; idx <= t; ++idx) 掩码_[t * T + idx] = 1.0f;
    }
}

void 训练器::前向后主机(std::uint32_t 步序号) {
    Qp_.下载(qh_.data(), qh_.size() * 4);
    BS_.下载(bsh_.data(), bsh_.size() * 4);
    Out_.下载(outh_.data(), outh_.size() * 4);
    主机路由();

    const std::uint32_t T = 参数_.T, DV = 参数_.DV;
    float 损 = 0.0f;
    for (std::uint32_t i = 0; i < T * DV; ++i) {
        const float d = outh_[i] - 目标_[i];
        损 += d * d;
        dyh_[i] = 2.0f * d / float(T * DV);
    }
    损 /= float(T * DV);
    损耗_[步序号] = 损;
    Dy_.上传(dyh_.data(), dyh_.size() * 4);

    const float 当前sink = 读sink();
    sm_.上传(&当前sink, 4);
    mFull_.上传(掩码_.data(), 掩码_.size() * 4);   // 批式反传的 T×T 掩码
}

void 训练器::反传输入上传(std::uint32_t t) {
    const std::uint32_t T = 参数_.T, DK = 参数_.DK, DV = 参数_.DV;
    std::vector<float> qa(DK, 0.f);
    for (std::uint32_t d = 0; d < DK; ++d) qa[d] = qh_[t * DK + d] / 参数_.tau;
    q1s_[t].上传(qa.data(), qa.size() * 4);
    m1s_[t].上传(&掩码_[t * T], T * 4);
    dy1s_[t].上传(&dyh_[t * DV], DV * 4);
}

void 训练器::反传收集(std::uint32_t t) {
    const std::uint32_t T = 参数_.T, DK = 参数_.DK, DV = 参数_.DV;
    std::vector<float> 缓冲(std::max(DK, std::max(T * DK, T * DV)), 0.f);
    dq1s_[t].下载(缓冲.data(), DK * 4);
    for (std::uint32_t d = 0; d < DK; ++d) dQh_[t * DK + d] = 缓冲[d] / 参数_.tau;
    std::fill(缓冲.begin(), 缓冲.end(), 0.f);
    dk1s_[t].下载(缓冲.data(), T * DK * 4);
    for (std::uint32_t i = 0; i < T * DK; ++i) dKh_[i] += 缓冲[i];
    std::fill(缓冲.begin(), 缓冲.end(), 0.f);
    dv1s_[t].下载(缓冲.data(), T * DV * 4);
    for (std::uint32_t i = 0; i < T * DV; ++i) dVh_[i] += 缓冲[i];
    float ds = 0.0f;
    ds1s_[t].下载(&ds, 4);
    dsink_ += ds;
}

void 训练器::反传上传梯度() {
    dQd_.上传(dQh_.data(), dQh_.size() * 4);
    dKd_.上传(dKh_.data(), dKh_.size() * 4);
    dVd_.上传(dVh_.data(), dVh_.size() * 4);
    dS_.上传(&dsink_, 4);
}

// ---- 两条调度路径 ----
void 训练器::启用Apollo() {
    用Apollo_ = true;
    ApolloSolver::配置 c;
    c.beta = beta1;
    c.学习率 = 1e-3f;
    c.sigma = 1e-3f;
    c.epsilon = eps;
    apolloWq_ = std::make_unique<ApolloSolver>(Wq_, dWq_, 参数_.DK * 参数_.DX, c);
    apolloWk_ = std::make_unique<ApolloSolver>(Wk_, dWk_, 参数_.DK * 参数_.DX, c);
    apolloWv_ = std::make_unique<ApolloSolver>(Wv_, dWv_, 参数_.DV * 参数_.DX, c);
    apolloSink_ = std::make_unique<ApolloSolver>(Sink_, dS_, 1u, c);
}

void 训练器::启用Int8前向() {
    用Int8前向_ = true;
    const std::uint32_t T = 参数_.T, DX = 参数_.DX, DK = 参数_.DK, DV = 参数_.DV;
    const std::uint32_t G = (DX + 127u) / 128u;
    auto 量化 = [&](const std::vector<float>& 源, std::uint32_t 行, std::uint32_t 列,
                    std::unique_ptr<vk张量>& q, std::unique_ptr<vk张量>& s, const char* 名) {
        q = std::make_unique<vk张量>(张量布局::连续({行 * 列}), 张量类型::整数8, std::string(名) + "_q");
        s = std::make_unique<vk张量>(张量布局::连续({行 * G}), 张量类型::浮点32, std::string(名) + "_s");
        std::vector<std::int8_t> qd(行 * 列);
        std::vector<float> sd(行 * G);
        for (std::uint32_t r = 0; r < 行; ++r) {
            for (std::uint32_t g = 0; g < G; ++g) {
                float 最大 = 0.0f;
                const std::uint32_t k0 = g * 128u;
                const std::uint32_t k1 = std::min(k0 + 128u, 列);
                for (std::uint32_t k = k0; k < k1; ++k)
                    最大 = std::max(最大, std::fabs(源[r * 列 + k]));
                const float sc = std::max(最大 / 127.0f, 1e-10f);
                sd[r * G + g] = sc;
                for (std::uint32_t k = k0; k < k1; ++k) {
                    int v = static_cast<int>(std::lround(源[r * 列 + k] / sc));
                    qd[r * 列 + k] = static_cast<std::int8_t>(std::clamp(v, -127, 127));
                }
            }
        }
        q->上传(qd.data(), qd.size());
        s->上传(sd.data(), sd.size() * sizeof(float));
    };
    量化(x_, T, DX, 量化X_, 尺度X_, "x");
    量化(Wq0_, DK, DX, 量化Wq_, 尺度Wq_, "wq");
    量化(Wk0_, DK, DX, 量化Wk_, 尺度Wk_, "wk");
    量化(Wv0_, DV, DX, 量化Wv_, 尺度Wv_, "wv");
}

void 训练器::刷新Int8权重() {
    const std::uint32_t DX = 参数_.DX, DK = 参数_.DK, DV = 参数_.DV;
    const std::uint32_t G = (DX + 127u) / 128u;
    auto 重量化 = [&](const vk张量& W, vk张量& qW, vk张量& sW,
                     std::uint32_t 行, std::uint32_t 列) {
        std::vector<float> host(行 * 列);
        W.下载(host.data(), host.size() * sizeof(float));
        std::vector<std::int8_t> qd(行 * 列);
        std::vector<float> sd(行 * G);
        for (std::uint32_t r = 0; r < 行; ++r) {
            for (std::uint32_t g = 0; g < G; ++g) {
                const std::uint32_t k0 = g * 128u;
                const std::uint32_t k1 = std::min(k0 + 128u, 列);
                float 最大 = 0.0f;
                for (std::uint32_t k = k0; k < k1; ++k)
                    最大 = std::max(最大, std::fabs(host[r * 列 + k]));
                const float sc = std::max(最大 / 127.0f, 1e-10f);
                sd[r * G + g] = sc;
                for (std::uint32_t k = k0; k < k1; ++k) {
                    int v = static_cast<int>(std::lround(host[r * 列 + k] / sc));
                    qd[r * 列 + k] = static_cast<std::int8_t>(std::clamp(v, -127, 127));
                }
            }
        }
        qW.上传(qd.data(), qd.size());
        sW.上传(sd.data(), sd.size() * sizeof(float));
    };
    重量化(Wq_, *量化Wq_, *尺度Wq_, DK, DX);
    重量化(Wk_, *量化Wk_, *尺度Wk_, DK, DX);
    重量化(Wv_, *量化Wv_, *尺度Wv_, DV, DX);
}

算子步骤 训练器::步线性int8(const vk张量& xq, const vk张量& sx,
                          const vk张量& wq, const vk张量& sw,
                          const vk张量& 偏, const vk张量& 出,
                          std::uint32_t N, std::uint32_t K, std::uint32_t M) const {
    推送Int8 pc{N, K, M, 128u, 1u};
    return 构造算子步骤("linear_fwd_int8_f32",
                        {xq.取句柄(), sx.取句柄(), wq.取句柄(), sw.取句柄(),
                         偏.取句柄(), 出.取句柄()},
                        pc, 派发规格{(N * M + 255u) / 256u, 1u, 1u});
}

float 训练器::前向损失() {
    提交串(前向步骤(读sink()));
    前向后主机(0);
    return 损耗_[0];
}

void 训练器::串行步(std::uint32_t 步序号) {
    if (用Int8前向_) 刷新Int8权重();
    const float 当前sink = 读sink();
    if (用Int8前向_) {
        提交单个(步线性int8(*量化X_, *尺度X_, *量化Wq_, *尺度Wq_,
                            零偏DK_, Qp_, 参数_.T, 参数_.DX, 参数_.DK));
        提交单个(步线性int8(*量化X_, *尺度X_, *量化Wk_, *尺度Wk_,
                            零偏DK_, Kp_, 参数_.T, 参数_.DX, 参数_.DK));
        提交单个(步线性int8(*量化X_, *尺度X_, *量化Wv_, *尺度Wv_,
                            零偏DV_, Vp_, 参数_.T, 参数_.DX, 参数_.DV));
    } else {
        提交单个(步线性(X_, Wq_, 零偏DK_, Qp_, 参数_.T, 参数_.DX, 参数_.DK));
        提交单个(步线性(X_, Wk_, 零偏DK_, Kp_, 参数_.T, 参数_.DX, 参数_.DK));
        提交单个(步线性(X_, Wv_, 零偏DV_, Vp_, 参数_.T, 参数_.DX, 参数_.DV));
    }
    提交单个(步块和());
    提交单个(步注意力(当前sink));

    前向后主机(步序号);

    std::fill(dQh_.begin(), dQh_.end(), 0.f);
    std::fill(dKh_.begin(), dKh_.end(), 0.f);
    std::fill(dVh_.begin(), dVh_.end(), 0.f);
    dsink_ = 0.0f;
    for (std::uint32_t t = 0; t < 参数_.T; ++t) {
        反传输入上传(t);
        提交单个(步反传(t));
        反传收集(t);
    }
    反传上传梯度();

    if (用Apollo_) {
        apolloWq_->步骤();
        apolloWk_->步骤();
        apolloWv_->步骤();
        apolloSink_->步骤();
    } else {
        const auto 更新 = 更新步骤(步序号);
        for (const auto& 步 : 更新) 提交单个(步);
    }
}

void 训练器::融合步(std::uint32_t 步序号, std::uint32_t 每串query数, bool 批式) {
    const auto 毫秒 = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    auto 标记 = std::chrono::steady_clock::now();

    const float 当前sink = 读sink();
    提交串(前向步骤(当前sink));
    auto 现在 = std::chrono::steady_clock::now();
    if (计时) { 计时->前向 += 毫秒(标记, 现在); 标记 = 现在; }

    前向后主机(步序号);
    现在 = std::chrono::steady_clock::now();
    if (计时) { 计时->主机 += 毫秒(标记, 现在); 标记 = 现在; }

    if (批式) {
        // 批式反传：dK/dV/dsink 设备侧跨 query 累加，先清零；dQ 每 query 一行直接覆盖
        std::vector<float> 零K(参数_.T * 参数_.DK, 0.f);
        std::vector<float> 零V(参数_.T * 参数_.DV, 0.f);
        dKd_.上传(零K.data(), 零K.size() * 4);
        dVd_.上传(零V.data(), 零V.size() * 4);
        dS_.上传(零K.data(), 4);
        现在 = std::chrono::steady_clock::now();
        if (计时) { 计时->反传准备 += 毫秒(标记, 现在); 标记 = 现在; }

        提交串({步批式反传()});
        现在 = std::chrono::steady_clock::now();
        if (计时) { 计时->反传提交 += 毫秒(标记, 现在); 标记 = 现在; }
    } else {
        std::fill(dQh_.begin(), dQh_.end(), 0.f);
        std::fill(dKh_.begin(), dKh_.end(), 0.f);
        std::fill(dVh_.begin(), dVh_.end(), 0.f);
        dsink_ = 0.0f;

        const std::uint32_t 块 = std::max<std::uint32_t>(1u, 每串query数);
        for (std::uint32_t 起点 = 0; 起点 < 参数_.T; 起点 += 块) {
            const std::uint32_t 终点 = std::min(起点 + 块, 参数_.T);
            std::vector<算子步骤> 串;
            串.reserve(终点 - 起点);
            for (std::uint32_t t = 起点; t < 终点; ++t) {
                反传输入上传(t);
                算子步骤 步 = 步反传(t);
                // 各 query 的反传互不读写对方输出，串内不需要逐步屏障
                步.不需前置屏障 = true;
                串.push_back(std::move(步));
            }
            现在 = std::chrono::steady_clock::now();
            if (计时) { 计时->反传准备 += 毫秒(标记, 现在); 标记 = 现在; }

            提交串(串);
            现在 = std::chrono::steady_clock::now();
            if (计时) { 计时->反传提交 += 毫秒(标记, 现在); 标记 = 现在; }

            for (std::uint32_t t = 起点; t < 终点; ++t) 反传收集(t);
            现在 = std::chrono::steady_clock::now();
            if (计时) { 计时->反传收集 += 毫秒(标记, 现在); 标记 = 现在; }
        }
        反传上传梯度();
        现在 = std::chrono::steady_clock::now();
        if (计时) { 计时->反传收集 += 毫秒(标记, 现在); 标记 = 现在; }
    }

    提交串(更新步骤(步序号));
    if (计时) 计时->更新 += 毫秒(标记, std::chrono::steady_clock::now());
}

// ---- 导出（格式与 train_m3c 完全一致，供 m3c_check.py） ----
void 训练器::导出(const char* 路径) const {
    std::ofstream f(路径, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error(std::string("无法写导出文件: ") + 路径);
    const std::uint32_t hdr[9] = {参数_.T, 参数_.DX, 参数_.DK, 参数_.DV,
                                  参数_.BLOCK, 参数_.NB, 参数_.TOPB, 参数_.WINDOW, 参数_.STEPS};
    f.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
    auto 写 = [&](const void* p, std::size_t 个) {
        f.write(static_cast<const char*>(p), static_cast<std::streamsize>(个 * 4));
    };
    写(&参数_.tau, 1);
    const float s0 = 初始sink;
    写(&s0, 1);
    const float 学习率 = lr, 衰减 = wd, b1 = beta1, b2 = beta2, e = eps;
    写(&学习率, 1); 写(&衰减, 1); 写(&b1, 1); 写(&b2, 1); 写(&e, 1);
    写(x_.data(), x_.size());
    写(目标_.data(), 目标_.size());
    写(Wq0_.data(), Wq0_.size());
    写(Wk0_.data(), Wk0_.size());
    写(Wv0_.data(), Wv0_.size());
    写(损耗_.data(), 损耗_.size());

    std::vector<float> Wq(Wq0_.size()), Wk(Wk0_.size()), Wv(Wv0_.size()), sink(1);
    取参数(Wq, Wk, Wv, sink);
    写(Wq.data(), Wq.size());
    写(Wk.data(), Wk.size());
    写(Wv.data(), Wv.size());
    写(sink.data(), 1);
}

// ---------------------------------------------------------------------------
// 基准：dispatch 开销 / 串行 vs 融合
// ---------------------------------------------------------------------------
struct 单步计时 {
    std::vector<double> 毫秒;
};

double 毫秒差(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

void 打印计时(const char* 标签, const 计时结果& 结果, std::uint32_t T) {
    const double 吞吐 = 结果.中位 > 0.0 ? (double(T) * 1000.0 / 结果.中位) : 0.0;
    std::printf("%-16s 中位 %8.3f ms/步  最小 %8.3f  最大 %8.3f  → %10.1f token/s\n",
                标签, 结果.中位, 结果.最小, 结果.最大, 吞吐);
}

// 映射内存实测带宽：1) 每次下载/上传的调用开销；2) 大块传输带宽
struct 带宽结果 { double 每次微秒 = 0.0; double 兆字节每秒 = 0.0; };

带宽结果 测读带宽(std::uint32_t 字节, std::uint32_t 次数) {
    const std::uint32_t 个数 = 字节 / 4u;
    vk张量 t = 建({个数}, "bench_dl");
    std::vector<float> 数据(个数, 1.0f);
    std::vector<float> 缓冲(个数, 0.f);
    t.上传(数据.data(), 字节);
    for (std::uint32_t i = 0; i < 64; ++i) t.下载(缓冲.data(), 字节);
    const auto t0 = std::chrono::steady_clock::now();
    for (std::uint32_t i = 0; i < 次数; ++i) t.下载(缓冲.data(), 字节);
    const auto t1 = std::chrono::steady_clock::now();
    const double 毫秒 = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double 总字节 = double(字节) * double(次数);
    带宽结果 结果;
    结果.每次微秒 = 毫秒 * 1000.0 / double(次数);
    结果.兆字节每秒 = 总字节 / (毫秒 / 1000.0) / (1024.0 * 1024.0);
    return 结果;
}

带宽结果 测写带宽(std::uint32_t 字节, std::uint32_t 次数) {
    const std::uint32_t 个数 = 字节 / 4u;
    vk张量 t = 建({个数}, "bench_ul");
    std::vector<float> 数据(个数, 1.0f);
    for (std::uint32_t i = 0; i < 64; ++i) t.上传(数据.data(), 字节);
    const auto t0 = std::chrono::steady_clock::now();
    for (std::uint32_t i = 0; i < 次数; ++i) t.上传(数据.data(), 字节);
    const auto t1 = std::chrono::steady_clock::now();
    const double 毫秒 = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double 总字节 = double(字节) * double(次数);
    带宽结果 结果;
    结果.每次微秒 = 毫秒 * 1000.0 / double(次数);
    结果.兆字节每秒 = 总字节 / (毫秒 / 1000.0) / (1024.0 * 1024.0);
    return 结果;
}

double 测dispatch开销(流水线调度器& 调度器, std::uint32_t 次数) {
    vk张量 g = 建({1}, "bench_g");
    vk张量 p = 建({1}, "bench_p");
    vk张量 m = 建({1}, "bench_m");
    vk张量 v = 建({1}, "bench_v");
    const float 一 = 1.0f, 零f = 0.0f;
    g.上传(&零f, 4); p.上传(&一, 4); m.上传(&零f, 4); v.上传(&零f, 4);
    推送AdamW pc{0.0f, 0.9f, 0.999f, 1e-8f, 1.0f, 1.0f, 1u, 0.0f, 0u};
    std::vector<double> 样本;
    样本.reserve(次数);
    for (std::uint32_t i = 0; i < 32; ++i) {
        const std::uint64_t 编号 = 调度器.提交自定义算子(
            "adamw_update", {g.取句柄(), p.取句柄(), m.取句柄(), v.取句柄()},
            &pc, sizeof(pc), 派发规格{1, 1, 1});
        调度器.等待(编号);
    }
    for (std::uint32_t i = 0; i < 次数; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        const std::uint64_t 编号 = 调度器.提交自定义算子(
            "adamw_update", {g.取句柄(), p.取句柄(), m.取句柄(), v.取句柄()},
            &pc, sizeof(pc), 派发规格{1, 1, 1});
        调度器.等待(编号);
        const auto t1 = std::chrono::steady_clock::now();
        样本.push_back(毫秒差(t0, t1));
    }
    return 汇总(std::move(样本)).中位;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "用法:\n"
                     "  %s <spv目录> check <导出.bin>\n"
                     "  %s <spv目录> bench [T] [步数] [每串query]\n", argv[0], argv[0]);
        return 2;
    }
    const std::string 模式 = argv[2];

    流水线调度器& 调度器 = 流水线调度器::获取();
    设备注册信息 注册;
    注册.应用名 = "train_m4";
    阶段配置 图形;
    阶段配置 计算{阶段配置::自动队列族, 0, 1};
    阶段配置 传输;
    try {
        调度器.启动(注册, {图形, 计算, 传输}, argv[1]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "启动失败: %s\n", e.what());
        return 1;
    }

    try {
        if (模式 == "check") {
            if (argc < 4) { std::fprintf(stderr, "check 需要导出路径\n"); return 2; }
            const 配置 c = M3c配置();
            训练器 甲(调度器, c);

            // 路径 A：逐算子提交 + 等待（M3c 基线）
            for (std::uint32_t s = 0; s < c.STEPS; ++s) 甲.串行步(s);
            const std::vector<float> 串行loss = 甲.损耗();
            std::vector<float> WqA, WkA, WvA, sinkA;
            甲.取参数(WqA, WkA, WvA, sinkA);

            // 路径 B：融合算子串，但仍逐 query 反传（与 A 同一个内核，应逐位一致）
            甲.重置();
            for (std::uint32_t s = 0; s < c.STEPS; ++s) 甲.融合步(s, c.T, false);
            const std::vector<float> 融合loss = 甲.损耗();
            std::vector<float> WqB, WkB, WvB, sinkB;
            甲.取参数(WqB, WkB, WvB, sinkB);

            // 路径 C：融合算子串 + 批式反传（一个 dispatch 扫全部 query）
            甲.重置();
            for (std::uint32_t s = 0; s < c.STEPS; ++s) 甲.融合步(s, c.T, true);
            const std::vector<float> 批式loss = 甲.损耗();
            std::vector<float> WqC, WkC, WvC, sinkC;
            甲.取参数(WqC, WkC, WvC, sinkC);

            auto 最大差 = [](const std::vector<float>& a, const std::vector<float>& b) {
                float m = 0.0f;
                for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
                    m = std::max(m, std::fabs(a[i] - b[i]));
                return m;
            };

            std::printf("== M4 路径对照（M3c 尺寸 T=%u，%u 步）==\n", c.T, c.STEPS);
            const float b_loss = 最大差(串行loss, 融合loss);
            const float b_p = std::max(std::max(最大差(WqA, WqB), 最大差(WkA, WkB)),
                                       std::max(最大差(WvA, WvB), 最大差(sinkA, sinkB)));
            // 链路本身有 ~1e-7 的 fp32 非确定性（同路径两遍也不完全相同），判据取 1e-6
            const bool b_ok = b_loss <= 1e-6f && b_p <= 1e-6f;
            std::printf("A 串行 vs B 融合逐query：loss %.3e  参数 %.3e  %s\n",
                        b_loss, b_p, b_ok ? "一致 PASS" : "FAIL");

            const float c_loss = 最大差(串行loss, 批式loss);
            const float c_p = std::max(std::max(最大差(WqA, WqC), 最大差(WkA, WkC)),
                                       std::max(最大差(WvA, WvC), 最大差(sinkA, sinkC)));
            const float 容差 = 2e-4f;   // 不同求和顺序，按梯度容差
            const bool c_ok = c_loss <= 容差 && c_p <= 容差;
            std::printf("A 串行 vs C 融合批式  ：loss %.3e  参数 %.3e  %s\n",
                        c_loss, c_p, c_ok ? "PASS" : "FAIL");
            std::printf("== M4 检查 %s ==\n", (b_ok && c_ok) ? "PASS" : "FAIL");
            if (!(b_ok && c_ok)) { 调度器.停止(); return 1; }

            甲.导出(argv[3]);
            std::printf("批式反传路径结果已导出 %s（供 m3c_check.py 对拍 PyTorch）\n", argv[3]);
            调度器.停止();
            return 0;
        }

        if (模式 == "int8train") {
            const 配置 c = M3c配置();
            训练器 甲(调度器, c);              // 纯 fp32
            训练器 乙(调度器, c);
            乙.启用Int8前向();
            std::printf("== M4 fp32 vs int8 训练（T=%u，%u 步）==\n", c.T, c.STEPS);
            for (std::uint32_t s = 0; s < c.STEPS; ++s) {
                甲.串行步(s);
                乙.串行步(s);
                std::printf("  step %2u: fp32=%.8f  int8=%.8f  Δ=%.3e\n",
                            s + 1, 甲.损耗()[s], 乙.损耗()[s],
                            乙.损耗()[s] - 甲.损耗()[s]);
            }
            std::printf("== 最终：fp32=%.8f  int8=%.8f ==\n",
                        甲.损耗().back(), 乙.损耗().back());
            调度器.停止();
            return 0;
        }

        if (模式 == "int8") {
            const 配置 c = M3c配置();
            训练器 甲(调度器, c);
            const float fp32loss = 甲.前向损失();
            甲.启用Int8前向();
            const float int8loss = 甲.前向损失();
            std::printf("== M4 前向量化对比（T=%u）==\n", c.T);
            std::printf("  fp32 前向 loss = %.8f\n", fp32loss);
            std::printf("  int8 前向 loss = %.8f  (Δ=%.3e)\n", int8loss, int8loss - fp32loss);
            调度器.停止();
            return 0;
        }

        if (模式 == "apollo") {
            const 配置 c = M3c配置();
            训练器 甲(调度器, c);
            甲.启用Apollo();
            std::printf("== M4 + Apollo（T=%u，%u 步，lr=%.3f）==\n", c.T, c.STEPS, 训练器::lr);
            for (std::uint32_t s = 0; s < c.STEPS; ++s) {
                甲.串行步(s);
                const auto& 历史 = 甲.损耗();
                std::printf("  step %u: loss=%.8f\n", s + 1, 历史[s]);
            }
            const auto& 历史 = 甲.损耗();
            std::printf("== Apollo 训练完成，最终 loss=%.6f ==\n", 历史.back());
            调度器.停止();
            return 0;
        }

        if (模式 == "bench") {
            const std::uint32_t T = (argc > 3) ? static_cast<std::uint32_t>(std::atoi(argv[3])) : 64u;
            const std::uint32_t 步数 = (argc > 4) ? static_cast<std::uint32_t>(std::atoi(argv[4])) : 20u;
            const std::uint32_t 每串 = (argc > 5) ? static_cast<std::uint32_t>(std::atoi(argv[5])) : T;
            if (T % 8 != 0) { std::fprintf(stderr, "T 必须是 8 的倍数（BLOCK=8）\n"); return 2; }
            const 配置 c = 基准配置(T, 步数);

            std::printf("== M4 基准 ==\n");
            std::printf("配置 T=%u DX=%u DK=%u DV=%u BLOCK=%u NB=%u TOPB=%u WINDOW=%u tau=%.3f 步数=%u\n",
                        c.T, c.DX, c.DK, c.DV, c.BLOCK, c.NB, c.TOPB, c.WINDOW, c.tau, c.STEPS);

            const double 单次dispatch = 测dispatch开销(调度器, 200);
            std::printf("dispatch 开销（adamw_update size=1，单次提交+等待）中位 %.4f ms\n",
                        单次dispatch);

            const 带宽结果 读16k = 测读带宽(16u * 1024u, 400u);
            const 带宽结果 写16k = 测写带宽(16u * 1024u, 400u);
            const 带宽结果 读4m = 测读带宽(4u * 1024u * 1024u, 8u);
            std::printf("映射内存 16KB：读 %.2f MB/s（%.1f µs/次）  写 %.2f MB/s（%.1f µs/次）"
                        " | 4MB 读 %.1f MB/s\n",
                        读16k.兆字节每秒, 读16k.每次微秒,
                        写16k.兆字节每秒, 写16k.每次微秒, 读4m.兆字节每秒);

            // 串行基线
            训练器 甲(调度器, c);
            for (std::uint32_t s = 0; s < 5; ++s) 甲.串行步(s);   // 预热
            甲.重置();
            std::vector<double> 串行样本;
            串行样本.reserve(步数);
            for (std::uint32_t s = 0; s < 步数; ++s) {
                const auto t0 = std::chrono::steady_clock::now();
                甲.串行步(s);
                const auto t1 = std::chrono::steady_clock::now();
                串行样本.push_back(毫秒差(t0, t1));
            }
            const 计时结果 串行 = 汇总(串行样本);
            const std::vector<float> 串行loss = 甲.损耗();
            打印计时("串行(逐算子等待)", 串行, c.T);

            // 自一致：同一路径再跑一遍，先判断内核本身是否逐位可复现
            甲.重置();
            for (std::uint32_t s = 0; s < 步数; ++s) 甲.串行步(s);
            const std::vector<float> 串行loss2 = 甲.损耗();
            float 自差 = 0.0f;
            for (std::size_t i = 0; i < 串行loss.size() && i < 串行loss2.size(); ++i)
                自差 = std::max(自差, std::fabs(串行loss[i] - 串行loss2[i]));
            std::printf("自一致：串行跑两遍 loss 最大差 %.3e（%s）\n",
                        自差, 自差 == 0.0f ? "逐位可复现" : "同一路径也不可复现，属内核/驱动级非确定性");

            // 融合路径 A：逐 query 反传，块大小取 1 / 每串 / T
            const std::vector<std::uint32_t> 块候选 = [&] {
                std::vector<std::uint32_t> v;
                v.push_back(1u);
                if (每串 > 1u && 每串 < c.T) v.push_back(每串);
                v.push_back(c.T);
                std::sort(v.begin(), v.end());
                v.erase(std::unique(v.begin(), v.end()), v.end());
                return v;
            }();
            std::vector<float> 逐queryloss;
            for (std::uint32_t 块 : 块候选) {
                甲.重置();
                for (std::uint32_t s = 0; s < 5; ++s) 甲.融合步(s, 块, false);
                甲.重置();
                分段耗时 分段;
                if (块 == c.T) 甲.计时 = &分段;   // 只给整块（最快路径）做分段
                std::vector<double> 样本;
                样本.reserve(步数);
                for (std::uint32_t s = 0; s < 步数; ++s) {
                    const auto t0 = std::chrono::steady_clock::now();
                    甲.融合步(s, 块, false);
                    const auto t1 = std::chrono::steady_clock::now();
                    样本.push_back(毫秒差(t0, t1));
                }
                甲.计时 = nullptr;
                const 计时结果 结果 = 汇总(std::move(样本));
                const std::string 标签 = "融合逐q(块=" + std::to_string(块) + ")";
                打印计时(标签.c_str(), 结果, c.T);
                if (块 == c.T) {
                    逐queryloss = 甲.损耗();
                    std::printf("  相对串行加速 %.2f×\n",
                                串行.中位 > 0.0 ? 串行.中位 / 结果.中位 : 0.0);
                    const double 反传 = 分段.反传准备 + 分段.反传提交 + 分段.反传收集;
                    const double 总 = 分段.前向 + 分段.主机 + 反传 + 分段.更新;
                    const double 分母 = 步数 > 0 ? double(步数) : 1.0;
                    std::printf("  分段均值：前向 %.3f  主机 %.3f  反传 %.3f（准备 %.3f + 提交 %.3f + 回收 %.3f）"
                                "  更新 %.3f ms/步  [合计 %.3f / 中位 %.3f]\n",
                                分段.前向 / 分母, 分段.主机 / 分母, 反传 / 分母,
                                分段.反传准备 / 分母, 分段.反传提交 / 分母, 分段.反传收集 / 分母,
                                分段.更新 / 分母, 总 / 分母, 结果.中位);
                }
            }

            // 融合路径 B：批式反传（一次 dispatch 扫全部 query，不再回读逐 query 梯度）
            甲.重置();
            for (std::uint32_t s = 0; s < 5; ++s) 甲.融合步(s, c.T, true);
            甲.重置();
            分段耗时 批式分段;
            甲.计时 = &批式分段;
            std::vector<double> 批式样本;
            批式样本.reserve(步数);
            for (std::uint32_t s = 0; s < 步数; ++s) {
                const auto t0 = std::chrono::steady_clock::now();
                甲.融合步(s, c.T, true);
                const auto t1 = std::chrono::steady_clock::now();
                批式样本.push_back(毫秒差(t0, t1));
            }
            甲.计时 = nullptr;
            const 计时结果 批式 = 汇总(std::move(批式样本));
            打印计时("融合批式", 批式, c.T);
            std::printf("  相对串行加速 %.2f×\n",
                        串行.中位 > 0.0 ? 串行.中位 / 批式.中位 : 0.0);
            {
                const double 反传 = 批式分段.反传准备 + 批式分段.反传提交 + 批式分段.反传收集;
                const double 总 = 批式分段.前向 + 批式分段.主机 + 反传 + 批式分段.更新;
                const double 分母 = 步数 > 0 ? double(步数) : 1.0;
                std::printf("  分段均值：前向 %.3f  主机 %.3f  反传 %.3f（准备 %.3f + 提交 %.3f + 回收 %.3f）"
                            "  更新 %.3f ms/步  [合计 %.3f / 中位 %.3f]\n",
                            批式分段.前向 / 分母, 批式分段.主机 / 分母, 反传 / 分母,
                            批式分段.反传准备 / 分母, 批式分段.反传提交 / 分母, 批式分段.反传收集 / 分母,
                            批式分段.更新 / 分母, 总 / 分母, 批式.中位);
            }
            const std::vector<float> 批式loss = 甲.损耗();

            // 一致性：逐 query 路径与串行同一个内核，应逐位一致；批式换了求和顺序，按容差
            auto 最大差 = [](const std::vector<float>& a, const std::vector<float>& b) {
                float m = 0.0f;
                for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
                    m = std::max(m, std::fabs(a[i] - b[i]));
                return m;
            };
            const float d逐位 = 最大差(串行loss, 逐queryloss);
            const float d批式 = 最大差(串行loss, 批式loss);
            std::printf("一致性：串行 vs 融合逐q loss 最大差 %.3e %s\n",
                        d逐位, d逐位 <= 1e-6f ? "PASS（链路本身 ~1e-7 非确定性）" : "FAIL");
            std::printf("一致性：串行 vs 融合批式 loss 最大差 %.3e %s\n",
                        d批式, d批式 <= 2e-4f ? "PASS（不同求和顺序）" : "FAIL");
            调度器.停止();
            return (d逐位 <= 1e-6f && d批式 <= 2e-4f) ? 0 : 1;
        }

        std::fprintf(stderr, "未知模式: %s\n", 模式.c_str());
        调度器.停止();
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "运行失败: %s\n", e.what());
        调度器.停止();
        return 1;
    }
}
