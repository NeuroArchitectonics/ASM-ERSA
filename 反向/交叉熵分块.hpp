#pragma once
// 分块交叉熵：任意词表大小下，单次 dispatch 的词表宽度和行数都不超过 64。
// 走两张新内核：
//   ce_stats_chunk64   —— 每行在 64 维词表块上合并 online max/sum，最后一块出 loss
//   ce_dlogits_chunk64 —— 用最终 max/sum 逐块写 dlogits
// 这样 780M 上不会出现「一个线程扫完整 248k 词表」的长 dispatch。
#include "流水线调度器.hpp"
#include "vk张量.hpp"
#include "ce_stats_chunk64.spv.hpp"
#include "ce_dlogits_chunk64.spv.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace vk张量系统 {

struct 分块交叉熵配置 {
    std::uint32_t 行块 = 64;   // 每次 dispatch 最多 64 条嵌入（token）
    std::uint32_t 词块 = 64;   // 每次 dispatch 最多 64 个词表维度
};

class 分块交叉熵 {
public:
    分块交叉熵(std::uint32_t 行数, std::uint32_t 词表大小,
               const std::string& 名称前缀 = "交叉熵")
        : 行数_(行数), 词表大小_(词表大小),
          row_max_(张量布局::连续({行数}), 张量类型::浮点32, 名称前缀 + ".row_max"),
          row_sum_(张量布局::连续({行数}), 张量类型::浮点32, 名称前缀 + ".row_sum"),
          target_logit_(张量布局::连续({行数}), 张量类型::浮点32, 名称前缀 + ".target_logit"),
          loss_(张量布局::连续({行数}), 张量类型::浮点32, 名称前缀 + ".loss"),
          dlogits_(张量布局::连续(校验形状(行数, 词表大小)), 张量类型::浮点32, 名称前缀 + ".dlogits") {
        if (行数 == 0 || 词表大小 == 0)
            throw std::invalid_argument("分块交叉熵：行数和词表大小必须大于 0");
    }

    // logits [R,V] F32；targets [R] I32，-100 表示忽略该行（prompt / padding）。
    // 返回时任务已提交，取平均损失或读 dlogits 会自动等待。
    void 计算(const vk张量& logits, const vk张量& targets,
              const 分块交叉熵配置& 配置 = {}) {
        校验配置(配置);
        if (logits.类型() != 张量类型::浮点32)
            throw std::invalid_argument("分块交叉熵：logits 必须是 float32");
        if (targets.类型() != 张量类型::整数32)
            throw std::invalid_argument("分块交叉熵：targets 必须是 int32");
        if (logits.布局().元素总数 != 行数_ * 词表大小_)
            throw std::invalid_argument("分块交叉熵：logits 元素数 != R*V");
        if (targets.布局().元素总数 != 行数_)
            throw std::invalid_argument("分块交叉熵：targets 元素数 != R");

        // 本轮要覆盖上一轮写的状态；挂任务只在上一个任务落地后才换新任务，
        // 这里先显式等到上一轮结束，避免读写同一块状态缓冲。
        logits.同步();
        targets.同步();
        row_max_.同步();
        row_sum_.同步();
        target_logit_.同步();
        loss_.同步();
        dlogits_.同步();

        流水线调度器& 调度器 = 流水线调度器::获取();
        const std::uint32_t 行块 = std::min(配置.行块, 行数_);
        const std::uint32_t 词块 = std::min(配置.词块, 词表大小_);
        const std::uint32_t 词块数 = (词表大小_ + 词块 - 1) / 词块;

        std::uint64_t 上一编号 = 0;
        for (std::uint32_t r0 = 0; r0 < 行数_; r0 += 行块) {
            const std::uint32_t 本块行数 = std::min(行块, 行数_ - r0);
            std::vector<算子步骤> 步骤;
            步骤.reserve(词块数 * 2);

            for (std::uint32_t v0 = 0; v0 < 词表大小_; v0 += 词块) {
                const std::uint32_t vn = std::min(词块, 词表大小_ - v0);
                const 规格ce 规格{行数_, 词表大小_, v0, vn, r0, 本块行数, 0, 0};
                步骤.push_back(构造算子步骤(
                    "ce_stats_chunk64",
                    {logits.取句柄(), targets.取句柄(), row_max_.取句柄(),
                     row_sum_.取句柄(), target_logit_.取句柄(), loss_.取句柄()},
                    规格, 派发规格{(本块行数 + 63u) / 64u, 1u, 1u}));
            }

            for (std::uint32_t v0 = 0; v0 < 词表大小_; v0 += 词块) {
                const std::uint32_t vn = std::min(词块, 词表大小_ - v0);
                const 规格ce 规格{行数_, 词表大小_, v0, vn, r0, 本块行数, 0, 0};
                步骤.push_back(构造算子步骤(
                    "ce_dlogits_chunk64",
                    {logits.取句柄(), targets.取句柄(), row_max_.取句柄(),
                     row_sum_.取句柄(), dlogits_.取句柄()},
                    规格, 派发规格{本块行数, 1u, 1u}));
            }

            std::vector<std::uint64_t> 依赖;
            if (上一编号 != 0) 依赖.push_back(上一编号);
            上一编号 = 调度器.提交算子串(std::move(步骤), 依赖);
        }

        if (上一编号 != 0) {
            // 多个行块是链式依赖，最后一个编号就代表全部块落地。
            row_max_.挂任务(上一编号);
            row_sum_.挂任务(上一编号);
            target_logit_.挂任务(上一编号);
            loss_.挂任务(上一编号);
            dlogits_.挂任务(上一编号);
            已计算_ = true;
        }
    }

    // 有效行（target>=0）的逐行 loss 平均；会等待 CE 任务完成。
    float 取平均损失(const std::vector<std::int32_t>& targets) {
        if (!已计算_) throw std::logic_error("分块交叉熵：计算() 尚未提交");
        if (targets.size() != 行数_)
            throw std::invalid_argument("分块交叉熵：targets 大小与 R 不一致");
        std::vector<float> 主机(行数_);
        loss_.下载(主机.data(), 主机.size() * sizeof(float));
        double 和 = 0.0;
        std::uint32_t 有效 = 0;
        for (std::uint32_t i = 0; i < 行数_; ++i) {
            if (targets[i] >= 0) {
                和 += 主机[i];
                ++有效;
            }
        }
        return 有效 ? static_cast<float>(和 / 有效) : 0.0f;
    }

    void 下载损失(std::vector<float>& 输出) const {
        if (!已计算_) throw std::logic_error("分块交叉熵：计算() 尚未提交");
        输出.resize(行数_);
        loss_.下载(输出.data(), 输出.size() * sizeof(float));
    }

    void 下载dlogits(std::vector<float>& 输出) const {
        if (!已计算_) throw std::logic_error("分块交叉熵：计算() 尚未提交");
        输出.resize(static_cast<std::size_t>(行数_) * 词表大小_);
        dlogits_.下载(输出.data(), 输出.size() * sizeof(float));
    }

    const vk张量& dlogits() const { return dlogits_; }
    const vk张量& loss() const { return loss_; }
    std::uint32_t 行数() const { return 行数_; }
    std::uint32_t 词表大小() const { return 词表大小_; }

private:
    // 与 ce_*_chunk64.comp 的 push_constant 逐字节一致（8 x uint32 = 32 B）
    struct 规格ce {
        std::uint32_t R;
        std::uint32_t V;
        std::uint32_t v0;
        std::uint32_t vn;
        std::uint32_t r0;
        std::uint32_t rn;
        std::uint32_t pad1;
        std::uint32_t pad2;
    };
    static_assert(sizeof(规格ce) == 32, "CE 推送常量必须 32 字节");

    static std::vector<std::uint32_t> 校验形状(std::uint32_t 行数, std::uint32_t 词表大小) {
        if (行数 == 0 || 词表大小 == 0)
            throw std::invalid_argument("分块交叉熵：行数和词表大小必须大于 0");
        if (static_cast<std::uint64_t>(行数) * 词表大小 >
            std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument("分块交叉熵：R*V 超出张量系统的 uint32 元素上限");
        return {行数, 词表大小};
    }

    static void 校验配置(const 分块交叉熵配置& 配置) {
        if (配置.行块 == 0 || 配置.行块 > 64)
            throw std::invalid_argument("分块交叉熵：行块必须在 1..64");
        if (配置.词块 == 0 || 配置.词块 > 64)
            throw std::invalid_argument("分块交叉熵：词块必须在 1..64");
    }

    std::uint32_t 行数_;
    std::uint32_t 词表大小_;
    vk张量 row_max_;
    vk张量 row_sum_;
    vk张量 target_logit_;
    vk张量 loss_;
    vk张量 dlogits_;
    bool 已计算_ = false;
};

} // namespace vk张量系统
