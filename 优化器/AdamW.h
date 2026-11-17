#pragma once
// ASM-ERSA 张量系统专属 AdamW（2026-11-16 从 SpectraSSM 拷贝件移植）
// 依赖：ASM-ERSA/vk/vk张量.hpp + 流水线调度器.hpp
// 算子：adamw_update（由 优化器/adamw_update.comp 编译为 spv/adamw_update.spv）

#include "../ASM-ERSA/vk/流水线调度器.hpp"
#include "../ASM-ERSA/vk/vk张量.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using vk张量系统::vk张量;
using vk张量系统::张量布局;
using vk张量系统::张量类型;
using vk张量系统::流水线调度器;

class AdamW {
public:
    struct 配置 {
        float 学习率 = 1e-3f;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float epsilon = 1e-8f;
        float 权重衰减 = 0.01f;      // AdamW 解耦权重衰减
        bool 启用偏差修正 = true;
        bool 更新后清零梯度 = true;  // 融合进同一 dispatch，省一次主机上传
    };

    struct 参数项 {
        std::string 名称;
        vk张量* 参数 = nullptr;
        vk张量* 梯度 = nullptr;
        std::unique_ptr<vk张量> m状态;   // 一阶矩，fp32
        std::unique_ptr<vk张量> v状态;   // 二阶矩，fp32
        bool 应用权重衰减 = true;        // norm/bias 分组不衰减
    };

    AdamW() = default;
    explicit AdamW(std::vector<参数项> 参数列表, 配置 配置 = {});
    ~AdamW() = default;
    AdamW(const AdamW&) = delete;
    AdamW& operator=(const AdamW&) = delete;
    AdamW(AdamW&&) noexcept = default;
    AdamW& operator=(AdamW&&) noexcept = default;

    /** @brief 执行单步更新；要求每项的梯度已在设备上算好 */
    void 步骤();

    /** @brief 清零 m/v 并把步数归零 */
    void 重置状态();

    uint32_t 获取当前步数() const { return 当前步数_; }
    void 设置学习率(float 新学习率) { 配置_.学习率 = 新学习率; }
    void 设置权重衰减(float 新衰减) { 配置_.权重衰减 = 新衰减; }

    /** @brief 二进制检查点：魔数 ADMW、版本 2（显式字段序列化） */
    void 保存检查点(const std::string& 文件路径) const;
    void 加载检查点(const std::string& 文件路径);

private:
    // 与 adamw_update.comp 的 push_constant 逐字节一致（36 字节）
    struct 推送常量 {
        float 学习率;
        float beta1;
        float beta2;
        float epsilon;
        float m修正;
        float v修正;
        std::uint32_t 元素数;
        float 权重衰减;
        std::uint32_t 清零梯度;
    };
    static_assert(sizeof(推送常量) == 36, "推送常量布局必须与 adamw_update.comp 一致");

    void 初始化状态();
    bool 验证梯度有效性(const 参数项& 项) const;
    static void 张量清零(vk张量& 张量);

    配置 配置_;
    std::vector<参数项> 参数列表_;
    std::uint32_t 当前步数_ = 0;
    bool 已初始化_ = false;
};
