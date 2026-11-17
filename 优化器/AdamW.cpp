#include "AdamW.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

// 检查点文件魔数与版本：格式与 SpectraSSM 拷贝件不兼容（字段显式序列化）
constexpr std::uint32_t 检查点魔数 = 0x41444D57;  // "ADMW"
constexpr std::uint32_t 检查点版本 = 2;

AdamW::AdamW(std::vector<参数项> 参数列表, 配置 配置)
    : 配置_(配置), 参数列表_(std::move(参数列表)) {
    if (参数列表_.empty()) {
        throw std::runtime_error("AdamW: 参数列表为空");
    }
    初始化状态();
}

void AdamW::张量清零(vk张量& 张量) {
    std::vector<float> 零(张量.布局().元素总数, 0.0f);
    张量.上传(零.data(), 零.size() * sizeof(float));
}

void AdamW::初始化状态() {
    参数列表_.erase(
        std::remove_if(参数列表_.begin(), 参数列表_.end(),
                       [](const 参数项& 项) { return 项.参数 == nullptr; }),
        参数列表_.end());

    for (auto& 项 : 参数列表_) {
        if (项.梯度 == nullptr) {
            throw std::runtime_error("AdamW: 参数 " + 项.名称 + " 未提供梯度张量");
        }
        if (!项.参数->有效() || !项.梯度->有效()) {
            throw std::runtime_error("AdamW: 参数或梯度张量无效: " + 项.名称);
        }
        if (!项.参数->布局().同形状(项.梯度->布局())) {
            throw std::runtime_error("AdamW: 参数与梯度形状不一致: " + 项.名称);
        }
        if (项.参数->类型() != 张量类型::浮点32) {
            throw std::runtime_error("AdamW: 仅支持 fp32 参数: " + 项.名称);
        }

        项.m状态 = std::make_unique<vk张量>(
            项.参数->布局(), 张量类型::浮点32, 项.名称 + "_m");
        项.v状态 = std::make_unique<vk张量>(
            项.参数->布局(), 张量类型::浮点32, 项.名称 + "_v");
        张量清零(*项.m状态);
        张量清零(*项.v状态);

        // 参数分组：归一化层与偏置不做解耦权重衰减
        const bool 是归一化 = 项.名称.find("norm") != std::string::npos ||
                              项.名称.find("Norm") != std::string::npos;
        const bool 是偏置 = 项.名称.find("bias") != std::string::npos;
        项.应用权重衰减 = !(是归一化 || 是偏置);
    }

    当前步数_ = 0;
    已初始化_ = true;
}

bool AdamW::验证梯度有效性(const 参数项& 项) const {
    return 项.参数 != nullptr && 项.梯度 != nullptr &&
           项.参数->有效() && 项.梯度->有效();
}

void AdamW::步骤() {
    if (!已初始化_) {
        throw std::runtime_error("AdamW: 优化器未初始化");
    }

    当前步数_++;
    const float beta1_t = std::pow(配置_.beta1, static_cast<float>(当前步数_));
    const float beta2_t = std::pow(配置_.beta2, static_cast<float>(当前步数_));

    for (auto& 项 : 参数列表_) {
        if (!验证梯度有效性(项)) continue;

        推送常量 推送{
            配置_.学习率,
            配置_.beta1,
            配置_.beta2,
            配置_.epsilon,
            配置_.启用偏差修正 ? (1.0f / (1.0f - beta1_t)) : 1.0f,
            配置_.启用偏差修正 ? (1.0f / (1.0f - beta2_t)) : 1.0f,
            static_cast<std::uint32_t>(项.参数->布局().元素总数),
            项.应用权重衰减 ? 配置_.权重衰减 : 0.0f,
            配置_.更新后清零梯度 ? 1u : 0u,
        };

        // 同步录制：返回即已执行完毕（调度器内部提交并等围栏）
        流水线调度器::获取().录制自定义推送算子(
            "adamw_update",
            {项.梯度->取句柄(), 项.参数->取句柄(),
             项.m状态->取句柄(), 项.v状态->取句柄()},
            &推送, sizeof(推送));
    }
}

void AdamW::重置状态() {
    for (auto& 项 : 参数列表_) {
        if (项.m状态) 张量清零(*项.m状态);
        if (项.v状态) 张量清零(*项.v状态);
    }
    当前步数_ = 0;
}

void AdamW::保存检查点(const std::string& 文件路径) const {
    if (!已初始化_) {
        throw std::runtime_error("AdamW: 优化器未初始化，无法保存检查点");
    }

    std::string 临时路径 = 文件路径 + ".tmp";
    std::ofstream 文件流(临时路径, std::ios::binary);
    if (!文件流.is_open()) {
        throw std::runtime_error("无法创建临时检查点文件: " + 临时路径);
    }

    文件流.write(reinterpret_cast<const char*>(&检查点魔数), sizeof(检查点魔数));
    文件流.write(reinterpret_cast<const char*>(&检查点版本), sizeof(检查点版本));

    // 显式序列化超参（不直接写结构体，避免填充字节与 ABI 差异）
    文件流.write(reinterpret_cast<const char*>(&配置_.学习率), sizeof(float));
    文件流.write(reinterpret_cast<const char*>(&配置_.beta1), sizeof(float));
    文件流.write(reinterpret_cast<const char*>(&配置_.beta2), sizeof(float));
    文件流.write(reinterpret_cast<const char*>(&配置_.epsilon), sizeof(float));
    文件流.write(reinterpret_cast<const char*>(&配置_.权重衰减), sizeof(float));
    std::uint32_t 标志 = (配置_.启用偏差修正 ? 1u : 0u) | (配置_.更新后清零梯度 ? 2u : 0u);
    文件流.write(reinterpret_cast<const char*>(&标志), sizeof(标志));
    文件流.write(reinterpret_cast<const char*>(&当前步数_), sizeof(当前步数_));

    std::uint32_t 参数数量 = static_cast<std::uint32_t>(参数列表_.size());
    文件流.write(reinterpret_cast<const char*>(&参数数量), sizeof(参数数量));

    for (const auto& 项 : 参数列表_) {
        if (!项.m状态 || !项.v状态) {
            throw std::runtime_error("AdamW: 优化器状态未初始化，无法保存");
        }
        std::size_t 字节数 = 项.m状态->字节大小();
        if (字节数 != 项.v状态->字节大小()) {
            throw std::runtime_error("AdamW: m/v 状态字节数不一致: " + 项.名称);
        }
        文件流.write(reinterpret_cast<const char*>(&字节数), sizeof(字节数));

        std::vector<float> m数据(项.m状态->布局().元素总数);
        std::vector<float> v数据(项.v状态->布局().元素总数);
        项.m状态->下载(m数据.data(), 字节数);
        项.v状态->下载(v数据.data(), 字节数);
        文件流.write(reinterpret_cast<const char*>(m数据.data()), 字节数);
        文件流.write(reinterpret_cast<const char*>(v数据.data()), 字节数);
    }

    文件流.close();

    if (std::filesystem::exists(文件路径)) {
        std::filesystem::remove(文件路径);
    }
    std::filesystem::rename(临时路径, 文件路径);
}

void AdamW::加载检查点(const std::string& 文件路径) {
    if (!已初始化_) {
        throw std::runtime_error("AdamW: 优化器未初始化，无法加载检查点");
    }

    std::ifstream 文件流(文件路径, std::ios::binary);
    if (!文件流.is_open()) {
        throw std::runtime_error("无法打开检查点文件: " + 文件路径);
    }

    std::uint32_t 读取魔数 = 0, 读取版本 = 0;
    文件流.read(reinterpret_cast<char*>(&读取魔数), sizeof(读取魔数));
    文件流.read(reinterpret_cast<char*>(&读取版本), sizeof(读取版本));
    if (读取魔数 != 检查点魔数) {
        throw std::runtime_error("检查点魔数不匹配，非 AdamW 检查点");
    }
    if (读取版本 != 检查点版本) {
        throw std::runtime_error("检查点版本不兼容: " + std::to_string(读取版本));
    }

    文件流.read(reinterpret_cast<char*>(&配置_.学习率), sizeof(float));
    文件流.read(reinterpret_cast<char*>(&配置_.beta1), sizeof(float));
    文件流.read(reinterpret_cast<char*>(&配置_.beta2), sizeof(float));
    文件流.read(reinterpret_cast<char*>(&配置_.epsilon), sizeof(float));
    文件流.read(reinterpret_cast<char*>(&配置_.权重衰减), sizeof(float));
    std::uint32_t 标志 = 0;
    文件流.read(reinterpret_cast<char*>(&标志), sizeof(标志));
    配置_.启用偏差修正 = (标志 & 1u) != 0;
    配置_.更新后清零梯度 = (标志 & 2u) != 0;
    文件流.read(reinterpret_cast<char*>(&当前步数_), sizeof(当前步数_));

    std::uint32_t 参数数量 = 0;
    文件流.read(reinterpret_cast<char*>(&参数数量), sizeof(参数数量));
    if (参数数量 != 参数列表_.size()) {
        throw std::runtime_error("检查点参数数量与当前模型不匹配！期望: " +
            std::to_string(参数列表_.size()) + ", 实际: " + std::to_string(参数数量));
    }

    for (auto& 项 : 参数列表_) {
        if (!项.m状态 || !项.v状态) {
            throw std::runtime_error("AdamW: 优化器状态未初始化，无法加载");
        }
        std::size_t 字节数 = 0;
        文件流.read(reinterpret_cast<char*>(&字节数), sizeof(字节数));
        if (字节数 != 项.m状态->字节大小() || 字节数 != 项.v状态->字节大小()) {
            throw std::runtime_error("检查点张量形状与当前模型不匹配: " + 项.名称);
        }
        std::vector<float> m数据(项.m状态->布局().元素总数);
        std::vector<float> v数据(项.v状态->布局().元素总数);
        文件流.read(reinterpret_cast<char*>(m数据.data()), 字节数);
        文件流.read(reinterpret_cast<char*>(v数据.data()), 字节数);
        项.m状态->上传(m数据.data(), 字节数);
        项.v状态->上传(v数据.data(), 字节数);
    }
}
