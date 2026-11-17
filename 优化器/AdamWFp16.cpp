#include "AdamWFp16.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
struct 推送 {
    float 学习率;
    float beta1;
    float beta2;
    float epsilon;
    float m修正;
    float v修正;
    float 权重衰减;
    std::uint32_t 元素数;
    std::uint32_t 清零梯度;
};
}  // namespace

AdamWFp16::AdamWFp16(std::size_t 元素数, 配置 配置)
    : 元素数_(元素数),
      配置_(配置),
      p_(张量布局::连续({static_cast<std::uint32_t>(元素数)}), 张量类型::浮点16, "adamw_fp16_p"),
      g_(张量布局::连续({static_cast<std::uint32_t>(元素数)}), 张量类型::浮点16, "adamw_fp16_g"),
      master_(张量布局::连续({static_cast<std::uint32_t>(元素数)}), 张量类型::浮点32, "adamw_fp16_master"),
      m_(张量布局::连续({static_cast<std::uint32_t>(元素数)}), 张量类型::浮点32, "adamw_fp16_m"),
      v_(张量布局::连续({static_cast<std::uint32_t>(元素数)}), 张量类型::浮点32, "adamw_fp16_v") {
    if (元素数_ == 0) throw std::runtime_error("AdamWFp16: 元素数为 0");
    std::vector<float> 零(元素数_, 0.0f);
    m_.上传(零.data(), 零.size() * sizeof(float));
    v_.上传(零.data(), 零.size() * sizeof(float));
}

void AdamWFp16::上传参数(const float* 数据) {
    master_.上传(数据, 元素数_ * sizeof(float));
    std::vector<std::uint16_t> 半(元素数_);
    for (std::size_t i = 0; i < 元素数_; ++i) {
        const float x = 数据[i];
        std::uint32_t u;
        std::memcpy(&u, &x, sizeof(u));
        std::uint32_t sign = (u >> 16) & 0x8000u;
        std::int32_t exp = static_cast<std::int32_t>((u >> 23) & 0xFFu) - 127 + 15;
        std::uint32_t mant = u & 0x7FFFFFu;
        std::uint16_t h;
        if (exp <= 0) {
            h = static_cast<std::uint16_t>(sign);
        } else if (exp >= 31) {
            h = static_cast<std::uint16_t>(sign | 0x7C00u);
        } else {
            h = static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) | (mant >> 13));
        }
        半[i] = h;
    }
    p_.上传(半.data(), 半.size() * sizeof(std::uint16_t));
}

void AdamWFp16::上传梯度(const float* 数据) {
    std::vector<std::uint16_t> 半(元素数_);
    for (std::size_t i = 0; i < 元素数_; ++i) {
        const float x = 数据[i];
        std::uint32_t u;
        std::memcpy(&u, &x, sizeof(u));
        std::uint32_t sign = (u >> 16) & 0x8000u;
        std::int32_t exp = static_cast<std::int32_t>((u >> 23) & 0xFFu) - 127 + 15;
        std::uint32_t mant = u & 0x7FFFFFu;
        std::uint16_t h;
        if (exp <= 0) {
            h = static_cast<std::uint16_t>(sign);
        } else if (exp >= 31) {
            h = static_cast<std::uint16_t>(sign | 0x7C00u);
        } else {
            h = static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) | (mant >> 13));
        }
        半[i] = h;
    }
    g_.上传(半.data(), 半.size() * sizeof(std::uint16_t));
}

void AdamWFp16::步骤() {
    当前步数_++;
    const float beta1_t = std::pow(配置_.beta1, static_cast<float>(当前步数_));
    const float beta2_t = std::pow(配置_.beta2, static_cast<float>(当前步数_));
    const 推送 参数{
        配置_.学习率,
        配置_.beta1,
        配置_.beta2,
        配置_.epsilon,
        配置_.启用偏差修正 ? (1.0f / (1.0f - beta1_t)) : 1.0f,
        配置_.启用偏差修正 ? (1.0f / (1.0f - beta2_t)) : 1.0f,
        配置_.权重衰减,
        static_cast<std::uint32_t>(元素数_),
        配置_.更新后清零梯度 ? 1u : 0u,
    };
    流水线调度器::获取().录制自定义推送算子(
        "adamw_update_fp16",
        {g_.取句柄(), master_.取句柄(), m_.取句柄(), v_.取句柄(), p_.取句柄()},
        &参数, sizeof(参数));
}

void AdamWFp16::下载参数(float* 输出) const {
    std::vector<std::uint16_t> 半(元素数_);
    p_.下载(半.data(), 半.size() * sizeof(std::uint16_t));
    for (std::size_t i = 0; i < 元素数_; ++i) {
        const std::uint16_t h = 半[i];
        const std::uint32_t sign = (h & 0x8000u) << 16;
        const std::uint32_t exp = (h >> 10) & 0x1Fu;
        const std::uint32_t mant = h & 0x3FFu;
        std::uint32_t u;
        if (exp == 0) {
            u = sign;
        } else if (exp == 31) {
            u = sign | 0x7F800000u | (mant << 13);
        } else {
            u = sign | ((exp - 15 + 127) << 23) | (mant << 13);
        }
        float f;
        std::memcpy(&f, &u, sizeof(f));
        输出[i] = f;
    }
}

void AdamWFp16::下载状态(float* m_out, float* v_out) const {
    m_.下载(m_out, 元素数_ * sizeof(float));
    v_.下载(v_out, 元素数_ * sizeof(float));
}
