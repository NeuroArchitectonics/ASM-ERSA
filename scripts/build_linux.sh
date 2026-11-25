#!/usr/bin/env bash
# Linux 构建（Cloud Studio / Ubuntu 22.04+）。与 Windows 的 build_train.bat 等价。
# 用法：在仓库根目录执行  bash scripts/build_linux.sh
set -euo pipefail
cd "$(dirname "$0")/.."

need() { command -v "$1" >/dev/null 2>&1; }

if ! need g++; then
    sudo apt-get update
    sudo apt-get install -y build-essential
fi
if ! ldconfig -p | grep -q 'libvulkan\.so'; then
    sudo apt-get install -y libvulkan-dev vulkan-tools
fi
if ! need glslangValidator; then
    sudo apt-get install -y glslang-tools spirv-tools
fi

# cpp/spv 里已有 Windows 侧编好的 .spv（SPIR-V 与平台无关），缺的才重编
if [ ! -f cpp/spv/apollo_mini_apply_i8.spv ]; then
    for f in ASM-ERSA-Design/vk/glsl_dev/*.comp; do
        out="cpp/spv/$(basename "${f%.comp}").spv"
        glslangValidator -V "$f" -o "$out"
    done
fi

mkdir -p cpp/build
g++ -std=c++20 -O2 -pthread -o cpp/build/train_main_linux \
    cpp/train_main.cpp cpp/asm_ersa_model.cpp cpp/cpu_reference.cpp \
    优化器/ApolloMini.cpp 优化器/ApolloMiniI8.cpp \
    -I. -Icpp -Icpp/spv -Ispv -I优化器 \
    -Ivk \
    -lvulkan

echo "[build] OK -> cpp/build/train_main_linux"
echo "先确认 Vulkan 设备： vulkaninfo --summary | head -30"
echo "3B 单步冒烟：       ./cpp/build/train_main_linux cpp/spv 3b_cache"
echo "3B 训练 10 步：     ASM_OPT=apollo_mini ASM_ADAMW_LR=1e-4 ASM_STEPS=10 ./cpp/build/train_main_linux cpp/spv 3b_train"
echo "3B INT8 直写：      ASM_OPT=apollo_mini ASM_PARAM=i8 ASM_ADAMW_LR=1e-4 ASM_STEPS=10 ./cpp/build/train_main_linux cpp/spv 3b_train"
