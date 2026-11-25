#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "[build] 此脚本只构建 Linux x86_64" >&2
    exit 1
fi

CXX="${CXX:-g++}"
OUT="dist/linux-x86_64"
OBJ="build/shared-linux-x86_64"

mkdir -p "$OUT/spv" "$OBJ"

"$CXX" -std=c++20 -O2 -fPIC -shared -fvisibility=hidden -pthread \
    -DASM_TRAINER_LIBRARY \
    -I. -Iinclude -Icpp -Icpp/spv -Ispv -I优化器 \
    -IASM-ERSA-Design/vk -IASM-ERSA-Design/vk/spv \
    cpp/train_main.cpp cpp/asm_ersa_model.cpp cpp/cpu_reference.cpp \
    cpp/trainer_bridge.cpp cpp/trainer_library.cpp \
    优化器/ApolloMini.cpp 优化器/ApolloMiniI8.cpp \
    -o "$OUT/libasm_ersa_trainer.so" \
    -lvulkan

cp include/asm_trainer.h "$OUT/asm_trainer.h"
cp bindings/python/asm_trainer.py bindings/python/example_train.py "$OUT/"

for f in spv/*.spv; do
    [[ -e "$f" ]] || continue
    cp -n "$f" "$OUT/spv/"
done
for f in cpp/spv/*.spv; do
    [[ -e "$f" ]] || continue
    cp -n "$f" "$OUT/spv/"
done

if [[ -n "$(find "$OUT" -type f -name '*.comp' -print -quit)" ]]; then
    echo "[build] 拒绝打包 .comp 源码" >&2
    exit 1
fi

(cd "$OUT" && sha256sum libasm_ersa_trainer.so > libasm_ersa_trainer.so.sha256)
ARCHIVE="dist/asm-ersa-trainer-linux-x86_64.tar.gz"
tar -czf "$ARCHIVE" -C "$OUT" .
(cd dist && sha256sum asm-ersa-trainer-linux-x86_64.tar.gz > asm-ersa-trainer-linux-x86_64.tar.gz.sha256)
echo "[build] OK -> $OUT/libasm_ersa_trainer.so"
echo "[build] package -> $ARCHIVE"
