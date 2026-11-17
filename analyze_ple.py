"""
PLE 架构分析
============
分析 PLE 组件的结构和工作原理
"""

import torch
from pathlib import Path
from safetensors import safe_open
import json
import os

src_dir = Path(os.environ.get("ASM_ERSA_SRC", "models/base-model"))

# 加载 config
with open(src_dir / "config.json") as f:
    config = json.load(f)

text_config = config.get("text_config", {})

print("="*60)
print("PLE 相关配置")
print("="*60)

ple_keys = [
    "ple_conv_kernel_size",
    "ple_embed_dim",
    "ple_layer_ids",
    "ngram_size",
    "heads_per_ngram",
    "split_ngram_parts",
    "ngram_vocab_size_base",
    "make_ngram_vocab_size_divisible_by",
    "linear_num_key_heads",
    "linear_key_head_dim",
    "linear_num_value_heads",
    "linear_value_head_dim",
]

for key in ple_keys:
    if key in text_config:
        print(f"  {key}: {text_config[key]}")

# 加载层 1 的 PLE 权重
print("\n" + "="*60)
print("层 1 PLE 权重")
print("="*60)

shard_files = sorted(src_dir.glob("*.safetensors"))
ple_keys_found = {}

for shard in shard_files:
    with safe_open(str(shard), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for key in keys:
            if "layers.1.ple." in key:
                tensor = f.get_tensor(key)
                ple_keys_found[key] = (list(tensor.shape), str(tensor.dtype))

for key in sorted(ple_keys_found.keys()):
    shape, dtype = ple_keys_found[key]
    short_key = key.replace("model.language_model.layers.1.ple.", "")
    print(f"  {short_key:60s}  {shape}  {dtype}")

# 分析 ngram 嵌入结构
print("\n" + "="*60)
print("Ngram 嵌入结构分析")
print("="*60)

ngram_size = text_config.get("ngram_size", 3)
heads_per_ngram = text_config.get("heads_per_ngram", 8)
split_ngram_parts = text_config.get("split_ngram_parts", 128)
ngram_vocab_size_base = text_config.get("ngram_vocab_size_base", 20000000)
make_divisible_by = text_config.get("make_ngram_vocab_size_divisible_by", 128)

print(f"ngram_size: {ngram_size} (unigram, bigram, trigram)")
print(f"heads_per_ngram: {heads_per_ngram}")
print(f"split_ngram_parts: {split_ngram_parts}")
print(f"ngram_vocab_size_base: {ngram_vocab_size_base:,}")
print(f"make_divisible_by: {make_divisible_by}")

# 计算每个 shard 的 vocab size
shard_vocab = 2500012  # from the shape we saw earlier
total_vocab = split_ngram_parts * shard_vocab
print(f"\n每个 shard 的 vocab size: {shard_vocab:,}")
print(f"总 vocab size: {total_vocab:,} ({split_ngram_parts} × {shard_vocab:,})")

# 加载 layer_multipliers, ngram_heads_offsets, ngram_heads_vocab_sizes
print("\n" + "="*60)
print("PLE 元数据")
print("="*60)

for shard in shard_files:
    with safe_open(str(shard), framework="pt", device="cpu") as f:
        keys = list(f.keys())

        if "model.language_model.layers.1.ple.ple_embedding.layer_multipliers" in keys:
            lm = f.get_tensor("model.language_model.layers.1.ple.ple_embedding.layer_multipliers")
            print(f"layer_multipliers: {lm.tolist()} (shape: {list(lm.shape)}, dtype: {lm.dtype})")

        if "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets" in keys:
            offsets = f.get_tensor("model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets")
            print(f"ngram_heads_offsets: {offsets.tolist()} (shape: {list(offsets.shape)}, dtype: {offsets.dtype})")

        if "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes" in keys:
            vocab_sizes = f.get_tensor("model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes")
            print(f"ngram_heads_vocab_sizes: {vocab_sizes.tolist()} (shape: {list(vocab_sizes.shape)}, dtype: {vocab_sizes.dtype})")

    if all(k in [k2 for k2 in keys] for k in [
        "model.language_model.layers.1.ple.ple_embedding.layer_multipliers",
        "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets",
        "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes"
    ]):
        break

print("\n" + "="*60)
print("推测的 PLE 工作原理")
print("="*60)

print("""
基于组件形状和配置，推测 PLE 的工作流程：

1. Ngram Hash 计算:
   - 对于每个 token 位置 t，计算 n-gram hash（n=1,2,3）
   - unigram: hash(token[t])
   - bigram: hash(token[t-1], token[t])
   - trigram: hash(token[t-2], token[t-1], token[t])

2. Embedding 查找:
   - 128 个 shard，每个 2.5M × 160 维
   - 根据 hash 值选择 shard 和 row
   - 16 个头，每个头有独立的 offset 和 vocab_size

3. Conv1d:
   - 输入: 10240 维（可能是 16 头 × 160 维 × 4 kernel_size）
   - 深度卷积，kernel=4
   - 输出: 10240 维

4. 投影:
   - key_proj: (10240, 2560) → 生成 keys
   - value_proj: (2560, 2560) → 生成 values

5. 集成到注意力:
   - PLE 生成的 keys/values 可能与线性注意力的 keys/values 拼接
   - 或者加到输入 embedding 上

需要进一步分析源模型代码来确认。
""")
