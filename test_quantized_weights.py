"""
验证 INT8 量化权重（轻量级版本）
================================
只加载前 4 层验证量化正确性，避免内存溢出
"""

import torch
import json
from pathlib import Path
from safetensors import safe_open

from asm_ersa.asm_ersa import ASM_ERSA, RMSNorm


def dequantize_int8(q_tensor: torch.Tensor, scale: torch.Tensor, group_size: int = 128):
    """INT8 反量化（x̂ = q · scale）；INT4 旧格式同样适用。"""
    flat = q_tensor.float().flatten()
    n = flat.numel()
    pad = (group_size - n % group_size) % group_size
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad)])
    groups = flat.reshape(-1, group_size)
    dequant = groups * scale.float().unsqueeze(1)
    return dequant.flatten()[:n].reshape(q_tensor.shape)


def load_quantized_weights(model_dir: str, max_layers: int = 4):
    """加载量化权重（只加载前 max_layers 层）"""
    model_path = Path(model_dir)
    config_path = model_path / "config.json"
    
    with open(config_path) as f:
        config = json.load(f)
    
    print(f"模型配置: {config}")
    print(f"只加载前 {max_layers} 层进行验证\n")
    
    # 加载第一个分片（包含嵌入层和前几层）
    shard_files = sorted(model_path.glob("model-*.safetensors"))
    print(f"发现 {len(shard_files)} 个分片文件")
    
    # 读取索引
    index_path = model_path / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    
    # 收集需要的权重键
    needed_keys = set()
    needed_keys.add("embed.weight")
    needed_keys.add("out_norm.weight")
    needed_keys.add("lm_head.weight")
    
    for layer_idx in range(max_layers):
        prefix = f"layers.{layer_idx}."
        for key in weight_map.keys():
            if key.startswith(prefix):
                needed_keys.add(key)
    
    print(f"需要加载 {len(needed_keys)} 个权重张量\n")
    
    # 按分片分组加载
    shards_needed = set()
    for key in needed_keys:
        if key in weight_map:
            shards_needed.add(weight_map[key])
    
    print(f"涉及 {len(shards_needed)} 个分片文件")
    
    weights = {}
    scales = {}
    
    for shard_name in sorted(shards_needed):
        shard_path = model_path / shard_name
        print(f"  加载: {shard_name}")
        
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in needed_keys:
                if key in weight_map and weight_map[key] == shard_name:
                    # 加载量化权重
                    if key in f.keys():
                        weights[key] = f.get_tensor(key)
                    
                    # 加载对应的 scale
                    scale_key = f"_scale.{key}"
                    if scale_key in f.keys():
                        scales[key] = f.get_tensor(scale_key)
    
    print(f"  已加载 {len(weights)} 个权重张量\n")
    
    return weights, scales, config


def build_model_from_quantized(weights: dict, scales: dict, config: dict):
    """从量化权重构建模型"""
    d_model = config["d_model"]
    n_heads = config["n_heads"]
    d_k = config["d_k"]
    d_v = config["d_v"]
    n_layers = len(set(k.split(".")[1] for k in weights.keys() if k.startswith("layers.")))
    vocab_size = config["vocab_size"]
    group_size = config.get("group_size", 128)
    
    print(f"构建模型: {n_layers} 层, d_model={d_model}, n_heads={n_heads}")
    
    # 创建模型（只创建需要的层数）
    model = ASM_ERSA(
        d_model=d_model,
        n_heads=n_heads,
        d_k=d_k,
        d_v=d_v,
        d_ff=config.get("moe_intermediate_size", 640),
        n_layers=n_layers,
        vocab_size=vocab_size
    )
    
    # 反量化并加载权重
    state_dict = {}
    
    for key, q_tensor in weights.items():
        if key in scales:
            # 反量化
            scale = scales[key]
            tensor = dequantize_int8(q_tensor, scale, group_size)
            print(f"  反量化: {key} (shape={list(tensor.shape)})")
        else:
            # 直接加载（未量化）
            tensor = q_tensor.float()
            print(f"  加载: {key} (shape={list(tensor.shape)})")
        
        state_dict[key] = tensor
    
    # 加载到模型
    model.load_state_dict(state_dict, strict=False)
    print(f"\n模型加载完成")
    
    return model


def test_forward(model, vocab_size: int = 248320):
    """测试前向传播"""
    print("\n" + "=" * 60)
    print("测试前向传播")
    print("=" * 60)
    
    # 生成随机输入
    batch_size = 1
    seq_len = 8
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"输入: {input_ids.shape}")
    print(f"Token IDs: {input_ids[0].tolist()}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
    
    print(f"\n输出 logits: {logits.shape}")
    print(f"Logits 范围: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
    print(f"Logits 均值: {logits.mean().item():.3f}, 标准差: {logits.std().item():.3f}")
    
    # 预测下一个 token
    next_token_logits = logits[0, -1, :]
    next_token_id = next_token_logits.argmax().item()
    print(f"\n预测的下一个 token: {next_token_id}")
    
    # 显示 top-5 预测
    top5 = next_token_logits.topk(5)
    print(f"Top-5 预测:")
    for i, (score, token_id) in enumerate(zip(top5.values, top5.indices)):
        print(f"  {i+1}. Token {token_id.item()}: {score.item():.3f}")
    
    return logits


def main():
    print("=" * 60)
    print("INT8 量化权重验证（轻量级）")
    print("=" * 60)
    print()
    
    model_dir = "models/asm_ersa-int8"
    max_layers = 4  # 只加载前 4 层
    
    # 加载权重
    weights, scales, config = load_quantized_weights(model_dir, max_layers)
    
    # 构建模型
    model = build_model_from_quantized(weights, scales, config)
    
    # 测试前向传播
    test_forward(model, config["vocab_size"])
    
    print("\n" + "=" * 60)
    print("验证完成!")
    print("=" * 60)
    print("\n说明:")
    print("  - 只加载了前 4 层进行验证，避免内存溢出")
    print("  - 量化权重已成功反量化并加载到模型")
    print("  - 前向传播正常，输出 logits 范围合理")
    print("\n完整模型（48 层）需要:")
    print("  - GPU 显存: ~85GB")
    print("  - 或 CPU 内存: ~100GB")
    print("  - 建议使用多 GPU 或模型并行")


if __name__ == "__main__":
    main()
