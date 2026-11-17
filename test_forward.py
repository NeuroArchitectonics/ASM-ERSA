"""
测试单步前向传播
==============
验证推理代码是否正确实现
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from safetensors import safe_open
import json
import gc

def dequant_int8(q_tensor, scale, group_size=128):
    """INT8 反量化（x̂ = q · scale）；INT4 旧格式同样适用。"""
    if q_tensor.dtype != torch.int8:
        return q_tensor.float()
    flat = q_tensor.half().flatten()
    n = flat.numel()
    pad = (group_size - n % group_size) % group_size
    if pad > 0:
        flat = F.pad(flat, (0, pad))
    groups = flat.reshape(-1, group_size)
    dequant = groups * scale.half().unsqueeze(1)
    return dequant.flatten()[:n].reshape(q_tensor.shape).float()

def load_all_weights(model_dir):
    """加载所有权重（包括反量化）"""
    model_path = Path(model_dir)
    with open(model_path / "config.json") as f:
        config = json.load(f)

    group_size = config.get("group_size", 128)
    shard_files = sorted(model_path.glob("model-*.safetensors"))

    all_scales = {}
    for shard_path in shard_files:
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("_scale."):
                    scale = f.get_tensor(key)
                    weight_key = key[7:]
                    all_scales[weight_key] = scale

    sd = {}
    for shard_path in shard_files:
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("_scale."):
                    continue
                tensor = f.get_tensor(key)
                if key in all_scales:
                    scale = all_scales[key]
                    tensor = dequant_int8(tensor, scale, group_size)
                sd[key] = tensor.float()
        gc.collect()

    return sd, config

def rms_norm(x, weight, eps=1e-6):
    rms = x.float().pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    return (weight * x * rms)

def test_single_layer_0(sd, config):
    """测试第 0 层的单步前向传播"""
    print("="*60)
    print("测试第 0 层（ASM-ERSA）单步前向")
    print("="*60)

    layer_idx = 0
    pfx = f"layers.{layer_idx}."

    # 检查关键权重是否存在
    required_keys = [
        f"{pfx}gr1.norms.0.weight",
        f"{pfx}gr1.read_proj.weight",
        f"{pfx}token_mixing.W_q.weight",
        f"{pfx}token_mixing.W_k.weight",
        f"{pfx}token_mixing.W_v.weight",
        f"{pfx}token_mixing.W_o.weight",
        f"{pfx}gr1.write_proj.weight",
        f"{pfx}moe.gate.weight",
    ]

    missing = [k for k in required_keys if k not in sd]
    if missing:
        print(f"缺少权重: {missing}")
        return

    print("所有关键权重存在")

    # 创建随机输入
    B = 1
    d_model = config["d_model"]
    x = torch.randn(B, d_model)

    print(f"\n输入: shape={list(x.shape)}, mean={x.mean().item():.4f}, std={x.std().item():.4f}")

    # GR1 Read
    try:
        normed = [rms_norm(x, sd[f"{pfx}gr1.norms.{i}.weight"]) for i in range(4)]
        concat = torch.cat(normed, dim=-1)
        print(f"GR1 concat: shape={list(concat.shape)}, mean={concat.mean().item():.4f}")

        gates = F.linear(concat, sd[f"{pfx}gr1.read_proj.weight"]).sigmoid()
        print(f"GR1 gates: shape={list(gates.shape)}, mean={gates.mean().item():.4f}, min={gates.min().item():.4f}, max={gates.max().item():.4f}")

        stacked = torch.stack(normed, dim=1)
        x_after_gr = (gates.unsqueeze(-1) * stacked).mean(dim=1)
        print(f"GR1 output: shape={list(x_after_gr.shape)}, mean={x_after_gr.mean().item():.4f}")
    except Exception as e:
        print(f"GR1 失败: {e}")
        return

    # Token Mixing (ASM-ERSA)
    try:
        H = config["n_heads"]
        dk = config["d_k"]
        dv = config["d_v"]

        q = F.linear(x_after_gr, sd[f"{pfx}W_q.weight"])
        k = F.linear(x_after_gr, sd[f"{pfx}W_k.weight"])
        v = F.linear(x_after_gr, sd[f"{pfx}W_v.weight"])

        print(f"\nToken Mixing:")
        print(f"  Q: shape={list(q.shape)}, mean={q.mean().item():.4f}, std={q.std().item():.4f}")
        print(f"  K: shape={list(k.shape)}, mean={k.mean().item():.4f}, std={k.std().item():.4f}")
        print(f"  V: shape={list(v.shape)}, mean={v.mean().item():.4f}, std={v.std().item():.4f}")

        # 检查是否有 NaN/Inf
        if torch.isnan(q).any() or torch.isinf(q).any():
            print("  WARNING: Q 包含 NaN 或 Inf!")
        if torch.isnan(k).any() or torch.isinf(k).any():
            print("  WARNING: K 包含 NaN 或 Inf!")
        if torch.isnan(v).any() or torch.isinf(v).any():
            print("  WARNING: V 包含 NaN 或 Inf!")

    except Exception as e:
        print(f"Token Mixing 失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n第 0 层单步前向传播成功!")

def test_embed_to_first_layer(sd, config):
    """测试 embed → 第 0 层"""
    print("\n" + "="*60)
    print("测试 embed → 第 0 层")
    print("="*60)

    if "embed.weight" not in sd:
        print("缺少 embed.weight")
        return

    # 模拟 token 输入
    token_id = 1000
    embed_w = sd["embed.weight"]

    x = F.embedding(torch.tensor([token_id]), embed_w)
    print(f"Token {token_id} embedding: shape={list(x.shape)}, mean={x.mean().item():.4f}, std={x.std().item():.4f}")

    # 检查 embedding 值
    if torch.isnan(x).any() or torch.isinf(x).any():
        print("WARNING: embedding 包含 NaN 或 Inf!")

    # 通过第 0 层的 GR1
    pfx = "layers.0."
    normed = [rms_norm(x, sd[f"{pfx}gr1.norms.{i}.weight"]) for i in range(4)]
    concat = torch.cat(normed, dim=-1)
    gates = F.linear(concat, sd[f"{pfx}gr1.read_proj.weight"]).sigmoid()

    print(f"GR1 gates: mean={gates.mean().item():.4f}, min={gates.min().item():.4f}, max={gates.max().item():.4f}")

    # 检查 gates 是否接近 0 或 1
    if gates.mean().item() < 0.1 or gates.mean().item() > 0.9:
        print("WARNING: gates 接近 0 或 1，可能导致信号丢失!")

if __name__ == "__main__":
    print("加载所有权重...")
    sd, config = load_all_weights("models/asm_ersa-int8")
    print(f"加载完成，共 {len(sd)} 个权重\n")

    test_embed_to_first_layer(sd, config)
    test_single_layer_0(sd, config)
