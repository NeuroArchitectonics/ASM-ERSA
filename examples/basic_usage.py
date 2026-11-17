"""
Example: Basic ASM-ERSA usage (tiny config, runs on CPU in seconds).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from asm_ersa import ASM_ERSA

# 64 维、4 层、16 专家的 tiny 配置，便于本地验证前向/反向
model = ASM_ERSA(
    d_model=64,
    n_heads=4,
    d_k=16,
    d_v=16,
    d_ff=128,
    n_layers=4,
    vocab_size=256,
    n_experts=16,
    moe_top_k=4,
)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params / 1e6:.2f}M")
print(f"ASM-ERSA layers: {sum(1 for l in model.layers if not l.is_ersa)}")
print(f"ERSA layers: {sum(1 for l in model.layers if l.is_ersa)}")

batch_size, seq_len = 2, 8
input_ids = torch.randint(0, 256, (batch_size, seq_len))
with torch.no_grad():
    outputs = model(input_ids)

print(f"\nInput shape:  {tuple(input_ids.shape)}")
print(f"Output shape: {tuple(outputs.shape)}")
print("✓ Forward pass successful")
