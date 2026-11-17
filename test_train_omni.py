"""
ASM-ERSA Omni 3B — 小数据集冒烟测试
===================================
用合成数据跑几步训练，验证整个流程无报错

用法:
  python test_train_omni.py
  python test_train_omni.py --steps 20 --batch-size 2
"""

import os
import sys
import json
import time
import math
import argparse
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from asm_ersa import ASM_ERSA_Omni, Modality, OmniTransformerBlock


# ═══════════════════════════════════════════════════════════════
# 缩小版模型配置（快速验证）
# ═══════════════════════════════════════════════════════════════

TINY_CONFIG = dict(
    d_model=256,
    n_heads=4,
    d_k=32,
    d_v=32,
    d_ff=512,
    n_layers=4,
    vocab_size=1000,
    n_shared_experts=2,
    n_text_experts=4,
    n_audio_experts=2,
    n_vision_experts=2,
    top_k=2,
    audio_output_mode='discrete',
    n_audio_tokens=256,
    audio_config={'n_mels': 80, 'patch_size': 4, 'stride': 2},
    vision_config={'image_size': 56, 'patch_size': 14, 'n_layers': 1, 'n_heads': 4},
)


# ═══════════════════════════════════════════════════════════════
# 合成数据生成
# ═══════════════════════════════════════════════════════════════

def make_batch(B: int, T_text: int, cfg: dict, device: torch.device,
               with_audio: bool = True, with_image: bool = True):
    """生成一个合成 batch"""
    input_ids = torch.randint(0, cfg['vocab_size'], (B, T_text), device=device)

    batch = {'input_ids': input_ids}

    if with_audio:
        batch['audio_mel'] = torch.randn(B, 80, 50, device=device)

    if with_image:
        img_size = cfg.get('vision_config', {}).get('image_size', 56)
        batch['images'] = torch.randn(B, 3, img_size, img_size, device=device)

    return batch


# ═══════════════════════════════════════════════════════════════
# 测试：前向传播
# ═══════════════════════════════════════════════════════════════

def test_forward(model, device):
    print("\n" + "=" * 60)
    print("测试 1: 前向传播 (text + audio + vision)")
    print("=" * 60)

    B, T = 2, 8
    batch = make_batch(B, T, TINY_CONFIG, device)

    t0 = time.time()
    with torch.no_grad():
        output = model(
            input_ids=batch['input_ids'],
            audio_mel=batch['audio_mel'],
            images=batch['images'],
        )
    elapsed = time.time() - t0

    print(f"  输入: text={batch['input_ids'].shape}, "
          f"audio={batch['audio_mel'].shape}, "
          f"image={batch['images'].shape}")
    print(f"  输出:")
    for k, v in output.items():
        print(f"    {k}: {v.shape}")
    print(f"  耗时: {elapsed:.3f}s")
    print("  ✓ 前向传播通过")


# ═══════════════════════════════════════════════════════════════
# 测试：纯文本前向
# ═══════════════════════════════════════════════════════════════

def test_text_only(model, device):
    print("\n" + "=" * 60)
    print("测试 2: 纯文本前向")
    print("=" * 60)

    B, T = 2, 16
    input_ids = torch.randint(0, TINY_CONFIG['vocab_size'], (B, T), device=device)

    t0 = time.time()
    with torch.no_grad():
        output = model(input_ids=input_ids)
    elapsed = time.time() - t0

    print(f"  输入: text={input_ids.shape}")
    print(f"  输出 text_logits: {output['text_logits'].shape}")
    print(f"  耗时: {elapsed:.3f}s")
    print("  ✓ 纯文本前向通过")


# ═══════════════════════════════════════════════════════════════
# 测试：损失计算 + 反向传播
# ═══════════════════════════════════════════════════════════════

def test_backward(model, device):
    print("\n" + "=" * 60)
    print("测试 3: 损失计算 + 反向传播")
    print("=" * 60)

    B, T = 2, 6
    batch = make_batch(B, T, TINY_CONFIG, device)

    output = model(
        input_ids=batch['input_ids'],
        audio_mel=batch['audio_mel'],
        images=batch['images'],
    )

    text_logits = output['text_logits']
    audio_output = output['audio_output']
    T_total = text_logits.shape[1]

    # 文本在序列末尾，计算文本位置的 loss
    T_text = batch['input_ids'].shape[1]
    text_start = T_total - T_text

    # 文本损失: 用文本位置的 logits 预测 next token
    text_logits_slice = text_logits[:, text_start:-1, :].contiguous()
    text_labels = batch['input_ids'][:, 1:].contiguous()
    text_loss = F.cross_entropy(
        text_logits_slice.view(-1, TINY_CONFIG['vocab_size']),
        text_labels.view(-1),
    )

    # 音频损失
    audio_loss = F.cross_entropy(
        audio_output[:, :-1, :].contiguous().view(-1, TINY_CONFIG['n_audio_tokens']),
        torch.randint(0, TINY_CONFIG['n_audio_tokens'],
                      (B, audio_output.shape[1] - 1), device=device).view(-1),
    )

    total_loss = text_loss + 0.5 * audio_loss

    print(f"  text_loss:  {text_loss.item():.4f}")
    print(f"  audio_loss: {audio_loss.item():.4f}")
    print(f"  total_loss: {total_loss.item():.4f}")

    t0 = time.time()
    total_loss.backward()
    elapsed = time.time() - t0

    # 检查梯度
    has_grad = False
    for name, p in model.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break

    print(f"  反向传播耗时: {elapsed:.3f}s")
    print(f"  梯度存在: {'✓' if has_grad else '✗'}")

    # 统计梯度范数
    total_norm = 0.0
    n_with_grad = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.norm().item() ** 2
            n_with_grad += 1
    total_norm = total_norm ** 0.5
    print(f"  梯度总范数: {total_norm:.4f} ({n_with_grad} 参数有梯度)")
    print("  ✓ 反向传播通过")


# ═══════════════════════════════════════════════════════════════
# 测试：多步训练循环
# ═══════════════════════════════════════════════════════════════

def test_training_loop(model, device, n_steps: int = 5, batch_size: int = 2):
    print("\n" + "=" * 60)
    print(f"测试 4: 训练循环 ({n_steps} steps)")
    print("=" * 60)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    losses = []
    t0 = time.time()

    for step in range(n_steps):
        seq_len = 4 + step % 4
        batch = make_batch(batch_size, seq_len, TINY_CONFIG, device)

        output = model(
            input_ids=batch['input_ids'],
            audio_mel=batch.get('audio_mel'),
            images=batch.get('images'),
        )

        text_logits = output['text_logits']
        T_total = text_logits.shape[1]
        T_text = batch['input_ids'].shape[1]
        text_start = T_total - T_text

        shift_logits = text_logits[:, text_start:-1, :].contiguous()
        shift_labels = batch['input_ids'][:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, TINY_CONFIG['vocab_size']),
            shift_labels.view(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        lr = scheduler.get_last_lr()[0]
        print(f"  step {step + 1}/{n_steps}: loss={loss.item():.4f}, lr={lr:.2e}")

    elapsed = time.time() - t0
    avg_step = elapsed / n_steps

    print(f"\n  总耗时: {elapsed:.2f}s")
    print(f"  平均 step 耗时: {avg_step:.3f}s")
    print(f"  损失变化: {losses[0]:.4f} → {losses[-1]:.4f}")

    if losses[-1] < losses[0] * 1.5:
        print("  ✓ 训练循环正常（损失未爆炸）")
    else:
        print("  ⚠ 损失增长过大，请检查")


# ═══════════════════════════════════════════════════════════════
# 测试：Gradient Checkpointing
# ═══════════════════════════════════════════════════════════════

def test_gradient_checkpointing(device):
    print("\n" + "=" * 60)
    print("测试 5: Gradient Checkpointing")
    print("=" * 60)

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    model = ASM_ERSA_Omni(**TINY_CONFIG).to(device)

    mem_before = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=checkpoint_wrapper,
        check_fn=lambda m: isinstance(m, OmniTransformerBlock),
    )

    B, T = 2, 8
    batch = make_batch(B, T, TINY_CONFIG, device)

    output = model(
        input_ids=batch['input_ids'],
        audio_mel=batch['audio_mel'],
        images=batch['images'],
    )
    loss = output['text_logits'][:, :-1].reshape(-1, TINY_CONFIG['vocab_size']).mean()
    loss.backward()

    mem_after = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0

    print(f"  显存 (前): {mem_before:.1f} MB")
    print(f"  显存 (后): {mem_after:.1f} MB")
    print("  ✓ Gradient checkpointing 通过")


# ═══════════════════════════════════════════════════════════════
# 测试：参数量统计
# ═══════════════════════════════════════════════════════════════

def test_param_count():
    print("\n" + "=" * 60)
    print("测试 6: 参数量统计")
    print("=" * 60)

    model = ASM_ERSA_Omni(**TINY_CONFIG)

    total = sum(p.numel() for p in model.parameters())
    encoder = (sum(p.numel() for p in model.audio_encoder.parameters()) +
               sum(p.numel() for p in model.vision_encoder.parameters()))
    heads = sum(p.numel() for p in model.audio_head.parameters())
    backbone = total - encoder - heads

    print(f"  总参数: {total / 1e6:.2f}M")
    print(f"  Backbone: {backbone / 1e6:.2f}M")
    print(f"  编码器: {encoder / 1e6:.2f}M (audio + vision)")
    print(f"  输出头: {heads / 1e6:.2f}M")

    moe_params = 0
    for layer in model.layers:
        if hasattr(layer, 'moe'):
            moe_params += sum(p.numel() for p in layer.moe.parameters())
    print(f"  MoE 总参数: {moe_params / 1e6:.2f}M")

    ersa_count = sum(1 for l in model.layers if hasattr(l, 'is_ersa') and l.is_ersa)
    gdn_count = len(model.layers) - ersa_count
    print(f"  层分布: ASM-ERSA={gdn_count}, ERSA-Omni={ersa_count}")
    print("  ✓ 参数统计通过")


# ═══════════════════════════════════════════════════════════════
# 测试：3B 规模模型实例化
# ═══════════════════════════════════════════════════════════════

def test_3b_instantiation(device):
    print("\n" + "=" * 60)
    print("测试 7: 3B Omni 模型实例化")
    print("=" * 60)

    config_3b = dict(
        d_model=2048,
        n_heads=16,
        d_k=128,
        d_v=128,
        d_ff=384,
        n_layers=32,
        vocab_size=152064,
        n_shared_experts=8,
        n_text_experts=12,
        n_audio_experts=6,
        n_vision_experts=6,
        top_k=4,
        audio_output_mode='discrete',
        n_audio_tokens=8192,
        audio_config={'n_mels': 80, 'patch_size': 4, 'stride': 2},
        vision_config={'image_size': 336, 'patch_size': 14, 'n_layers': 2, 'n_heads': 8},
    )

    t0 = time.time()

    if torch.cuda.is_available():
        mem_before = torch.cuda.memory_allocated() / 1e9
    else:
        mem_before = 0

    model = ASM_ERSA_Omni(**config_3b).to(device)
    model.eval()

    elapsed = time.time() - t0

    total = sum(p.numel() for p in model.parameters())

    if torch.cuda.is_available():
        mem_after = torch.cuda.memory_allocated() / 1e9
        mem_used = mem_after - mem_before
    else:
        mem_used = 0

    print(f"  实例化耗时: {elapsed:.2f}s")
    print(f"  总参数: {total / 1e9:.2f}B")
    if mem_used > 0:
        print(f"  显存占用: {mem_used:.2f} GB")

    ersa_count = sum(1 for l in model.layers if hasattr(l, 'is_ersa') and l.is_ersa)
    gdn_count = len(model.layers) - ersa_count
    print(f"  层分布: ASM-ERSA={gdn_count}, ERSA-Omni={ersa_count}")

    moe_per_layer = sum(p.numel() for p in model.layers[0].moe.parameters())
    active_experts = 4 + 1
    active_moe = moe_per_layer * active_experts / (32 + 48 + 24 + 24)
    total_active = active_moe * 32
    non_moe = total - sum(
        sum(p.numel() for p in l.moe.parameters())
        for l in model.layers if hasattr(l, 'moe')
    )
    print(f"  每 token 激活参数 (估算): ~{(non_moe + total_active) / 1e9:.2f}B")

    print("  ✓ 3B 模型实例化通过")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='ASM-ERSA Omni 训练冒烟测试')
    parser.add_argument('--steps', type=int, default=5, help='训练步数')
    parser.add_argument('--batch-size', type=int, default=2, help='batch size')
    parser.add_argument('--skip-3b', action='store_true', help='跳过 3B 模型测试（显存不足时）')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    model = ASM_ERSA_Omni(**TINY_CONFIG).to(device)

    results = {}

    try:
        test_param_count()
        results['param_count'] = 'PASS'
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results['param_count'] = f'FAIL: {e}'

    try:
        test_forward(model, device)
        results['forward'] = 'PASS'
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        import traceback; traceback.print_exc()
        results['forward'] = f'FAIL: {e}'

    try:
        test_text_only(model, device)
        results['text_only'] = 'PASS'
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        import traceback; traceback.print_exc()
        results['text_only'] = f'FAIL: {e}'

    try:
        test_backward(model, device)
        results['backward'] = 'PASS'
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        import traceback; traceback.print_exc()
        results['backward'] = f'FAIL: {e}'

    try:
        test_training_loop(model, device, n_steps=args.steps, batch_size=args.batch_size)
        results['training_loop'] = 'PASS'
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        import traceback; traceback.print_exc()
        results['training_loop'] = f'FAIL: {e}'

    if torch.cuda.is_available():
        try:
            test_gradient_checkpointing(device)
            results['grad_ckpt'] = 'PASS'
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            import traceback; traceback.print_exc()
            results['grad_ckpt'] = f'FAIL: {e}'

    if not args.skip_3b:
        try:
            test_3b_instantiation(device)
            results['3b_model'] = 'PASS'
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            import traceback; traceback.print_exc()
            results['3b_model'] = f'FAIL: {e}'

    # 汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    all_pass = True
    for name, result in results.items():
        status = "✓" if result == 'PASS' else "✗"
        print(f"  {status} {name}: {result}")
        if result != 'PASS':
            all_pass = False

    print()
    if all_pass:
        print("所有测试通过 ✓")
    else:
        print("部分测试失败 ✗")
        sys.exit(1)


if __name__ == '__main__':
    main()
