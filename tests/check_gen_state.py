"""检查 generate_audio 的增量递推与整段 forward 是否一致。

若两者 logits 一致 → 生成路径的状态传递没有 bug，自由生成坍塌属模型/采样问题；
若不一致 → 定位到具体第几帧开始分叉。
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from asm_ersa import ASM_ERSA_Omni, Modality                                   # noqa: E402
from train_omni_3b import TrainConfig, OmniDataset, collate_fn, read_jsonl_samples  # noqa: E402
from overfit_audio_small import MEDIA_JSONL, select_subset, CharTokenizer  # noqa: E402


def main():
    torch.manual_seed(42)
    cfg = TrainConfig(
        d_model=512, n_heads=8, d_k=64, d_v=64, d_ff=384,
        n_layers=4, top_k=4, max_audio_seconds=1.0,
        data_path='.cache/overfit8.jsonl', num_workers=0, seed=42)
    cfg.audio_codes_cache = '.cache/overfit8_codes.pt'

    all_samples = read_jsonl_samples(MEDIA_JSONL)
    subset, subset_jsonl = select_subset(all_samples, 8, 1.0, Path('.cache/overfit8.jsonl'))
    tokenizer = CharTokenizer([s['text'] for s in subset])
    cfg.vocab_size = tokenizer.vocab_size
    dataset = OmniDataset(str(subset_jsonl), tokenizer, cfg)

    model = ASM_ERSA_Omni.from_pretrained('.cache/overfit_audio_ckpt_asm')
    model.eval()

    batch = collate_fn([dataset[0]])
    codes = batch['audio_codes'][0]
    T = int(codes.shape[0])

    with torch.no_grad():
        out = model(
            input_ids=batch['input_ids'],
            audio_mel=batch['audio_mel'],
            attention_mask=batch['attention_mask'].bool(),
            audio_valid_frames=batch.get('audio_valid_frames'),
            audio_codes_target=batch['audio_codes'],
            audio_codes_valid=batch['audio_codes_valid'],
        )
        full_logits = out['audio_target_logits'][0]              # [T, V]

        # ── 增量路径：前缀一次 + audio_start + 逐步喂真值码 ──
        all_tokens, all_mod_ids, seg_masks, n_prefix = model._build_prefix_segments(
            batch['audio_mel'], None, batch['input_ids'],
            batch['attention_mask'].bool(), batch.get('audio_valid_frames'))
        x, modality_ids, key_mask = model._pack_segments(all_tokens, all_mod_ids, seg_masks)
        B, Tp, _ = x.shape

        states = [None] * len(model.layers)
        buffers = [None] * len(model.layers)
        positions = torch.zeros(B, 0, dtype=torch.long)
        for t in range(Tp):
            positions = torch.cat(
                [positions, torch.full((B, 1), t, dtype=torch.long)], dim=1)
            _, states, buffers = model._forward_one_token(
                x[:, t], modality_ids[:, t], positions, modality_ids[:, :t + 1],
                n_prefix, states, buffers,
                None if key_mask is None else key_mask[:, :t + 1])

        cur = model.modality_embed(
            model.audio_start_embed.view(1, 1, -1).expand(B, 1, -1), Modality.AUDIO)[:, 0]
        inc_logits = []
        for t in range(T):
            positions = torch.cat(
                [positions, torch.full((B, 1), positions.shape[1], dtype=torch.long)], dim=1)
            mod_t = torch.full((B,), Modality.AUDIO, dtype=torch.long)
            modality_ids = torch.cat([modality_ids, mod_t.unsqueeze(1)], dim=1)
            if key_mask is not None:
                key_mask = torch.cat(
                    [key_mask, torch.ones(B, 1, dtype=torch.bool)], dim=1)
            h_t, states, buffers = model._forward_one_token(
                cur, mod_t, positions, modality_ids, n_prefix, states, buffers, key_mask)
            inc_logits.append(model.audio_head(model.out_norm(h_t)))
            code = codes[t].view(1)
            cur = model.modality_embed(
                model.audio_code_embed(code).view(1, 1, -1), Modality.AUDIO)[:, 0]

        inc = torch.stack(inc_logits, dim=0)[:, 0]               # [T, V]

    acc_full = (full_logits.argmax(-1) == codes).float().mean().item()
    acc_inc = (inc.argmax(-1) == codes).float().mean().item()
    diff = (full_logits - inc).abs()
    print(f"T={T}")

    # 额外诊断：训练侧 code 嵌入 vs 增量侧逐 token 嵌入
    with torch.no_grad():
        emb_full = model.encode_audio_target(batch['audio_codes'], batch['audio_codes_valid'])[0][0]
        emb_inc = model.modality_embed(
            model.audio_code_embed(codes[0].view(1)).view(1, 1, -1), Modality.AUDIO)[0, 0]
        print(f"code_0 嵌入差（训练 vs 增量）: {(emb_full[1] - emb_inc).abs().max().item():.3e}")

        # 长度对照：只喂 [start, code_0] 的短序列，看第 1 位 logits 是否与长序列一致
        short = model(
            input_ids=batch['input_ids'],
            audio_mel=batch['audio_mel'],
            attention_mask=batch['attention_mask'].bool(),
            audio_valid_frames=batch.get('audio_valid_frames'),
            audio_codes_target=batch['audio_codes'][:, :2],
            audio_codes_valid=batch['audio_codes_valid'][:, :2] if batch.get('audio_codes_valid') is not None else None,
        )['audio_target_logits'][0]
        print(f"短序列 vs 长序列 第1位 logits 差: {(short[1] - full_logits[1]).abs().max().item():.3e}")
        print(f"短序列 vs 增量   第1位 logits 差: {(short[1] - inc[1]).abs().max().item():.3e}")
    print(f"整段 forward 准确率: {acc_full:.4f} | 增量递推准确率: {acc_inc:.4f}")
    print(f"logits 最大绝对差: {diff.max().item():.3e}")
    print(f"逐帧最大差(前10): {[f'{v:.1e}' for v in diff.max(-1).values[:10].tolist()]}")
    print(f"首帧 argmax 一致: {(full_logits[0].argmax() == inc[0].argmax()).item()}")

    ok = diff.max().item() < 1e-3 and abs(acc_full - acc_inc) < 1e-6
    print("判定:", "状态传递一致（生成坍塌属模型/采样问题）" if ok else "增量路径与整段 forward 不一致（存在状态 bug）")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
