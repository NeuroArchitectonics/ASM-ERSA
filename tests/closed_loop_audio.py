"""
P3 §4-11 端到端闭环（后半：预测码 → 可播放 wav）。

链路：真实 wav → mel + Encodec 码(真值) → 小模型 → 预测码 → Encodec 解码 → wav

三条路径逐样本对比：
  A) codec 往返   真值码 → decode          —— 量化误差下界（模型不可能好过它）
  B) teacher-forced  逐步喂真值的历史，取每帧 argmax → decode
  C) generate_audio  自由生成（无真值泄漏）→ decode

指标：
  - 帧准确率 vs 真值码，并给出「单一最常码」基线（第 0 帧之后全靠猜，需高于基线）
  - 归一化 log-mel L1 距离（与 asm_ersa.audio_io.load_audio_mel 同一归一化），以 A 为下界参照
  - 生成码的取值分布（用了多少码 / 最高频码占比），用于识别 mode collapse

前置：先跑 tests/overfit_audio_small.py 得到 ckpt（默认 .cache/overfit_audio_ckpt）。
用法：python tests/closed_loop_audio.py            # 参数见 --help
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from asm_ersa import ASM_ERSA_Omni                                    # noqa: E402
from asm_ersa.audio_codec import EncodecTokenizer                 # noqa: E402
from asm_ersa.audio_io import log_mel, resample                   # noqa: E402
from train_omni_3b import (                                    # noqa: E402
    TrainConfig, OmniDataset, collate_fn, read_jsonl_samples,
)
from overfit_audio_small import (                              # noqa: E402
    MEDIA_JSONL, CharTokenizer, select_subset, label_baselines,
)

MEL_REF_SR = 16000


# ═══════════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════════

def norm_mel(wav_1xT: torch.Tensor, sr: int) -> torch.Tensor:
    """波形（模型输出）→ 与 load_audio_mel 同口径的归一化 log-mel [n_mels, T]"""
    x = wav_1xT.to(torch.float32)
    if sr != MEL_REF_SR:
        x = resample(x, sr, MEL_REF_SR)
    mel = log_mel(x, MEL_REF_SR)
    return (mel - mel.mean()) / (mel.std() + 1e-6)


def l1_aligned(a: torch.Tensor, b: torch.Tensor) -> float:
    """两个 [n_mels, T] 取公共帧数后的平均绝对差"""
    T = min(a.shape[-1], b.shape[-1])
    if T == 0:
        return float('nan')
    return float((a[:, :T] - b[:, :T]).abs().mean())


def code_stats(codes: torch.Tensor) -> dict:
    cnt = Counter(codes.tolist())
    top = cnt.most_common(1)[0]
    return {'n_unique': len(cnt), 'top_code': int(top[0]),
            'top_share': top[1] / int(codes.numel())}


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, default='.cache/overfit_audio_ckpt')
    p.add_argument('--samples', type=int, default=8)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--max-seconds', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out-dir', type=str, default='.cache/closed_loop')
    p.add_argument('--json', type=str, default='.cache/closed_loop_report.json')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 与 overfit 脚本完全同源的语料/配置/词表 ────────────────
    all_samples = read_jsonl_samples(MEDIA_JSONL)
    subset_jsonl = Path(f'.cache/overfit{args.samples}.jsonl')
    subset, subset_jsonl = select_subset(
        all_samples, args.samples, args.max_seconds, subset_jsonl)

    cfg = TrainConfig(
        d_model=512, n_heads=8, d_k=64, d_v=64, d_ff=384,
        n_layers=args.layers, top_k=4,
        max_audio_seconds=args.max_seconds,
        data_path=str(subset_jsonl), num_workers=0,
        seed=args.seed,
    )
    cfg.audio_codes_cache = f'.cache/overfit{args.samples}_codes.pt'

    tokenizer = CharTokenizer([s['text'] for s in subset])
    cfg.vocab_size = tokenizer.vocab_size
    dataset = OmniDataset(str(subset_jsonl), tokenizer, cfg)
    codes_list = [dataset[i]['audio_codes'] for i in range(len(dataset))]

    _, ce_const, n_codes, n_frames = label_baselines(codes_list)
    const_acc = 1.0 / math.exp(ce_const)      # 单一最常码的命中率

    # ── 模型 + 编解码器 ───────────────────────────────────────
    model = ASM_ERSA_Omni.from_pretrained(args.ckpt)
    model.eval()
    n_vocab_ckpt = model.text_embed.weight.shape[0]
    assert n_vocab_ckpt == tokenizer.vocab_size, (
        f"词表不匹配: ckpt={n_vocab_ckpt} 本脚本={tokenizer.vocab_size}；"
        f"ckpt 与语料子集不同源，需先用同一 {subset_jsonl} 重训")
    assert model.audio_head.mode == 'discrete', \
        f"audio_head.mode={model.audio_head.mode}，非离散模式无法比较码"
    codec = EncodecTokenizer()

    print(f"语料 {len(dataset)} 条 / {n_frames} 帧（子集 {subset_jsonl}）")
    print(f"ckpt {args.ckpt}  (参数量 "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M, "
          f"vocab={n_vocab_ckpt})")
    print(f"码本 {codec!r}")
    print(f"基线: ln(1024)={math.log(1024):.3f}  最优常数码命中率={const_acc:.4f}  "
          f"语料用到 {n_codes}/1024 个码\n")

    rows = []
    for i in range(len(dataset)):
        batch = collate_fn([dataset[i]])
        codes = batch['audio_codes'][0]                       # [T] 真值
        T_code = int(codes.shape[0])
        mel_in = batch['audio_mel']                           # [1, n_mels, T_mel]

        # teacher-forced：逐步喂真值历史，每帧取 argmax
        with torch.no_grad():
            out = model(
                input_ids=batch['input_ids'],
                audio_mel=mel_in,
                attention_mask=batch['attention_mask'].bool(),
                audio_valid_frames=batch.get('audio_valid_frames'),
                audio_codes_target=batch['audio_codes'],
                audio_codes_valid=batch['audio_codes_valid'],
            )
            tf_pred = out['audio_target_logits'][0].argmax(-1)      # [T] 与真值逐位对齐

            # 自由生成：前缀只前向一次，之后每帧只喂上一帧的预测码（无真值泄漏）
            gen = model.generate_audio(
                audio_mel=mel_in,
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'].bool(),
                audio_valid_frames=batch.get('audio_valid_frames'),
                n_frames=T_code, greedy=True,
            )[0]                                                    # [T]

        # 三条路径解码 → wav
        with torch.no_grad():
            wav_true = codec.decode(codes.unsqueeze(0))[0]
            wav_tf = codec.decode(tf_pred.unsqueeze(0))[0]
            wav_gen = codec.decode(gen.unsqueeze(0))[0]

        stem = Path(subset[i]['audio_path']).stem
        for tag, wav in (('reftrue', wav_true), ('tf', wav_tf), ('gen', wav_gen)):
            sf.write(out_dir / f"{i:02d}_{stem}_{tag}.wav",
                     wav.cpu().numpy(), codec.sample_rate)

        mel_ref = mel_in[0]                                     # 模型实际看到的输入
        mel_rt = norm_mel(wav_true.unsqueeze(0), codec.sample_rate)
        mel_tf = norm_mel(wav_tf.unsqueeze(0), codec.sample_rate)
        mel_gn = norm_mel(wav_gen.unsqueeze(0), codec.sample_rate)

        acc_tf = float((tf_pred == codes).float().mean())
        acc_gen = float((gen == codes).float().mean())
        # 首帧之外（不含 <audio_start> 提供的第 0 帧）的命中率更能反映真实预测能力
        acc_tf_tail = float((tf_pred[1:] == codes[1:]).float().mean())
        acc_gen_tail = float((gen[1:] == codes[1:]).float().mean())

        row = {
            'idx': i, 'name': stem, 'audio_path': subset[i]['audio_path'],
            'n_frames': T_code,
            'acc_tf': acc_tf, 'acc_gen': acc_gen,
            'acc_tf_tail': acc_tf_tail, 'acc_gen_tail': acc_gen_tail,
            'mel_l1_codec': l1_aligned(mel_rt, mel_ref),
            'mel_l1_tf': l1_aligned(mel_tf, mel_ref),
            'mel_l1_gen': l1_aligned(mel_gn, mel_ref),
            'rms_true': float(wav_true.pow(2).mean().sqrt()),
            'rms_tf': float(wav_tf.pow(2).mean().sqrt()),
            'rms_gen': float(wav_gen.pow(2).mean().sqrt()),
            'finite': bool(torch.isfinite(wav_tf).all() and torch.isfinite(wav_gen).all()),
            'sec_gen': wav_gen.shape[-1] / codec.sample_rate,
            'stats_tf': code_stats(tf_pred),
            'stats_gen': code_stats(gen),
        }
        rows.append(row)
        print(f"[{i}] {stem[:26]:26s} T={T_code:3d}  "
              f"acc tf={acc_tf:.3f}/gen={acc_gen:.3f}  "
              f"mel L1 codec={row['mel_l1_codec']:.3f} tf={row['mel_l1_tf']:.3f} "
              f"gen={row['mel_l1_gen']:.3f}  "
              f"gen用码 {row['stats_gen']['n_unique']:3d} "
              f"(top占比 {row['stats_gen']['top_share']:.2f})  rms={row['rms_gen']:.4f}")

    def mean_of(key):
        return sum(r[key] for r in rows) / len(rows)

    print(f"\n均值: acc tf={mean_of('acc_tf'):.3f} / gen={mean_of('acc_gen'):.3f}"
          f"   (尾部 acc tf={mean_of('acc_tf_tail'):.3f} / "
          f"gen={mean_of('acc_gen_tail'):.3f})")
    print(f"      mel L1  codec={mean_of('mel_l1_codec'):.3f} < "
          f"tf={mean_of('mel_l1_tf'):.3f}   gen={mean_of('mel_l1_gen'):.3f}")

    # ── 判据（§4-11 后半：解码预测码得到可播放 wav） ───────────
    dur_ok = all(abs(r['sec_gen'] - args.max_seconds) < 0.05 for r in rows)
    loud_ok = all(r['rms_gen'] > 1e-4 for r in rows)
    finite_ok = all(r['finite'] for r in rows)
    beat_const = mean_of('acc_gen') > const_acc
    checks = [
        ('时长 == 截断长度', dur_ok, f"{[round(r['sec_gen'], 3) for r in rows]}"),
        ('非静音 (rms>1e-4)', loud_ok, f"min={min(r['rms_gen'] for r in rows):.5f}"),
        ('数值有限（无 NaN/Inf）', finite_ok, ''),
        ('生成准确率 > 常数基线', beat_const,
         f"{mean_of('acc_gen'):.4f} vs {const_acc:.4f}"),
    ]
    print()
    for name, ok, detail in checks:
        print(f"  [{'✓' if ok else '✗'}] {name}  {detail}")

    report = {
        'args': vars(args), 'ckpt': args.ckpt,
        'vocab_size': n_vocab_ckpt, 'n_frames': n_frames,
        'baseline_const_acc': const_acc, 'baseline_ln1024': math.log(1024),
        'means': {k: mean_of(k) for k in
                  ('acc_tf', 'acc_gen', 'acc_tf_tail', 'acc_gen_tail',
                   'mel_l1_codec', 'mel_l1_tf', 'mel_l1_gen', 'rms_gen')},
        'checks': {name: bool(ok) for name, ok, _ in checks},
        'rows': rows,
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.json, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print(f"\nwav 目录 {out_dir}\n报告已存 {args.json}")


if __name__ == '__main__':
    main()
