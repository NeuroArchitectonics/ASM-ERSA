"""
§4-11 补充诊断：自由生成退化的成因（不是验收项，是给结论提供证据）。

`tests/closed_loop_audio.py` 报出 teacher-forced 命中率 0.71、自由生成 0.04。
本脚本回答两个问题：

1. 自由生成的码序列是「与条件无关的同一吸引子」吗？（8 条真值码互不相关，
   若生成序列彼此高度相同，则说明生成时条件失效）
2. 退化是误差累积（exposure bias）还是管线不一致？做法：把前 k 帧的历史换成
   模型自己的自由生成码，只测位置 k 的命中率——若随 k 单调衰减到 0，则历史里
   一旦混入自身错误，预测即失效；这与「训练只用真值历史」一致。

管线一致性另有硬证据：`tests/test_fixes.py::test_generate_audio_matches_teacher_forcing`
（增量递推 vs 整段前向逐位误差 < 1e-5，实测 7.2e-07）。

用法：python tests/audio_exposure_curve.py            # 参数见 --help
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from asm_ersa import ASM_ERSA_Omni                                     # noqa: E402
from train_omni_3b import (                                     # noqa: E402
    TrainConfig, OmniDataset, collate_fn, read_jsonl_samples,
)
from overfit_audio_small import (                               # noqa: E402
    MEDIA_JSONL, CharTokenizer, select_subset,
)

KS = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, default='.cache/overfit_audio_ckpt')
    p.add_argument('--samples', type=int, default=8)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--max-seconds', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    all_samples = read_jsonl_samples(MEDIA_JSONL)
    subset, subset_jsonl = select_subset(
        all_samples, args.samples, args.max_seconds,
        Path(f'.cache/overfit{args.samples}.jsonl'))

    cfg = TrainConfig(
        d_model=512, n_heads=8, d_k=64, d_v=64, d_ff=384,
        n_layers=args.layers, top_k=4,
        max_audio_seconds=args.max_seconds,
        data_path=str(subset_jsonl), num_workers=0, seed=args.seed,
    )
    cfg.audio_codes_cache = f'.cache/overfit{args.samples}_codes.pt'
    tokenizer = CharTokenizer([s['text'] for s in subset])
    cfg.vocab_size = tokenizer.vocab_size
    dataset = OmniDataset(str(subset_jsonl), tokenizer, cfg)

    model = ASM_ERSA_Omni.from_pretrained(args.ckpt)
    model.eval()

    lengths = {int(dataset[i]['audio_codes'].shape[0]) for i in range(len(dataset))}
    if len(lengths) != 1:
        raise SystemExit(f"本诊断需同长度样本，实测 {sorted(lengths)}；"
                         f"请减小 --max-seconds 或换 --samples")
    batch = collate_fn([dataset[i] for i in range(len(dataset))])
    codes = batch['audio_codes']
    B, T = codes.shape

    with torch.no_grad():
        gens = model.generate_audio(
            audio_mel=batch['audio_mel'], input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'].bool(),
            audio_valid_frames=batch.get('audio_valid_frames'),
            n_frames=T, greedy=True)

    # ── 1) 生成序列 vs 真值序列（跨样本） ──────────────────────
    print(f'batch {B} x T={T}  自由生成整体命中率 '
          f'{float((gens == codes).float().mean()):.3f}\n')
    print('与样本 0 的比较 [生成 / 真值]:')
    for i in range(B):
        same_g = float((gens[i] == gens[0]).float().mean())
        same_c = float((codes[i] == codes[0]).float().mean())
        pre_g = next((j for j in range(T) if gens[i, j] != gens[0, j]), T)
        pre_c = next((j for j in range(T) if codes[i, j] != codes[0, j]), T)
        acc = float((gens[i] == codes[i]).float().mean())
        print(f'  [{i}] 生成: 相同率={same_g:.3f} 共同前缀={pre_g:2d} | '
              f'真值: 相同率={same_c:.3f} 共同前缀={pre_c:2d} | '
              f'命中率={acc:.3f} 用码={len(set(gens[i].tolist()))}')

    # ── 2) 暴露偏差曲线 ───────────────────────────────────────
    print('\n位置 k 的命中率（历史 = 前 k 帧自身生成码，其余符号不参与）')
    print('  k    命中率   历史里自身预测码占比')
    for k in KS:
        mixed = codes.clone()
        if k:
            mixed[:, :k] = gens[:, :k]
        with torch.no_grad():
            out = model(input_ids=batch['input_ids'], audio_mel=batch['audio_mel'],
                        attention_mask=batch['attention_mask'].bool(),
                        audio_valid_frames=batch.get('audio_valid_frames'),
                        audio_codes_target=mixed,
                        audio_codes_valid=batch['audio_codes_valid'])
        pred = out['audio_target_logits'].argmax(-1)
        hit = float((pred[:, k] == codes[:, k]).float().mean())
        frac = float('nan') if k == 0 else \
            float((mixed[:, :k] != codes[:, :k]).float().mean())
        print(f'  {k:3d}   {hit:.3f}    {frac:.3f}')


if __name__ == '__main__':
    main()
