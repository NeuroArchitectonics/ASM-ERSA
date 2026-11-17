"""
P2 #20 可学性自检 + P3 #11 端到端闭环（前半：真实音频上过拟合）。

目的不是证明模型效果好，而是证明这条链路真的通了、且真的在学：
  真实 wav → Encodec 码(离线缓存) → 模型 next-frame 预测 → CE 下降
并以「打乱 target → loss 回到 ~ln1024」证明 loss 确实由 codes 驱动。

沿用 train_omni_3b.py 的真实管线（OmniDataset / collate_fn / compute_loss /
build_code_cache / build_model），不另写一套简化实现，否则验不到真实链路。

离线、可复现：不需要网络（char-level tokenizer，不依赖外部分词器）。

用法:
  # 1) 建子集 jsonl + 1s 截断的码缓存（首次运行自动做）
  python tests/overfit_audio_small.py --steps 200 --samples 8 --layers 4
  # 2) 完整参数见 --help
"""

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_omni_3b import (                                   # noqa: E402
    TrainConfig, OmniDataset, collate_fn, compute_loss, build_model,
    get_cosine_schedule, build_code_cache, read_jsonl_samples,
)

MEDIA_JSONL = '.cache/media70.jsonl'


# ═══════════════════════════════════════════════════════════════
# 离线 char-level tokenizer（不依赖任何外部词表）
# ═══════════════════════════════════════════════════════════════

class CharTokenizer:
    """text → ids，id 0 保留给未知字符。只为让实验完全离线可复现。"""

    def __init__(self, texts):
        chars = sorted({ch for t in texts for ch in t})
        self.itos = ['<unk>'] + chars
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}
        self.vocab_size = len(self.itos)

    def __call__(self, text, truncation=True, max_length=2048,
                 return_tensors='pt'):
        ids = [self.stoi.get(ch, 0) for ch in text[:max_length]] or [0]
        t = torch.tensor(ids, dtype=torch.long)
        return {'input_ids': t.unsqueeze(0)}


# ═══════════════════════════════════════════════════════════════
# 语料准备
# ═══════════════════════════════════════════════════════════════

def select_subset(all_samples, k: int, min_seconds: float, out_path: Path):
    """按文件名顺序取 k 条长度 ≥ min_seconds 的样本（确定性，不做挑选）"""
    picked = []
    for s in all_samples:
        p = s.get('audio_path')
        if not p:
            continue
        try:
            dur = sf.info(p).duration
        except Exception:
            continue
        if dur >= min_seconds:
            picked.append(s)
        if len(picked) == k:
            break
    if len(picked) < k:
        raise RuntimeError(f"只有 {len(picked)} 条 ≥{min_seconds}s，需要 {k} 条")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        for s in picked:
            fh.write(json.dumps(s, ensure_ascii=False) + '\n')
    return picked, out_path


def label_baselines(codes_list):
    """两个对照基线：边缘熵 H_marg、最优常数预测器 -log p_max"""
    cnt = Counter()
    for c in codes_list:
        cnt.update(c.tolist())
    total = sum(cnt.values())
    probs = torch.tensor([v / total for v in cnt.values()])
    h_marg = float(-(probs * probs.log()).sum())
    p_max = float(probs.max())
    return h_marg, -math.log(p_max), len(cnt), total


# ═══════════════════════════════════════════════════════════════
# 评估：同一份 logits 上做 真实 / 打乱 / 随机 三种 target
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_all(model, dataset, n_audio_tokens: int, seed: int = 0):
    """逐样本前向（B=1），返回 CE(真实) / CE(打乱 target) / CE(随机 target)"""
    model.eval()
    rows = []
    for i in range(len(dataset)):
        batch = collate_fn([dataset[i]])
        out = model(
            input_ids=batch['input_ids'],
            audio_mel=batch.get('audio_mel'),
            attention_mask=batch['attention_mask'].bool(),
            audio_valid_frames=batch.get('audio_valid_frames'),
            audio_codes_target=batch['audio_codes'],
            audio_codes_valid=batch['audio_codes_valid'],
        )
        logits = out['audio_target_logits'][0]                  # [T_code, K]
        codes = batch['audio_codes'][0]
        valid = batch['audio_codes_valid'][0]
        n_valid = int(valid.sum())

        def ce(labels):
            lab = labels.masked_fill(~valid, -100)
            return float(F.cross_entropy(logits, lab, ignore_index=-100))

        g = torch.Generator().manual_seed(seed * 1000 + i)
        perm = torch.randperm(codes.shape[0], generator=g)
        rand = torch.randint(0, n_audio_tokens, codes.shape, generator=g)

        rows.append({
            'idx': i,
            'n_frames': int(codes.shape[0]),
            'n_valid': n_valid,
            'ce_true': ce(codes),
            'ce_shuffled': ce(codes[perm]),
            'ce_random': ce(rand),
        })
    model.train()
    return rows


def mean_of(rows, key):
    return sum(r[key] for r in rows) / len(rows)


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--samples', type=int, default=8)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--max-seconds', type=float, default=1.0)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--out', type=str, default='.cache/overfit_audio_log.json')
    p.add_argument('--save-model', type=str, default='.cache/overfit_audio_ckpt')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # ── 语料子集 + 码缓存 ──────────────────────────────────────
    all_samples = read_jsonl_samples(MEDIA_JSONL)
    subset_jsonl = Path(f'.cache/overfit{args.samples}.jsonl')
    subset, subset_jsonl = select_subset(
        all_samples, args.samples, args.max_seconds, subset_jsonl)

    cfg = TrainConfig(
        d_model=512, n_heads=8, d_k=64, d_v=64, d_ff=384,
        n_layers=args.layers, top_k=4,
        max_audio_seconds=args.max_seconds,
        max_steps=args.steps, warmup_steps=args.warmup, lr_peak=args.lr,
        data_path=str(subset_jsonl),
        num_workers=0,
        output_dir='.cache/overfit_ckpt',
        save_every_steps=10 ** 9,
        log_every_steps=args.log_every,
        seed=args.seed,
    )
    cache_pt = build_code_cache(
        str(subset_jsonl), f'.cache/overfit{args.samples}_codes.pt', cfg)
    cfg.audio_codes_cache = str(cache_pt)

    tokenizer = CharTokenizer([s['text'] for s in subset])
    cfg.vocab_size = tokenizer.vocab_size
    dataset = OmniDataset(str(subset_jsonl), tokenizer, cfg)

    codes_list = [dataset[i]['audio_codes'] for i in range(len(dataset))]
    h_marg, ce_const, n_codes, n_frames = label_baselines(codes_list)
    print(f"语料: {len(subset)} 条, {n_frames} 帧, 用到 {n_codes}/1024 个码")
    print(f"基线: ln(1024)={math.log(1024):.3f}  边缘熵={h_marg:.3f}  "
          f"常数预测器={ce_const:.3f}")
    print("帧长:", [int(c.shape[0]) for c in codes_list])

    # ── 模型 / 优化器 ─────────────────────────────────────────
    model = build_model(cfg)
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    nodecay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    optimizer = torch.optim.AdamW([
        {'params': decay, 'weight_decay': cfg.weight_decay},
        {'params': nodecay, 'weight_decay': 0.0},
    ], lr=cfg.lr_peak, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)
    scheduler = get_cosine_schedule(
        optimizer, cfg.warmup_steps, cfg.max_steps, cfg.lr_min_ratio)

    # ── 训练前评估（随机初始化） ──────────────────────────────
    device = torch.device('cpu')
    t0 = time.time()
    init_rows = eval_all(model, dataset, cfg.n_audio_tokens, args.seed)
    print(f"\n[init] audio_loss(真实 target) = {mean_of(init_rows, 'ce_true'):.3f}"
          f"  打乱 = {mean_of(init_rows, 'ce_shuffled'):.3f}"
          f"  随机 = {mean_of(init_rows, 'ce_random'):.3f}"
          f"  ({time.time() - t0:.0f}s)")

    # compute_loss 与手工 CE 必须逐位一致（在真实链路上再验一次）
    b0 = collate_fn([dataset[0]])
    with torch.no_grad():
        ref = compute_loss(model, b0, cfg, device)
        out0 = model(
            input_ids=b0['input_ids'], audio_mel=b0.get('audio_mel'),
            attention_mask=b0['attention_mask'].bool(),
            audio_valid_frames=b0.get('audio_valid_frames'),
            audio_codes_target=b0['audio_codes'],
            audio_codes_valid=b0['audio_codes_valid'])
        manual = F.cross_entropy(
            out0['audio_target_logits'][0],
            b0['audio_codes'][0].masked_fill(~b0['audio_codes_valid'][0], -100),
            ignore_index=-100)
    print(f"[check] compute_loss={float(ref['audio_loss']):.8f} "
          f"手工CE={float(manual):.8f}")
    assert abs(float(ref['audio_loss']) - float(manual)) < 1e-6, \
        "compute_loss 的音频分支与手工 CE 不一致"

    # ── 训练循环 ──────────────────────────────────────────────
    print(f"\n训练 {args.steps} 步 (d_model=512, n_layers={args.layers}, "
          f"lr={args.lr}, warmup={args.warmup}) ...")
    history, log_buf = [], []
    model.train()
    optimizer.zero_grad()
    idx = 0
    t0 = time.time()
    for step in range(1, args.steps + 1):
        sample = dataset[idx % len(dataset)]
        idx += 1
        batch = collate_fn([sample])
        losses = compute_loss(model, batch, cfg, device)
        losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        def _f(v):
            return float(v.detach())

        rec = {'step': step,
               'lr': scheduler.get_last_lr()[0],
               'text_loss': _f(losses['text_loss']),
               'audio_loss': _f(losses['audio_loss']),
               'total_loss': _f(losses['total_loss'])}
        if 'balance_loss' in losses:
            rec['balance_loss'] = _f(losses['balance_loss'])
        history.append(rec)
        log_buf.append(rec)

        if step % args.log_every == 0 or step == 1:
            n = len(log_buf)
            avg = {k: sum(r[k] for r in log_buf) / n
                   for k in ('text_loss', 'audio_loss', 'total_loss')}
            el = time.time() - t0
            print(f"step {step:4d}/{args.steps}  text={avg['text_loss']:6.3f}  "
                  f"audio={avg['audio_loss']:6.3f}  total={avg['total_loss']:6.3f}"
                  f"  lr={rec['lr']:.2e}  [{el:.0f}s, {el / step:.2f}s/step]")
            log_buf = []

    train_sec = time.time() - t0

    # ── 训练后评估 ────────────────────────────────────────────
    final_rows = eval_all(model, dataset, cfg.n_audio_tokens, args.seed)
    print(f"\n[final] audio_loss(真实 target) = {mean_of(final_rows, 'ce_true'):.3f}"
          f"  打乱 = {mean_of(final_rows, 'ce_shuffled'):.3f}"
          f"  随机 = {mean_of(final_rows, 'ce_random'):.3f}")
    print("逐样本 ce_true:",
          ' '.join(f"{r['ce_true']:.2f}" for r in final_rows))

    init_true = mean_of(init_rows, 'ce_true')
    final_true = mean_of(final_rows, 'ce_true')
    final_shuf = mean_of(final_rows, 'ce_shuffled')
    bar = 5.5   # 计划 §4-11 的判据
    print(f"\n判据: 6.93 → {init_true:.3f} → {final_true:.3f} "
          f"(< {bar} ? {'通过' if final_true < bar else '未达'})")
    print(f"      target 打乱后 {final_shuf:.3f} "
          f"(上升 {final_shuf - final_true:.3f} nats)")

    if args.save_model:
        Path(args.save_model).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.save_model)
        print(f"权重已存 {args.save_model}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump({
            'args': vars(args),
            'baselines': {'ln1024': math.log(1024), 'h_marginal': h_marg,
                          'ce_const': ce_const, 'n_codes_used': n_codes,
                          'n_frames': n_frames,
                          'frames_per_sample': [int(c.shape[0]) for c in codes_list]},
            'init': init_rows, 'final': final_rows,
            'history': history, 'train_sec': train_sec,
        }, fh, ensure_ascii=False, indent=1)
    print(f"日志已存 {args.out}")


if __name__ == '__main__':
    main()
