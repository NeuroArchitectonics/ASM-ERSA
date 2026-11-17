"""采样参数探针：贪心 vs temperature/top-k，看自由生成能否逃离单码坍塌。"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from asm_ersa import ASM_ERSA_Omni                                    # noqa: E402
from train_omni_3b import TrainConfig, OmniDataset, collate_fn, read_jsonl_samples  # noqa: E402
from overfit_audio_small import MEDIA_JSONL, select_subset, CharTokenizer  # noqa: E402


def main():
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

    配置列表 = [
        ("greedy", dict(greedy=True)),
        ("T=1.0 k=8", dict(greedy=False, temperature=1.0, top_k=8)),
        ("T=0.8 k=8", dict(greedy=False, temperature=0.8, top_k=8)),
        ("T=1.2 k=16", dict(greedy=False, temperature=1.2, top_k=16)),
    ]

    for idx in (0, 3):
        batch = collate_fn([dataset[idx]])
        codes = batch['audio_codes'][0]
        T = int(codes.shape[0])
        print(f"\n样本[{idx}] T={T}  真值用码 {codes.unique().numel()} 个")
        for 名称, 参数 in 配置列表:
            torch.manual_seed(42)
            with torch.no_grad():
                gen = model.generate_audio(
                    audio_mel=batch['audio_mel'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'].bool(),
                    audio_valid_frames=batch.get('audio_valid_frames'),
                    n_frames=T, **参数)[0]
            准确率 = (gen == codes).float().mean().item()
            用码数 = gen.unique().numel()
            最高占比 = gen.bincount(minlength=1024).max().item() / T
            print(f"  {名称:<12} acc={准确率:.3f}  用码={用码数:>3}  最高频占比={最高占比:.2f}")


if __name__ == '__main__':
    raise SystemExit(main())
