"""
text ↔ audio 跨模态语义对齐验证（真实音频）

数据：`C:/Windows/Media/*.wav`（Windows 自带 70 个真实音频，无需下载）
  - 音频：soundfile 读取 → log-mel → ASM_ERSA_Omni 音频编码器
  - 文本：文件名切词 → 词表 id（词表由文件名全集构建，含数字切分，
          故 "Alarm01" 与 "Alarm02" 可区分）

任务：对称 InfoNCE（批内负样本，对角线为正），检验 SemanticModalityEmbedding
      + 主干能否把同一段音频的两种模态表示拉近。

读出头：
  - 音频表示 = 前 T_audio 个位置中 padding_mask 为 True 的 hidden 均值
    （T_audio 由 ASM_ERSA_Omni.audio_token_len 精确计算，不用 //4 猜测）
  - 文本表示 = 文本段中 attention_mask 为 True 的 hidden 均值

指标：loss（随机基线 ln(N)）与 top-1 检索准确率（随机基线 1/N），
      两者都打印，避免只看 loss 下不去就误判。

用法:
  python train_semantic_alignment.py --dir C:/Windows/Media --epochs 30
"""

import argparse
import glob
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from asm_ersa.asm_ersa_omni import ASM_ERSA_Omni
from asm_ersa.audio_io import load_audio_mel

PAD_ID = 0
BOS_ID = 1
N_SPECIAL = 2

SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 256
PATCH_SIZE = 4
STRIDE = 2
MAX_SECONDS = 2.0


def tokenize_name(path: str) -> List[str]:
    """文件名 → 词列表。驼峰与数字边界处切开，使 'Alarm01' → ['alarm', '01']。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    words: List[str] = []
    for chunk in re.split(r'[^0-9A-Za-z]+', stem):
        if not chunk:
            continue
        # 在 小写→大写、字母→数字、数字→字母 的边界插入分隔
        spaced = re.sub(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])',
                        ' ', chunk)
        words.extend(w.lower() for w in spaced.split() if w)
    return words


class WindowsSoundDataset(Dataset):
    """
    每个样本 = 一个 wav 文件。
    音频右补 0 到固定帧数，文本右补 PAD_ID 到固定长度；
    两者都用 mask 记录真实区，模型与读出头据此排除补齐区。
    """

    def __init__(self, audio_dir: str):
        files = sorted(glob.glob(os.path.join(audio_dir, '*.wav')))
        if not files:
            raise FileNotFoundError(f"{audio_dir} 下没有 .wav 文件")

        mels: List[Tensor] = []
        self.names: List[str] = []
        self.word_lists: List[List[str]] = []
        self.n_valid_frames: List[int] = []
        for path in files:
            mel = load_audio_mel(path, sample_rate=SAMPLE_RATE, n_mels=N_MELS,
                                 n_fft=N_FFT, hop_length=HOP_LENGTH,
                                 max_seconds=MAX_SECONDS)
            if mel is None:
                print(f"跳过无法解码的文件: {path}")
                continue
            mels.append(mel)
            self.names.append(os.path.basename(path))
            self.word_lists.append(tokenize_name(path))
            self.n_valid_frames.append(mel.shape[1])

        if not mels:
            raise RuntimeError("所有音频都解码失败")

        vocab: Dict[str, int] = {}
        for words in self.word_lists:
            for w in words:
                if w not in vocab:
                    vocab[w] = len(vocab) + N_SPECIAL
        self.vocab = vocab
        self.vocab_size = len(vocab) + N_SPECIAL

        self.mel_frames = max(self.n_valid_frames)
        self.text_len = 1 + max(len(w) for w in self.word_lists)

        self.mels = torch.zeros(len(mels), N_MELS, self.mel_frames)
        for i, mel in enumerate(mels):
            self.mels[i, :, :mel.shape[1]] = mel

        self.text_ids = torch.full((len(mels), self.text_len), PAD_ID, dtype=torch.long)
        self.text_mask = torch.zeros((len(mels), self.text_len), dtype=torch.long)
        for i, words in enumerate(self.word_lists):
            ids = [BOS_ID] + [vocab[w] for w in words]
            self.text_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            self.text_mask[i, :len(ids)] = 1

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {
            'text_ids': self.text_ids[idx],
            'text_mask': self.text_mask[idx],
            'audio_mel': self.mels[idx],
            'n_valid_frames': torch.tensor(self.n_valid_frames[idx], dtype=torch.long),
        }


def masked_mean(hidden: Tensor, mask: Tensor, eps: float = 1e-6) -> Tensor:
    """hidden [B, T, D], mask [B, T] bool → [B, D]，全 False 的行返回 0。"""
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=eps)


def info_nce(t2a: Tensor, a2t: Tensor) -> Tensor:
    """对称 InfoNCE，对角线为正样本。t2a/a2t 形状 [B, N]。"""
    b = t2a.shape[0]
    target = torch.arange(b, device=t2a.device)
    return 0.5 * (F.cross_entropy(t2a, target) + F.cross_entropy(a2t, target))


def run_epoch(model: ASM_ERSA_Omni, loader: DataLoader, device: torch.device,
              optimizer: Optional[torch.optim.Optimizer], temperature: float,
              mel_frames: int) -> Tuple[float, float, int, int]:
    """返回 (平均 loss, top-1 准确率, 批内候选总数, batch 数)。"""
    train = optimizer is not None
    model.train(train)

    total_loss, total_correct, total_cand, n_batches = 0.0, 0, 0, 0
    for batch in loader:
        text_ids = batch['text_ids'].to(device)
        text_mask = batch['text_mask'].to(device)
        mel = batch['audio_mel'].to(device)
        n_valid = batch['n_valid_frames'].to(device)

        with torch.set_grad_enabled(train):
            out = model(input_ids=text_ids, audio_mel=mel,
                        attention_mask=text_mask, audio_valid_frames=n_valid)
            hidden = out['hidden_states']
            pad = out['padding_mask']

            t_audio = model.audio_token_len(mel_frames, PATCH_SIZE, STRIDE)
            # 位置切分必须与编码器实际输出一致：一旦不符，说明 token 数又被"猜"错了
            assert hidden.shape[1] == t_audio + text_ids.shape[1], (
                f"总长 {hidden.shape[1]} ≠ 音频 {t_audio} + 文本 {text_ids.shape[1]}，"
                f"audio_token_len 与实际编码器不符")
            audio_emb = masked_mean(hidden[:, :t_audio], pad[:, :t_audio])
            text_emb = masked_mean(hidden[:, t_audio:], pad[:, t_audio:])

            audio_emb = F.normalize(audio_emb, dim=-1)
            text_emb = F.normalize(text_emb, dim=-1)
            logits = text_emb @ audio_emb.T / temperature
            loss = info_nce(logits, logits.T)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        # 对角线唯一正样本，故 top-1 检索命中数 = 对角线被判为最大的行数
        correct = int((logits.argmax(dim=1) == torch.arange(logits.shape[0],
                                                            device=device)).sum())
        total_loss += loss.item()
        total_correct += correct
        total_cand += logits.shape[0]
        n_batches += 1

    return total_loss / n_batches, total_correct / max(total_cand, 1), total_cand, n_batches


def train_semantic_alignment(audio_dir: str, epochs: int, batch_size: int,
                             lr: float, temperature: float, seed: int,
                             d_model: int, n_layers: int, d_ff: int) -> None:
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    dataset = WindowsSoundDataset(audio_dir)
    print(f"真实音频 {len(dataset)} 条，词表 {dataset.vocab_size} 个词元，"
          f"mel {dataset.mel_frames} 帧，文本 {dataset.text_len} token")
    # 名字不重复时对角线才是唯一正样本，InfoNCE 才能用"对角线命中"当准确率
    assert len(set(dataset.names)) == len(dataset), "存在同名文件，批内会出现假负样本"

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = ASM_ERSA_Omni(
        d_model=d_model,
        n_heads=4,
        d_k=d_model // 4,
        d_v=d_model // 4,
        d_ff=d_ff,
        n_layers=n_layers,
        vocab_size=dataset.vocab_size,
        n_scales=4,
        n_shared_experts=2,
        n_text_experts=2,
        n_audio_experts=2,
        n_vision_experts=2,
        top_k=2,
        audio_config={'n_mels': N_MELS, 'patch_size': PATCH_SIZE, 'stride': STRIDE},
        audio_output_mode='continuous',  # 本脚本不用音频生成头，省掉离散词表参数
        state_grad_chunk=32,
    ).to(device)

    n_param = sum(p.numel() for p in model.parameters())
    n_mod = sum(p.numel() for p in model.modality_embed.parameters())
    print(f"Model params: {n_param / 1e6:.2f}M "
          f"(其中 SemanticModalityEmbedding {n_mod / 1e6:.3f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_cand = min(batch_size, len(dataset))
    print(f"\n随机基线：loss = ln({n_cand}) = {math.log(n_cand):.3f}，"
          f"top-1 准确率 = 1/{n_cand} = {1 / n_cand:.3f}\n")

    for epoch in range(epochs):
        loss, acc, cand, n_batches = run_epoch(
            model, loader, device, optimizer, temperature, dataset.mel_frames)
        scheduler.step()
        print(f"Epoch {epoch + 1:3d}/{epochs}  loss={loss:.4f}  "
              f"top1={acc:.3f}  (批内候选 {cand}，{n_batches} 批)")

    # 收尾评测：关 dropout、不更新参数，与训练曲线可比
    loss, acc, cand, _ = run_epoch(model, loader, device, None, temperature,
                                   dataset.mel_frames)
    print(f"\n最终（eval 模式）: loss={loss:.4f}  top1={acc:.3f}")

    # 这里的数字只能读作"梯度通路是通的、目标可学"，不能读作泛化能力：
    # 全量 70 条做全批训练 30 epoch 属于记忆而非泛化；且文件名共享词族
    # （Alarm01..05 共有 "alarm"），部分正确可由词族结构白拿。
    print("注意：70 条全批训练 = 记忆性验证，不是泛化性验证。")


def main():
    ap = argparse.ArgumentParser(description="text↔audio 跨模态对齐验证（真实音频）")
    ap.add_argument('--dir', default='C:/Windows/Media', help='真实 wav 所在目录')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=70)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--temperature', type=float, default=0.07)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--d-model', type=int, default=256)
    ap.add_argument('--n-layers', type=int, default=4)
    ap.add_argument('--d-ff', type=int, default=512)
    args = ap.parse_args()

    train_semantic_alignment(
        audio_dir=args.dir, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, temperature=args.temperature, seed=args.seed,
        d_model=args.d_model, n_layers=args.n_layers, d_ff=args.d_ff)


if __name__ == '__main__':
    main()
