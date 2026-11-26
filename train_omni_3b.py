"""
ASM-ERSA Omni 3B — 多模态训练脚本
================================
支持 text + audio + vision 联合训练

用法:
  # 单机 8 卡
  torchrun --nproc_per_node=8 train_omni_3b.py \
      --data-path /data/omni_train \
      --output-dir /checkpoints/asm_ersa-omni-3b

  # 多机 (2 节点 × 8 卡)
  torchrun --nproc_per_node=8 --nnodes=2 \
      --node_rank=$RANK --master_addr=$MASTER_ADDR \
      train_omni_3b.py --data-path /data/omni_train

数据格式 (jsonl 每行):
  {"text": "hello", "audio_path": "a.wav", "image_path": "b.jpg"}
  {"text": "describe this", "image_path": "c.png"}
  {"text": "transcribe", "audio_path": "d.wav"}
  {"text": "pure text sample"}

音频码预缓存 (P2，正式训练必须先做):
  # 1) 只有 wav 目录时先生成 jsonl
  python train_omni_3b.py --make-jsonl C:/Windows/Media data/media.jsonl
  # 2) 把全部 audio_path 编码成单码本 codes（可重入，已编码的会跳过）
  python train_omni_3b.py --data-path data/media.jsonl --build-code-cache data/codes.pt
  # 3) 训练时挂上缓存；不挂则每个样本现场跑 Encodec（很慢，仅兜底）
  python train_omni_3b.py --data-path data/media.jsonl --audio-codes-cache data/codes.pt
"""

import os
import json
import math
import time
import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, size_based_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)

from torchvision import transforms
from PIL import Image

from asm_ersa import ASM_ERSA_Omni, Modality, OmniTransformerBlock
from asm_ersa.audio_io import load_audio_mel
from asm_ersa.audio_codec import EncodecTokenizer, ENCODEC_24KHZ

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 训练配置
# ═══════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    # 模型（3B Omni）
    d_model: int = 2048
    n_heads: int = 16
    d_k: int = 128
    d_v: int = 128
    d_ff: int = 384
    n_layers: int = 32
    vocab_size: int = 152064
    n_shared_experts: int = 8
    n_text_experts: int = 12
    n_audio_experts: int = 6
    n_vision_experts: int = 6
    top_k: int = 4
    optimizer: str = 'adamw'            # 'adamw' | 'apollo'（6B-lite 走 apollo）
    tie_embeddings: bool = False        # True: lm_head 与词嵌入共享（6B-lite 用）
    audio_output_mode: str = 'discrete'
    n_audio_tokens: int = 1024          # = EncodecTokenizer.codebook_size（asm_ersa/audio_codec.py）

    # 音频
    n_mels: int = 80
    sample_rate: int = 16000
    max_audio_seconds: float = 30.0

    # 视觉
    image_size: int = 336

    # 训练
    seed: int = 42
    max_steps: int = 100_000
    global_batch_size: int = 256
    micro_batch_size: int = 1
    max_seq_len: int = 2048
    warmup_steps: int = 2000
    lr_peak: float = 3e-4
    lr_min_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    warmup_tokens: int = 0

    # 损失权重
    loss_text_weight: float = 1.0
    loss_audio_weight: float = 0.5
    loss_moe_balance: float = 0.01

    # 精度
    dtype: str = 'bfloat16'

    # 检查点
    save_every_steps: int = 2000
    keep_last_n: int = 5
    resume_from: Optional[str] = None

    # 日志
    log_every_steps: int = 10
    wandb_project: str = 'asm_ersa-omni'
    wandb_run_name: Optional[str] = None

    # 数据
    data_path: str = ''
    num_workers: int = 4
    prefetch_factor: int = 2
    # 音频码离线缓存（codes.pt）。空字符串 = 每个样本现场跑 Encodec（慢，仅兜底）
    audio_codes_cache: str = ''
    codec_model: str = ENCODEC_24KHZ
    codec_bandwidth: float = 1.5

    # 输出
    output_dir: str = './checkpoints/asm_ersa-omni-3b'


def TrainConfig6BLite() -> TrainConfig:
    """6B-lite：总参 ~5.99B、每 token 激活 ~0.93B。

    与 3B 同宽（D=2048）、同 expert 尺寸（F=384），靠三处改动把总参顶到 6B 同时压住激活：
      - n_layers 32 → 24（token_mixing 是全激活项，与层数线性）
      - 专家 32 → 138（top-4，激活比例 4/138 ≈ 2.9%），按 3B 的模态分区比例放大
      - tie_embeddings=True（lm_head 与词嵌入共享，省一份 311M）
    """
    c = TrainConfig()
    c.n_layers = 24
    c.n_shared_experts = 34
    c.n_text_experts = 52
    c.n_audio_experts = 26
    c.n_vision_experts = 26
    c.tie_embeddings = True
    c.optimizer = 'apollo_mini'
    c.output_dir = './checkpoints/asm_ersa-omni-6b-lite'
    return c


# ═══════════════════════════════════════════════════════════════
# 音频码离线预缓存
# ═══════════════════════════════════════════════════════════════

def read_jsonl_samples(data_path: str) -> List[Dict[str, Any]]:
    """data_path 是 jsonl 文件或含 jsonl 的目录 → 样本列表"""
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"数据路径不存在: {data_path}")
    files = [data_file] if data_file.is_file() else sorted(data_file.glob('*.jsonl'))
    if not files:
        raise FileNotFoundError(f"{data_path} 下没有 .jsonl 文件")

    samples = []
    for f in files:
        with open(f, 'r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
    log.info(f"加载 {len(samples)} 条样本 from {len(files)} 文件")
    return samples


def collect_audio_paths(samples: List[Dict[str, Any]]) -> List[str]:
    """去重后的 audio_path 列表（保持出现顺序）"""
    seen, paths = set(), []
    for s in samples:
        p = s.get('audio_path') or ''
        if p and p not in seen:
            seen.add(p)
            paths.append(p)
    return paths


def make_code_cache_meta(cfg: TrainConfig, codebook_size: int, frame_rate: float,
                         sample_rate: int) -> Dict[str, Any]:
    return {
        'codec_model': cfg.codec_model,
        'codec_bandwidth': float(cfg.codec_bandwidth),
        'codebook_size': int(codebook_size),
        'frame_rate': float(frame_rate),
        'sample_rate': int(sample_rate),
        'max_seconds': float(cfg.max_audio_seconds),
    }


def verify_code_cache_meta(meta: Dict[str, Any], cfg: TrainConfig) -> None:
    """
    校验缓存与当前配置兼容。

    这四项不一致都会造成**静默错位**：码本大小不符 → 输出头与 target 空间不匹配；
    max_seconds 不符 → 缓存码长与截断后的音频长度不一致。宁可报错也不要静默复用。
    """
    checks = [
        ('codec_model', cfg.codec_model, '编解码器'),
        ('codec_bandwidth', float(cfg.codec_bandwidth), '带宽'),
        ('codebook_size', cfg.n_audio_tokens, '码本大小'),
        ('max_seconds', float(cfg.max_audio_seconds), '截断秒数'),
    ]
    for key, expect, desc in checks:
        got = meta.get(key)
        got_cmp = float(got) if isinstance(expect, float) else got
        if got_cmp != expect:
            raise ValueError(
                f"音频码缓存与配置不匹配（{desc}）: 缓存 {key}={got!r}，"
                f"当前配置={expect!r}。请用 --build-code-cache 重建。")


def build_code_cache(data_path: str, out_path: str, cfg: TrainConfig,
                     rebuild: bool = False) -> Path:
    """
    把数据集中全部 audio_path 编码为单码本 codes，存成一个 .pt。

    产物: {'codes': {path: LongTensor[T_frames]}, 'meta': {...}}
    可重入：已存在的 .pt 会校验 meta 后复用，只补编码缺失的路径。
    """
    samples = read_jsonl_samples(data_path)
    paths = collect_audio_paths(samples)
    if not paths:
        raise ValueError(f"{data_path} 中没有 audio_path，无需预缓存")

    out = Path(out_path)
    codes: Dict[str, Tensor] = {}
    if out.exists() and not rebuild:
        blob = torch.load(out, map_location='cpu', weights_only=False)
        verify_code_cache_meta(blob['meta'], cfg)
        codes = {p: blob['codes'][p] for p in paths if p in blob['codes']}
        log.info(f"复用已有缓存 {out}: 命中 {len(codes)}/{len(paths)}")

    todo = [p for p in paths if p not in codes]
    if not todo:
        log.info(f"音频码缓存已完整: {out}（{len(codes)} 条）")
        return out

    tok = EncodecTokenizer(cfg.codec_model, cfg.codec_bandwidth)
    log.info(f"编码 {len(todo)} 条音频 (codec={cfg.codec_model}, "
             f"bandwidth={cfg.codec_bandwidth}, max_seconds={cfg.max_audio_seconds})")

    failed, t0 = [], time.time()
    for i, p in enumerate(todo, 1):
        c = tok.encode_file(p, max_seconds=cfg.max_audio_seconds)
        if c is None or c.numel() == 0:
            failed.append(p)
        else:
            codes[p] = c
        if i % 50 == 0 or i == len(todo):
            log.info(f"  {i}/{len(todo)}  ({time.time() - t0:.1f}s)")

    if failed:
        log.warning(f"{len(failed)} 条编码失败，训练时这些样本会退化为纯文本: "
                    f"{failed[:10]}{' ...' if len(failed) > 10 else ''}")

    meta = make_code_cache_meta(cfg, tok.codebook_size, tok.frame_rate, tok.sample_rate)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + '.tmp')
    torch.save({'codes': codes, 'meta': meta}, tmp)
    tmp.replace(out)                                     # 原子替换，避免半截缓存
    frames = sum(int(c.numel()) for c in codes.values())
    log.info(f"缓存已写入 {out}: {len(codes)} 条, {frames} 帧, meta={meta}")
    return out


def build_jsonl_from_wav_dir(wav_dir: str, out_path: str) -> Path:
    """
    目录下 *.wav → jsonl，text 取文件名主干（真实词，不再用随机数占位）。
    用于 C:/Windows/Media 这类只有音频的语料。
    """
    wavs = sorted(Path(wav_dir).glob('*.wav'))
    if not wavs:
        raise FileNotFoundError(f"{wav_dir} 下没有 .wav")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as fh:
        for w in wavs:
            fh.write(json.dumps(
                {'text': w.stem, 'audio_path': str(w)},
                ensure_ascii=False) + '\n')
    log.info(f"已写出 {len(wavs)} 行 → {out}")
    return out


# ═══════════════════════════════════════════════════════════════
# 数据集
# ═══════════════════════════════════════════════════════════════

class OmniDataset(Dataset):
    """
    多模态 JSONL 数据集

    每行 JSON:
      {"text": str, "audio_path"?: str, "image_path"?: str}

    至少包含 text，可选 audio 和/或 image。
    """

    def __init__(self, data_path: str, tokenizer, cfg: TrainConfig):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.samples = read_jsonl_samples(data_path)

        self.image_transform = transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])

        self.code_cache: Dict[str, Tensor] = {}
        if cfg.audio_codes_cache:
            blob = torch.load(cfg.audio_codes_cache, map_location='cpu',
                              weights_only=False)
            verify_code_cache_meta(blob['meta'], cfg)
            self.code_cache = blob['codes']
            log.info(f"音频码缓存: {len(self.code_cache)} 条 "
                     f"from {cfg.audio_codes_cache}")

        self._codec: Optional[EncodecTokenizer] = None
        self._warned_miss = False

    def __len__(self):
        return len(self.samples)

    def _load_audio_mel(self, path: str) -> Optional[torch.Tensor]:
        mel = load_audio_mel(
            path,
            sample_rate=self.cfg.sample_rate,
            n_mels=self.cfg.n_mels,
            max_seconds=self.cfg.max_audio_seconds,
        )
        if mel is None:
            log.warning(f"音频加载失败 {path}")
        return mel

    def _load_audio_codes(self, path: str) -> Optional[Tensor]:
        codes = self.code_cache.get(path)
        if codes is not None:
            return codes

        # 缓存未命中：现场编码。multiprocessing worker 各持一份 Encodec（100MB 级），
        # 且每步都要跑一次编码器前向，只作兜底，正式训练必须预缓存。
        if not self._warned_miss:
            log.warning("音频码缓存未命中，将现场跑 Encodec（很慢）。"
                        "建议先执行 --build-code-cache 预缓存。")
            self._warned_miss = True
        if self._codec is None:
            self._codec = EncodecTokenizer(self.cfg.codec_model,
                                           self.cfg.codec_bandwidth)
        codes = self._codec.encode_file(path, max_seconds=self.cfg.max_audio_seconds)
        if codes is None or codes.numel() == 0:
            log.warning(f"音频编码失败 {path}")
            return None
        return codes

    def _load_image(self, path: str) -> Optional[torch.Tensor]:
        try:
            img = Image.open(path).convert('RGB')
            return self.image_transform(img)
        except Exception as e:
            log.warning(f"图像加载失败 {path}: {e}")
            return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        text = sample.get('text', '')

        input_ids = self.tokenizer(
            text, truncation=True,
            max_length=self.cfg.max_seq_len,
            return_tensors='pt',
        )['input_ids'].squeeze(0)

        result = {
            'input_ids': input_ids,
            'has_audio': False,
            'has_image': False,
        }

        if 'audio_path' in sample and sample['audio_path']:
            path = sample['audio_path']
            mel = self._load_audio_mel(path)
            codes = self._load_audio_codes(path) if mel is not None else None
            # mel 与 codes 全有才算有效音频样本：缺 codes 就没有监督目标，
            # 只喂 mel 会让该样本白占一个音频段，不如退化为纯文本。
            if mel is not None and codes is not None:
                result['audio_mel'] = mel
                result['audio_codes'] = codes
                result['has_audio'] = True

        if 'image_path' in sample and sample['image_path']:
            img = self._load_image(sample['image_path'])
            if img is not None:
                result['images'] = img
                result['has_image'] = True

        return result


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """变长 batch 整理 — 文本右 padding"""
    max_len = max(item['input_ids'].shape[0] for item in batch)
    B = len(batch)

    input_ids = torch.zeros(B, max_len, dtype=torch.long)
    attention_mask = torch.zeros(B, max_len, dtype=torch.long)

    has_audio = any(item['has_audio'] for item in batch)
    has_image = any(item['has_image'] for item in batch)

    # 模型按 [audio | vision | text] 固定拼接，段长必须与 batch 维一致：
    # 部分样本缺模态会直接造成 shape 崩溃，这里提前显式拒绝。
    if has_audio and not all(item['has_audio'] for item in batch):
        raise ValueError(
            "batch 内音频必须全有或全无：模型按固定顺序拼接各模态，"
            "部分缺失会导致 shape 不一致。请在数据侧按模态分桶。")
    if has_image and not all(item['has_image'] for item in batch):
        raise ValueError(
            "batch 内图像必须全有或全无：模型按固定顺序拼接各模态，"
            "部分缺失会导致 shape 不一致。请在数据侧按模态分桶。")

    for i, item in enumerate(batch):
        L = item['input_ids'].shape[0]
        input_ids[i, :L] = item['input_ids']
        attention_mask[i, :L] = 1

    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'has_audio': has_audio,
        'has_image': has_image,
    }

    if has_audio:
        mels = [item['audio_mel'] for item in batch]
        valid_frames = torch.tensor([m.shape[-1] for m in mels], dtype=torch.long)
        mel_batch = torch.zeros(
            B, mels[0].shape[0], max(valid_frames).item(),
        )
        for i, m in enumerate(mels):
            mel_batch[i, :, :m.shape[-1]] = m
        result['audio_mel'] = mel_batch
        result['audio_valid_frames'] = valid_frames

        # 音频码：padding 只可能落在生成段尾部（causal 段，无后继位置），
        # 因此补齐值不会泄漏到任何被监督位；但仍用 valid 掩掉其 loss。
        codes = [item['audio_codes'] for item in batch]
        code_lens = torch.tensor([c.shape[0] for c in codes], dtype=torch.long)
        code_batch = torch.zeros(B, int(code_lens.max()), dtype=torch.long)
        code_valid = torch.zeros(B, int(code_lens.max()), dtype=torch.bool)
        for i, c in enumerate(codes):
            code_batch[i, :c.shape[0]] = c
            code_valid[i, :c.shape[0]] = True
        result['audio_codes'] = code_batch
        result['audio_codes_valid'] = code_valid

    if has_image:
        result['images'] = torch.stack([item['images'] for item in batch])

    return result


# ═══════════════════════════════════════════════════════════════
# 学习率调度
# ═══════════════════════════════════════════════════════════════

def get_cosine_schedule(optimizer, warmup_steps: int, max_steps: int,
                        min_ratio: float = 0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(progress, 1.0)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ═══════════════════════════════════════════════════════════════
# MoE 负载均衡损失
# ═══════════════════════════════════════════════════════════════

def compute_moe_balance_loss(model: nn.Module) -> torch.Tensor:
    """
    计算 MoE 负载均衡损失（Switch Transformer 风格）

    鼓励 expert 被均匀选择，防止 router collapse
    """
    balance_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    n_moe_layers = 0

    for layer in model.layers:
        moe = layer.moe
        gate_logits = moe.gate
        if not hasattr(gate_logits, 'weight'):
            continue

        n_experts = moe.n_experts
        top_k = moe.top_k

        router_weight = gate_logits.weight
        n = n_experts
        balance = (n ** 0.5) * F.normalize(
            router_weight.float(), dim=0
        ).norm() - 1.0
        balance_loss = balance_loss + balance
        n_moe_layers += 1

    if n_moe_layers > 0:
        balance_loss = balance_loss / n_moe_layers

    return balance_loss


# ═══════════════════════════════════════════════════════════════
# 损失计算
# ═══════════════════════════════════════════════════════════════

def compute_loss(
    model: ASM_ERSA_Omni,
    batch: Dict[str, Any],
    cfg: TrainConfig,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    计算多模态训练损失

    文本: 标准 next-token prediction CE
    音频: 生成段 next-frame CE（target = 真实 Encodec codes）
    MoE: 负载均衡正则
    """
    input_ids = batch['input_ids'].to(device)
    text_mask = batch['attention_mask'].to(device).bool()

    audio_mel = batch.get('audio_mel', None)
    audio_valid_frames = batch.get('audio_valid_frames', None)
    if audio_mel is not None:
        audio_mel = audio_mel.to(device)
    if audio_valid_frames is not None:
        audio_valid_frames = audio_valid_frames.to(device)

    audio_codes = batch.get('audio_codes', None)
    audio_codes_valid = batch.get('audio_codes_valid', None)
    if audio_codes is not None:
        audio_codes = audio_codes.to(device)
        audio_codes_valid = audio_codes_valid.to(device)

    images = batch.get('images', None)
    if images is not None:
        images = images.to(device)

    output = model(
        input_ids=input_ids,
        audio_mel=audio_mel,
        images=images,
        attention_mask=text_mask,
        audio_valid_frames=audio_valid_frames,
        audio_codes_target=audio_codes,
        audio_codes_valid=audio_codes_valid,
    )

    text_logits = output['text_logits']

    T_total = text_logits.shape[1]
    T_text = input_ids.shape[1]
    text_start = T_total - T_text

    # 真实 token 掩码（[audio | vision | text]），padding 位置不进 loss
    pad_mask = output.get('padding_mask')
    text_valid = pad_mask[:, text_start:] if pad_mask is not None else text_mask

    # 文本损失: next-token prediction（仅真实文本位置）
    shift_logits = text_logits[:, text_start:-1, :].contiguous()
    shift_labels = input_ids.clone()
    shift_labels[~text_valid] = -100
    shift_labels = shift_labels[:, 1:].contiguous()

    text_loss = F.cross_entropy(
        shift_logits.view(-1, cfg.vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    losses = {'text_loss': text_loss}
    total_loss = cfg.loss_text_weight * text_loss

    # 音频损失: 生成段 next-frame CE。
    # 模型侧已把生成段输入右移为 [<audio_start>, c_0..c_{T-2}]，故 audio_target_logits
    # 与真实码逐位对齐（位置 j 的 logits 就是 c_j 的预测），这里不再做任何 shift。
    # 位置是否被监督由 audio_codes_valid 决定（collate 补齐帧置 -100）。
    if 'audio_target_logits' in output:
        audio_logits = output['audio_target_logits']
        if tuple(audio_logits.shape[:2]) != tuple(audio_codes.shape):
            raise ValueError(
                f"audio_target_logits {tuple(audio_logits.shape)} 与 "
                f"audio_codes {tuple(audio_codes.shape)} 不对齐")
        audio_labels = audio_codes.masked_fill(~audio_codes_valid, -100)
        audio_loss = F.cross_entropy(
            audio_logits.reshape(-1, cfg.n_audio_tokens),
            audio_labels.reshape(-1),
            ignore_index=-100,
        )
        losses['audio_loss'] = audio_loss
        total_loss = total_loss + cfg.loss_audio_weight * audio_loss

    # MoE 负载均衡
    if cfg.loss_moe_balance > 0:
        balance = compute_moe_balance_loss(model)
        losses['balance_loss'] = balance
        total_loss = total_loss + cfg.loss_moe_balance * balance

    losses['total_loss'] = total_loss
    return losses


# ═══════════════════════════════════════════════════════════════
# 分布式工具
# ═══════════════════════════════════════════════════════════════

def setup_distributed():
    if 'RANK' in os.environ:
        dist.init_process_group('nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank, world_size, local_rank = 0, 1, 0
        if torch.cuda.is_available():
            dist.init_process_group('nccl', rank=0, world_size=1,
                                    init_method='tcp://localhost:29500')

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


# ═══════════════════════════════════════════════════════════════
# 检查点
# ═══════════════════════════════════════════════════════════════

def save_checkpoint(
    model, optimizer, scheduler, step: int,
    output_dir: str, cfg: TrainConfig, rank: int,
):
    if rank != 0:
        return

    ckpt_dir = Path(output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / f'checkpoint-{step}'
    ckpt_path.mkdir(parents=True, exist_ok=True)

    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': vars(cfg),
    }, ckpt_path / 'training_state.pt')

    model.save_pretrained(ckpt_path / 'model')

    latest = ckpt_dir / 'latest'
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    try:
        latest.symlink_to(ckpt_path.name)
    except OSError:
        # Windows 无符号链接权限（WinError 1314）时退化为文本指针
        latest.write_text(ckpt_path.name, encoding='utf-8')

    log.info(f"保存检查点: {ckpt_path}")

    ckpts = sorted(
        [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')],
        key=lambda d: int(d.name.split('-')[1]),
    )
    while len(ckpts) > cfg.keep_last_n:
        old = ckpts.pop(0)
        import shutil
        shutil.rmtree(old)
        log.info(f"删除旧检查点: {old}")


def _strip_fsdp_prefix(key: str) -> str:
    return key.replace('_fsdp_wrapped_module.', '').replace('_orig_mod.', '')


def load_checkpoint(path: str, model, optimizer=None, scheduler=None):
    ckpt_path = Path(path) / 'training_state.pt'
    if not ckpt_path.exists():
        log.warning(f"检查点不存在: {ckpt_path}")
        return 0

    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # strict=False + 前缀归一化后的核对：FSDP 包装差异可容忍，其余 key 不匹配必须报错
    incompat = model.load_state_dict(state['model_state_dict'], strict=False)
    want = {_strip_fsdp_prefix(k) for k in model.state_dict()}
    got = {_strip_fsdp_prefix(k) for k in state['model_state_dict']}
    missing = sorted(want - got)
    unexpected = sorted(got - want)
    if missing or unexpected:
        raise RuntimeError(
            f"检查点与模型不匹配: {ckpt_path}\n"
            f"  missing({len(missing)}): {missing[:10]}\n"
            f"  unexpected({len(unexpected)}): {unexpected[:10]}\n"
            f"  load_state_dict 报告: missing={len(incompat.missing_keys)} "
            f"unexpected={len(incompat.unexpected_keys)}")

    if optimizer and 'optimizer_state_dict' in state:
        optimizer.load_state_dict(state['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in state:
        scheduler.load_state_dict(state['scheduler_state_dict'])

    step = state.get('step', 0)
    log.info(f"从 step {step} 恢复训练")
    return step


# ═══════════════════════════════════════════════════════════════
# 模型构建
# ═══════════════════════════════════════════════════════════════

def build_model(cfg: TrainConfig) -> ASM_ERSA_Omni:
    model = ASM_ERSA_Omni(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_k=cfg.d_k,
        d_v=cfg.d_v,
        d_ff=cfg.d_ff,
        n_layers=cfg.n_layers,
        vocab_size=cfg.vocab_size,
        n_shared_experts=cfg.n_shared_experts,
        n_text_experts=cfg.n_text_experts,
        n_audio_experts=cfg.n_audio_experts,
        n_vision_experts=cfg.n_vision_experts,
        top_k=cfg.top_k,
        audio_output_mode=cfg.audio_output_mode,
        n_audio_tokens=cfg.n_audio_tokens,
        tie_embeddings=cfg.tie_embeddings,
        audio_config={
            'n_mels': cfg.n_mels,
            'patch_size': 4,
            'stride': 2,
        },
        vision_config={
            'image_size': cfg.image_size,
            'patch_size': 14,
            'n_layers': 2,
            'n_heads': 8,
        },
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expert = sum(p.numel() for n, p in model.named_parameters()
                 if '.moe.W1' in n or '.moe.W2' in n)
    n_exp = (cfg.n_shared_experts + cfg.n_text_experts
             + cfg.n_audio_experts + cfg.n_vision_experts)
    active = total - expert + expert * cfg.top_k // max(1, n_exp)
    log.info(f"模型参数: total={total / 1e9:.2f}B, trainable={trainable / 1e9:.2f}B, "
             f"active={active / 1e9:.3f}B (top{cfg.top_k}/{n_exp}), tied_embed={cfg.tie_embeddings}")

    ersa_count = sum(1 for l in model.layers if hasattr(l, 'is_ersa') and l.is_ersa)
    gdn_count = len(model.layers) - ersa_count
    log.info(f"层分布: ASM-ERSA={gdn_count}, ERSA-Omni={ersa_count}, total={len(model.layers)}")

    return model


def wrap_fsdp(model: ASM_ERSA_Omni, local_rank: int, dtype: torch.dtype) -> FSDP:
    mp_policy = MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=dtype,
        buffer_dtype=dtype,
    )

    auto_wrap_policy = size_based_auto_wrap_policy(
        model,
        min_num_params=1e6,
    )

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=local_rank,
        use_orig_params=True,
    )

    return model


def apply_gradient_checkpointing(model: nn.Module):
    non_reentrant = checkpoint_wrapper
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant,
        check_fn=lambda m: isinstance(m, OmniTransformerBlock),
    )
    log.info("已启用 gradient checkpointing (OmniTransformerBlock)")


# ═══════════════════════════════════════════════════════════════
# Tokenizer
# ═══════════════════════════════════════════════════════════════

def get_tokenizer(vocab_size: int):
    """
    加载分词器，优先使用 C++ 分词器（`分词器入口.hpp` 编译的 DLL）：

      ASM_ERSA_TOKENIZER_DLL  DLL 路径（设置后启用，不再走 HF）
      ASM_ERSA_TOKENIZER      分词器目录（含 vocab.json / tokenizer.json）

    否则回退到本地 HuggingFace 分词器目录。
    """
    dll_path = os.environ.get("ASM_ERSA_TOKENIZER_DLL", "")
    if dll_path:
        from asm_ersa.tokenizer_cpp import CppTokenizer

        model_dir = os.environ.get("ASM_ERSA_TOKENIZER", "tokenizer")
        tok = CppTokenizer(dll_path, model_dir)
        if tok.vocab_size != vocab_size:
            raise ValueError(
                f"C++ 分词器词表 {tok.vocab_size} 与 vocab_size={vocab_size} 不一致")
        return tok

    from transformers import AutoTokenizer

    path = os.environ.get("ASM_ERSA_TOKENIZER", "tokenizer")
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"未找到分词器目录：{path}；请设置 ASM_ERSA_TOKENIZER 指向与 "
            f"vocab_size={vocab_size} 兼容的 tokenizer 目录")
    tok = AutoTokenizer.from_pretrained(path)
    if tok.vocab_size != vocab_size:
        raise ValueError(
            f"分词器词表 {tok.vocab_size} 与 vocab_size={vocab_size} 不一致")
    return tok


# ═══════════════════════════════════════════════════════════════
# 训练主循环
# ═══════════════════════════════════════════════════════════════

def train(cfg: TrainConfig):
    if cfg.optimizer != 'adamw':
        raise SystemExit(
            f"train_omni_3b.py 只提供 AdamW 参考训练；optimizer={cfg.optimizer!r} 请走 "
            "VK 训练器（cpp/train_main.cpp，ASM_OPT=apollo）")
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(cfg.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed + rank)

    dtype_map = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32,
    }
    dtype = dtype_map[cfg.dtype]

    if is_main():
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        log.info(f"训练配置:\n{json.dumps(vars(cfg), indent=2, ensure_ascii=False)}")

    # Tokenizer & 数据
    tokenizer = get_tokenizer(cfg.vocab_size)
    dataset = OmniDataset(cfg.data_path, tokenizer, cfg)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True,
    ) if dist.is_initialized() else None

    loader = DataLoader(
        dataset,
        batch_size=cfg.micro_batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        pin_memory=True,
        drop_last=True,
    )

    # 模型
    model = build_model(cfg)

    if cfg.resume_from:
        load_checkpoint(cfg.resume_from, model)

    if torch.cuda.is_available():
        model = model.to(device)

    apply_gradient_checkpointing(model)

    if world_size > 1:
        model = wrap_fsdp(model, local_rank, dtype)
        log.info(f"FSDP 包装完成 (world_size={world_size})")

    # 优化器
    param_groups = [
        {'params': [p for p in model.parameters() if p.requires_grad and p.ndim >= 2],
         'weight_decay': cfg.weight_decay},
        {'params': [p for p in model.parameters() if p.requires_grad and p.ndim < 2],
         'weight_decay': 0.0},
    ]
    optimizer = torch.optim.AdamW(
        param_groups, lr=cfg.lr_peak,
        betas=(cfg.beta1, cfg.beta2), eps=cfg.eps,
    )
    scheduler = get_cosine_schedule(optimizer, cfg.warmup_steps, cfg.max_steps, cfg.lr_min_ratio)

    if cfg.resume_from and Path(cfg.resume_from, 'training_state.pt').exists():
        load_checkpoint(cfg.resume_from, model, optimizer, scheduler)

    # 训练循环
    grad_accum_steps = max(1, cfg.global_batch_size // (cfg.micro_batch_size * world_size))
    scaler = torch.amp.GradScaler('cuda', enabled=(cfg.dtype == 'float16'))

    step = 0
    tokens_seen = 0
    model.train()
    optimizer.zero_grad()

    log.info(f"开始训练: max_steps={cfg.max_steps}, "
             f"global_batch={cfg.global_batch_size}, "
             f"grad_accum={grad_accum_steps}, "
             f"world_size={world_size}")

    t0 = time.time()
    running_loss = 0.0

    while step < cfg.max_steps:
        if sampler is not None:
            sampler.set_epoch(step)

        for batch_idx, batch in enumerate(loader):
            if step >= cfg.max_steps:
                break

            is_accum_step = (batch_idx + 1) % grad_accum_steps != 0

            with torch.autocast(device_type='cuda', dtype=dtype, enabled=True):
                losses = compute_loss(model, batch, cfg, device)

            loss = losses['total_loss'] / grad_accum_steps
            scaler.scale(loss).backward()

            running_loss += losses['total_loss'].item()

            if not is_accum_step:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_clip,
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                step += 1
                batch_tokens = batch['attention_mask'].sum().item()
                tokens_seen += batch_tokens

                if is_main() and step % cfg.log_every_steps == 0:
                    elapsed = time.time() - t0
                    avg_loss = running_loss / cfg.log_every_steps
                    lr = scheduler.get_last_lr()[0]
                    tps = tokens_seen / elapsed

                    log.info(
                        f"step={step:>6d} | "
                        f"loss={avg_loss:.4f} | "
                        f"lr={lr:.2e} | "
                        f"grad_norm={grad_norm:.3f} | "
                        f"tokens={tokens_seen / 1e9:.2f}B | "
                        f"tok/s={tps:.0f} | "
                        f"elapsed={elapsed / 60:.1f}min"
                    )

                    try:
                        import wandb
                        if wandb.run is not None:
                            wandb.log({
                                'train/loss': avg_loss,
                                'train/lr': lr,
                                'train/grad_norm': grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                                'train/tokens_seen_B': tokens_seen / 1e9,
                                'train/tokens_per_sec': tps,
                            }, step=step)
                    except ImportError:
                        pass

                    running_loss = 0.0

                if step % cfg.save_every_steps == 0:
                    unwrapped = model.module if hasattr(model, 'module') else model
                    save_checkpoint(
                        unwrapped, optimizer, scheduler, step,
                        cfg.output_dir, cfg, rank,
                    )

    # 最终保存
    if is_main():
        unwrapped = model.module if hasattr(model, 'module') else model
        save_checkpoint(
            unwrapped, optimizer, scheduler, step,
            cfg.output_dir, cfg, rank,
        )
        log.info(f"训练完成! 总 tokens: {tokens_seen / 1e9:.2f}B, 总 steps: {step}")

    cleanup_distributed()


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='ASM-ERSA Omni 3B 训练')

    # 数据
    p.add_argument('--smoke', action='store_true',
                   help='随机数据冒烟：不需要 --data-path/tokenizer，跑几步前向+反向后退出')
    p.add_argument('--smoke-steps', type=int, default=3)
    p.add_argument('--data-path', type=str, default='',
                   help='训练数据路径 (jsonl 文件或包含 jsonl 的目录)')
    p.add_argument('--output-dir', type=str, default='./checkpoints/asm_ersa-omni-3b')

    # 音频码缓存
    p.add_argument('--audio-codes-cache', type=str, default='',
                   help='音频码预缓存 codes.pt 路径（训练用）')
    p.add_argument('--build-code-cache', type=str, default=None,
                   metavar='OUT_PT',
                   help='离线预缓存模式：编码全部 audio_path 到 OUT_PT 后退出')
    p.add_argument('--rebuild-code-cache', action='store_true',
                   help='配合 --build-code-cache，忽略已有缓存全量重编')
    p.add_argument('--make-jsonl', type=str, nargs=2, default=None,
                   metavar=('WAV_DIR', 'OUT_JSONL'),
                   help='从 wav 目录生成 jsonl（text 取文件名）后退出')
    p.add_argument('--codec-model', type=str, default=ENCODEC_24KHZ)
    p.add_argument('--codec-bandwidth', type=float, default=1.5)

    # 模型规模
    p.add_argument('--d-model', type=int, default=2048)
    p.add_argument('--n-heads', type=int, default=16)
    p.add_argument('--n-layers', type=int, default=32)
    p.add_argument('--d-ff', type=int, default=5504)
    p.add_argument('--vocab-size', type=int, default=152064)
    p.add_argument('--n-audio-tokens', type=int, default=1024,
                   help='音频码本大小，必须等于 codec 的 codebook_size')
    p.add_argument('--max-audio-seconds', type=float, default=30.0,
                   help='音频截断秒数（缓存与 mel 必须用同一值）')

    # MoE
    p.add_argument('--n-shared-experts', type=int, default=32)
    p.add_argument('--n-text-experts', type=int, default=48)
    p.add_argument('--n-audio-experts', type=int, default=24)
    p.add_argument('--n-vision-experts', type=int, default=24)
    p.add_argument('--top-k', type=int, default=4)

    # 训练超参
    p.add_argument('--max-steps', type=int, default=100_000)
    p.add_argument('--global-batch-size', type=int, default=256)
    p.add_argument('--micro-batch-size', type=int, default=1)
    p.add_argument('--max-seq-len', type=int, default=2048)
    p.add_argument('--warmup-steps', type=int, default=2000)
    p.add_argument('--lr-peak', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--dtype', type=str, default='bfloat16',
                   choices=['bfloat16', 'float16', 'float32'])

    # 损失权重
    p.add_argument('--loss-text-weight', type=float, default=1.0)
    p.add_argument('--loss-audio-weight', type=float, default=0.5)
    p.add_argument('--loss-moe-balance', type=float, default=0.01)

    # 检查点
    p.add_argument('--save-every-steps', type=int, default=2000)
    p.add_argument('--keep-last-n', type=int, default=5)
    p.add_argument('--resume-from', type=str, default=None)

    # 日志
    p.add_argument('--log-every-steps', type=int, default=10)
    p.add_argument('--wandb-project', type=str, default='asm_ersa-omni')
    p.add_argument('--wandb-run-name', type=str, default=None)

    # 数据加载
    p.add_argument('--num-workers', type=int, default=4)

    # 种子
    p.add_argument('--seed', type=int, default=42)

    return p.parse_args()


def run_smoke(steps: int = 3) -> None:
    """随机数据冒烟：不依赖数据集/分词器，验证 前向→loss→反向→优化器 闭环。"""
    torch.manual_seed(0)
    cfg = TrainConfig(
        d_model=128, n_heads=4, d_k=32, d_v=32, d_ff=256, n_layers=4,
        vocab_size=1000, n_shared_experts=2, n_text_experts=2,
        n_audio_experts=2, n_vision_experts=2, top_k=2,
        audio_output_mode='discrete', n_audio_tokens=256,
        n_mels=80, image_size=56, max_seq_len=64, max_steps=steps,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    B, T = 2, 16
    for step in range(1, steps + 1):
        ids = torch.randint(0, cfg.vocab_size, (B, T), device=device)
        mel = torch.randn(B, cfg.n_mels, 200, device=device)
        images = torch.randn(B, 3, cfg.image_size, cfg.image_size, device=device)
        out = model(input_ids=ids, audio_mel=mel, images=images)
        logits = out['text_logits']
        text_start = logits.shape[1] - T
        loss = F.cross_entropy(
            logits[:, text_start:-1, :].reshape(-1, cfg.vocab_size),
            ids[:, 1:].reshape(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        log.info("[smoke] step %d/%d loss=%.4f", step, steps, loss.item())
    log.info("[smoke] OK：随机数据闭环通过（无数据集/分词器依赖）")


def main():
    args = parse_args()

    if args.smoke:
        run_smoke(args.smoke_steps)
        return

    if args.make_jsonl is not None:
        build_jsonl_from_wav_dir(args.make_jsonl[0], args.make_jsonl[1])
        return

    cfg = TrainConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        vocab_size=args.vocab_size,
        n_shared_experts=args.n_shared_experts,
        n_text_experts=args.n_text_experts,
        n_audio_experts=args.n_audio_experts,
        n_vision_experts=args.n_vision_experts,
        top_k=args.top_k,
        max_steps=args.max_steps,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        max_seq_len=args.max_seq_len,
        warmup_steps=args.warmup_steps,
        lr_peak=args.lr_peak,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        dtype=args.dtype,
        loss_text_weight=args.loss_text_weight,
        loss_audio_weight=args.loss_audio_weight,
        loss_moe_balance=args.loss_moe_balance,
        save_every_steps=args.save_every_steps,
        keep_last_n=args.keep_last_n,
        resume_from=args.resume_from,
        log_every_steps=args.log_every_steps,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        num_workers=args.num_workers,
        seed=args.seed,
        audio_codes_cache=args.audio_codes_cache,
        codec_model=args.codec_model,
        codec_bandwidth=args.codec_bandwidth,
        max_audio_seconds=args.max_audio_seconds,
        n_audio_tokens=args.n_audio_tokens,
    )

    if args.build_code_cache is not None:
        if not cfg.data_path:
            raise SystemExit("--build-code-cache 需要同时给出 --data-path")
        build_code_cache(cfg.data_path, args.build_code_cache, cfg,
                         rebuild=args.rebuild_code_cache)
        return

    if not cfg.data_path:
        raise SystemExit("--data-path 必填（或用 --make-jsonl / --build-code-cache）")

    if is_main() or not dist.is_initialized():
        try:
            import wandb
            wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name,
                config=vars(cfg),
            )
        except (ImportError, Exception):
            pass

    train(cfg)


if __name__ == '__main__':
    main()
