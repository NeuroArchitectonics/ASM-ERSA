"""
音频离散码本 — AudioTokenizer 抽象接口 + Encodec 实现
====================================================

P2 交付：把音频监督从"随机 target"（旧 train_omni_3b.py:403-422 的
`torch.randint`）换成真实音频码本。

契约（冻结）：

    codebook_size : int                  单本码本大小
    frame_rate    : float                码帧率（Hz）
    sample_rate   : int                  输入音频采样率
    encode(wav)   -> LongTensor  [B, T_frames]
    decode(codes) -> FloatTensor [B, T_samples]

后续自训的 VQ-GAN 令牌器只要实现同一接口即可整体替换，训练脚本与模型
代码零改动——这是本轮把接口抽出来的唯一原因。除上表外还提供
`n_frames()` / `n_samples()` 做样本↔码帧的长度换算（target 对齐必须用它，
不允许再出现 `//4` 之类的猜测）。

不依赖 torchaudio（它与 torch 2.14 无版本匹配，强装会破坏 torch）：
Encodec 走 transformers，wav 读取/重采样走 asm_ersa.audio_io。

已知实现细节（2026-09-25 在 transformers 5.17.0 上实测，非文档推测）：
  - `EncodecModel.encode()` 返回的 `audio_codes` 形状为
    `[nb_frames, B, Q, T]`。本模型 `config.chunk_length_s is None`，
    nb_frames 恒为 1；取第 1 本码本得 `[B, T]`。
  - `EncodecModel.decode()` 的 `audio_codes` 形参是**帧的序列**，
    单帧须写成 `[codes[B, Q, T]]`，且 `audio_scales` 也必须是序列。
  - 本模型 `config.normalize is False`，`encode` 返回 `audio_scales=[None]`，
    故 `decode(..., [None])` 是 encode 的严格逆路径、**不需要**估计 scale；
    这对"解码预测出的码本"是必要条件（预测阶段拿不到真值 scale）。
  - 24kHz 下 hop_length=320 → frame_rate=75.0；帧数 = ceil(n_samples/320)
    （0.5s → 38 帧，1s → 75 帧，与 ceil 一致）。
  - CPU 上 encode 完全确定（同输入两次结果逐位相同），故离线预缓存安全。
"""

import math
from typing import Optional

import torch
from torch import Tensor

from .audio_io import load_wav, resample

__all__ = ["AudioTokenizer", "EncodecTokenizer", "ENCODEC_24KHZ"]

ENCODEC_24KHZ = "facebook/encodec_24khz"


# ═══════════════════════════════════════════════════════════════
# 抽象接口
# ═══════════════════════════════════════════════════════════════

class AudioTokenizer:
    """
    音频分词器契约。任何实现（Encodec / 自训 VQ-GAN）都必须满足：

      codebook_size : int   单本码本大小 K，模型输出头维度 = K
      frame_rate    : float 码帧率，决定 1 秒音频占多少 token
      sample_rate   : int   期望的输入采样率
      encode(wav)   : [B, T] float @ sample_rate  → [B, T_frames] long
      decode(codes) : [B, T_frames] long          → [B, T_samples] float

    作为基类提供（而非 typing.Protocol），以便携带 `n_frames`/`n_samples`
    的默认换算式；接口本身仍是鸭子类型，不要求继承。
    """

    codebook_size: int
    frame_rate: float
    sample_rate: int

    def n_frames(self, n_samples: int) -> int:
        """样本数 → 码帧数（向上取整，与卷积编码器实际输出一致）。"""
        hop = self.sample_rate / self.frame_rate
        return int(math.ceil(n_samples / hop))

    def n_samples(self, n_frames: int) -> int:
        """码帧数 → 样本数。"""
        return int(round(int(n_frames) * self.sample_rate / self.frame_rate))

    def encode(self, wav: Tensor) -> Tensor:
        raise NotImplementedError

    def decode(self, codes: Tensor) -> Tensor:
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════
# Encodec 实现
# ═══════════════════════════════════════════════════════════════

class EncodecTokenizer(AudioTokenizer):
    """
    Meta Encodec（`facebook/encodec_24khz`）包装，仅取 RVQ 第 1 本码本。

      - 权重冻结、eval 模式，不产生梯度
      - 带宽默认 1.5 kbps：24kHz 下用 2 本码本，但只取第 1 本（K=1024、75Hz）
      - 多码本问题见计划 §7（本轮不做），故 `encode` 只返回单本码

    权重首次使用需联网下载（约 80MB）；已缓存时可离线运行。
    """

    def __init__(self, model_name: str = ENCODEC_24KHZ, bandwidth: float = 1.5,
                 device: str = "cpu"):
        from transformers import EncodecModel

        self.model_name = model_name
        self.bandwidth = float(bandwidth)
        self.device = torch.device(device)

        model = EncodecModel.from_pretrained(model_name)
        model.eval().requires_grad_(False)
        self.model = model.to(self.device)

        cfg = model.config
        if cfg.chunk_length_s is not None:
            # 分块编码会让 audio_codes 出现 nb_frames>1，本实现按单帧假设取码
            raise NotImplementedError(
                f"{model_name} 配置了 chunk_length_s={cfg.chunk_length_s}，"
                f"分块编码未实现；EncodecTokenizer 假定 nb_frames == 1")

        self.sample_rate = int(cfg.sampling_rate)
        self.codebook_size = int(cfg.codebook_size)
        self.hop_length = int(cfg.hop_length)
        self.n_quantizers = int(
            model.quantizer.get_num_quantizers_for_bandwidth(self.bandwidth))
        self.frame_rate = self.sample_rate / self.hop_length

    def __repr__(self) -> str:
        return (f"EncodecTokenizer({self.model_name}, K={self.codebook_size}, "
                f"{self.frame_rate:.1f}Hz, bw={self.bandwidth}, "
                f"n_q={self.n_quantizers}→取第1本, sr={self.sample_rate})")

    # ───────────────────────────────────────────────────────────
    # 码本
    # ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(self, wav: Tensor) -> Tensor:
        """
        wav: [T] 或 [B, T] float，必须已重采样到 self.sample_rate、单声道。
        返回 [B, T_frames] int64，取值 [0, codebook_size)。
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.dim() != 2:
            raise ValueError(f"encode 期望 [T] 或 [B, T]，收到 {tuple(wav.shape)}")
        if wav.shape[-1] == 0:
            return torch.zeros(wav.shape[0], 0, dtype=torch.long, device=wav.device)

        wav = wav.to(self.device, dtype=torch.float32)
        out = self.model.encode(wav.unsqueeze(1), bandwidth=self.bandwidth)
        codes = out.audio_codes                      # [nb_frames=1, B, Q, T]
        if codes.shape[0] != 1:
            raise RuntimeError(f"意外的 nb_frames={codes.shape[0]}（期望 1）")
        return codes[0, :, 0, :].contiguous().to(torch.long)   # 第 1 本码本

    @torch.no_grad()
    def decode(self, codes: Tensor) -> Tensor:
        """
        codes: [T_frames] 或 [B, T_frames] int。
        返回 [B, T_samples] float32 单声道波形（未做幅度归一化）。
        """
        if codes.dim() == 1:
            codes = codes.unsqueeze(0)
        if codes.dim() != 2:
            raise ValueError(f"decode 期望 [T] 或 [B, T]，收到 {tuple(codes.shape)}")

        codes = codes.to(self.device, dtype=torch.long)
        # 单帧须包成序列；audio_scales=[None] 即不做 scale 估计
        out = self.model.decode([codes.unsqueeze(1)], [None])
        return out.audio_values.squeeze(1).to(torch.float32)    # [B, 1, S] → [B, S]

    # ───────────────────────────────────────────────────────────
    # 文件级便捷路径（预缓存脚本用）
    # ───────────────────────────────────────────────────────────

    def encode_file(self, path: str,
                    max_seconds: Optional[float] = None) -> Optional[Tensor]:
        """
        音频路径 → [T_frames] long（单条，已 squeeze）。

        读取 → 截断 → 重采样到 self.sample_rate → 混单声道 → encode。
        任何一步失败返回 None（与 audio_io.load_audio_mel 的坏样本约定一致）。
        """
        try:
            waveform, sr = load_wav(path)            # [C, T]
            if waveform.numel() == 0:
                return None

            if max_seconds is not None and max_seconds > 0:
                # 先按目标长度裁剪，避免对将被丢弃的尾部做重采样
                limit = int(math.ceil(max_seconds * sr)) + 2 * sr
                if waveform.shape[1] > limit:
                    waveform = waveform[:, :limit]

            if sr != self.sample_rate:
                waveform = resample(waveform, sr, self.sample_rate)

            if max_seconds is not None and max_seconds > 0:
                max_samples = int(max_seconds * self.sample_rate)
                if waveform.shape[1] > max_samples:
                    waveform = waveform[:, :max_samples]

            mono = waveform.mean(dim=0)              # [T]
            if mono.shape[0] < self.hop_length:      # 太短，编码器无输出帧
                return None
            return self.encode(mono)[0]
        except Exception:
            return None
