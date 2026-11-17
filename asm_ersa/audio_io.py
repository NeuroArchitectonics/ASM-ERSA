"""
音频 I/O — soundfile + torch 实现
==================================
替代 torchaudio（torchaudio 与 torch 2.14 无版本匹配，强行安装会破坏 torch）。

提供：
  load_wav        — 读 wav/flac/ogg → [C, T] float32
  resample        — 窗化 sinc 重采样
  mel_filterbank  — HTK mel 三角滤波器组（peak=1，与 torchaudio norm=None 一致）
  log_mel         — log-mel 频谱，对齐 torchaudio MelSpectrogram + AmplitudeToDB
  load_audio_mel  — 端到端：路径 → 归一化 log-mel [n_mels, T]

数值语义对齐 torchaudio 默认参数：
  MelSpectrogram(n_fft=1024, hop_length=256, win_length=n_fft, power=2.0,
                 center=True, pad_mode='reflect', mel_scale='htk', norm=None)
  AmplitudeToDB(multiplier=10.0, amin=1e-10, top_db=80.0)
"""

import math
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "load_wav",
    "resample",
    "mel_filterbank",
    "log_mel",
    "load_audio_mel",
]


# ═══════════════════════════════════════════════════════════════
# 读取
# ═══════════════════════════════════════════════════════════════

def load_wav(path: str) -> Tuple[Tensor, int]:
    """
    读取音频文件。

    Returns:
        waveform: [C, T] float32，范围约 [-1, 1]
        sample_rate: int
    """
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # soundfile 返回 [T, C]，转成 [C, T]
    waveform = torch.from_numpy(data.T.copy())
    return waveform, int(sr)


# ═══════════════════════════════════════════════════════════════
# 重采样（窗化 sinc 插值）
# ═══════════════════════════════════════════════════════════════

def resample(waveform: Tensor, orig_sr: int, target_sr: int,
             lowpass_width: int = 6, rolloff: float = 0.99) -> Tensor:
    """
    窗化 sinc 重采样。waveform: [C, T] → [C, T']

    升采样与降采样统一处理：截止频率取 min(1, target/orig) * rolloff，
    窗函数用 sinc 窗（Lanczos 形式），逐输出点归一化保证直流增益为 1。
    """
    if orig_sr == target_sr:
        return waveform
    if waveform.numel() == 0:
        return waveform

    C, T = waveform.shape
    ratio = orig_sr / target_sr
    T_out = int(math.floor(T / ratio))
    if T_out <= 0:
        return waveform.new_zeros(C, 0)

    cutoff = min(1.0, 1.0 / ratio) * rolloff
    half = max(int(math.ceil(lowpass_width / cutoff)), 1)

    n = torch.arange(-half + 1, half + 1, dtype=torch.float64)
    t_out = torch.arange(T_out, dtype=torch.float64) * ratio
    base = torch.floor(t_out)
    frac = t_out - base

    d = frac[:, None] + n[None, :]
    w = torch.sinc(cutoff * d) * torch.sinc(cutoff * d / half) * cutoff
    w = w / w.sum(-1, keepdim=True)

    idx = base[:, None].long() - n[None, :].long()
    valid = (idx >= 0) & (idx < T)
    idx = idx.clamp_(0, T - 1)

    x = waveform.to(torch.float64)
    gathered = x[:, idx.reshape(-1)].reshape(C, T_out, -1)
    gathered = gathered * valid.unsqueeze(0)
    out = (gathered * w.unsqueeze(0)).sum(-1)
    return out.to(waveform.dtype)


# ═══════════════════════════════════════════════════════════════
# Mel 滤波器组
# ═══════════════════════════════════════════════════════════════

def _hz_to_mel(f: Tensor) -> Tensor:
    return 2595.0 * torch.log10(1.0 + f / 700.0)


def _mel_to_hz(m: Tensor) -> Tensor:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int,
                   f_min: float = 0.0, f_max: Optional[float] = None) -> Tensor:
    """
    HTK mel 三角滤波器组，峰值 1（等价 torchaudio mel_scale='htk', norm=None）。

    Returns: [n_freqs, n_mels]，n_freqs = n_fft // 2 + 1
    """
    n_freqs = n_fft // 2 + 1
    if f_max is None:
        f_max = sample_rate / 2.0

    all_freqs = torch.linspace(0.0, sample_rate / 2.0, n_freqs, dtype=torch.float64)
    m_pts = torch.linspace(float(_hz_to_mel(torch.tensor(f_min))),
                           float(_hz_to_mel(torch.tensor(f_max))),
                           n_mels + 2, dtype=torch.float64)
    f_pts = _mel_to_hz(m_pts)

    fb = torch.zeros(n_freqs, n_mels, dtype=torch.float64)
    for i in range(n_mels):
        left, center, right = f_pts[i], f_pts[i + 1], f_pts[i + 2]
        down = (all_freqs - left) / (center - left)
        up = (right - all_freqs) / (right - center)
        fb[:, i] = torch.clamp(torch.minimum(down, up), min=0.0)
    return fb


# ═══════════════════════════════════════════════════════════════
# log-mel
# ═══════════════════════════════════════════════════════════════

def log_mel(waveform: Tensor, sample_rate: int, n_fft: int = 1024,
            hop_length: int = 256, n_mels: int = 80,
            f_min: float = 0.0, f_max: Optional[float] = None,
            top_db: float = 80.0, amin: float = 1e-10) -> Tensor:
    """
    waveform [C, T] 或 [T] → log-mel [n_mels, n_frames]

    对齐 torchaudio MelSpectrogram(power=2.0, center=True, pad_mode='reflect')
    + AmplitudeToDB(multiplier=10.0, amin=1e-10, top_db=80.0)
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    x = waveform.mean(dim=0, keepdim=True)  # 混合为单声道

    # center=True 的 reflect padding 要求 pad 宽度 < 信号长度；
    # 极短片段（< n_fft）补齐到 n_fft，否则 torch.stft 直接报错
    if x.shape[-1] < n_fft:
        x = F.pad(x, (0, n_fft - x.shape[-1]))

    window = torch.hann_window(n_fft, periodic=True, dtype=x.dtype, device=x.device)
    spec = torch.stft(
        x, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
        window=window, center=True, pad_mode="reflect",
        normalized=False, onesided=True, return_complex=True,
    )
    power = spec.abs().pow(2.0).squeeze(0)  # [n_freqs, n_frames]

    fb = mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max)
    fb = fb.to(power.dtype).to(power.device)
    mel = fb.transpose(0, 1) @ power  # [n_mels, n_frames]

    mel_db = 10.0 * torch.log10(torch.clamp(mel, min=amin))
    if top_db is not None and mel_db.numel() > 0:
        mel_db = torch.clamp(mel_db, min=mel_db.max() - top_db)
    return mel_db


def load_audio_mel(path: str, sample_rate: int = 16000, n_mels: int = 80,
                   n_fft: int = 1024, hop_length: int = 256,
                   max_seconds: Optional[float] = 30.0) -> Optional[Tensor]:
    """
    端到端：音频路径 → 归一化 log-mel [n_mels, T]

    失败返回 None（与原 torchaudio 实现行为一致，便于数据集跳过坏样本）。
    """
    try:
        waveform, sr = load_wav(path)

        if max_seconds is not None and max_seconds > 0:
            # 先按目标长度裁剪，避免对后面会被丢弃的尾部做重采样
            limit = int(math.ceil(max_seconds * sr)) + 2 * sr
            if waveform.shape[1] > limit:
                waveform = waveform[:, :limit]

        if sr != sample_rate:
            waveform = resample(waveform, sr, sample_rate)

        max_samples = int(max_seconds * sample_rate) if max_seconds else None
        if max_samples is not None and waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]

        mel = log_mel(waveform, sample_rate, n_fft=n_fft,
                      hop_length=hop_length, n_mels=n_mels)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        return mel
    except Exception:
        return None
