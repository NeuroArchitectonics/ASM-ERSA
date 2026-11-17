"""
ASM-ERSA Omni — 多模态版本
========================
在 ASM-ERSA v4.0 基础上扩展为 Omni 模型（文本 + 音频 + 视觉）

核心改造：
  - ModalityEmbedding：可学习的模态标识嵌入
  - AudioEncoder：轻量 CNN 音频编码器（Mel → patch tokens）
  - VisionEncoder：轻量 ViT 视觉编码器（image patches → tokens）
  - ERSA-Omni：跨模态精确回召稀疏注意力（模态感知 mask）
  - ModalityMoE：模态感知路由（expert 按模态分区）
  - AudioHead：音频输出生成头（预测 audio tokens）

3B Omni 配置：
  d_model=2048, n_heads=16, n_layers=32, d_ff=384
  n_experts=32 (8 shared + 12 text + 6 audio + 6 vision), top_k=4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List, Dict, Tuple
from enum import IntEnum
import math

from .asm_ersa import (
    RMSNorm,
    GatedResidual, ASM_ERSA_Layer
)


# ═══════════════════════════════════════════════════════════════
# 模态定义
# ═══════════════════════════════════════════════════════════════

class Modality(IntEnum):
    TEXT = 0
    AUDIO = 1
    VISION = 2


# ═══════════════════════════════════════════════════════════════
# 语义模态嵌入（改进版）
# ═══════════════════════════════════════════════════════════════

class SemanticModalityEmbedding(nn.Module):
    """
    语义感知的模态嵌入：
      - 每个模态有独立的投影网络，保留输入语义信息
      - 可学习的模态标识嵌入作为偏置
      - 支持跨模态语义对齐

    相比原版 ModalityEmbedding：
      - 原版：3 个静态向量，无语义能力
      - 改进版：模态特定投影 + 语义保留
    """
    def __init__(self, d_model: int, n_modalities: int = 3, use_projection: bool = True):
        super().__init__()
        self.d_model = d_model
        self.use_projection = use_projection

        # 可学习的模态标识嵌入（保留，作为偏置）
        self.modality_bias = nn.Embedding(n_modalities, d_model)
        nn.init.normal_(self.modality_bias.weight, std=0.02)

        # 模态特定投影网络（核心改进）
        if use_projection:
            self.text_proj = nn.Sequential(
                nn.Linear(d_model, d_model * 2, bias=False),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model, bias=False),
                RMSNorm(d_model)
            )
            self.audio_proj = nn.Sequential(
                nn.Linear(d_model, d_model * 2, bias=False),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model, bias=False),
                RMSNorm(d_model)
            )
            self.vision_proj = nn.Sequential(
                nn.Linear(d_model, d_model * 2, bias=False),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model, bias=False),
                RMSNorm(d_model)
            )

    def forward(self, tokens: Tensor, modality: int) -> Tensor:
        """
        tokens: [B, T, d_model] — 编码器输出的 token embeddings
        modality: int — Modality.TEXT / AUDIO / VISION
        返回: [B, T, d_model] — 语义增强的模态嵌入
        """
        B, T, _ = tokens.shape

        # 模态特定投影（保留语义）
        if self.use_projection:
            if modality == Modality.TEXT:
                projected = self.text_proj(tokens)
            elif modality == Modality.AUDIO:
                projected = self.audio_proj(tokens)
            else:  # VISION
                projected = self.vision_proj(tokens)
        else:
            projected = tokens

        # 加上模态标识偏置
        mod_ids = torch.full((B, T), modality, dtype=torch.long, device=tokens.device)
        bias = self.modality_bias(mod_ids)

        return projected + bias


class ModalityEmbedding(nn.Module):
    """
    [已弃用] 原版模态嵌入，保留用于向后兼容
    建议使用 SemanticModalityEmbedding
    """
    def __init__(self, d_model: int, n_modalities: int = 3):
        super().__init__()
        self.emb = nn.Embedding(n_modalities, d_model)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, modality_ids: Tensor) -> Tensor:
        """modality_ids: [B] or [B, T] → [B, d_model] or [B, T, d_model]"""
        return self.emb(modality_ids)


# ═══════════════════════════════════════════════════════════════
# 音频编码器（轻量 CNN）
# ═══════════════════════════════════════════════════════════════

class AudioEncoder(nn.Module):
    """
    轻量 CNN 音频编码器：Mel 频谱 → patch tokens
    输入: [B, n_mels, T_audio] Mel 频谱图
    输出: [B, T_patches, d_model]

    架构：Conv1d 下采样 + 线性投影
    约 10M 参数（n_mels=80, d_model=2048）
    """
    def __init__(self, n_mels: int = 80, d_model: int = 2048,
                 patch_size: int = 4, stride: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, 256, kernel_size=patch_size, stride=stride, padding=patch_size // 2),
            nn.GELU(),
            nn.Conv1d(256, 512, kernel_size=patch_size, stride=stride, padding=patch_size // 2),
            nn.GELU(),
        )
        self.proj = nn.Linear(512, d_model, bias=False)
        self.norm = RMSNorm(d_model)

    def forward(self, mel: Tensor) -> Tensor:
        """mel: [B, n_mels, T] → [B, T_patches, d_model]"""
        x = self.conv(mel)
        x = x.transpose(1, 2)
        x = self.proj(x)
        return self.norm(x)


# ═══════════════════════════════════════════════════════════════
# 视觉编码器（轻量 Patch Embedding + Transformer）
# ═══════════════════════════════════════════════════════════════

class VisionEncoder(nn.Module):
    """
    轻量视觉编码器：image patches → tokens
    输入: [B, C, H, W] 图像
    输出: [B, n_patches, d_model]

    使用简单 patch embedding + 2 层轻量 self-attention
    约 50M 参数（patch_size=14, d_model=2048, image_size=336）
    """
    def __init__(self, image_size: int = 336, patch_size: int = 14,
                 in_channels: int = 3, d_model: int = 2048,
                 n_vision_layers: int = 2, n_heads: int = 8):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.n_patches, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)

        self.layers = nn.ModuleList()
        for _ in range(n_vision_layers):
            self.layers.append(nn.ModuleDict({
                'norm': RMSNorm(d_model),
                'attn_q': nn.Linear(d_model, n_heads * 64, bias=False),
                'attn_k': nn.Linear(d_model, n_heads * 64, bias=False),
                'attn_v': nn.Linear(d_model, n_heads * 64, bias=False),
                'attn_o': nn.Linear(n_heads * 64, d_model, bias=False),
                'ff_norm': RMSNorm(d_model),
                'ff1': nn.Linear(d_model, d_model * 4, bias=False),
                'ff2': nn.Linear(d_model * 4, d_model, bias=False),
            }))
        self.n_heads = n_heads
        self.head_dim = 64
        self.out_norm = RMSNorm(d_model)

    def forward(self, images: Tensor) -> Tensor:
        """images: [B, C, H, W] → [B, n_patches, d_model]"""
        B = images.shape[0]
        x = self.patch_embed(images)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed

        for layer in self.layers:
            h = layer['norm'](x)
            q = layer['attn_q'](h).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
            k = layer['attn_k'](h).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
            v = layer['attn_v'](h).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
            attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn = F.softmax(attn, dim=-1)
            x = x + layer['attn_o']((attn @ v).transpose(1, 2).flatten(2))

            h = layer['ff_norm'](x)
            x = x + layer['ff2'](F.gelu(layer['ff1'](h)))

        return self.out_norm(x)


# ═══════════════════════════════════════════════════════════════
# ERSA-Omni（跨模态精确回召稀疏注意力）
# ═══════════════════════════════════════════════════════════════

class ERSA_Omni(nn.Module):
    """
    跨模态 ERSA：块级 sketch 路由 + 模态感知可见性 + 原始 K/V 精确重打分。

    三阶段（与 asm_ersa.ERSA 同一套结构）：
      1. 块级 sketch（随机 JL 投影）只负责选候选块；
      2. 对候选块 ∪ 局部窗口用**原始 K/V** 精确重打分（+ sink）；
      3. 输出 RMSNorm。

    与纯文本 ERSA 的唯一区别是掩码：用 _build_causal_mask 的模态感知可见性
    （TEXT 严格因果；AUDIO / VISION 前缀内同模态双向）。
    """
    def __init__(self, d_model: int, n_heads: int, d_k: int = 128, d_v: int = 128,
                 block_size: int = 64, top_blocks: int = 4, window: int = 64,
                 sketch_dim: int = 32, max_seq_len: int = 2048):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.block_size = block_size
        self.top_blocks = top_blocks
        self.window = window
        self.max_seq_len = max_seq_len

        self.W_q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.W_o = nn.Linear(n_heads * d_v, d_model, bias=False)

        # JL sketch 投影：固定随机（不可训练）
        self.register_buffer('sketch', torch.randn(sketch_dim, d_k) / math.sqrt(d_k))
        self.sink = nn.Parameter(torch.tensor(-4.0))
        self.out_norm = RMSNorm(d_model)

    def _build_causal_mask(self, modality_ids: Tensor, n_prefix: int,
                           query_index: Optional[int] = None) -> Tensor:
        """
        查询位置 i 的可见性行，[B, t]，True = 可见。

        规则（唯一标准，全项目共用）：
          1. TEXT 严格因果：TEXT 查询只能看 j <= i；
          2. AUDIO / VISION 前缀内同模态双向：i、j 均在同一前缀区间 [0, n_prefix)
             且同一模态时互相可见（并行感知所需）；
          3. 任何 token 不得看到未来的 TEXT（规则 1 已保证）；
          4. 生成段（text 及其后）一律 j <= i，无双向。

        modality_ids: [B, t]   n_prefix: 前导的 audio + vision token 数
        query_index:  查询位置，None = 末位。

        注意：线上 forward 逐 token 调用，buffer 只含 0..i，故 i 恒为末位，
        规则 1/3/4 由 buffer 长度结构性保证（j <= i 在该情形下恒真，i 之后的
        位置根本不在 buffer 里）。规则 2 的"前缀内双向"要到并行前缀前向才真正
        生效。query_index 存在的意义是让规则本身可单测（tests/test_fixes.py）。
        """
        B, t = modality_ids.shape
        i = t - 1 if query_index is None else query_index
        j = torch.arange(t, device=modality_ids.device)

        visible = (j <= i).unsqueeze(0).expand(B, t)          # 规则 1、4

        if n_prefix > 0:
            mod_i = modality_ids[:, i]
            is_av = (mod_i == Modality.AUDIO) | (mod_i == Modality.VISION)
            same_mod = (modality_ids == mod_i.unsqueeze(-1))
            in_prefix = (j < n_prefix).unsqueeze(0).expand(B, t)
            visible = visible | (is_av.unsqueeze(-1) & same_mod & in_prefix)

        return visible

    @staticmethod
    def _sparse_scores(q: Tensor, K: Tensor, dk: int) -> Tensor:
        """
        q: [B, H, dk]，K: [B, n, H, dk] → [B, n]。

        打分就是 q·k / sqrt(dk)，不做 ReLU：负相关必须保留符号。
        """
        return (q.unsqueeze(1) * K).sum(-1).sum(-1) / math.sqrt(dk)

    def forward(self, x: Tensor, hidden_buffer: Tensor, positions: Tensor,
                modality_ids: Optional[Tensor] = None, n_prefix: int = 0,
                key_mask: Optional[Tensor] = None) -> Tensor:
        """
        x:              [B, d_model] 当前 token 的本层输入（残差流，非原始嵌入）
        hidden_buffer:  [B, t, d_model] — 本层历史输入 + 当前 token
        positions:      [B, t]（接口保留；ERSA 走内容路由，不需要位置编码）
        modality_ids:   [B, t] optional — 每个位置的模态标签
        n_prefix:       前导 audio/vision token 数，决定前缀内双向的范围
        key_mask:       [B, t] optional — False 表示 padding，不可被 attend
        """
        B, t = hidden_buffer.shape[0], hidden_buffer.shape[1]
        H, dk, dv = self.n_heads, self.d_k, self.d_v
        dev = x.device

        if t > self.max_seq_len:
            raise ValueError(
                f"序列长度 {t} 超过 max_seq_len={self.max_seq_len}")

        q = F.normalize(self.W_q(x).view(B, H, dk).float(), dim=-1)
        K = self.W_k(hidden_buffer).view(B, t, H, dk).float()
        V = self.W_v(hidden_buffer).view(B, t, H, dv).float()

        # 可见性行 [B, t]：模态感知；padding 一律不可见
        if modality_ids is not None:
            vis = self._build_causal_mask(modality_ids, n_prefix)
        else:
            vis = torch.ones(B, t, dtype=torch.bool, device=dev)
        if key_mask is not None:
            vis = vis & key_mask

        # ── 阶段 1：块级 sketch 路由（只在可见 token 上取块分）──
        nb = (t + self.block_size - 1) // self.block_size
        pad = nb * self.block_size - t
        Kp = F.pad(K, (0, 0, 0, 0, 0, pad)) if pad else K
        phi = torch.einsum('bthk,rk->bthr', Kp, self.sketch)
        phiq = torch.einsum('bhk,rk->bhr', q, self.sketch)
        sc = torch.einsum('bhr,bthr->bth', phiq, phi)
        vis_pad = F.pad(vis, (0, pad), value=False) if pad else vis
        sc = sc.masked_fill(~vis_pad.unsqueeze(-1), float('-inf'))
        block_sc = sc.reshape(B, nb, self.block_size, H).amax(dim=2)   # [B, nb, H]
        kb = min(self.top_blocks, nb)
        tb = block_sc.topk(kb, dim=1).indices
        block_mask = torch.zeros(B, H, nb, dtype=torch.bool, device=dev)
        block_mask.scatter_(2, tb.permute(0, 2, 1), True)
        tok_mask = block_mask.repeat_interleave(self.block_size, dim=-1)[:, :, :t]

        # ── 局部窗口（仍受模态可见性约束）──
        w_start = max(0, t - self.window)
        win = (torch.arange(t, device=dev) >= w_start).view(1, 1, t) & vis.unsqueeze(1)
        tok_mask = (tok_mask & vis.unsqueeze(1)) | win

        # ── 阶段 2：原始 K/V 精确重打分 + sink ──
        e = torch.einsum('bhk,bthk->bht', q, K) / math.sqrt(dk)
        e = e.masked_fill(~tok_mask, float('-inf'))
        e = torch.cat([e, self.sink.expand(B, H, 1)], dim=-1)
        a = torch.softmax(e, dim=-1)[..., :t]
        y = torch.einsum('bht,bthv->bhv', a, V)

        return self.out_norm(self.W_o(y.to(x.dtype).reshape(B, H * dv)))

# ═══════════════════════════════════════════════════════════════
# 模态感知 MoE
# ═══════════════════════════════════════════════════════════════

class ModalityMoE(nn.Module):
    """
    模态感知 MoE：expert 按模态分区

    分区策略：
      - shared_experts: 所有模态共用
      - text_experts: 文本专用
      - audio_experts: 音频专用
      - vision_experts: 视觉专用

    Router 输入加模态偏置，引导 routing 到对应模态的 expert
    """
    def __init__(self, d_model: int, d_ff: int,
                 n_shared: int = 32, n_text: int = 48,
                 n_audio: int = 24, n_vision: int = 24,
                 top_k: int = 4):
        super().__init__()
        self.n_shared = n_shared
        self.n_text = n_text
        self.n_audio = n_audio
        self.n_vision = n_vision
        self.n_experts = n_shared + n_text + n_audio + n_vision
        self.top_k = top_k

        self.gate = nn.Linear(d_model, self.n_experts, bias=False)

        self.modality_bias = nn.Parameter(torch.zeros(3, self.n_experts))
        nn.init.zeros_(self.modality_bias)
        bias_init = torch.full((self.n_experts,), -2.0)
        bias_init[n_shared:n_shared + n_text] = 2.0
        self.modality_bias.data[Modality.TEXT] = bias_init
        bias_init2 = torch.full((self.n_experts,), -2.0)
        bias_init2[n_shared + n_text:n_shared + n_text + n_audio] = 2.0
        self.modality_bias.data[Modality.AUDIO] = bias_init2
        bias_init3 = torch.full((self.n_experts,), -2.0)
        bias_init3[n_shared + n_text + n_audio:] = 2.0
        self.modality_bias.data[Modality.VISION] = bias_init3

        self.W1 = nn.Parameter(torch.randn(self.n_experts, d_model, d_ff) * 0.02)
        self.W2 = nn.Parameter(torch.randn(self.n_experts, d_ff, d_model) * 0.02)

        self.shared_W1 = nn.Linear(d_model, d_ff, bias=False)
        self.shared_W_up = nn.Linear(d_model, d_ff, bias=False)
        self.shared_W2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor, modality: Optional[Tensor] = None) -> Tensor:
        """
        x: [B, d_model]
        modality: [B] long tensor 或 None — 每个样本自己的模态标签。
                  传张量而非 Python int，避免 device→host 同步，也避免整批
                  共用样本 0 的模态。
        """
        scores = self.gate(x)

        if modality is not None:
            scores = scores + self.modality_bias[modality]

        topk_vals, topk_idx = scores.topk(self.top_k, dim=-1)
        topk_weights = F.softmax(topk_vals, dim=-1)

        out = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = topk_idx[:, k]
            weight = topk_weights[:, k:k+1]
            h = F.silu(torch.einsum('bd,bdf->bf', x, self.W1[expert_idx]))
            e = torch.einsum('bf,bfd->bd', h, self.W2[expert_idx])
            out = out + weight * e

        shared = self.shared_W2(F.silu(self.shared_W1(x)) * self.shared_W_up(x))
        return out + shared


# ═══════════════════════════════════════════════════════════════
# Omni Transformer 块
# ═══════════════════════════════════════════════════════════════

class OmniTransformerBlock(nn.Module):
    """
    Omni 版 Transformer 块：
      - ASM-ERSA层：维护多模态状态
      - ERSA-Omni 层：跨模态精确回召稀疏注意力
      - ModalityMoE：模态感知路由
    """
    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int,
                 d_ff: int, is_ersa: bool = False, layer_idx: int = 0,
                 n_slots: int = 512, top_r: int = 16, n_scales: int = 4,
                 block_size: int = 64, top_blocks: int = 4, window: int = 64,
                 sketch_dim: int = 32,
                 max_seq_len: int = 2048,
                 n_shared_experts: int = 32, n_text_experts: int = 48,
                 n_audio_experts: int = 24, n_vision_experts: int = 24,
                 top_k: int = 4):
        super().__init__()
        self.is_ersa = is_ersa
        self.layer_idx = layer_idx
        self.d_model = d_model

        self.gr1 = GatedResidual(d_model, n_branches=4)
        self.gr2 = GatedResidual(d_model, n_branches=4)

        if is_ersa:
            self.token_mixing = ERSA_Omni(
                d_model, n_heads, d_k, d_v,
                block_size=block_size, top_blocks=top_blocks,
                window=window, sketch_dim=sketch_dim,
                max_seq_len=max_seq_len)
        else:
            self.token_mixing = ASM_ERSA_Layer(
                d_model, n_heads, d_k, d_v,
                n_slots=n_slots, top_r=top_r, n_scales=n_scales)

        self.moe = ModalityMoE(
            d_model, d_ff,
            n_shared=n_shared_experts, n_text=n_text_experts,
            n_audio=n_audio_experts, n_vision=n_vision_experts,
            top_k=top_k)

    def forward(self, branches: list, hidden_buffer: Optional[Tensor] = None,
                positions: Optional[Tensor] = None,
                modality_ids: Optional[Tensor] = None,
                modality: Optional[Tensor] = None,
                n_prefix: int = 0,
                key_mask: Optional[Tensor] = None,
                state: Optional[Tensor] = None):
        """
        返回 (branches, new_state, hidden_buffer)。
        ASM-ERSA层返回新的 per-sample 状态；ERSA 层返回 None。
        ERSA 的 buffer 存的是**本层自己的输入** x（残差流），不是原始嵌入：
        这样每个 ERSA 层投影的是深度不同的表示，K/V 才随深度变化。
        """
        x = self.gr1.read(branches)

        if self.is_ersa:
            hidden_buffer = x.unsqueeze(1) if hidden_buffer is None else \
                torch.cat([hidden_buffer, x.unsqueeze(1)], dim=1)
            y = self.token_mixing(x, hidden_buffer, positions,
                                  modality_ids=modality_ids, n_prefix=n_prefix,
                                  key_mask=key_mask)
            new_state = None
        else:
            y, new_state = self.token_mixing(x, positions, state)

        branches = self.gr1.write(branches, y)

        x = self.gr2.read(branches)
        y = self.moe(x, modality=modality)
        branches = self.gr2.write(branches, y)

        return branches, new_state, hidden_buffer


# ═══════════════════════════════════════════════════════════════
# 音频输出头
# ═══════════════════════════════════════════════════════════════

class AudioOutputHead(nn.Module):
    """
    音频输出生成头：hidden states → audio token logits

    支持两种模式：
      1. 离散 audio tokens（如 EnCodec/SoundStream codebook）
      2. 连续 Mel 频谱回归
    """
    def __init__(self, d_model: int, n_audio_tokens: int = 8192,
                 mode: str = 'discrete'):
        super().__init__()
        self.mode = mode
        self.norm = RMSNorm(d_model)

        if mode == 'discrete':
            self.head = nn.Linear(d_model, n_audio_tokens, bias=False)
        elif mode == 'continuous':
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model * 2, bias=False),
                nn.GELU(),
                nn.Linear(d_model * 2, 80, bias=False)
            )
        else:
            raise ValueError(f"Unknown audio output mode: {mode}")

    def forward(self, hidden: Tensor) -> Tensor:
        h = self.norm(hidden)
        return self.head(h)


# ═══════════════════════════════════════════════════════════════
# ASM-ERSA Omni 完整模型
# ═══════════════════════════════════════════════════════════════

class ASM_ERSA_Omni(nn.Module):
    """
    ASM-ERSA Omni — 多模态模型

    支持输入：
      - text: input_ids [B, T_text]
      - audio: mel_spectrogram [B, n_mels, T_audio]（条件段）
      - audio 生成目标: audio_codes_target [B, T_frames]（生成段，teacher forcing）
      - vision: images [B, C, H, W]

    拼接顺序（固定）：
      [audio_cond?] [vision?] [text] [<audio_start>] [audio_codes...]

    支持输出：
      - text logits [B, T, vocab_size]
      - audio tokens/mel [B, T, n_audio_tokens] or [B, T, 80]
      - audio_target_logits [B, T_frames, n_audio_tokens]（与生成目标逐位对齐）

    3B Omni 配置：
      d_model=2048, n_heads=16, n_layers=32, d_ff=384,
      n_shared=8, n_text=12, n_audio=6, n_vision=6, top_k=4
    """
    def __init__(
        self,
        d_model: int = 2048,
        n_heads: int = 16,
        d_k: int = 128,
        d_v: int = 128,
        d_ff: int = 384,
        n_layers: int = 32,
        vocab_size: int = 152064,
        n_slots: int = 512,
        top_r: int = 16,
        n_scales: int = 4,
        block_size: int = 64,
        top_blocks: int = 4,
        window: int = 64,
        sketch_dim: int = 32,
        max_seq_len: int = 2048,
        n_shared_experts: int = 8,
        n_text_experts: int = 12,
        n_audio_experts: int = 6,
        n_vision_experts: int = 6,
        top_k: int = 4,
        audio_config: Optional[Dict] = None,
        vision_config: Optional[Dict] = None,
        audio_output_mode: str = 'discrete',
        n_audio_tokens: int = 8192,
        state_grad_chunk: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.state_grad_chunk = state_grad_chunk
        self._init_kwargs = dict(
            d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
            n_layers=n_layers, vocab_size=vocab_size,
            n_slots=n_slots, top_r=top_r,
            n_scales=n_scales, block_size=block_size,
            max_seq_len=max_seq_len, top_blocks=top_blocks,
            window=window, sketch_dim=sketch_dim,
            n_shared_experts=n_shared_experts, n_text_experts=n_text_experts,
            n_audio_experts=n_audio_experts, n_vision_experts=n_vision_experts,
            top_k=top_k, audio_config=audio_config, vision_config=vision_config,
            audio_output_mode=audio_output_mode, n_audio_tokens=n_audio_tokens,
            state_grad_chunk=state_grad_chunk,
        )

        self.text_embed = nn.Embedding(vocab_size, d_model)
        self.modality_embed = SemanticModalityEmbedding(d_model)

        audio_cfg = audio_config or {}
        self.audio_encoder = AudioEncoder(
            n_mels=audio_cfg.get('n_mels', 80),
            d_model=d_model,
            patch_size=audio_cfg.get('patch_size', 4),
            stride=audio_cfg.get('stride', 2),
        )

        vision_cfg = vision_config or {}
        self.vision_encoder = VisionEncoder(
            image_size=vision_cfg.get('image_size', 336),
            patch_size=vision_cfg.get('patch_size', 14),
            in_channels=vision_cfg.get('in_channels', 3),
            d_model=d_model,
            n_vision_layers=vision_cfg.get('n_layers', 2),
            n_heads=vision_cfg.get('n_heads', 8),
        )

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            is_ersa = (i % 4 == 3)
            self.layers.append(
                OmniTransformerBlock(
                    d_model, n_heads, d_k, d_v, d_ff,
                    is_ersa=is_ersa, layer_idx=i,
                    n_slots=n_slots, top_r=top_r,
                    n_scales=n_scales,
                    block_size=block_size,
                    top_blocks=top_blocks,
                    window=window,
                    sketch_dim=sketch_dim,
                    max_seq_len=max_seq_len,
                    n_shared_experts=n_shared_experts,
                    n_text_experts=n_text_experts,
                    n_audio_experts=n_audio_experts,
                    n_vision_experts=n_vision_experts,
                    top_k=top_k)
            )

        self.out_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.audio_head = AudioOutputHead(d_model, n_audio_tokens, audio_output_mode)

        # 音频生成段：<audio_start> 特殊 token + 音频码本嵌入表。
        # 码本嵌入是 teacher forcing 必需的——生成段输入的是"上一帧码"，
        # 没有这张表就只能拿 padding 或零向量充当输入（N/A 梯度会污染状态）。
        self.audio_start_embed = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.audio_start_embed, std=0.02)
        self.audio_code_embed = nn.Embedding(n_audio_tokens, d_model)
        nn.init.normal_(self.audio_code_embed.weight, std=0.02)

        # padding 占位向量。保持非零：ASM-ERSA里 q/k 走 F.normalize，精确零向量的
        # 反向是 1/eps = 1e12 倍放大，会把梯度推成 inf 再变 NaN。
        self.pad_embed = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.pad_embed, std=0.02)

    def encode_audio(self, mel: Tensor) -> Tuple[Tensor, Tensor]:
        """编码音频 → 语义增强 tokens + modality ids"""
        audio_tokens = self.audio_encoder(mel)
        B, T, _ = audio_tokens.shape
        mod_ids = torch.full((B, T), Modality.AUDIO, dtype=torch.long, device=mel.device)
        # 使用语义模态嵌入（投影 + 偏置）
        audio_emb = self.modality_embed(audio_tokens, Modality.AUDIO)
        return audio_emb, mod_ids

    def encode_vision(self, images: Tensor) -> Tuple[Tensor, Tensor]:
        """编码图像 → 语义增强 tokens + modality ids"""
        vision_tokens = self.vision_encoder(images)
        B, T, _ = vision_tokens.shape
        mod_ids = torch.full((B, T), Modality.VISION, dtype=torch.long, device=images.device)
        # 使用语义模态嵌入（投影 + 偏置）
        vision_emb = self.modality_embed(vision_tokens, Modality.VISION)
        return vision_emb, mod_ids

    def encode_text(self, input_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """编码文本 → 语义增强 tokens + modality ids"""
        text_tokens = self.text_embed(input_ids)
        B, T = input_ids.shape
        mod_ids = torch.full((B, T), Modality.TEXT, dtype=torch.long, device=input_ids.device)
        # 使用语义模态嵌入（投影 + 偏置）
        text_emb = self.modality_embed(text_tokens, Modality.TEXT)
        return text_emb, mod_ids

    def encode_audio_target(
        self,
        codes: Tensor,
        codes_valid: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        音频生成段输入：`[<audio_start>, c_0, …, c_{T-2}]`，长度 T。

        右移一位（teacher forcing）：`<audio_start>` 位置预测第 0 帧，位置 c_j 预测
        第 j+1 帧，故整段 T 个位置各自对应目标 `codes[:, 0..T-1]`——预测与标签是
        逐位对齐的，不需要 loss 侧再做偏移。最后一帧 c_{T-1} 只作为标签，不作为输入。

        codes:       [B, T] long，右对齐补齐（尾部为 padding）
        codes_valid: [B, T] bool，True = 真实帧。None 表示全部真实。

        返回 (emb [B, T, d], mod_ids [B, T], seg_valid [B, T])。
        """
        if self.audio_head.mode != 'discrete':
            raise ValueError(
                f"audio_codes_target 仅适用于离散音频头，当前 mode="
                f"{self.audio_head.mode}")
        if codes.dim() != 2:
            raise ValueError(f"audio_codes_target 期望 [B, T]，收到 {tuple(codes.shape)}")

        B, T = codes.shape
        if T == 0:
            raise ValueError("audio_codes_target 长度不能为 0")
        if codes_valid is not None:
            assert tuple(codes_valid.shape) == (B, T), (
                f"audio_codes_valid 形状 {tuple(codes_valid.shape)} 与 target {(B, T)} 不一致")

        start = self.audio_start_embed.to(self.text_embed.weight.dtype)
        start = start.view(1, 1, -1).expand(B, 1, -1)
        if T > 1:
            prev = self.audio_code_embed(codes[:, :-1])     # 第 0..T-2 帧作为输入
            seg = torch.cat([start, prev], dim=1)
        else:
            seg = start

        mod_ids = torch.full((B, T), Modality.AUDIO, dtype=torch.long, device=codes.device)
        emb = self.modality_embed(seg, Modality.AUDIO)

        seg_valid = torch.ones(B, T, dtype=torch.bool, device=codes.device)
        if codes_valid is not None:
            # <audio_start> 恒有效；输入位的有效性继承自对应帧（c_j 来自第 j 帧）
            seg_valid[:, 1:] = codes_valid[:, :-1].bool()
        return emb, mod_ids, seg_valid

    def _audio_seg_mask(
        self,
        audio_valid_frames: Optional[Tensor],
        n_audio_tokens: int,
        mel_width: int,
    ) -> Optional[Tensor]:
        """
        音频段 key mask [B, n_audio_tokens]，True=真实 token。返回 None 表示整段都真实。

        每个样本的 mel 尾部若被 collate 补齐（n_valid < mel_width），补齐区参与卷积产生
        的 token 必须屏蔽；n_valid == mel_width 的样本没有补齐区，其末尾 token 依赖的是
        卷积自身的 zero padding（编码器固有行为），正常监督。
        """
        if audio_valid_frames is None:
            return None

        conv = self.audio_encoder.conv[0]
        k, s = conv.kernel_size[0], conv.stride[0]

        counts = [
            n_audio_tokens if int(n) >= mel_width
            else self.audio_valid_len(int(n), k, s)
            for n in audio_valid_frames
        ]
        if min(counts) >= n_audio_tokens:
            return None

        idx = torch.arange(n_audio_tokens, device=audio_valid_frames.device).unsqueeze(0)
        n_clean = torch.tensor(counts, device=audio_valid_frames.device).unsqueeze(1)
        return idx < n_clean

    @staticmethod
    def audio_token_len(mel_frames: int, patch_size: int = 4, stride: int = 2) -> int:
        """
        两层 Conv1d(kernel=patch_size, stride=stride, padding=patch_size//2) 对 L 帧
        mel 的输出帧数。数据侧据此构造与音频 token 对齐的 mask，不做 //4 之类的猜测。
        """
        def out_len(n: int) -> int:
            return (n + 2 * (patch_size // 2) - patch_size) // stride + 1
        return out_len(out_len(mel_frames))

    @staticmethod
    def audio_valid_len(n_valid_mel_frames: int, patch_size: int = 4, stride: int = 2) -> int:
        """
        前 n_valid_mel_frames 帧为真实数据（其余为 collate padding）时，感受野完全落在
        真实区域内的 audio token 数。超出的 token 由 padding 参与卷积产生，必须 mask。

        推导：单层 out i ← in [i*s - p, i*s - p + k - 1]（p = k//2），两层复合后
        上界为 i*s² + ((k-1-p)*(s+1))，要求 <= n_valid - 1。
        """
        k, s, p = patch_size, stride, patch_size // 2
        hi = (k - 1 - p) * (s + 1)
        return max(0, (n_valid_mel_frames - 1 - hi) // (s * s) + 1)

    # ───────────────────────────────────────────────────────────
    # 段拼接 / 单 token 递推：forward 与 generate_audio 的唯一共用点
    # ───────────────────────────────────────────────────────────

    def _build_prefix_segments(
        self,
        audio_mel: Optional[Tensor],
        images: Optional[Tensor],
        input_ids: Optional[Tensor],
        attention_mask: Optional[Tensor],
        audio_valid_frames: Optional[Tensor],
    ) -> Tuple[List[Tensor], List[Tensor], List[Optional[Tensor]], int]:
        """
        构造前导段 `[audio_cond?] [vision?] [text]`（不含生成段）。

        返回 (tokens, mod_ids, seg_masks, n_prefix)：前两者是逐段的 token 嵌入与模态
        id 列表，seg_masks 是同长度的逐段可见性 mask（None = 整段可见）。
        """
        all_tokens: List[Tensor] = []
        all_mod_ids: List[Tensor] = []
        seg_masks: List[Optional[Tensor]] = []
        n_prefix = 0

        if audio_mel is not None:
            audio_tokens, audio_mods = self.encode_audio(audio_mel)
            all_tokens.append(audio_tokens)
            all_mod_ids.append(audio_mods)
            n_prefix += audio_tokens.shape[1]
            seg_masks.append(self._audio_seg_mask(
                audio_valid_frames, audio_tokens.shape[1], audio_mel.shape[-1]))

        if images is not None:
            vision_tokens, vision_mods = self.encode_vision(images)
            all_tokens.append(vision_tokens)
            all_mod_ids.append(vision_mods)
            n_prefix += vision_tokens.shape[1]
            seg_masks.append(None)

        if input_ids is not None:
            text_tokens, text_mods = self.encode_text(input_ids)
            all_tokens.append(text_tokens)
            all_mod_ids.append(text_mods)
            if attention_mask is None:
                seg_masks.append(None)
            else:
                assert tuple(attention_mask.shape) == tuple(input_ids.shape), (
                    f"attention_mask 形状 {tuple(attention_mask.shape)} 与文本段 "
                    f"{tuple(input_ids.shape)} 不一致（前缀的可见性由模型自行推导）")
                seg_masks.append(attention_mask.bool().to(text_tokens.device))

        if not all_tokens:
            raise ValueError("At least one modality must be provided")
        return all_tokens, all_mod_ids, seg_masks, n_prefix

    def _pack_segments(
        self,
        all_tokens: List[Tensor],
        all_mod_ids: List[Tensor],
        seg_masks: List[Optional[Tensor]],
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        逐段列表 → 拼接后的 (x, modality_ids, key_mask)，并做 padding 占位替换。

        padding 位置替换为固定的可学习占位向量，内容与输入无关，因此"改 pad 内容 →
        输出/loss 不变"仍成立。它们已被排除在 loss 之外，但若放任原值参与 ASM-ERSA的
        状态递推，pad 区的噪声会顺着状态污染其后所有 token（音频 pad 在文本之前，
        正是这种情形）。这里不能用精确零向量，原因见 __init__ 中 pad_embed 的注释。
        """
        x = torch.cat(all_tokens, dim=1)
        modality_ids = torch.cat(all_mod_ids, dim=1)

        # 任一模态段有 padding 时才构造总 key mask，顺序与拼接一致 [audio | vision | text | codes]
        key_mask = None
        if any(m is not None for m in seg_masks):
            parts = []
            for m, tok in zip(seg_masks, all_tokens):
                if m is None:
                    parts.append(torch.ones(
                        tok.shape[0], tok.shape[1], dtype=torch.bool, device=x.device))
                else:
                    parts.append(m)
            key_mask = torch.cat(parts, dim=1)

        if key_mask is not None:
            x = torch.where(key_mask.unsqueeze(-1), x,
                            self.pad_embed.to(x.dtype).view(1, 1, -1))
        return x, modality_ids, key_mask

    def _forward_one_token(
        self,
        x_t: Tensor,
        mod_t: Tensor,
        positions: Tensor,
        modality_ids: Tensor,
        n_prefix: int,
        states: List[Optional[Tensor]],
        buffers: List[Optional[Tensor]],
        key_mask: Optional[Tensor],
    ) -> Tuple[Tensor, List[Optional[Tensor]], List[Optional[Tensor]]]:
        """
        单个 token 穿过全部层（含 ASM-ERSA状态与 ERSA buffer 的递推）。

        参数中的 positions / modality_ids / key_mask 都必须**已包含当前 token**；
        modality_ids 走的是"整段历史"语义（ERSA 的模态感知稀疏打分需要它）。
        返回 (hidden_t, states, buffers)；states/buffers 是就地更新后的同一个列表。
        """
        branches = [x_t] * 4
        for li, layer in enumerate(self.layers):
            branches, states[li], buffers[li] = layer(
                branches, buffers[li], positions,
                modality_ids=modality_ids,
                modality=mod_t, n_prefix=n_prefix,
                key_mask=key_mask,
                state=states[li])
        return branches[0], states, buffers

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        audio_mel: Optional[Tensor] = None,
        images: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        audio_valid_frames: Optional[Tensor] = None,
        audio_codes_target: Optional[Tensor] = None,
        audio_codes_valid: Optional[Tensor] = None,
        states: Optional[List[Optional[Tensor]]] = None,
        return_states: bool = False,
    ) -> Dict[str, Tensor]:
        """
        多模态前向传播

        拼接顺序（固定）：
          [audio_cond?] [vision?] [text] [<audio_start>] [audio_codes...]

        参数：
          input_ids:  [B, T_text] 文本 token ids
          audio_mel:  [B, n_mels, T_audio] 音频 Mel 频谱（条件段）
          images:     [B, C, H, W] 图像
          attention_mask: [B, T_text] 文本区 1=真实 / 0=padding。前缀（音频/视觉）的
                          可见性由模型自己按感受野推导，不需要调用方拼总长。
          audio_valid_frames: [B] 每个样本真实（非 collate padding）的 mel 帧数。
                          尾部补齐产生的 audio token 会被自动 mask 掉。
          audio_codes_target: [B, T_frames] 真实音频码（生成段监督目标）。
                          输入段为 `<audio_start>` + 前 T-1 帧码，逐位预测对应帧。
          audio_codes_valid: [B, T_frames] bool，True = 真实帧（其余为 collate 补齐）
          states:     每层 ASM-ERSA的 per-sample 状态列表（长度 = n_layers，ERSA 层为 None）。
                      传 None 表示序列起点；传回上一次的 'states' 可延续记忆。
          return_states: True 时返回值中含 'states'。

        返回：
          dict with:
            'text_logits':  [B, T_total, vocab_size]
            'audio_output': [B, T_total, n_audio_tokens or 80]
            'hidden_states': [B, T_total, d_model]
            'padding_mask': [B, T_total] bool，True=真实 token（有 padding 时）
            'audio_target_logits': [B, T_frames, n_audio_tokens]（给 codes 时）
                                   已按生成段切好，与 audio_codes_target 逐位对齐
            'states':       List[Optional[Tensor]]（仅 return_states=True）
        """
        all_tokens, all_mod_ids, seg_masks, n_prefix = self._build_prefix_segments(
            audio_mel, images, input_ids, attention_mask, audio_valid_frames)

        # 生成段（后缀，不计入 n_prefix）：<audio_start> + 前 T-1 帧码。
        # 它在 text 之后，故 n_prefix 保持不变——n_prefix 只标记"可双向的前导段"。
        n_code_frames = 0
        if audio_codes_target is not None:
            code_emb, code_mods, code_valid = self.encode_audio_target(
                audio_codes_target, audio_codes_valid)
            all_tokens.append(code_emb)
            all_mod_ids.append(code_mods)
            seg_masks.append(code_valid)
            n_code_frames = audio_codes_target.shape[1]

        x, modality_ids, key_mask = self._pack_segments(
            all_tokens, all_mod_ids, seg_masks)
        B, T_total, _ = x.shape

        # 每层自己的 ASM-ERSA状态与 ERSA hidden buffer
        states = [None] * len(self.layers) if states is None else list(states)
        buffers: List[Optional[Tensor]] = [None] * len(self.layers)

        positions = torch.zeros(B, 0, dtype=torch.long, device=x.device)

        outputs = []
        for token_idx in range(T_total):
            pos_t = torch.full((B, 1), token_idx, dtype=torch.long, device=x.device)
            positions = torch.cat([positions, pos_t], dim=1)

            hidden_t, states, buffers = self._forward_one_token(
                x[:, token_idx], modality_ids[:, token_idx], positions,
                modality_ids[:, :token_idx + 1], n_prefix, states, buffers,
                None if key_mask is None else key_mask[:, :token_idx + 1])
            outputs.append(hidden_t)

            # 截断 BPTT：限定跨 token 的信用分配窗口（不减少每 token 自身激活的显存）
            if self.state_grad_chunk > 0 and (token_idx + 1) % self.state_grad_chunk == 0:
                def _detach(s):
                    if s is None:
                        return None
                    if isinstance(s, tuple):
                        return tuple(t.detach() for t in s)
                    return s.detach()
                states = [_detach(s) for s in states]

        hidden = torch.stack(outputs, dim=1)
        hidden = self.out_norm(hidden)

        result = {
            'text_logits': self.lm_head(hidden),
            'audio_output': self.audio_head(hidden),
            'hidden_states': hidden,
        }
        if n_code_frames > 0:
            # 生成段是序列末尾，直接切尾；模型自己保证与 audio_codes_target 对齐
            result['audio_target_logits'] = result['audio_output'][:, -n_code_frames:, :]
        if key_mask is not None:
            result['padding_mask'] = key_mask
        if return_states:
            result['states'] = states
        return result

    def save_pretrained(self, save_dir) -> None:
        """权重存 safetensors + 结构参数存 config.json（不依赖 transformers 的 API）。"""
        import json
        from pathlib import Path
        from safetensors.torch import save_file

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        state = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        save_file(state, str(save_dir / 'model.safetensors'))
        with open(save_dir / 'config.json', 'w', encoding='utf-8') as f:
            json.dump({'model_type': 'asm_ersa_omni', 'kwargs': self._init_kwargs},
                      f, indent=2)

    @classmethod
    def from_pretrained(cls, load_dir, map_location: str = 'cpu', **overrides):
        """与 save_pretrained 往返一致；strict=True 加载，shape/key 不匹配直接报错。"""
        import json
        from pathlib import Path
        from safetensors.torch import load_file

        load_dir = Path(load_dir)
        with open(load_dir / 'config.json', encoding='utf-8') as f:
            cfg = json.load(f)

        kwargs = dict(cfg['kwargs'])
        kwargs.update(overrides)
        model = cls(**kwargs)
        state = load_file(str(load_dir / 'model.safetensors'), device=map_location)
        model.load_state_dict(state, strict=True)
        return model

    def generate_text(
        self,
        audio_mel: Optional[Tensor] = None,
        images: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        eos_token_id: Optional[int] = None,
    ) -> Tensor:
        """
        自回归文本生成（支持音频/视觉输入条件）

        返回: [B, T_text + max_new_tokens] token ids
        注：每个 step 重算全序列（O(T²)），增量 KV cache 见规划文档附录 B。
        """
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            prefix_parts = []
            prefix_mod_parts = []
            if audio_mel is not None:
                audio_tokens, audio_mods = self.encode_audio(audio_mel)
                prefix_parts.append(audio_tokens)
                prefix_mod_parts.append(audio_mods)
            if images is not None:
                vision_tokens, vision_mods = self.encode_vision(images)
                prefix_parts.append(vision_tokens)
                prefix_mod_parts.append(vision_mods)

            if input_ids is None:
                input_ids = torch.tensor([[1]], device=device)

            # 前缀的模态 ID 直接来自编码器，不再整段标成 TEXT
            prefix = torch.cat(prefix_parts, dim=1) if prefix_parts else None
            prefix_mods = torch.cat(prefix_mod_parts, dim=1) if prefix_mod_parts else None
            n_prefix = 0 if prefix is None else prefix.shape[1]

            generated = input_ids.clone()

            for _ in range(max_new_tokens):
                text_tokens, text_mods = self.encode_text(generated)

                if prefix is None:
                    x, modality_ids = text_tokens, text_mods
                else:
                    x = torch.cat([prefix, text_tokens], dim=1)
                    modality_ids = torch.cat([prefix_mods, text_mods], dim=1)

                B, T_total, _ = x.shape
                states: List[Optional[Tensor]] = [None] * len(self.layers)
                buffers: List[Optional[Tensor]] = [None] * len(self.layers)
                positions = torch.zeros(B, 0, dtype=torch.long, device=device)

                last_hidden = None
                for token_idx in range(T_total):
                    x_t = x[:, token_idx]
                    branches = [x_t] * 4

                    pos_t = torch.full((B, 1), token_idx, dtype=torch.long, device=device)
                    positions = torch.cat([positions, pos_t], dim=1)

                    mod_t = modality_ids[:, token_idx]
                    for li, layer in enumerate(self.layers):
                        branches, states[li], buffers[li] = layer(
                            branches, buffers[li], positions,
                            modality_ids=modality_ids[:, :token_idx + 1],
                            modality=mod_t, n_prefix=n_prefix,
                            state=states[li])

                    last_hidden = branches[0]

                logits = self.lm_head(self.out_norm(last_hidden))

                logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('inf')

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

                if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                    break

        return generated

    @torch.no_grad()
    def generate_audio(
        self,
        audio_mel: Optional[Tensor] = None,
        images: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        audio_valid_frames: Optional[Tensor] = None,
        n_frames: int = 75,
        temperature: float = 1.0,
        top_k: int = 0,
        greedy: bool = True,
    ) -> Tensor:
        """
        自回归音频生成：`[audio_cond?] [vision?] [text]` → `n_frames` 个预测码。

        返回 [B, n_frames] long（取值 [0, n_audio_tokens)），可直接交给
        AudioTokenizer.decode 得到波形。生成段与训练时的 teacher forcing 逐位对齐：
        `<audio_start>` 位置出第 0 帧，其后每帧的输入是上一帧的预测码。

        与 generate_text 的"每步重算全序列"不同，这里利用 ASM-ERSA的定长状态与 ERSA 的
        hidden buffer 做增量递推：前缀只前向一次，之后每帧只喂 1 个 token。
        """
        if self.audio_head.mode != 'discrete':
            raise ValueError(
                f"generate_audio 仅适用于离散音频头，当前 mode={self.audio_head.mode}")
        if n_frames <= 0:
            raise ValueError(f"n_frames 必须为正，收到 {n_frames}")

        self.eval()
        device = next(self.parameters()).device
        dtype = self.text_embed.weight.dtype

        all_tokens, all_mod_ids, seg_masks, n_prefix = self._build_prefix_segments(
            audio_mel, images, input_ids, attention_mask, audio_valid_frames)
        x, modality_ids, key_mask = self._pack_segments(all_tokens, all_mod_ids, seg_masks)
        B, T_prefix, _ = x.shape

        states: List[Optional[Tensor]] = [None] * len(self.layers)
        buffers: List[Optional[Tensor]] = [None] * len(self.layers)
        positions = torch.zeros(B, 0, dtype=torch.long, device=device)

        for t in range(T_prefix):
            positions = torch.cat(
                [positions, torch.full((B, 1), t, dtype=torch.long, device=device)], dim=1)
            _, states, buffers = self._forward_one_token(
                x[:, t], modality_ids[:, t], positions,
                modality_ids[:, :t + 1], n_prefix, states, buffers,
                None if key_mask is None else key_mask[:, :t + 1])

        cur = self.modality_embed(
            self.audio_start_embed.to(dtype).view(1, 1, -1).expand(B, 1, -1),
            Modality.AUDIO)[:, 0]

        codes = []
        for _ in range(n_frames):
            positions = torch.cat(
                [positions, torch.full((B, 1), positions.shape[1], dtype=torch.long,
                                       device=device)], dim=1)
            mod_t = torch.full((B,), Modality.AUDIO, dtype=torch.long, device=device)
            modality_ids = torch.cat([modality_ids, mod_t.unsqueeze(1)], dim=1)
            if key_mask is not None:
                key_mask = torch.cat(
                    [key_mask, torch.ones(B, 1, dtype=torch.bool, device=device)], dim=1)

            h_t, states, buffers = self._forward_one_token(
                cur, mod_t, positions, modality_ids, n_prefix, states, buffers, key_mask)
            logits = self.audio_head(self.out_norm(h_t))

            if greedy or temperature <= 0:
                next_code = logits.argmax(dim=-1)
            else:
                scaled = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                    scaled = scaled.masked_fill(scaled < v[:, [-1]], -float('inf'))
                next_code = torch.multinomial(
                    F.softmax(scaled, dim=-1), num_samples=1).squeeze(-1)

            codes.append(next_code)
            cur = self.modality_embed(
                self.audio_code_embed(next_code).unsqueeze(1), Modality.AUDIO)[:, 0]

        return torch.stack(codes, dim=1)


# ═══════════════════════════════════════════════════════════════
# 使用示例
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    model = ASM_ERSA_Omni(
        d_model=512,
        n_heads=8,
        d_k=64,
        d_v=64,
        d_ff=1408,
        n_layers=8,
        vocab_size=32000,
        n_slots=128,
        top_r=8,
        n_scales=4,
        stride_choices=[8, 16, 32],
        n_shared_experts=8,
        n_text_experts=16,
        n_audio_experts=8,
        n_vision_experts=8,
        top_k=4,
        audio_config={'n_mels': 80, 'patch_size': 4, 'stride': 2},
        vision_config={'image_size': 56, 'patch_size': 14, 'n_layers': 1, 'n_heads': 4},
        audio_output_mode='discrete',
        n_audio_tokens=1024,
    )

    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.audio_encoder.parameters()) + \
                     sum(p.numel() for p in model.vision_encoder.parameters())
    backbone_params = total_params - encoder_params - \
                      sum(p.numel() for p in model.audio_head.parameters())

    print(f"总参数量: {total_params / 1e6:.1f}M")
    print(f"  编码器: {encoder_params / 1e6:.1f}M (音频 + 视觉)")
    print(f"  Backbone: {backbone_params / 1e6:.1f}M")
    print(f"  音频头: {sum(p.numel() for p in model.audio_head.parameters()) / 1e6:.1f}M")
    print()

    ersa_layers = sum(1 for l in model.layers if l.is_ersa)
    gdn_layers = sum(1 for l in model.layers if not l.is_ersa)
    print(f"ASM-ERSA层数: {gdn_layers}")
    print(f"ERSA-Omni 层数: {ersa_layers}")
    print()

    moe = model.layers[0].moe
    print(f"MoE 配置:")
    print(f"  共享专家: {moe.n_shared}")
    print(f"  文本专家: {moe.n_text}")
    print(f"  音频专家: {moe.n_audio}")
    print(f"  视觉专家: {moe.n_vision}")
    print(f"  Top-k: {moe.top_k}")
    print()

    B = 2
    text_ids = torch.randint(0, 32000, (B, 8))
    audio_mel = torch.randn(B, 80, 200)
    images = torch.randn(B, 3, 56, 56)

    with torch.no_grad():
        output = model(input_ids=text_ids, audio_mel=audio_mel, images=images)

    print(f"输入:")
    print(f"  text: {text_ids.shape}")
    print(f"  audio_mel: {audio_mel.shape}")
    print(f"  images: {images.shape}")
    print()
    print(f"输出:")
    print(f"  text_logits: {output['text_logits'].shape}")
    print(f"  audio_output: {output['audio_output'].shape}")
    print(f"  hidden_states: {output['hidden_states'].shape}")
    print()
    print("测试通过 ✓")
