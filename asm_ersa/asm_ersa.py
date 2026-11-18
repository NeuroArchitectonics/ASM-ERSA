"""
ASM-ERSA（Addressable Slot Memory + Exact-Recall Sparse Attention）v4.0
=========================================
ASM-ERSA 混合架构：36 层 ASM-ERSA + 12 层 ERSA（每 4 层一个 ERSA）
48 层 Transformer，4 条 Gated Residual 分支，MoE 512 专家 top-10

ASM-ERSA 状态（2026-11-17）：
  - 旧的 GDN3（通道级门控线性记忆）Python 实现已删除；
  - `ASM_ERSA_Layer` 是本文件里的机制参考实现（ASM 槽位读/写 + ERSA 回召），
    已在 tiny 配置上验证前向/反传；Vulkan 侧反向与训练器见 `反向/`、`优化器/`；
  - 设计目标：可寻址槽位记忆（ASM）+ 精确回召稀疏注意力（ERSA）。

ERSA（Exact-Recall Sparse Attention）：
  块级 sketch 路由（随机 JL 投影）只选候选块，候选块 ∪ 局部窗口用原始 K/V
  精确重打分（+ sink），输出 RMSNorm。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List
import math


# ═══════════════════════════════════════════════════════════════
# 基础组件
# ═══════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * rms


def apply_rope(x: Tensor, positions: Tensor, d_k: int) -> Tensor:
    """RoPE 位置编码。x: [..., d_k]"""
    half = d_k // 2
    freq = 1.0 / (10000 ** (torch.arange(0, d_k, 2, device=x.device).float() / d_k))
    angles = positions.unsqueeze(-1).float() * freq
    cos, sin = angles.cos(), angles.sin()
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).type_as(x)


# ═══════════════════════════════════════════════════════════════
# Gated Residual（4 条残差分支）
# ═══════════════════════════════════════════════════════════════

class GatedResidual(nn.Module):
    """
    GR Read:  4 条分支 → 加权平均 → 子层输入
    GR Write: 子层输出 → 标量门控 → 残差加法
    """
    def __init__(self, d_model: int, n_branches: int = 4):
        super().__init__()
        self.n_branches = n_branches
        self.norms = nn.ModuleList([RMSNorm(d_model) for _ in range(n_branches)])
        self.read_proj = nn.Linear(d_model * n_branches, n_branches)
        self.write_proj = nn.Linear(d_model, n_branches)

    def read(self, branches: list[Tensor]) -> Tensor:
        normed = [self.norms[i](branches[i]) for i in range(self.n_branches)]
        concat = torch.cat(normed, dim=-1)
        gates = self.read_proj(concat).sigmoid()
        stacked = torch.stack(normed, dim=1)
        return (gates.unsqueeze(-1) * stacked).mean(dim=1)

    def write(self, branches: list[Tensor], y: Tensor) -> list[Tensor]:
        scales = 2.0 * self.write_proj(y).sigmoid()
        return [branches[i] + scales[:, i:i+1] * y for i in range(self.n_branches)]


# ═══════════════════════════════════════════════════════════════
# ASM-ERSA（可寻址槽位记忆 + 精确回召稀疏注意力）
# 设计依据：ASM-ERSA 设计文档 v0.1（2026-11-15）
# 本文件是 CPU/GPU PyTorch 参考实现；Vulkan 内核见 ASM-ERSA-Design/vk（同接口）
# ═══════════════════════════════════════════════════════════════

class ASM_ERSA_Layer(nn.Module):
    """
    ASM：可寻址槽位记忆（设计 §3）。

    每个头维护 M 个槽位 {(κ_m, υ_m, γ_m, β_m)}：
      - 读：top-r 槽位精确注意力 + 空槽位 ∅（§3.2）
      - 写：选中槽位的凸组合，η_t 为 token 级标量（§3.3）
      - 衰减：槽位级标量 γ_m，按时间尺度分组（不是通道级对角矩阵）
    状态为 (K [B,H,M,dk], V [B,H,M,dv], beta [B,H,M])，fp32 递推。
    """
    def __init__(self, d_model: int, n_heads: int, d_k: int = 128, d_v: int = 128,
                 n_slots: int = 512, top_r: int = 16, n_scales: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.n_slots = n_slots
        self.top_r = min(top_r, n_slots)
        self.n_scales = n_scales
        self.tau = 0.005
        self.tau_w = 0.005
        # 新颖性门控：选中槽位的最大 cos 低于该值时，写入改投 beta 最低的冷槽，
        # 避免不相关的新 key 覆盖旧内容（负值关闭）。
        self.tau_new = 0.15
        self.rho = 0.9
        self.lam = 0.01

        self.W_q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.W_o = nn.Linear(n_heads * d_v, d_model, bias=False)

        # η_t = sigmoid(w_eta^T x)，初始化约 0.1
        self.w_eta = nn.Linear(d_model, 1, bias=True)
        nn.init.zeros_(self.w_eta.weight)
        nn.init.constant_(self.w_eta.bias, -2.197)

        # 槽位初始化 key（共享初值，配合 k-means 预训练可选）
        self.slot_init_k = nn.Parameter(F.normalize(torch.randn(n_slots, d_k), dim=-1))

        # 槽位级标量衰减：按 n_scales 组初始化，sigmoid 参数化
        groups = max(1, n_scales)
        per = max(1, n_slots // groups)
        vals = []
        for gi in range(groups):
            decay = [0.99, 0.999, 0.9999, 0.99999][gi % 4]
            vals.append(torch.full((per,), decay))
        g = torch.cat(vals)[:n_slots]
        if g.numel() < n_slots:
            g = torch.cat([g, g[-1:].expand(n_slots - g.numel())])
        self.gamma_logit = nn.Parameter(torch.log(g / (1 - g)))

        # 空槽位 β_∅（可学习），υ_∅ = 0
        self.beta_null = nn.Parameter(torch.tensor(-2.0))
        self.out_norm = RMSNorm(d_model)

    def initial_state(self, batch_size: int, device=None,
                      dtype: torch.dtype = torch.float32) -> tuple:
        K = self.slot_init_k.unsqueeze(0).unsqueeze(0).expand(
            batch_size, self.n_heads, self.n_slots, self.d_k).to(device=device, dtype=dtype)
        V = torch.zeros(batch_size, self.n_heads, self.n_slots, self.d_v,
                        device=device, dtype=dtype)
        beta = torch.zeros(batch_size, self.n_heads, self.n_slots,
                           device=device, dtype=dtype)
        return K, V, beta

    def forward(self, x: Tensor, positions: Optional[Tensor] = None,
                state: Optional[tuple] = None):
        """x: [B, d_model] → (out [B, d_model], (K, V, beta))"""
        B = x.shape[0]
        H, dk, dv, M, r = self.n_heads, self.d_k, self.d_v, self.n_slots, self.top_r

        K, V, beta = state if state is not None else self.initial_state(B, device=x.device)
        K, V, beta = K.float(), V.float(), beta.float()

        q = F.normalize(self.W_q(x).view(B, H, dk).float(), dim=-1)
        k = F.normalize(self.W_k(x).view(B, H, dk).float(), dim=-1)
        v = self.W_v(x).view(B, H, dv).float()

        # ── 读：top-r 槽位 + 空槽位（§3.2）──
        s = torch.einsum('bhk,bhmk->bhm', q, K) / self.tau + beta
        idx = s.topk(r, dim=-1).indices                       # [B,H,r]
        sel_s = torch.gather(s, -1, idx)
        sel_v = torch.gather(V, 2, idx.unsqueeze(-1).expand(-1, -1, -1, dv))
        null_s = self.beta_null.expand(B, H, 1)
        a = torch.softmax(torch.cat([sel_s, null_s], dim=-1), dim=-1)
        o = torch.einsum('bhr,bhrv->bhv', a[..., :r], sel_v)   # 空槽位 υ=0

        # ── 写：选中槽位的凸组合（§3.3）──
        beta_sel = torch.gather(beta, 2, idx)
        p = torch.softmax(
            torch.einsum('bhk,bhrk->bhr', k, torch.gather(
                K, 2, idx.unsqueeze(-1).expand(-1, -1, -1, dk))) / self.tau_w + beta_sel,
            dim=-1)                                            # [B,H,r]
        eta = torch.sigmoid(self.w_eta(x)).view(B, 1, 1, 1)    # token 级标量，按 batch 广播
        gamma = torch.sigmoid(self.gamma_logit)                # [M]
        g_sel = gamma.expand(B, H, M).gather(2, idx)           # [B,H,r]

        K_sel = torch.gather(K, 2, idx.unsqueeze(-1).expand(-1, -1, -1, dk))
        V_sel = sel_v
        new_K = F.normalize(g_sel.unsqueeze(-1) * K_sel + eta * p.unsqueeze(-1) * k.unsqueeze(2), dim=-1)
        new_V = g_sel.unsqueeze(-1) * V_sel + eta * p.unsqueeze(-1) * v.unsqueeze(2)
        new_beta = self.rho * beta_sel + self.lam * p

        # 新颖性门控：k 与选中槽位最大 cos 低于 tau_new 时不覆盖，
        # 改写 beta 最低且未被选中的冷槽（p = 1）。
        kdot = torch.einsum('bhk,bhrk->bhr', k, torch.gather(
            K, 2, idx.unsqueeze(-1).expand(-1, -1, -1, dk)))
        novel = kdot.max(dim=-1).values < self.tau_new
        sel_mask = torch.zeros(B, H, M, dtype=torch.bool, device=x.device)
        sel_mask.scatter_(2, idx, True)
        cold = beta.masked_fill(sel_mask, float('inf')).argmin(dim=-1)
        if novel.any():
            nf = novel.view(B, H, 1, 1)
            cold4k = cold.view(B, H, 1, 1).expand(-1, -1, -1, dk)
            cold4v = cold.view(B, H, 1, 1).expand(-1, -1, -1, dv)
            K_cold = torch.gather(K, 2, cold4k)
            V_cold = torch.gather(V, 2, cold4v)
            b_cold = torch.gather(beta, 2, cold.unsqueeze(-1)).squeeze(-1)
            g_cold = gamma.expand(B, H, M).gather(2, cold.unsqueeze(-1)).squeeze(-1)
            K_cold_new = F.normalize(g_cold.unsqueeze(-1) * K_cold + eta * k.unsqueeze(2), dim=-1)
            V_cold_new = g_cold.unsqueeze(-1) * V_cold + eta * v.unsqueeze(2)
            b_cold_new = self.rho * b_cold + self.lam
            new_K = torch.where(nf, K_sel, new_K)
            new_V = torch.where(nf, V_sel, new_V)
            new_beta = torch.where(novel.unsqueeze(-1), beta_sel, new_beta)
            K.scatter_(2, cold4k, torch.where(nf, K_cold_new, K_cold))
            V.scatter_(2, cold4v, torch.where(nf, V_cold_new, V_cold))
            beta.scatter_(2, cold.unsqueeze(-1),
                          torch.where(novel.unsqueeze(-1), b_cold_new, b_cold))

        K = K.scatter(2, idx.unsqueeze(-1).expand(-1, -1, -1, dk), new_K)
        V = V.scatter(2, idx.unsqueeze(-1).expand(-1, -1, -1, dv), new_V)
        beta = beta.scatter(2, idx, new_beta)

        out = self.W_o(o.to(x.dtype).reshape(B, H * dv))
        return self.out_norm(out), (K, V, beta)


class ERSA(nn.Module):
    """
    ERSA：精确回召稀疏注意力（设计 §4）。

    阶段 1：块级 sketch（随机 JL 投影）只负责选候选块；
    阶段 2：对候选块 ∪ 局部窗口用**原始 K/V** 精确重打分（+ sink）；
    阶段 3：输出 RMSNorm。

    本实现为参考版：掩码后对全序列做精确打分（O(T)），
    稀疏 gather 版本见 ASM-ERSA-Design/vk 的 ersa_attn_fused / ersa_hier_ivf_attn_fused。
    """
    def __init__(self, d_model: int, n_heads: int, d_k: int = 128, d_v: int = 128,
                 block_size: int = 64, top_blocks: int = 4, window: int = 64,
                 sketch_dim: int = 32, max_seq_len: int = 4096):
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

        # JL sketch 投影：P ∈ R^{r_s × d_k}，固定随机（不可训练）
        self.register_buffer('sketch', torch.randn(sketch_dim, d_k) / math.sqrt(d_k))
        self.sink = nn.Parameter(torch.tensor(-4.0))
        self.out_norm = RMSNorm(d_model)

    def forward(self, x: Tensor, hidden_buffer: Tensor, positions: Tensor) -> Tensor:
        B, T = hidden_buffer.shape[0], hidden_buffer.shape[1]
        H, dk, dv = self.n_heads, self.d_k, self.d_v
        dev = x.device

        q = F.normalize(self.W_q(x).view(B, H, dk).float(), dim=-1)
        K = self.W_k(hidden_buffer).view(B, T, H, dk).float()
        V = self.W_v(hidden_buffer).view(B, T, H, dv).float()

        # ── 阶段 1：块级 sketch 路由 ──
        nb = (T + self.block_size - 1) // self.block_size
        pad = nb * self.block_size - T
        Kp = F.pad(K, (0, 0, 0, 0, 0, pad)) if pad else K
        phi = torch.einsum('bthk,rk->bthr', Kp, self.sketch)
        phiq = torch.einsum('bhk,rk->bhr', q, self.sketch)
        sc = torch.einsum('bhr,bthr->bth', phiq, phi).reshape(
            B, nb, self.block_size, H).amax(dim=2)             # [B,nb,H]
        kb = min(self.top_blocks, nb)
        tb = sc.topk(kb, dim=1).indices                        # [B,kb,H]
        block_mask = torch.zeros(B, H, nb, dtype=torch.bool, device=dev)
        block_mask.scatter_(2, tb.permute(0, 2, 1), True)
        tok_mask = block_mask.repeat_interleave(self.block_size, dim=-1)[:, :, :T]

        # ── 局部窗口 ──
        win = (torch.arange(T, device=dev) >= max(0, T - self.window)).view(1, 1, T)
        tok_mask = tok_mask | win

        # ── 阶段 2：原始 K/V 精确重打分 + sink ──
        e = torch.einsum('bhk,bthk->bht', q, K) / math.sqrt(dk)
        e = e.masked_fill(~tok_mask, float('-inf'))
        e = torch.cat([e, self.sink.expand(B, H, 1)], dim=-1)
        a = torch.softmax(e, dim=-1)[..., :T]
        y = torch.einsum('bht,bthv->bhv', a, V)

        return self.out_norm(self.W_o(y.to(x.dtype).reshape(B, H * dv)))


# ═══════════════════════════════════════════════════════════════
# MoE（简化版）
# ═══════════════════════════════════════════════════════════════

class MoE(nn.Module):
    """512 路由专家 + 1 共享专家，top-10 路由"""
    def __init__(self, d_model: int, d_ff: int, n_experts: int = 512, top_k: int = 10):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.W1 = nn.Parameter(torch.randn(n_experts, d_model, d_ff) * 0.02)
        self.W2 = nn.Parameter(torch.randn(n_experts, d_ff, d_model) * 0.02)
        self.shared_W1 = nn.Linear(d_model, d_ff, bias=False)
        self.shared_W_up = nn.Linear(d_model, d_ff, bias=False)
        self.shared_W2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        scores = self.gate(x)
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
# Transformer 块（ASM-ERSA 层 / ERSA 层）
# ═══════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """
    一个完整的 Transformer 块：
      GR Read → Token Mixing (ASM-ERSA 或 ERSA) → GR Write
      → GR Read → MoE → GR Write
    """
    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int,
                 d_ff: int, is_ersa: bool = False, layer_idx: int = 0,
                 n_slots: int = 512, top_r: int = 16, n_scales: int = 4,
                 block_size: int = 64, top_blocks: int = 4,
                 window: int = 64, sketch_dim: int = 32,
                 n_experts: int = 512, moe_top_k: int = 10,
                 max_seq_len: int = 4096):
        super().__init__()
        self.is_ersa = is_ersa
        self.layer_idx = layer_idx

        self.gr1 = GatedResidual(d_model, n_branches=4)
        self.gr2 = GatedResidual(d_model, n_branches=4)

        if is_ersa:
            self.token_mixing = ERSA(
                d_model, n_heads, d_k, d_v,
                block_size=block_size, top_blocks=top_blocks,
                window=window, sketch_dim=sketch_dim,
                max_seq_len=max_seq_len)
        else:
            self.token_mixing = ASM_ERSA_Layer(
                d_model, n_heads, d_k, d_v,
                n_slots=n_slots, top_r=top_r, n_scales=n_scales)

        self.moe = MoE(d_model, d_ff, n_experts=n_experts, top_k=moe_top_k)

    def forward(self, branches: list[Tensor],
                hidden_buffer: Optional[Tensor] = None,
                positions: Optional[Tensor] = None,
                state: Optional[Tensor] = None):
        """
        branches:       4 条残差分支，各 [B, d_model]
        hidden_buffer:  [B, t, d_model] 本层截至当前 token 的输入（仅 ERSA 层维护）
        state:          ASM-ERSA 上一步状态 (K, V, beta)（仅 ASM 层使用）
        返回:           (branches, new_state, new_hidden_buffer)
        """
        x = self.gr1.read(branches)

        if self.is_ersa:
            hidden_buffer = x.unsqueeze(1) if hidden_buffer is None else \
                torch.cat([hidden_buffer, x.unsqueeze(1)], dim=1)
            y = self.token_mixing(x, hidden_buffer, positions)
            new_state = None
        else:
            y, new_state = self.token_mixing(x, positions, state)

        branches = self.gr1.write(branches, y)

        x = self.gr2.read(branches)
        y = self.moe(x)
        branches = self.gr2.write(branches, y)

        return branches, new_state, hidden_buffer


# ═══════════════════════════════════════════════════════════════
# 完整模型
# ═══════════════════════════════════════════════════════════════

class ASM_ERSA(nn.Module):
    """
    ASM-ERSA v4.0
    48 层混合架构：每 4 层一个 ERSA（第 3, 7, 11, ... 层），其余为 ASM
    """
    def __init__(self, d_model: int = 4096, n_heads: int = 32,
                 d_k: int = 128, d_v: int = 128, d_ff: int = 11264,
                 n_layers: int = 48, vocab_size: int = 152064,
                 n_slots: int = 512, top_r: int = 16, n_scales: int = 4,
                 block_size: int = 64, top_blocks: int = 4,
                 window: int = 64, sketch_dim: int = 32,
                 n_experts: int = 512, moe_top_k: int = 10,
                 max_seq_len: int = 4096):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        self.embed = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            is_ersa = (i % 4 == 3)
            self.layers.append(
                TransformerBlock(
                    d_model, n_heads, d_k, d_v, d_ff,
                    is_ersa=is_ersa, layer_idx=i,
                    n_slots=n_slots, top_r=top_r, n_scales=n_scales,
                    block_size=block_size, top_blocks=top_blocks,
                    window=window, sketch_dim=sketch_dim,
                    n_experts=n_experts, moe_top_k=moe_top_k,
                    max_seq_len=max_seq_len)
            )

        self.out_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        逐样本递归前向。ASM-ERSA 状态在每个序列起点置零，序列内递推、梯度可回传；
        ERSA 层用本层输入构成因果 K/V 序列。返回 [B, t, vocab_size]。
        """
        B, t = input_ids.shape
        device = input_ids.device

        x = self.embed(input_ids)

        states: List[Optional[Tensor]] = [None] * len(self.layers)
        hidden_buffers: List[Optional[Tensor]] = [None] * len(self.layers)

        outputs = []
        positions = torch.zeros(B, 0, dtype=torch.long, device=device)

        for token_idx in range(t):
            x_t = x[:, token_idx]
            branches = [x_t] * 4

            pos_t = torch.full((B, 1), token_idx, dtype=torch.long, device=device)
            positions = torch.cat([positions, pos_t], dim=1)

            for li, layer in enumerate(self.layers):
                branches, states[li], hidden_buffers[li] = layer(
                    branches, hidden_buffers[li], positions, states[li])

            outputs.append(branches[0])

        stacked = torch.stack(outputs, dim=1)
        stacked = self.out_norm(stacked)
        return self.lm_head(stacked)


# ═══════════════════════════════════════════════════════════════
# 使用示例已移除：ASM-ERSA 参考实现落地后恢复（2026-11-16）
# ═══════════════════════════════════════════════════════════════
