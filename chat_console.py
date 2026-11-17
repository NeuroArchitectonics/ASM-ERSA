"""
ASM-ERSA 交互式控制台对话
======================
在终端中与模型对话，查看预测效果
"""

import torch
import torch.nn.functional as F
import json, os, gc, time, math
from pathlib import Path
from safetensors import safe_open
from collections import OrderedDict
from tokenizers import Tokenizer

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["OMP_NUM_THREADS"] = "8"
torch.set_num_threads(8)


def log(msg):
    print(msg, flush=True)


# ═══════════════════════════════════════════════════════════════
# INT8 反量化
# ═══════════════════════════════════════════════════════════════

def dequantize_int8(q_tensor, scale, group_size=128):
    """INT8 对称量化反量化（x̂ = q · scale）；INT4 旧格式同样适用。"""
    flat = q_tensor.half().flatten()
    n = flat.numel()
    pad = (group_size - n % group_size) % group_size
    if pad > 0:
        flat = F.pad(flat, (0, pad))
    groups = flat.reshape(-1, group_size)
    dequant = groups * scale.half().unsqueeze(1)
    return dequant.flatten()[:n].reshape(q_tensor.shape)


# ═══════════════════════════════════════════════════════════════
# 专家索引
# ═══════════════════════════════════════════════════════════════

class ExpertIndex:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.index = {}
        self._build_index()
    
    def _build_index(self):
        shard_files = sorted(self.model_dir.glob("model-*.safetensors"))
        
        scale_locations = {}
        for shard_path in shard_files:
            with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                keys = list(f.keys())
                for key in keys:
                    if key.startswith("_scale.") and (key.endswith(".moe.W1") or key.endswith(".moe.W2")):
                        weight_key = key[7:]
                        scale_locations[weight_key] = str(shard_path)
        
        for shard_path in shard_files:
            with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                keys = list(f.keys())
                for key in keys:
                    if (key.endswith(".moe.W1") or key.endswith(".moe.W2")) and not key.startswith("_scale."):
                        if key in scale_locations:
                            scale_shard = scale_locations[key]
                            self.index[key] = (str(shard_path), key, f"_scale.{key}", scale_shard)
    
    def load_expert(self, layer_idx, expert_idx, group_size=128):
        w1_key = f"layers.{layer_idx}.moe.W1"
        w2_key = f"layers.{layer_idx}.moe.W2"
        
        w1 = self._load_tensor_slice(w1_key, expert_idx, group_size)
        w2 = self._load_tensor_slice(w2_key, expert_idx, group_size)
        
        return w1, w2
    
    def _load_tensor_slice(self, base_key, expert_idx, group_size):
        if base_key not in self.index:
            raise KeyError(f"未找到 {base_key}")
        
        weight_shard, tensor_key, scale_key, scale_shard = self.index[base_key]
        
        with safe_open(weight_shard, framework="pt", device="cpu") as f:
            full_tensor = f.get_tensor(tensor_key)
            expert_tensor = full_tensor[expert_idx]
        
        with safe_open(scale_shard, framework="pt", device="cpu") as f:
            full_scale = f.get_tensor(scale_key)
        
        elements_per_expert = expert_tensor.numel()
        scales_per_expert = elements_per_expert // group_size
        
        start = expert_idx * scales_per_expert
        end = start + scales_per_expert
        expert_scale = full_scale[start:end]
        
        return dequantize_int8(expert_tensor, expert_scale, group_size)


class LRUCache:
    def __init__(self, maxsize=200):
        self.cache = OrderedDict()
        self.maxsize = maxsize
    
    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)


class ExpertLoader:
    def __init__(self, expert_index, cache_size=200):
        self.expert_index = expert_index
        self.cache = LRUCache(maxsize=cache_size)
        self.group_size = 128
    
    def get_expert(self, layer_idx, expert_idx):
        key = (layer_idx, expert_idx)
        
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        
        w1, w2 = self.expert_index.load_expert(layer_idx, expert_idx, self.group_size)
        self.cache.put(key, (w1, w2))
        
        return w1, w2


# ═══════════════════════════════════════════════════════════════
# 加载基础权重
# ═══════════════════════════════════════════════════════════════

def load_compact(model_dir):
    model_path = Path(model_dir)
    with open(model_path / "config.json") as f:
        config = json.load(f)
    
    group_size = config.get("group_size", 128)
    shard_files = sorted(model_path.glob("model-*.safetensors"))
    
    all_scales = {}
    for shard_path in shard_files:
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            for key in keys:
                if key.startswith("_scale."):
                    scale = f.get_tensor(key)
                    weight_key = key[7:]
                    all_scales[weight_key] = scale
    
    sd = {}
    for shard_path in shard_files:
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            
            for key in keys:
                if key.startswith("_scale."):
                    continue
                if key.endswith(".moe.W1") or key.endswith(".moe.W2"):
                    continue
                
                tensor = f.get_tensor(key)
                
                if key in all_scales:
                    scale = all_scales[key]
                    tensor = dequantize_int8(tensor, scale, group_size)
                    del all_scales[key]
                
                sd[key] = tensor
        
        gc.collect()
    
    return sd, config


# ═══════════════════════════════════════════════════════════════
# PLE (Progressive Layer-wise Embedding)
# ═══════════════════════════════════════════════════════════════

class PLEEmbedding:
    """Hash-based ngram embedding lookup with shard-based loading."""

    def __init__(self, sd, prefix, model_dir):
        self.layer_multipliers = sd[f"{prefix}ple_embedding.layer_multipliers"]
        self.head_offsets = sd[f"{prefix}ple_embedding.ngram_heads_offsets"]
        self.head_vocab_sizes = sd[f"{prefix}ple_embedding.ngram_heads_vocab_sizes"]
        self.ngram_heads = len(self.head_offsets)
        self.model_dir = Path(model_dir)
        self.n_shards = 128
        shard0 = sd.get(f"{prefix}ple_embedding.ngram_embedding.shard_0.weight")
        self.shard_size = shard0.shape[0] if shard0 is not None else 2500012
        self.head_dim = shard0.shape[1] if shard0 is not None else 160
        self.shard_cache = OrderedDict()
        self.max_cache = 2
        self._build_shard_index(prefix)

    def _build_shard_index(self, prefix):
        self.shard_to_file = {}
        for f in sorted(self.model_dir.glob("model-*.safetensors")):
            with safe_open(str(f), framework="pt", device="cpu") as sf:
                for k in sf.keys():
                    if "ngram_embedding.shard_" in k and ".weight" in k:
                        for si in range(self.n_shards):
                            if f"shard_{si}.weight" in k:
                                self.shard_to_file[si] = str(f)
                                break

    def _load_shard(self, shard_idx):
        if shard_idx in self.shard_cache:
            self.shard_cache.move_to_end(shard_idx)
            return self.shard_cache[shard_idx]
        fp = self.shard_to_file.get(shard_idx)
        if fp is None:
            raise FileNotFoundError(f"Shard {shard_idx} not found in index")
        with safe_open(fp, framework="pt", device="cpu") as sf:
            for k in sf.keys():
                if f"shard_{shard_idx}.weight" in k:
                    w = sf.get_tensor(k)
                    if len(self.shard_cache) >= self.max_cache:
                        self.shard_cache.popitem(last=False)
                    self.shard_cache[shard_idx] = w
                    return w
        raise FileNotFoundError(f"Shard {shard_idx} weight not found in {fp}")

    def _lookup(self, abs_indices):
        results = torch.zeros(abs_indices.shape[0], abs_indices.shape[1],
                              self.head_dim, dtype=torch.float32)
        for h in range(self.ngram_heads):
            idx = abs_indices[:, h].long()
            shard_idx = (idx // self.shard_size).item()
            local_idx = (idx % self.shard_size).item()
            shard_w = self._load_shard(shard_idx)
            results[:, h] = shard_w[local_idx].float()
        return results

    def forward(self, token_ids, ngram_context):
        B = 1
        device = token_ids.device
        cur_id = token_ids.view(-1).long()
        prev1 = ngram_context.get("prev1")
        prev2 = ngram_context.get("prev2")

        blocks = []
        for ngram_level, shift_tokens in enumerate([
            lambda: cur_id,
            lambda: prev1 if prev1 is not None else cur_id,
        ]):
            h_start = ngram_level * 8
            h_end = h_start + 8
            t_cur = cur_id
            t_shift = shift_tokens()

            mixed = t_cur * self.layer_multipliers[0]
            mixed = torch.bitwise_xor(mixed, t_shift * self.layer_multipliers[1])

            h_vs = self.head_vocab_sizes[h_start:h_end]
            h_off = self.head_offsets[h_start:h_end]
            ngram_ids = torch.remainder(mixed.unsqueeze(-1), h_vs.view(1, -1))
            abs_ids = ngram_ids + h_off.view(1, -1)
            blocks.append(abs_ids)

        abs_ids_all = torch.cat(blocks, dim=-1)
        embeddings = self._lookup(abs_ids_all)
        return embeddings.flatten(-2)


def ple_fwd(sd, ple_prefix, ple_embedding, branches, token_id, ngram_context, d_model):
    """
    PLE forward pass.
    branches: list of 4 tensors (1, d_model) — concatenated = (1, 10240)
    Returns: updated branches list
    """
    hidden_flat = torch.cat(branches, dim=-1)

    embeddings = ple_embedding.forward(token_id, ngram_context)

    key_proj_w = sd[f"{ple_prefix}key_proj.weight"]
    value_proj_w = sd[f"{ple_prefix}value_proj.weight"]
    norm_key_w = sd[f"{ple_prefix}norm_key.weight"]
    norm_query_w = sd[f"{ple_prefix}norm_query.weight"]
    norm_conv_w = sd[f"{ple_prefix}norm_conv.weight"]
    conv_w = sd[f"{ple_prefix}conv1d.weight"]

    embeddings = embeddings.to(key_proj_w.dtype)
    key_proj = F.linear(embeddings, key_proj_w)
    key_rms = key_proj.float().pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
    key_normed = (key_proj * norm_key_w * key_rms).half()
    key_normed = key_normed.unflatten(-1, (4, d_model))

    value = F.linear(embeddings, value_proj_w)

    query_rms = hidden_flat.float().pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
    query_normed = (hidden_flat * norm_query_w * query_rms).half()
    query_normed = query_normed.unflatten(-1, (4, d_model))

    gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(d_model)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)

    gated_flat = gated_value.flatten(-2)
    gated_rms = gated_flat.float().pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
    gated_normed = (gated_flat * norm_conv_w * gated_rms).half()

    ple_state = ngram_context.get("conv_state")
    if ple_state is None:
        ple_state = torch.zeros(1, 10240, 9, dtype=torch.float16)

    conv_input = torch.cat([ple_state, gated_normed.unsqueeze(-1)], dim=-1)
    conv_w_sq = conv_w.squeeze(1)
    positions = torch.arange(0, 10, 3)
    conv_out = (conv_input[:, :, positions] * conv_w_sq).sum(dim=-1)
    conv_out = F.silu(conv_out)
    ple_state = conv_input[:, :, 1:]

    output = gated_flat + conv_out
    ngram_context["conv_state"] = ple_state

    output_4 = output.unflatten(-1, (4, d_model))
    new_branches = [branches[i] + output_4[:, i] for i in range(4)]
    return new_branches


# ═══════════════════════════════════════════════════════════════
# 推理引擎
# ═══════════════════════════════════════════════════════════════

def rms_norm(x, weight, eps=1e-6):
    rms = x.float().pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    return (weight * x * rms).half()


def gdn2_fwd(x, sd, pfx, H, dk, dv, state, conv_state):
    """
    ASM-ERSA 前向传播
    x: [1, d_model]
    state: [1, H, dv, dk] — 循环状态
    conv_state: [1, 12288, 3] — conv1d 缓存（最近 3 步 qkv）
    返回: (output, state, conv_state)
    """
    # 投影
    q = F.linear(x, sd[f"{pfx}W_q.weight"])   # (1, 4096)
    k = F.linear(x, sd[f"{pfx}W_k.weight"])   # (1, 4096)
    v = F.linear(x, sd[f"{pfx}W_v.weight"])   # (1, 4096)
    
    # Conv1d: 在 QKV 拼接后应用因果深度卷积
    if f"{pfx}conv1d.weight" in sd:
        qkv = torch.cat([q, k, v], dim=-1).unsqueeze(-1)  # (1, 12288, 1)
        # 拼接 conv_state: [1, 12288, 3] + [1, 12288, 1] → [1, 12288, 4]
        conv_input = torch.cat([conv_state, qkv], dim=-1)  # (1, 12288, 4)
        # 深度卷积: 对每个通道做 4-tap FIR
        conv_w = sd[f"{pfx}conv1d.weight"]  # (12288, 1, 4)
        conv_out = (conv_input * conv_w.squeeze(1)).sum(dim=-1)  # (1, 12288)
        # 更新 conv_state: 保留最近 3 步
        conv_state = conv_input[:, :, 1:]  # (1, 12288, 3)
        # 拆分回 q, k, v
        q = conv_out[:, :H*dk]
        k = conv_out[:, H*dk:2*H*dk]
        v = conv_out[:, 2*H*dk:]
    
    q = q.view(1, H, dk)
    k = k.view(1, H, dk)
    v = v.view(1, H, dv)
    
    # L2 归一化
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    
    # 通道级缩放（如果有 norm.weight）
    if f"{pfx}norm.weight" in sd:
        norm_w = sd[f"{pfx}norm.weight"]  # (128,) = (dk,)
        # 广播到 (1, H, dk)
        q = q * norm_w
        k = k * norm_w
    
    b = F.linear(x, sd[f"{pfx}W_b.weight"]).view(1, H, dk).sigmoid()
    w = F.linear(x, sd[f"{pfx}W_w.weight"]).view(1, H, dv).sigmoid()
    a = F.linear(x, sd[f"{pfx}W_a.weight"]).view(1, H, dk)
    
    A_log = sd[f"{pfx}A_log"]
    dt_bias = sd[f"{pfx}dt_bias"]
    alpha = (-A_log.exp() * F.softplus(a + dt_bias)).exp()
    
    bk_f = (b * k).float()
    D_state_f = (alpha.unsqueeze(-2) * state).float()
    bk_dot_D = torch.einsum('bhk,bhvk->bhv', bk_f, D_state_f)
    correction = torch.einsum('bhk,bhv->bhvk', k.float(), bk_dot_D)
    state = (D_state_f - correction).half()
    
    wv = w * v
    state = state + torch.einsum('bhk,bhv->bhvk', k.float(), wv.float()).half()
    
    out = torch.einsum('bhk,bhvk->bhv', q.float(), state.float()).half()
    out = out.reshape(1, H * dv)
    out = F.linear(out, sd[f"{pfx}W_o.weight"])

    return out, state, conv_state


def apply_rope(x, positions, d_k):
    """RoPE 位置编码。x: [..., d_k], positions: [...]"""
    half = d_k // 2
    freq = 1.0 / (10000 ** (torch.arange(0, d_k, 2, device=x.device).float() / d_k))
    angles = positions.unsqueeze(-1).float() * freq
    cos, sin = angles.cos(), angles.sin()
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).type_as(x)


def ersa_fwd(x, sd, pfx, H, dk, dv, kv_cache, step,
             block_size=64, top_blocks=4, window=1024):
    """单 token 的 ERSA 前向：块级 sketch 路由 + 原始 K/V 精确重打分 + sink。

    x: [1, d_model]
    kv_cache: [1, t, H, dk+dv]（含当前 token；首次为 None）
    返回: (output [1, d_model], updated kv_cache)
    """
    B = 1
    q = F.linear(x, sd[f"{pfx}W_q.weight"]).view(B, H, dk).float()
    q = F.normalize(q, dim=-1)
    k_cur = F.linear(x, sd[f"{pfx}W_k.weight"]).view(B, 1, H, dk)
    v_cur = F.linear(x, sd[f"{pfx}W_v.weight"]).view(B, 1, H, dv)

    kv_new = torch.cat([k_cur, v_cur], dim=-1)
    kv_cache = kv_new if kv_cache is None else torch.cat([kv_cache, kv_new], dim=1)

    t = kv_cache.shape[1]
    K_all = kv_cache[:, :, :, :dk].float()
    V_all = kv_cache[:, :, :, dk:].float()

    sketch = sd[f"{pfx}sketch"].float()
    nb = (t + block_size - 1) // block_size
    pad = nb * block_size - t
    Kp = F.pad(K_all, (0, 0, 0, 0, 0, pad)) if pad else K_all
    phi = torch.einsum('bthk,rk->bthr', Kp, sketch)
    phiq = torch.einsum('bhk,rk->bhr', q, sketch)
    sc = torch.einsum('bhr,bthr->bth', phiq, phi)
    sc = sc.reshape(B, nb, block_size, H).amax(dim=2)
    kb = min(top_blocks, nb)
    tb = sc.topk(kb, dim=1).indices
    block_mask = torch.zeros(B, H, nb, dtype=torch.bool)
    block_mask.scatter_(2, tb.permute(0, 2, 1), True)
    tok_mask = block_mask.repeat_interleave(block_size, dim=-1)[:, :, :t]

    w_start = max(0, t - window)
    tok_mask = tok_mask | (torch.arange(t) >= w_start).view(1, 1, t)

    e = torch.einsum('bhk,bthk->bht', q, K_all) / math.sqrt(dk)
    e = e.masked_fill(~tok_mask, float('-inf'))
    if f"{pfx}sink" in sd:
        e = torch.cat([e, sd[f"{pfx}sink"].float().view(1, 1, 1).expand(B, H, 1)], dim=-1)
    a = F.softmax(e, dim=-1)[..., :t]
    y = torch.einsum('bht,bthv->bhv', a, V_all)

    out = F.linear(y.reshape(B, H * dv), sd[f"{pfx}W_o.weight"])
    if f"{pfx}out_norm.weight" in sd:
        out = rms_norm(out, sd[f"{pfx}out_norm.weight"])
    return out, kv_cache

def moe_fwd_on_demand(x, sd, pfx, expert_loader, layer_idx, top_k=10):
    scores = F.linear(x, sd[f"{pfx}gate.weight"])
    tv, ti = scores.topk(top_k, dim=-1)
    tw = F.softmax(tv, dim=-1)
    
    out = torch.zeros_like(x)
    for k in range(top_k):
        ei = ti[:, k].item()
        wt = tw[:, k:k+1]
        
        W1, W2 = expert_loader.get_expert(layer_idx, ei)
        
        # W1: (2560, 1280) = gate_up_proj 转置
        # 拆为 gate(,:640) + up(:,640:)，实现 SwiGLU
        W_gate_t = W1[:, :640]    # (2560, 640)
        W_up_t = W1[:, 640:]      # (2560, 640)
        gate = F.silu(F.linear(x, W_gate_t.T))   # silu(x @ gate_proj.T)
        up = F.linear(x, W_up_t.T)               # x @ up_proj.T
        h = gate * up
        e = F.linear(h, W2.T)
        out = out + wt * e
    
    gate_x = F.silu(F.linear(x, sd[f"{pfx}shared_W1.weight"]))
    if f"{pfx}shared_W_up.weight" in sd:
        up_x = F.linear(x, sd[f"{pfx}shared_W_up.weight"])
        gate_x = gate_x * up_x
    shared = F.linear(gate_x, sd[f"{pfx}shared_W2.weight"])
    if f"{pfx}shared_gate" in sd:
        shared = shared * sd[f"{pfx}shared_gate"]
    return out + shared


def init_gdn_states(n_layers, H, dk, dv):
    """初始化 ASM-ERSA 状态（循环状态 + conv1d 缓存）"""
    states = {}
    conv_states = {}
    for i in range(n_layers):
        if i % 4 != 3:
            states[i] = torch.zeros(1, H, dv, dk)
            conv_states[i] = torch.zeros(1, H*dk + H*dk + H*dv, 3)  # Q+K+V, 最近3步
    return states, conv_states


def single_step(x_t, sd, config, expert_loader, gdn_states, conv_states, ersa_caches, step,
                ple_embedding=None, ngram_context=None, token_id=None):
    """单 token 前向传播，返回 (logits, branches, gdn_states, conv_states, ersa_caches)"""
    H = config["n_heads"]
    dk = config["d_k"]
    dv = config["d_v"]
    n_layers = config["n_layers"]
    d_model = config["d_model"]

    branches = [x_t.clone() for _ in range(4)]

    for layer_idx in range(n_layers):
        pfx = f"layers.{layer_idx}."
        is_ersa = (layer_idx % 4 == 3)

        # PLE: 仅在层 1，在 GR Read 之前
        if layer_idx == 1 and ple_embedding is not None:
            branches = ple_fwd(sd, f"layers.{layer_idx}.ple.", ple_embedding, branches, token_id, ngram_context, d_model)

        # GR Read 1
        normed = [rms_norm(branches[i], sd[f"{pfx}gr1.norms.{i}.weight"]) for i in range(4)]
        concat = torch.cat(normed, dim=-1)
        gates = F.linear(concat, sd[f"{pfx}gr1.read_proj.weight"]).sigmoid()
        stacked = torch.stack(normed, dim=1)
        x = (gates.unsqueeze(-1) * stacked).mean(dim=1)

        # Token Mixing
        if not is_ersa:
            y, gdn_states[layer_idx], conv_states[layer_idx] = gdn2_fwd(
                x, sd, f"{pfx}token_mixing.", H, dk, dv,
                gdn_states[layer_idx], conv_states[layer_idx]
            )
        else:
            y, ersa_caches[layer_idx] = ersa_fwd(
                x, sd, f"{pfx}token_mixing.", H, dk, dv,
                ersa_caches.get(layer_idx), step
            )

        # GR Write 1
        wp = sd[f"{pfx}gr1.write_proj.weight"]
        wb = sd.get(f"{pfx}gr1.write_proj.bias")
        sc = 2.0 * F.linear(y, wp, wb).sigmoid()
        branches = [branches[i] + sc[:, i:i+1] * y for i in range(4)]

        # GR Read 2
        normed = [rms_norm(branches[i], sd[f"{pfx}gr2.norms.{i}.weight"]) for i in range(4)]
        concat = torch.cat(normed, dim=-1)
        gates = F.linear(concat, sd[f"{pfx}gr2.read_proj.weight"]).sigmoid()
        stacked = torch.stack(normed, dim=1)
        x = (gates.unsqueeze(-1) * stacked).mean(dim=1)

        # MoE
        y = moe_fwd_on_demand(x, sd, f"{pfx}moe.", expert_loader, layer_idx)

        # GR Write 2
        wp = sd[f"{pfx}gr2.write_proj.weight"]
        wb = sd.get(f"{pfx}gr2.write_proj.bias")
        sc = 2.0 * F.linear(y, wp, wb).sigmoid()
        branches = [branches[i] + sc[:, i:i+1] * y for i in range(4)]

    final = torch.stack(branches, dim=1).mean(dim=1)
    final = rms_norm(final, sd["out_norm.weight"])
    logits = F.linear(final, sd["lm_head.weight"])

    return logits[0], branches, gdn_states, conv_states, ersa_caches


# ═══════════════════════════════════════════════════════════════
# 交互式对话
# ═══════════════════════════════════════════════════════════════

def main():
    log("=" * 60)
    log("ASM-ERSA 交互式控制台（SSD 按需加载版）")
    log("=" * 60)

    model_dir = "C:\\models\\asm_ersa-int8-ple"
    tokenizer_path = os.environ.get("ASM_ERSA_TOKENIZER", "tokenizer.json")

    log("\n[初始化] 加载 tokenizer...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    log(f"  vocab 大小: {tokenizer.get_vocab_size()}")

    log("\n[初始化] 加载基础权重...")
    t0 = time.time()
    sd, config = load_compact(model_dir)
    log(f"  完成: {time.time()-t0:.1f}s")

    log("\n[初始化] 构建专家索引...")
    t0 = time.time()
    expert_index = ExpertIndex(model_dir)
    expert_loader = ExpertLoader(expert_index, cache_size=500)
    log(f"  完成: {time.time()-t0:.1f}s")

    log("\n[初始化] 构建 PLE 索引...")
    t0 = time.time()
    ple_embedding = None
    has_ple = any("ple." in k for k in sd if k.startswith("layers.1."))
    if has_ple:
        ple_embedding = PLEEmbedding(sd, "layers.1.ple.", model_dir)
        log(f"  PLE 就绪: head_dim={ple_embedding.head_dim}, shards={ple_embedding.n_shards}")
    else:
        log("  无 PLE 权重，跳过")
    log(f"  完成: {time.time()-t0:.1f}s")

    vocab_size = config["vocab_size"]
    n_layers = config["n_layers"]
    H = config["n_heads"]
    dk = config["d_k"]
    dv = config["d_v"]
    embed_w = sd["embed.weight"]

    log("\n" + "=" * 60)
    log("就绪！输入文本开始对话（输入 'quit' 退出）")
    log("注意：INT8 量化 + CPU 推理，结果质量可能有限")
    log("=" * 60)

    while True:
        try:
            user_input = input("\n你: ").strip()

            if user_input.lower() in ('quit', 'exit', 'q'):
                log("\n再见！")
                break

            if not user_input:
                continue

            token_ids = tokenizer.encode(user_input).ids
            if not token_ids:
                continue

            log(f"\n模型处理中（{len(token_ids)} 个 token）...")
            t0 = time.time()

            gdn_states, conv_states = init_gdn_states(n_layers, H, dk, dv)
            ersa_caches = {}
            logits = None
            ngram_context = {"prev1": None, "prev2": None, "conv_state": None}

            with torch.no_grad():
                for i, tid in enumerate(token_ids):
                    x_t = F.embedding(torch.tensor([tid]), embed_w)
                    logits, _, gdn_states, conv_states, ersa_caches = single_step(
                        x_t, sd, config, expert_loader, gdn_states, conv_states, ersa_caches, i,
                        ple_embedding=ple_embedding, ngram_context=ngram_context,
                        token_id=torch.tensor([tid])
                    )
                    ngram_context["prev2"] = ngram_context["prev1"]
                    ngram_context["prev1"] = torch.tensor([tid])
                    if (i + 1) % 10 == 0:
                        log(f"  已处理 {i+1}/{len(token_ids)} token...")

            elapsed = time.time() - t0

            top10 = logits.topk(10)

            log(f"\n预测结果 ({elapsed:.1f}s, {elapsed/max(len(token_ids),1)*1000:.0f}ms/token):")
            log("  Top-10 预测:")
            for i, (score, token_id) in enumerate(zip(top10.values, top10.indices)):
                tid = token_id.item()
                text = tokenizer.decode([tid])
                log(f"    {i+1}. [{tid:6d}] score={score.item():.2f}  →  \"{text}\"")

            best_id = top10.indices[0].item()
            best_text = tokenizer.decode([best_id])
            log(f"\n  最佳预测: \"{best_text}\" (token {best_id})")

        except KeyboardInterrupt:
            log("\n\n中断！")
            break
        except Exception as e:
            log(f"\n错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
