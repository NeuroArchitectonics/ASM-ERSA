"""
ASM-ERSA INT8 CPU 推理（内存优化版）
==================================
权重保持 INT8 紧凑格式（~81GB），推理时逐张量按需反量化
避免全量展开至 FP32（~320GB）的内存开销

核心改动：
  1. 加载时保留 INT8 + FP16 scales，不提前反量化
  2. 前向传播中按需反量化单个权重，用完即释放
  3. 嵌入/lm_head 反量化一次后缓存（反复使用）
  4. KV 缓冲区预分配，避免 O(t²) 拼接
"""

import torch
import torch.nn.functional as F
import json, os, gc, time
from pathlib import Path
from safetensors import safe_open

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["OMP_NUM_THREADS"] = "8"
torch.set_num_threads(8)


def log(msg):
    print(msg, flush=True)


# ═══════════════════════════════════════════════════════════════
# INT8 → FP16 反量化（快速路径，无 FP32 中间值）
# ═══════════════════════════════════════════════════════════════

def dequantize_int8(q_tensor, scale, group_size=128):
    """INT8 对称量化反量化：q ∈ [-127, 127]，x̂ = q · scale。

    INT4 旧权重（q ∈ [-8, 7]，scale = max/7）同样满足 x̂ = q · scale，
    因此本函数也兼容 INT4 checkpoint。
    """
    flat = q_tensor.half().flatten()
    n = flat.numel()
    pad = (group_size - n % group_size) % group_size
    if pad > 0:
        flat = F.pad(flat, (0, pad))
    groups = flat.reshape(-1, group_size)
    dequant = groups * scale.half().unsqueeze(1)
    return dequant.flatten()[:n].reshape(q_tensor.shape)


# ═══════════════════════════════════════════════════════════════
# 紧凑加载
# ═══════════════════════════════════════════════════════════════

def load_compact(model_dir):
    model_path = Path(model_dir)
    with open(model_path / "config.json") as f:
        config = json.load(f)

    shard_files = sorted(model_path.glob("model-*.safetensors"))
    log(f"  分片: {len(shard_files)} 个")

    sd = {}
    scales = {}
    total_bytes = 0

    for idx, sp in enumerate(shard_files):
        sz = sp.stat().st_size
        log(f"  [{idx+1}/{len(shard_files)}] {sp.name} ({sz/1024**3:.1f}GB)")

        with safe_open(str(sp), framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                if key.startswith("_scale."):
                    scales[key[7:]] = t
                else:
                    sd[key] = t
                total_bytes += t.numel() * t.element_size()

        gc.collect()
        log(f"    累计: {total_bytes/1024**3:.1f}GB")

    log(f"  完成: {len(sd)} 权重 + {len(scales)} 尺度 = {total_bytes/1024**3:.1f}GB")
    return sd, scales, config


# ═══════════════════════════════════════════════════════════════
# 权重访问器（按需反量化 + 缓存）
# ═══════════════════════════════════════════════════════════════

class Weights:
    def __init__(self, sd, scales, group_size):
        self.sd = sd
        self.scales = scales
        self.group_size = group_size
        self._cache = {}

    def get(self, name):
        if name in self._cache:
            return self._cache[name]
        if name in self.scales:
            return dequantize_int8(self.sd[name], self.scales[name], self.group_size)
        if name in self.sd:
            return self.sd[name]
        return None

    def cache(self, name):
        if name not in self._cache:
            self._cache[name] = self.get(name)
        return self._cache[name]

    def mem_report(self):
        sd_bytes = sum(t.numel() * t.element_size() for t in self.sd.values())
        sc_bytes = sum(t.numel() * t.element_size() for t in self.scales.values())
        cache_bytes = sum(t.numel() * t.element_size() for t in self._cache.values())
        return sd_bytes, sc_bytes, cache_bytes


# ═══════════════════════════════════════════════════════════════
# 推理原语
# ═══════════════════════════════════════════════════════════════

def rms_norm(x, weight, eps=1e-6):
    rms = x.float().pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    return (weight * x * rms).half()


def apply_rope(x, positions, d_k):
    half = d_k // 2
    freq = 1.0 / (10000 ** (torch.arange(0, d_k, 2, device=x.device).float() / d_k))
    angles = positions.unsqueeze(-1).float() * freq
    cos_a, sin_a = angles.cos(), angles.sin()
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos_a - x2 * sin_a, x1 * sin_a + x2 * cos_a], dim=-1).type_as(x)


# ── ASM-ERSA ──

def gdn2_fwd(x, w, pfx, H, dk, dv, state):
    q = F.linear(x, w.get(f"{pfx}W_q.weight")).view(1, H, dk)
    k = F.linear(x, w.get(f"{pfx}W_k.weight")).view(1, H, dk)
    v = F.linear(x, w.get(f"{pfx}W_v.weight")).view(1, H, dv)
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)

    b = F.linear(x, w.get(f"{pfx}W_b.weight")).view(1, H, dk).sigmoid()
    ww = F.linear(x, w.get(f"{pfx}W_w.weight")).view(1, H, dv).sigmoid()
    a = F.linear(x, w.get(f"{pfx}W_a.weight")).view(1, H, dk)

    A_log = w.get(f"{pfx}A_log")
    dt_bias = w.get(f"{pfx}dt_bias")
    alpha = (-A_log.exp() * F.softplus(a + dt_bias)).exp()

    bk = b * k
    D_state = alpha.unsqueeze(-2) * state
    
    # einsum 需要 float32
    bk_f = bk.float()
    D_state_f = D_state.float()
    bk_dot_D = torch.einsum('bhk,bhvk->bhv', bk_f, D_state_f)
    correction = torch.einsum('bhk,bhv->bhvk', k.float(), bk_dot_D)
    state = (D_state_f - correction).half()
    
    wv = ww * v
    state = state + torch.einsum('bhk,bhv->bhvk', k.float(), (ww * v).float()).half()

    out = torch.einsum('bhk,bhvk->bhv', q.float(), state.float()).half()
    return F.linear(out.reshape(1, H * dv), w.get(f"{pfx}W_o.weight")), state


# ── ERSA（精确回召稀疏注意力）──

def ersa_fwd(x, w, pfx, H, dk, dv, kv_buf, kv_len,
             block_size=64, top_blocks=4, window=1024):
    """单 token 的 ERSA 前向：块级 sketch 路由 + 原始 K/V 精确重打分 + sink。"""
    t = kv_len
    q = F.linear(x, w.get(f"{pfx}W_q.weight")).view(1, H, dk).float()
    q = F.normalize(q, dim=-1)
    K_all = kv_buf[:, :t, :, :dk].float()
    V_all = kv_buf[:, :t, :, dk:].float()

    sketch = w.get(f"{pfx}sketch").float()          # [r, dk]
    nb = (t + block_size - 1) // block_size
    pad = nb * block_size - t
    Kp = F.pad(K_all, (0, 0, 0, 0, 0, pad)) if pad else K_all
    phi = torch.einsum('bthk,rk->bthr', Kp, sketch)
    phiq = torch.einsum('bhk,rk->bhr', q, sketch)
    sc = torch.einsum('bhr,bthr->bth', phiq, phi)
    sc = sc.reshape(1, nb, block_size, H).amax(dim=2)          # [1, nb, H]
    kb = min(top_blocks, nb)
    tb = sc.topk(kb, dim=1).indices
    block_mask = torch.zeros(1, H, nb, dtype=torch.bool)
    block_mask.scatter_(2, tb.permute(0, 2, 1), True)
    tok_mask = block_mask.repeat_interleave(block_size, dim=-1)[:, :, :t]

    w_start = max(0, t - window)
    tok_mask = tok_mask | (torch.arange(t) >= w_start).view(1, 1, t)

    e = torch.einsum('bhk,bthk->bht', q, K_all) / (dk ** 0.5)
    e = e.masked_fill(~tok_mask, float('-inf'))
    sink = w.get(f"{pfx}sink")
    if sink is not None:
        e = torch.cat([e, sink.float().view(1, 1, 1).expand(1, H, 1)], dim=-1)
    a = F.softmax(e, dim=-1)[..., :t]
    y = torch.einsum('bht,bthv->bhv', a, V_all)

    out = F.linear(y.reshape(1, H * dv).to(x.dtype), w.get(f"{pfx}W_o.weight"))
    ow = w.get(f"{pfx}out_norm.weight")
    if ow is not None:
        out = rms_norm(out, ow)
    return out

# ── Gated Residual ──

def gr_read(w, pfx, branches):
    normed = [rms_norm(branches[i], w.get(f"{pfx}norms.{i}.weight")) for i in range(4)]
    concat = torch.cat(normed, dim=-1)
    gates = F.linear(concat, w.get(f"{pfx}read_proj.weight")).sigmoid()
    stacked = torch.stack(normed, dim=1)
    return (gates.unsqueeze(-1) * stacked).mean(dim=1)


def gr_write(w, pfx, branches, y):
    wp = w.get(f"{pfx}write_proj.weight")
    wb = w.get(f"{pfx}write_proj.bias")
    sc = 2.0 * F.linear(y, wp, wb).sigmoid()
    return [branches[i] + sc[:, i:i+1] * y for i in range(4)]


# ── MoE ──

def moe_fwd(x, w, pfx, top_k=10):
    scores = F.linear(x, w.get(f"{pfx}gate.weight"))
    tv, ti = scores.topk(top_k, dim=-1)
    tw = F.softmax(tv, dim=-1)

    W1 = w.get(f"{pfx}W1")
    W2 = w.get(f"{pfx}W2")

    out = torch.zeros_like(x)
    for k in range(top_k):
        ei = ti[:, k]
        wt = tw[:, k:k+1]
        h = F.silu(torch.einsum('bd,bdf->bf', x, W1[ei]))
        e = torch.einsum('bf,bfd->bd', h, W2[ei])
        out = out + wt * e

    del W1, W2
    shared = F.linear(F.silu(F.linear(x, w.get(f"{pfx}shared_W1.weight"))),
                      w.get(f"{pfx}shared_W2.weight"))
    return out + shared


# ═══════════════════════════════════════════════════════════════
# 完整前向
# ═══════════════════════════════════════════════════════════════

def run_inference(w, config, input_ids, max_tokens=None):
    d_model = config["d_model"]
    H = config["n_heads"]
    dk = config["d_k"]
    dv = config["d_v"]
    n_layers = config["n_layers"]
    vocab = config["vocab_size"]

    B, T = input_ids.shape
    if max_tokens and max_tokens < T:
        T = max_tokens

    log(f"  嵌入 + lm_head 反量化（缓存）...")
    embed_w = w.cache("embed.weight")
    lm_head_w = w.cache("lm_head.weight")
    sd_b, sc_b, ca_b = w.mem_report()
    log(f"  缓存后内存: 权重 {sd_b/1024**3:.1f}GB + 尺度 {sc_b/1024**3:.1f}GB + 缓存 {ca_b/1024**3:.1f}GB")

    # ASM-ERSA 状态 [1, H, dv, dk]
    gdn_st = {}
    for i in range(n_layers):
        if i % 4 != 3:
            gdn_st[i] = torch.zeros(1, H, dv, dk)

    # ERSA KV 缓冲区预分配 [1, max_len, H, dk+dv]
    max_kv = T
    ersa_kv = {}
    ersa_len = {}
    for i in range(n_layers):
        if i % 4 == 3:
            ersa_kv[i] = torch.zeros(1, max_kv, H, dk + dv)
            ersa_len[i] = 0

    sample_logits = {}

    for t_step in range(T):
        x_t = F.embedding(input_ids[:, t_step], embed_w)  # [1, d_model]
        branches = [x_t.clone() for _ in range(4)]

        for li in range(n_layers):
            pfx = f"layers.{li}."
            is_ersa = (li % 4 == 3)

            x = gr_read(w, f"{pfx}gr1.", branches)

            if not is_ersa:
                y, gdn_st[li] = gdn2_fwd(x, w, f"{pfx}token_mixing.", H, dk, dv, gdn_st[li])
            else:
                k_proj = F.linear(x, w.get(f"{pfx}token_mixing.W_k.weight")).view(1, 1, H, dk)
                v_proj = F.linear(x, w.get(f"{pfx}token_mixing.W_v.weight")).view(1, 1, H, dv)
                pos = ersa_len[li]
                ersa_kv[li][:, pos:pos+1] = torch.cat([k_proj, v_proj], dim=-1)
                ersa_len[li] = pos + 1
                y = ersa_fwd(x, w, f"{pfx}token_mixing.", H, dk, dv,
                            ersa_kv[li], ersa_len[li])

            branches = gr_write(w, f"{pfx}gr1.", branches, y)

            x = gr_read(w, f"{pfx}gr2.", branches)
            y = moe_fwd(x, w, f"{pfx}moe.")
            branches = gr_write(w, f"{pfx}gr2.", branches, y)

        final = sum(branches) / 4.0
        final = rms_norm(final, w.get("out_norm.weight"))
        logits = F.linear(final, lm_head_w)  # [1, vocab]

        if t_step in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                      T // 4, T // 2, 3 * T // 4,
                      T - 3, T - 2, T - 1):
            sample_logits[t_step] = logits[0].detach().clone()

        gc.collect()

        if (t_step + 1) % 10 == 0 or t_step == 0:
            sd_b, sc_b, ca_b = w.mem_report()
            log(f"  [{t_step+1}/{T}] 内存: 权重{sd_b/1024**3:.1f} + 尺度{sc_b/1024**3:.1f} + 缓存{ca_b/1024**3:.1f} GB")

    return sample_logits


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    log("=" * 60)
    log("ASM-ERSA INT8 CPU 推理（内存优化版）")
    log("=" * 60)

    model_dir = "models/asm_ersa-int8"
    num_tokens = 3  # 快速测试

    log(f"\n[1/4] 紧凑加载权重（保持 INT8 格式）...")
    t0 = time.time()
    sd, scales, config = load_compact(model_dir)
    group_size = config.get("group_size", 128)
    w = Weights(sd, scales, group_size)
    log(f"  耗时: {time.time()-t0:.1f}s")

    log(f"\n[2/4] 准备输入...")
    vocab = config["vocab_size"]
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab, (1, num_tokens))
    log(f"  tokens: {num_tokens}, vocab: {vocab}")
    log(f"  前 10 token: {input_ids[0, :10].tolist()}")

    log(f"\n[3/4] 前向传播（{num_tokens} tokens × {config['n_layers']} 层）...")
    t0 = time.time()

    with torch.no_grad():
        sample_logits = run_inference(w, config, input_ids)

    elapsed = time.time() - t0
    log(f"  耗时: {elapsed:.1f}s ({elapsed/len(sample_logits)*1000:.0f}ms/token)")

    log(f"\n[4/4] 结果")
    log("=" * 60)

    for step in sorted(sample_logits.keys()):
        lg = sample_logits[step]
        top5 = lg.topk(5)
        tokens_str = ", ".join(
            f"{t.item()}({s.item():.1f})"
            for t, s in zip(top5.indices, top5.values)
        )
        log(f"  [{step:4d}] range=[{lg.min().item():.1f}, {lg.max().item():.1f}] "
            f"μ={lg.mean().item():.1f} σ={lg.std().item():.1f}")
        log(f"         top-5: {tokens_str}")

    log(f"\n{'='*60}")
    log(f"性能: {elapsed/len(sample_logits)*1000:.0f} ms/token, "
        f"{1000/(elapsed/len(sample_logits)):.1f} tok/s")

    sd_b, sc_b, ca_b = w.mem_report()
    total = (sd_b + sc_b + ca_b) / 1024**3
    log(f"内存: 权重{sd_b/1024**3:.1f} + 尺度{sc_b/1024**3:.1f} + 缓存{ca_b/1024**3:.1f} = {total:.1f}GB")


if __name__ == "__main__":
    main()
