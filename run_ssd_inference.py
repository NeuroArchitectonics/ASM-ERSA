"""
ASM-ERSA INT8 SSD 按需加载推理
============================
专家权重保留在 SSD，推理时按需加载
内存占用：~15-20GB（而非 83GB）
"""

import torch
import torch.nn.functional as F
import json, os, gc, time, struct
from pathlib import Path
from safetensors import safe_open
from collections import OrderedDict

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
# 专家索引构建
# ═══════════════════════════════════════════════════════════════

class ExpertIndex:
    """专家权重索引，支持快速随机访问"""
    
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.index = {}  # key -> (shard_file, tensor_key, scale_key)
        self._build_index()
    
    def _build_index(self):
        log("  构建专家索引...")
        shard_files = sorted(self.model_dir.glob("model-*.safetensors"))
        
        # 第一遍：收集所有 scale 的位置
        scale_locations = {}
        for shard_path in shard_files:
            with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                keys = list(f.keys())
                for key in keys:
                    if key.startswith("_scale.") and (key.endswith(".moe.W1") or key.endswith(".moe.W2")):
                        weight_key = key[7:]  # 去掉 "_scale."
                        scale_locations[weight_key] = str(shard_path)
        
        # 第二遍：收集权重位置
        for shard_path in shard_files:
            with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                keys = list(f.keys())
                for key in keys:
                    if (key.endswith(".moe.W1") or key.endswith(".moe.W2")) and not key.startswith("_scale."):
                        scale_key = f"_scale.{key}"
                        if key in scale_locations:
                            scale_shard = scale_locations[key]
                            self.index[key] = (str(shard_path), key, scale_key, scale_shard)
        
        log(f"    索引完成: {len(self.index)} 个专家权重张量")
    
    def load_expert(self, layer_idx, expert_idx, group_size=128):
        """加载单个专家的权重"""
        w1_key = f"layers.{layer_idx}.moe.W1"
        w2_key = f"layers.{layer_idx}.moe.W2"
        
        w1 = self._load_tensor_slice(w1_key, expert_idx, group_size)
        w2 = self._load_tensor_slice(w2_key, expert_idx, group_size)
        
        return w1, w2
    
    def _load_tensor_slice(self, base_key, expert_idx, group_size):
        """加载专家切片（W1[expert_idx] 或 W2[expert_idx]）"""
        if base_key not in self.index:
            raise KeyError(f"未找到 {base_key}")
        
        weight_shard, tensor_key, scale_key, scale_shard = self.index[base_key]
        
        # 加载权重
        with safe_open(weight_shard, framework="pt", device="cpu") as f:
            full_tensor = f.get_tensor(tensor_key)  # [512, ...]
            expert_tensor = full_tensor[expert_idx]
        
        # 加载 scale
        with safe_open(scale_shard, framework="pt", device="cpu") as f:
            full_scale = f.get_tensor(scale_key)
        
        # 计算每个专家的 scale 数量
        elements_per_expert = expert_tensor.numel()
        scales_per_expert = elements_per_expert // group_size
        
        # 提取该专家的 scale
        start = expert_idx * scales_per_expert
        end = start + scales_per_expert
        expert_scale = full_scale[start:end]
        
        # 反量化
        return dequantize_int8(expert_tensor, expert_scale, group_size)


# ═══════════════════════════════════════════════════════════════
# LRU 缓存
# ═══════════════════════════════════════════════════════════════

class LRUCache:
    """最近最少使用缓存"""
    
    def __init__(self, maxsize=200):
        self.cache = OrderedDict()
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0
    
    def get(self, key):
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        return None
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)
    
    def stats(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return f"缓存: {len(self.cache)}/{self.maxsize}, 命中率: {hit_rate:.1%}"


# ═══════════════════════════════════════════════════════════════
# 专家加载器
# ═══════════════════════════════════════════════════════════════

class ExpertLoader:
    """按需从 SSD 加载专家"""
    
    def __init__(self, expert_index, cache_size=200):
        self.expert_index = expert_index
        self.cache = LRUCache(maxsize=cache_size)
        self.group_size = 128
    
    def get_expert(self, layer_idx, expert_idx):
        """获取专家权重（缓存优先）"""
        key = (layer_idx, expert_idx)
        
        # 缓存命中
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        
        # SSD 加载
        w1, w2 = self.expert_index.load_expert(layer_idx, expert_idx, self.group_size)
        self.cache.put(key, (w1, w2))
        
        return w1, w2
    
    def preload_experts(self, layer_idx, expert_indices):
        """预加载一批专家"""
        for idx in expert_indices:
            key = (layer_idx, idx)
            if self.cache.get(key) is None:
                w1, w2 = self.expert_index.load_expert(layer_idx, idx, self.group_size)
                self.cache.put(key, (w1, w2))
    
    def memory_usage(self):
        """估算缓存内存占用"""
        total_params = 0
        for (w1, w2) in self.cache.cache.values():
            total_params += w1.numel() + w2.numel()
        return total_params * 2  # FP16 = 2 bytes


# ═══════════════════════════════════════════════════════════════
# 紧凑加载（非专家权重）
# ═══════════════════════════════════════════════════════════════

def load_compact(model_dir):
    """加载非专家权重到内存"""
    model_path = Path(model_dir)
    with open(model_path / "config.json") as f:
        config = json.load(f)
    
    group_size = config.get("group_size", 128)
    shard_files = sorted(model_path.glob("model-*.safetensors"))
    
    # 第一遍：收集所有 scales
    log("  第一遍：收集 scales...")
    all_scales = {}
    for shard_path in shard_files:
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            for key in keys:
                if key.startswith("_scale."):
                    scale = f.get_tensor(key)
                    # 去掉 _scale. 前缀
                    weight_key = key[7:]  # len("_scale.") == 7
                    all_scales[weight_key] = scale
    
    log(f"    收集到 {len(all_scales)} 个 scales")
    
    # 第二遍：加载并反量化权重
    sd = {}
    total_bytes = 0
    
    for shard_idx, shard_path in enumerate(shard_files):
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            
            for key in keys:
                if key.startswith("_scale."):
                    continue
                
                # 跳过 MoE 专家权重（它们留在 SSD）
                if (key.endswith(".moe.W1") or key.endswith(".moe.W2")):
                    continue
                
                tensor = f.get_tensor(key)
                
                # 反量化
                if key in all_scales:
                    scale = all_scales[key]
                    tensor = dequantize_int8(tensor, scale, group_size)
                    del all_scales[key]  # 释放内存
                
                sd[key] = tensor
                total_bytes += tensor.numel() * tensor.element_size()
        
        if (shard_idx + 1) % 5 == 0:
            log(f"    已加载 {shard_idx+1}/{len(shard_files)} 分片, {total_bytes/1024**3:.1f}GB")
        gc.collect()
    
    log(f"  完成: {len(sd)} 个权重, {total_bytes/1024**3:.1f}GB")
    
    # 调试：检查一个权重的 dtype
    if "layers.0.token_mixing.W_q.weight" in sd:
        w = sd["layers.0.token_mixing.W_q.weight"]
        log(f"  调试: W_q.dtype = {w.dtype}")
    if "embed.weight" in sd:
        e = sd["embed.weight"]
        log(f"  调试: embed.dtype = {e.dtype}")
    
    return sd, config


# ═══════════════════════════════════════════════════════════════
# 推理引擎（按需加载版）
# ═══════════════════════════════════════════════════════════════

def rms_norm(x, weight, eps=1e-6):
    rms = x.float().pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    return (weight * x * rms).half()


def gdn2_fwd(x, sd, pfx, H, dk, dv, state):
    # 调试
    wq = sd[f"{pfx}W_q.weight"]
    if x.dtype != wq.dtype:
        print(f"DEBUG: x.dtype={x.dtype}, W_q.dtype={wq.dtype}, key={pfx}W_q.weight")
    
    q = F.linear(x, sd[f"{pfx}W_q.weight"])
    k = F.linear(x, sd[f"{pfx}W_k.weight"])
    v = F.linear(x, sd[f"{pfx}W_v.weight"])
    
    q = q.view(1, H, dk)
    k = k.view(1, H, dk)
    v = v.view(1, H, dv)
    
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    
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
    
    return out, state


def ersa_fwd(x, sd, pfx, H, dk, dv, kv_buf, kv_len,
             block_size=64, top_blocks=4, window=1024):
    """单 token 的 ERSA 前向：块级 sketch 路由 + 原始 K/V 精确重打分 + sink。"""
    t = kv_len
    q = F.normalize(F.linear(x, sd[f"{pfx}W_q.weight"]).view(1, H, dk).float(), dim=-1)
    K_all = kv_buf[:, :t, :, :dk].float()
    V_all = kv_buf[:, :t, :, dk:].float()

    sketch = sd[f"{pfx}sketch"].float()
    nb = (t + block_size - 1) // block_size
    pad = nb * block_size - t
    Kp = F.pad(K_all, (0, 0, 0, 0, 0, pad)) if pad else K_all
    phi = torch.einsum('bthk,rk->bthr', Kp, sketch)
    phiq = torch.einsum('bhk,rk->bhr', q, sketch)
    sc = torch.einsum('bhr,bthr->bth', phiq, phi)
    sc = sc.reshape(1, nb, block_size, H).amax(dim=2)
    tb = sc.topk(min(top_blocks, nb), dim=1).indices
    block_mask = torch.zeros(1, H, nb, dtype=torch.bool)
    block_mask.scatter_(2, tb.permute(0, 2, 1), True)
    tok_mask = block_mask.repeat_interleave(block_size, dim=-1)[:, :, :t]

    w_start = max(0, t - window)
    tok_mask = tok_mask | (torch.arange(t) >= w_start).view(1, 1, t)

    e = torch.einsum('bhk,bthk->bht', q, K_all) / (dk ** 0.5)
    e = e.masked_fill(~tok_mask, float('-inf'))
    if f"{pfx}sink" in sd:
        e = torch.cat([e, sd[f"{pfx}sink"].float().view(1, 1, 1).expand(1, H, 1)], dim=-1)
    a = F.softmax(e, dim=-1)[..., :t]
    y = torch.einsum('bht,bthv->bhv', a, V_all)

    out = F.linear(y.reshape(1, H * dv).to(x.dtype), sd[f"{pfx}W_o.weight"])
    if f"{pfx}out_norm.weight" in sd:
        out = rms_norm(out, sd[f"{pfx}out_norm.weight"])
    return out


def moe_fwd_on_demand(x, sd, pfx, expert_loader, layer_idx, top_k=10):
    """按需加载专家的 MoE 前向"""
    scores = F.linear(x, sd[f"{pfx}gate.weight"])
    tv, ti = scores.topk(top_k, dim=-1)
    tw = F.softmax(tv, dim=-1)
    
    out = torch.zeros_like(x)
    for k in range(top_k):
        ei = ti[:, k].item()
        wt = tw[:, k:k+1]
        
        # 按需加载专家
        W1, W2 = expert_loader.get_expert(layer_idx, ei)
        
        # W1: [d_model, intermediate], W2: [intermediate, d_model]
        h = F.silu(F.linear(x, W1.T))  # [1, intermediate]
        e = F.linear(h, W2.T)  # [1, d_model]
        out = out + wt * e
    
    gate_x = F.silu(F.linear(x, sd[f"{pfx}shared_W1.weight"]))
    if f"{pfx}shared_W_up.weight" in sd:
        up_x = F.linear(x, sd[f"{pfx}shared_W_up.weight"])
        gate_x = gate_x * up_x
    shared = F.linear(gate_x, sd[f"{pfx}shared_W2.weight"])
    if f"{pfx}shared_gate" in sd:
        shared = shared * sd[f"{pfx}shared_gate"]
    return out + shared


def run_inference(sd, config, expert_loader, input_ids):
    """按需加载版推理"""
    d_model = config["d_model"]
    H = config["n_heads"]
    dk = config["d_k"]
    dv = config["d_v"]
    n_layers = config["n_layers"]
    vocab = config["vocab_size"]
    
    B, T = input_ids.shape
    
    # 嵌入
    embed_w = sd["embed.weight"]
    
    # 初始化状态
    gdn_states = {}
    for i in range(n_layers):
        if i % 4 != 3:
            gdn_states[i] = torch.zeros(1, H, dv, dk)

    # ERSA KV 缓冲区预分配 [1, max_len, H, dk+dv]
    max_kv = T
    ersa_kv = {}
    ersa_len = {}
    for i in range(n_layers):
        if i % 4 == 3:
            ersa_kv[i] = torch.zeros(1, max_kv, H, dk + dv,
                                     dtype=sd["embed.weight"].dtype)
            ersa_len[i] = 0
    
    sample_logits = {}
    
    for t_step in range(T):
        x_t = F.embedding(input_ids[:, t_step], embed_w)
        branches = [x_t.clone() for _ in range(4)]
        
        for layer_idx in range(n_layers):
            pfx = f"layers.{layer_idx}."
            is_ersa = (layer_idx % 4 == 3)
            
            # GR Read 1
            normed = [rms_norm(branches[i], sd[f"{pfx}gr1.norms.{i}.weight"]) for i in range(4)]
            concat = torch.cat(normed, dim=-1)
            gates = F.linear(concat, sd[f"{pfx}gr1.read_proj.weight"]).sigmoid()
            stacked = torch.stack(normed, dim=1)
            x = (gates.unsqueeze(-1) * stacked).mean(dim=1)
            
            # Token Mixing
            if not is_ersa:
                y, gdn_states[layer_idx] = gdn2_fwd(
                    x, sd, f"{pfx}token_mixing.", H, dk, dv, gdn_states[layer_idx]
                )
            else:
                k_proj = F.linear(x, sd[f"{pfx}token_mixing.W_k.weight"]).view(1, 1, H, dk)
                v_proj = F.linear(x, sd[f"{pfx}token_mixing.W_v.weight"]).view(1, 1, H, dv)
                pos = ersa_len[layer_idx]
                ersa_kv[layer_idx][:, pos:pos+1] = torch.cat([k_proj, v_proj], dim=-1)
                ersa_len[layer_idx] = pos + 1
                y = ersa_fwd(x, sd, f"{pfx}token_mixing.", H, dk, dv,
                             ersa_kv[layer_idx], ersa_len[layer_idx])
            
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
            
            # MoE（按需加载专家）
            y = moe_fwd_on_demand(x, sd, f"{pfx}moe.", expert_loader, layer_idx)
            
            # GR Write 2
            wp = sd[f"{pfx}gr2.write_proj.weight"]
            wb = sd.get(f"{pfx}gr2.write_proj.bias")
            sc = 2.0 * F.linear(y, wp, wb).sigmoid()
            branches = [branches[i] + sc[:, i:i+1] * y for i in range(4)]
        
        # 输出：取 4 分支均值 → 最终归一化 → lm_head
        final = torch.stack(branches, dim=1).mean(dim=1)
        final = rms_norm(final, sd["out_norm.weight"])
        logits = F.linear(final, sd["lm_head.weight"])
        
        if t_step in (0, 1, 2, T-3, T-2, T-1):
            sample_logits[t_step] = logits[0].detach().clone()
        
        if (t_step + 1) % 10 == 0 or t_step == 0:
            mem_gb = expert_loader.memory_usage() / 1024**3
            log(f"  [{t_step+1}/{T}] 专家缓存: {mem_gb:.2f}GB | {expert_loader.cache.stats()}")
        
        gc.collect()
    
    return sample_logits


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    log("=" * 60)
    log("ASM-ERSA INT8 SSD 按需加载推理")
    log("=" * 60)
    
    model_dir = "models/asm_ersa-int8"
    num_tokens = 10  # 测试 10 个 token
    
    # 加载非专家权重
    log(f"\n[1/5] 加载基础权重（非专家）...")
    t0 = time.time()
    sd, config = load_compact(model_dir)
    log(f"  耗时: {time.time()-t0:.1f}s")
    
    # 构建专家索引
    log(f"\n[2/5] 构建专家索引...")
    t0 = time.time()
    expert_index = ExpertIndex(model_dir)
    log(f"  耗时: {time.time()-t0:.1f}s")
    
    # 创建专家加载器
    log(f"\n[3/5] 初始化专家加载器（缓存 200 个专家）...")
    expert_loader = ExpertLoader(expert_index, cache_size=200)
    
    # 准备输入
    log(f"\n[4/5] 准备输入...")
    vocab = config["vocab_size"]
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab, (1, num_tokens))
    log(f"  tokens: {num_tokens}")
    
    # 推理
    log(f"\n[5/5] 前向传播（按需加载专家）...")
    t0 = time.time()
    
    with torch.no_grad():
        sample_logits = run_inference(sd, config, expert_loader, input_ids)
    
    elapsed = time.time() - t0
    log(f"  耗时: {elapsed:.1f}s ({elapsed/len(sample_logits)*1000:.0f}ms/token)")
    
    # 结果
    log(f"\n{'='*60}")
    log("推理结果")
    log("=" * 60)
    
    for step in sorted(sample_logits.keys()):
        lg = sample_logits[step]
        top5 = lg.topk(5)
        tokens_str = ", ".join(
            f"{t.item()}({s.item():.1f})"
            for t, s in zip(top5.indices, top5.values)
        )
        log(f"  [{step:4d}] top-5: {tokens_str}")
    
    # 性能统计
    log(f"\n{'='*60}")
    log("性能统计")
    log("=" * 60)
    log(f"  推理速度: {elapsed/len(sample_logits)*1000:.0f} ms/token")
    log(f"  专家缓存: {expert_loader.cache.stats()}")
    log(f"  缓存内存: {expert_loader.memory_usage()/1024**3:.2f}GB")


if __name__ == "__main__":
    main()
