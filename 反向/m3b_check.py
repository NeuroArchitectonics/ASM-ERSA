# M3b 对拍：ERSA 层逐 token 训练环（KV cache 增长）—— 前向/反向 PyTorch 复算
# 用法：python 反向/m3b_check.py <导出.bin>
import sys
import numpy as np
import torch

raw = open(sys.argv[1], 'rb').read()
pos = 0
def take(n):
    global pos
    b = raw[pos:pos + n]
    pos += n
    return b
def u():
    return int(np.frombuffer(take(4), np.uint32)[0])
def f(n):
    return np.frombuffer(take(4 * n), np.float32).astype(np.float64)
def i32(n):
    return np.frombuffer(take(4 * n), np.uint32)

T, B, H, N, BLOCK, TOPB, WINDOW, DK, DV, NBMAX = [u() for _ in range(10)]
tau, sink = f(2)
q = f(T * DK).reshape(T, DK)
k = f(T * DK).reshape(T, DK)
v = f(T * DV).reshape(T, DV)
dO = f(T * DV).reshape(T, DV)

out_t = np.zeros((T, DV))
sel_t = np.zeros((T, TOPB), dtype=int)
mask_t = np.zeros((T, T))
bs_g = np.zeros((T, NBMAX, DK))
bs_h = np.zeros((T, NBMAX, DK))
for t in range(T):
    out_t[t] = f(DV)
    sel_t[t] = i32(TOPB)
    mask_t[t] = f(T)
    bs_g[t] = f(NBMAX * DK).reshape(NBMAX, DK)
    bs_h[t] = f(NBMAX * DK).reshape(NBMAX, DK)

dq_gpu = f(T * DK).reshape(T, DK)
dK_gpu = f(T * DK).reshape(T, DK)
dV_gpu = f(T * DV).reshape(T, DV)
dsink_t = f(T)
dsink_total = f(1)[0]
assert pos == len(raw), "导出文件长度与读序不符"

print(f"T={T} BLOCK={BLOCK} TOPB={TOPB} WINDOW={WINDOW} DK={DK} DV={DV} "
      f"tau={tau:.4f} sink={sink:.4f}")

def report(name, got, ref, tol=2e-4):
    err = float(np.max(np.abs(got - ref)))
    rel = err / max(1e-12, float(np.max(np.abs(ref))))
    ok = err <= tol or rel <= tol
    print(f"{name:10s} max|err|={err:.3e} rel={rel:.3e} {'PASS' if ok else 'FAIL'}")
    return ok

all_ok = True
scale = 1.0 / np.sqrt(DK)

# ── 1. 每步块摘要：GPU 全量重算 vs 主机增量式（前 NBt 块有效）──
bs_err = 0.0
for t in range(T):
    NBt = ((t + 1) + BLOCK - 1) // BLOCK
    bs_err = max(bs_err, float(np.max(np.abs(bs_g[t, :NBt] - bs_h[t, :NBt]))))
print(f"bsum       max|err|={bs_err:.3e} {'PASS' if bs_err <= 1e-6 else 'FAIL'}")
all_ok &= bs_err <= 1e-6

# ── 2. 路由/掩码复算 ──
sel_bad, mask_bad = 0, 0
for t in range(T):
    Nt, NBt = t + 1, ((t + 1) + BLOCK - 1) // BLOCK
    topn = min(TOPB, NBt)
    s = (q[t] @ bs_g[t, :NBt].T) * scale / tau
    ref = []
    for _ in range(topn):
        bi = int(np.argmax(s))
        ref.append(bi)
        s[bi] = -3.402823e38
    sel_bad += int(np.sum(np.array(ref) != sel_t[t, :topn]))
    m = np.zeros(T)
    for bi in ref:
        for idx in range(bi * BLOCK, min(bi * BLOCK + BLOCK, Nt)):
            m[idx] = 1.0
    for idx in range(max(0, Nt - WINDOW), Nt):
        m[idx] = 1.0
    mask_bad += int(np.sum(m != mask_t[t]))
print(f"sel        mismatch={sel_bad}  mask mismatch={mask_bad}")
all_ok &= sel_bad == 0 and mask_bad == 0

# ── 3. 前向 + autograd ──
qt = torch.tensor(q, dtype=torch.float64, requires_grad=True)
kt = torch.tensor(k, dtype=torch.float64, requires_grad=True)
vt = torch.tensor(v, dtype=torch.float64, requires_grad=True)
st = torch.tensor(float(sink), dtype=torch.float64, requires_grad=True)
dOt = torch.tensor(dO, dtype=torch.float64)

loss = torch.zeros((), dtype=torch.float64)
outs_ref = []
P_list = []
for t in range(T):
    Nt = t + 1
    Kc, Vc = kt[:Nt], vt[:Nt]
    S = (qt[t] @ Kc.T) * scale / tau
    m = torch.tensor(mask_t[t, :Nt].astype(bool))
    S = S.masked_fill(~m, float('-inf'))
    logits = torch.cat([S, st.view(1)])
    P = torch.softmax(logits, dim=0)
    out = P[:Nt] @ Vc
    outs_ref.append(out)
    P_list.append(P)
    loss = loss + (out * dOt[t]).sum()

gq, gk, gv, gs = torch.autograd.grad(loss, [qt, kt, vt, st])

all_ok &= report("out", out_t, torch.stack(outs_ref).detach().numpy())
all_ok &= report("dq", dq_gpu, gq.numpy())
all_ok &= report("dK", dK_gpu, gk.numpy())
all_ok &= report("dV", dV_gpu, gv.numpy())

# 逐 token 的 sink 梯度（P[:, Nt] * (0 - a·dA)）
contrib = np.zeros(T)
for t in range(T):
    Nt = t + 1
    dA = v[:Nt] @ dO[t]
    contrib[t] = float(P_list[t][Nt]) * (0.0 - float((P_list[t][:Nt] * torch.tensor(dA)).sum()))
all_ok &= report("dsink/t", dsink_t, contrib)
all_ok &= report("dsink", np.array([dsink_total]), np.array([float(gs)]))

print("=== M3b: " + ("PASS" if all_ok else "FAIL") + " ===")
sys.exit(0 if all_ok else 1)
