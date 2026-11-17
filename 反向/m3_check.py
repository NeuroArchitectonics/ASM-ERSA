# M3 第 1/2 步对拍：ERSA 前向（块路由+窗口+sink）+ 逐 query 反向
# 用法：python 反向/m3_check.py <导出.bin>
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

T, B, H, N, BLOCK, TOPB, WINDOW, DK, DV, NB = [u() for _ in range(10)]
tau, sink = f(2)
q = f(T * DK).reshape(T, DK)
k = f(N * DK).reshape(N, DK)
v = f(N * DV).reshape(N, DV)
dO = f(T * DV).reshape(T, DV)
bs_gpu = f(NB * DK).reshape(NB, DK)
bs_host = f(NB * DK).reshape(NB, DK)
out_gpu = f(T * DV).reshape(T, DV)
masks = f(T * N).reshape(T, N)
sel_gpu = i32(T * TOPB).reshape(T, TOPB)
dq_gpu = f(T * DK).reshape(T, DK)
dK_gpu = f(N * DK).reshape(N, DK)
dV_gpu = f(N * DV).reshape(N, DV)
dsink_gpu = f(T)
assert pos == len(raw), "导出文件长度与读序不符"

print(f"T={T} B={B} H={H} N={N} BLOCK={BLOCK} NB={NB} TOPB={TOPB} WINDOW={WINDOW} "
      f"DK={DK} DV={DV} tau={tau} sink={sink}")

def report(name, got, ref, tol=2e-4):
    err = float(np.max(np.abs(got - ref)))
    rel = err / max(1e-12, float(np.max(np.abs(ref))))
    ok = err <= tol or rel <= tol
    print(f"{name:8s} max|err|={err:.3e} rel={rel:.3e} {'PASS' if ok else 'FAIL'}")
    return ok

all_ok = True

# ── 1. 块摘要：GPU vs 主机 mean ──
bs_err = float(np.max(np.abs(bs_gpu - bs_host)))
print(f"bsum     max|err|={bs_err:.3e} {'PASS' if bs_err <= 1e-6 else 'FAIL'}")
all_ok &= bs_err <= 1e-6

# ── 2. 路由复算（与着色器同规则：argmax 首次命中，选中后置 -FLT_MAX）──
scale = 1.0 / np.sqrt(DK)
score = (q @ bs_gpu.T) * scale / tau
sel_ref = np.zeros((T, TOPB), dtype=int)
for t in range(T):
    s = score[t].copy()
    for kk in range(min(TOPB, NB)):
        bi = int(np.argmax(s))
        sel_ref[t, kk] = bi
        s[bi] = -3.402823e38
sel_ok = np.array_equal(sel_ref, sel_gpu)
print(f"sel      {'PASS' if sel_ok else 'FAIL'}  ref={sel_ref.tolist()}")

mask_ref = np.zeros((T, N))
for t in range(T):
    for bi in sel_ref[t]:
        for idx in range(bi * BLOCK, min(bi * BLOCK + BLOCK, N)):
            if idx <= t:
                mask_ref[t, idx] = 1.0
    for idx in range(max(0, t + 1 - WINDOW), t + 1):
        mask_ref[t, idx] = 1.0
mask_ok = np.array_equal(mask_ref, masks)
n_bad = int(np.sum(mask_ref != masks))
print(f"mask     {'PASS' if mask_ok else 'FAIL'}  mismatch={n_bad}/{T * N}")
all_ok &= sel_ok and mask_ok

# ── 3. 前向 + autograd ──
qt = torch.tensor(q, dtype=torch.float64, requires_grad=True)
kt = torch.tensor(k, dtype=torch.float64, requires_grad=True)
vt = torch.tensor(v, dtype=torch.float64, requires_grad=True)
st = torch.tensor(float(sink), dtype=torch.float64, requires_grad=True)
mt = torch.tensor(masks.astype(bool))
dOt = torch.tensor(dO, dtype=torch.float64)

S = qt @ kt.T * scale / tau
S = S.masked_fill(~mt, float('-inf'))
logits = torch.cat([S, st.expand(T, 1)], dim=1)
P = torch.softmax(logits, dim=1)
out_ref = P[:, :N] @ vt
loss = (out_ref * dOt).sum()
gq, gk, gv, gs = torch.autograd.grad(loss, [qt, kt, vt, st])

all_ok &= report("out", out_gpu, out_ref.detach().numpy())
all_ok &= report("dq", dq_gpu, gq.numpy())
all_ok &= report("dK", dK_gpu, gk.numpy())
all_ok &= report("dV", dV_gpu, gv.numpy())
# dsink 是逐 query 的值（每次调用算该 token 的 sink 梯度），总量 = 求和
dA = v @ dO.T
dotAdA_t = (P[:, :N].detach().numpy() * dA.T).sum(1)
contrib = P[:, N].detach().numpy() * (0.0 - dotAdA_t)
all_ok &= report("dsink/t", dsink_gpu, contrib)
all_ok &= report("dsink", np.array([dsink_gpu.sum()]), np.array([float(gs)]))

print("=== M3 第 1/2 步: " + ("PASS" if all_ok else "FAIL") + " ===")
sys.exit(0 if all_ok else 1)
