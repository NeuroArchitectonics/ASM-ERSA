# M3c 对拍：冻结多模态输入下的 ERSA 块复合训练环（投影+ERSA+AdamW，20 步）
# 参考：float64 PyTorch，路由规则与 ersa_attn_fused 完全一致，AdamW 与 adamw_update 同序
# 用法：python 反向/m3c_check.py <导出.bin>
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

T, DX, DK, DV, BLOCK, NB, TOPB, WINDOW, STEPS = [u() for _ in range(9)]
tau, sink0, lr, wd, beta1, beta2, eps = f(7)
x = f(T * DX).reshape(T, DX)
target = f(T * DV).reshape(T, DV)
Wq0 = f(DK * DX).reshape(DK, DX)
Wk0 = f(DK * DX).reshape(DK, DX)
Wv0 = f(DV * DX).reshape(DV, DX)
losses_vk = f(STEPS)
WqF = f(DK * DX).reshape(DK, DX)
WkF = f(DK * DX).reshape(DK, DX)
WvF = f(DV * DX).reshape(DV, DX)
sinkF = f(1)
assert pos == len(raw), "导出文件长度与读序不符"

print(f"T={T} DX={DX} DK={DK} DV={DV} BLOCK={BLOCK} NB={NB} TOPB={TOPB} WINDOW={WINDOW} "
      f"STEPS={STEPS} tau={tau} lr={lr} wd={wd}")

def report(name, got, ref, tol=2e-4):
    err = float(np.max(np.abs(got - ref)))
    rel = err / max(1e-12, float(np.max(np.abs(ref))))
    ok = err <= tol or rel <= tol
    print(f"{name:10s} max|err|={err:.3e} rel={rel:.3e} {'PASS' if ok else 'FAIL'}")
    return ok

all_ok = True
scale = 1.0 / np.sqrt(DK)
xt = torch.tensor(x)          # 冻结：不建图
tt = torch.tensor(target)
Wq = torch.tensor(Wq0)
Wk = torch.tensor(Wk0)
Wv = torch.tensor(Wv0)
sink = torch.tensor(float(sink0))
mWq = torch.zeros_like(Wq); vWq = torch.zeros_like(Wq)
mWk = torch.zeros_like(Wk); vWk = torch.zeros_like(Wk)
mWv = torch.zeros_like(Wv); vWv = torch.zeros_like(Wv)
mS = torch.zeros(1); vS = torch.zeros(1)
losses = []

def adamw_step(p, g, m, v, t, decay):
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g * g
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    p = p - lr * m_hat / (torch.sqrt(v_hat) + eps)
    if decay > 0:
        p = p - lr * decay * p
    return p, m, v

for s in range(1, STEPS + 1):
    Wq_ = Wq.clone().requires_grad_()
    Wk_ = Wk.clone().requires_grad_()
    Wv_ = Wv.clone().requires_grad_()
    sink_ = sink.clone().requires_grad_()

    q = xt @ Wq_.T
    k = xt @ Wk_.T
    v = xt @ Wv_.T
    bs = k.reshape(NB, BLOCK, DK).mean(dim=1)          # [NB, DK]

    qd = q.detach().numpy()
    bsd = bs.detach().numpy()
    S = q @ k.T * scale / tau
    mask = torch.zeros(T, T, dtype=torch.bool)
    for t in range(T):
        sc = (qd[t] @ bsd.T) * scale / tau
        ref = []
        for _ in range(min(TOPB, NB)):
            bi = int(np.argmax(sc))
            ref.append(bi)
            sc[bi] = -3.402823e38
        for bi in ref:
            for idx in range(bi * BLOCK, min(bi * BLOCK + BLOCK, T)):
                if idx <= t:
                    mask[t, idx] = True
        for idx in range(max(0, t + 1 - WINDOW), t + 1):
            mask[t, idx] = True

    S = S.masked_fill(~mask, float('-inf'))
    logits = torch.cat([S, sink_.expand(T, 1)], dim=1)
    P = torch.softmax(logits, dim=1)
    out = P[:, :T] @ v
    loss = ((out - tt) ** 2).mean()

    gWq, gWk, gWv, gs = torch.autograd.grad(loss, [Wq_, Wk_, Wv_, sink_])
    losses.append(float(loss))

    Wq, mWq, vWq = adamw_step(Wq, gWq, mWq, vWq, s, wd)
    Wk, mWk, vWk = adamw_step(Wk, gWk, mWk, vWk, s, wd)
    Wv, mWv, vWv = adamw_step(Wv, gWv, mWv, vWv, s, wd)
    sink, mS, vS = adamw_step(sink, gs.reshape(1), mS, vS, s, 0.0)

losses = np.array(losses)
all_ok &= report("loss", losses_vk, losses, tol=1e-4)
print("  loss 轨迹:", " ".join(f"{v:.6f}" for v in losses_vk[::4]), "...")
all_ok &= report("Wq", WqF, Wq.numpy(), tol=1e-3)
all_ok &= report("Wk", WkF, Wk.numpy(), tol=1e-3)
all_ok &= report("Wv", WvF, Wv.numpy(), tol=1e-3)
all_ok &= report("sink", sinkF, sink.numpy(), tol=1e-3)

print("=== M3c: " + ("PASS" if all_ok else "FAIL") + " ===")
sys.exit(0 if all_ok else 1)
