"""P1 融合反向的 PyTorch 梯度对拍。

读取 bwd_gradcheck.exe 导出的二进制（输入 + Vulkan kernel 输出），
用 PyTorch autograd 复算同一组梯度并逐项比对。

用法：python 反向/gradcheck.py 反向/out/gradcheck.bin
"""

import struct
import sys
from pathlib import Path

import numpy as np
import torch


def 读取(path):
    数据 = {}
    with open(path, "rb") as f:
        (个数,) = struct.unpack("<I", f.read(4))
        for _ in range(个数):
            (名长,) = struct.unpack("<I", f.read(4))
            名 = f.read(名长).decode("utf-8")
            类型, 维数 = struct.unpack("<II", f.read(8))
            形状 = struct.unpack(f"<{维数}I", f.read(4 * 维数))
            元素数 = int(np.prod(形状))
            if 类型 == 0:
                数组 = np.frombuffer(f.read(4 * 元素数), dtype=np.float32).copy()
                数据[名] = torch.from_numpy(数组.reshape(形状))
            else:
                数组 = np.frombuffer(f.read(4 * 元素数), dtype=np.int32).copy()
                数据[名] = torch.from_numpy(数组.reshape(形状))
    return 数据


def 对比(名, 参考, 实测, 容差=2e-4):
    参考 = 参考.detach().to(torch.float64).reshape(-1)
    实测 = 实测.detach().to(torch.float64).reshape(-1)
    误差 = (参考 - 实测).abs().max().item()
    相对 = 误差 / max(参考.abs().max().item(), 1e-8)
    结论 = "PASS" if 误差 <= 容差 else "FAIL"
    print(f"  {名:<16} max_abs_err={误差:.3e}  rel={相对:.3e}  {结论}")
    return 结论 == "PASS"


def main():
    路径 = sys.argv[1] if len(sys.argv) > 1 else "反向/out/gradcheck.bin"
    d = 读取(路径)
    全部通过 = True

    # ── linear + bias ──
    x = d["linear.x"].to(torch.float64).requires_grad_(True)
    w = d["linear.w"].to(torch.float64).requires_grad_(True)
    b = torch.zeros(w.shape[0], dtype=torch.float64, requires_grad=True)
    dy = d["linear.dy"].to(torch.float64)
    y = x @ w.T + b
    dx_ref, dw_ref, db_ref = torch.autograd.grad(y, (x, w, b), grad_outputs=dy)
    print("[linear+bias]")
    全部通过 &= 对比("dx", dx_ref, d["linear.dx"])
    全部通过 &= 对比("dw", dw_ref, d["linear.dw"])
    全部通过 &= 对比("db", db_ref, d["linear.db"])

    # ── rmsnorm ──
    eps = float(d["rmsnorm.eps"][0])
    x = d["rmsnorm.x"].to(torch.float64).requires_grad_(True)
    w = d["rmsnorm.w"].to(torch.float64).requires_grad_(True)
    dy = d["rmsnorm.dy"].to(torch.float64)
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
    dx_ref, dw_ref = torch.autograd.grad(y, (x, w), grad_outputs=dy)
    print("[rms_norm]")
    全部通过 &= 对比("dx", dx_ref, d["rmsnorm.dx"])
    全部通过 &= 对比("dw", dw_ref, d["rmsnorm.dw"])

    # ── masked softmax ──
    a = d["softmax.a"].to(torch.float64)
    da = d["softmax.da"].to(torch.float64)
    掩码 = (a > 0).to(torch.float64)
    logits = torch.where(掩码 > 0, torch.log(a.clamp_min(1e-30)), torch.full_like(a, -1e30))
    logits.requires_grad_(True)
    a_ref = torch.softmax(logits, dim=-1) * 掩码
    a_ref = a_ref / a_ref.sum(-1, keepdim=True).clamp_min(1e-30)
    (ds_ref,) = torch.autograd.grad(a_ref, (logits,), grad_outputs=da)
    ds_ref = torch.where(掩码 > 0, ds_ref, torch.zeros_like(ds_ref))
    print("[masked_softmax]")
    全部通过 &= 对比("ds", ds_ref, d["softmax.ds"])

    # ── fused cross-entropy ──
    logits = d["ce.logits"].to(torch.float64).requires_grad_(True)
    targets = d["ce.targets"].long()
    loss = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    (dlogits_ref,) = torch.autograd.grad(loss.sum(), (logits,))
    print("[fused_CE]")
    全部通过 &= 对比("dlogits", dlogits_ref, d["ce.dlogits"])
    全部通过 &= 对比("loss", loss, d["ce.loss"], 容差=1e-5)

    # ── ERSA attention ──
    q = d["ersa.q"].to(torch.float64).requires_grad_(True)
    k = d["ersa.k"].to(torch.float64).requires_grad_(True)
    v = d["ersa.v"].to(torch.float64).requires_grad_(True)
    sink = d["ersa.sink"].to(torch.float64).requires_grad_(True)
    掩码 = d["ersa.mask"].to(torch.float64)
    dy = d["ersa.dy"].to(torch.float64)
    dk_dim = q.shape[-1]
    e = torch.einsum("bhk,thk->bht", q, k) / (dk_dim ** 0.5)
    e = e.masked_fill(掩码 < 0.5, -1e30)
    e = torch.cat([e, sink.unsqueeze(-1)], dim=-1)
    a = torch.softmax(e, dim=-1)
    y = torch.einsum("bht,thv->bhv", a[..., :-1], v)
    dq_ref, dk_ref, dv_ref, dsink_ref = torch.autograd.grad(y, (q, k, v, sink), grad_outputs=dy)
    print("[ERSA attention]")
    全部通过 &= 对比("dq", dq_ref, d["ersa.dq"])
    全部通过 &= 对比("dk", dk_ref, d["ersa.dk"])
    全部通过 &= 对比("dv", dv_ref, d["ersa.dv"])
    全部通过 &= 对比("dsink", dsink_ref, d["ersa.dsink"])

    # ── ASM 单步（读端 + 写端）──
    tau, tau_w, rho, lam, bnull = 1.0, 1.0, 0.9, 0.01, -2.0
    q = d["asm.q"].to(torch.float64).reshape(-1).requires_grad_(True)
    k = d["asm.k"].to(torch.float64).reshape(-1).requires_grad_(True)
    v = d["asm.v"].to(torch.float64).reshape(-1).requires_grad_(True)
    eta = d["asm.eta"].to(torch.float64).reshape(-1)[0].clone().requires_grad_(True)
    K = d["asm.K"].to(torch.float64).reshape(-1, d["asm.K"].shape[-1]).requires_grad_(True)
    V = d["asm.V"].to(torch.float64).reshape(-1, d["asm.V"].shape[-1]).requires_grad_(True)
    beta = d["asm.beta"].to(torch.float64).reshape(-1).requires_grad_(True)
    gamma = d["asm.gamma"].to(torch.float64).requires_grad_(True)
    dO = d["asm.dO"].to(torch.float64).reshape(-1)
    dKn = d["asm.dKn"].to(torch.float64).reshape(-1, d["asm.K"].shape[-1])
    dVn = d["asm.dVn"].to(torch.float64).reshape(-1, d["asm.V"].shape[-1])
    dBn = d["asm.dBn"].to(torch.float64).reshape(-1)
    idx = d["asm.idx"].reshape(-1).long()
    R = int(d["asm.idx"].shape[-1])

    s = (K[idx] @ q) / tau + beta[idx]
    a = torch.softmax(torch.cat([s, torch.tensor([bnull], dtype=torch.float64)]), dim=-1)
    o = a[:R] @ V[idx]
    p = torch.softmax((K[idx] @ k) / tau_w + beta[idx], dim=-1)
    u = gamma[idx].unsqueeze(-1) * K[idx] + eta * p.unsqueeze(-1) * k
    Kn = u / u.norm(dim=-1, keepdim=True)
    Vn = gamma[idx].unsqueeze(-1) * V[idx] + eta * p.unsqueeze(-1) * v
    Bn = rho * beta[idx] + lam * p
    Knew = K.index_copy(0, idx, Kn)
    Vnew = V.index_copy(0, idx, Vn)
    Bnew = beta.clone(); Bnew = Bnew.index_copy(0, idx, Bn)
    loss = (o * dO).sum() + (Knew * dKn).sum() + (Vnew * dVn).sum() + (Bnew * dBn).sum()
    (dq_r, dk_r, dv_r, deta_r, dK_r, dV_r, dB_r, dgamma_r) = torch.autograd.grad(
        loss, (q, k, v, eta, K, V, beta, gamma))
    print("[ASM step]")
    全部通过 &= 对比("dq", dq_r, d["asm.dq"])
    全部通过 &= 对比("dk", dk_r, d["asm.dk"])
    全部通过 &= 对比("dv", dv_r, d["asm.dv"])
    全部通过 &= 对比("deta", deta_r, d["asm.deta"].reshape(-1)[0])
    全部通过 &= 对比("dKold", dK_r, d["asm.dKo"].reshape_as(dK_r))
    全部通过 &= 对比("dVold", dV_r, d["asm.dVo"].reshape_as(dV_r))
    全部通过 &= 对比("dBold", dB_r, d["asm.dBo"].reshape(-1))
    全部通过 &= 对比("dgamma", dgamma_r, d["asm.dgamma"])

    # ── embedding scatter-add ──
    de = d["emb.dEmbed"].to(torch.float64)
    ids = d["emb.ids"].long().reshape(-1)
    ref = torch.zeros(d["emb.dTable"].shape, dtype=torch.float64)
    ref.index_add_(0, ids, de.reshape(ids.shape[0], -1))
    print("[embedding]")
    全部通过 &= 对比("dTable", ref, d["emb.dTable"])

    # ── MoE token 梯度（dScore + dX）──
    x = d["moe.x"].to(torch.float64).requires_grad_(True)
    gateW = d["moe.gateW"].to(torch.float64).requires_grad_(True)
    W1 = d["moe.W1"].to(torch.float64).requires_grad_(True)
    W2 = d["moe.W2"].to(torch.float64).requires_grad_(True)
    W1s = d["moe.W1s"].to(torch.float64).requires_grad_(True)
    Wup = d["moe.Wup"].to(torch.float64).requires_grad_(True)
    W2s = d["moe.W2s"].to(torch.float64).requires_grad_(True)
    dy = d["moe.dy"].to(torch.float64)
    idx = d["moe.idx"].long()
    kw = d["moe.kw"].to(torch.float64)
    N, D = x.shape
    K = idx.shape[1]
    scores = x @ gateW.T
    e_out = torch.zeros(N, D, dtype=torch.float64)
    g_list = []
    for k in range(K):
        e = idx[:, k]
        pre = torch.einsum("nd,ndf->nf", x, W1[e])
        h = torch.nn.functional.silu(pre)
        e_k = torch.einsum("nf,nfd->nd", h, W2[e])
        e_out = e_out + kw[:, k:k + 1] * e_k
        g_list.append((e_k * dy).sum(-1))
    s1 = torch.nn.functional.silu(x @ W1s)
    sh = s1 * (x @ Wup)
    out = e_out + sh @ W2s
    loss = (out * dy).sum()
    (dx_ref, dW1_ref, dW2_ref, dW1s_ref, dWup_ref, dW2s_ref) = torch.autograd.grad(
        loss, (x, W1, W2, W1s, Wup, W2s))
    # router 梯度参考：dtopk = kw*(g - Σ kw·g)，再 scatter 回选中专家
    g = torch.stack(g_list, dim=-1).detach()
    dotwg = (kw * g).sum(-1, keepdim=True)
    dtopk = kw * (g - dotwg)
    dscore_ref = torch.zeros(N, d["moe.gateW"].shape[0], dtype=torch.float64)
    dscore_ref.scatter_add_(1, idx, dtopk)
    # 参考补上 gate 路径：dX += dScore @ gateW（kernel 里有，这里漏了）
    dx_ref = dx_ref + dscore_ref @ gateW.detach()
    print("[MoE token]")
    全部通过 &= 对比("dx", dx_ref, d["moe.dx"])
    全部通过 &= 对比("dscore", dscore_ref, d["moe.dscore"])
    print("[MoE weights]")
    全部通过 &= 对比("dW1", dW1_ref, d["moe.dW1"])
    全部通过 &= 对比("dW2", dW2_ref, d["moe.dW2"])
    全部通过 &= 对比("dW1s", dW1s_ref, d["moe.dW1s"])
    全部通过 &= 对比("dWup", dWup_ref, d["moe.dWup"])
    全部通过 &= 对比("dW2s", dW2s_ref, d["moe.dW2s"])

    # 隔离：W1/W2=0 → 只剩 shared 路径
    x2 = d["moe.x"].to(torch.float64).requires_grad_(True)
    s1b = torch.nn.functional.silu(x2 @ W1s.detach())
    shb = s1b * (x2 @ Wup.detach())
    loss2 = ((shb @ W2s.detach()) * dy).sum()
    (dx_shared_ref,) = torch.autograd.grad(loss2, (x2,))
    全部通过 &= 对比("dx(shared只)", dx_shared_ref, d["moe.dx_shared_only"])

    print("\n总判定:", "全部 PASS" if 全部通过 else "存在 FAIL")
    return 0 if 全部通过 else 1


if __name__ == "__main__":
    raise SystemExit(main())
