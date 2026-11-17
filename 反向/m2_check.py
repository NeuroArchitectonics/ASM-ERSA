"""M2：单层 ASM 的 BPTT 对照（Vulkan 串联反向 vs PyTorch autograd）。"""

import struct
import sys

import numpy as np
import torch

TAU, TAU_W, RHO, LAM, BNULL = 1.0, 1.0, 0.9, 0.01, -2.0


def 对比(名, ref, got, 容差=2e-4):
    ref = ref.detach().to(torch.float64).reshape(-1)
    got = got.detach().to(torch.float64).reshape(-1)
    err = (ref - got).abs().max().item()
    print(f"  {名:<10} max_abs_err={err:.3e}  {'PASS' if err <= 容差 else 'FAIL'}")
    return err <= 容差


def main():
    with open(sys.argv[1], "rb") as f:
        T, B, H, M, R, DK, DV = struct.unpack("<7I", f.read(28))

        def 读(n):
            return torch.from_numpy(np.frombuffer(f.read(4 * n), dtype=np.float32).copy())

        K = 读(M * DK).reshape(M, DK).double().requires_grad_(True)
        V = 读(M * DV).reshape(M, DV).double().requires_grad_(True)
        beta = 读(M).double().requires_grad_(True)
        gamma = 读(M).double()
        qs, ks, vs, dOs = [], [], [], []
        for _ in range(T):
            qs.append(读(DK).double().requires_grad_(True))
            ks.append(读(DK).double().requires_grad_(True))
            vs.append(读(DV).double().requires_grad_(True))
            dOs.append(读(DV).double())
        etas = 读(T).double()
        dq_vk, dk_vk, dv_vk = [], [], []
        for _ in range(T):
            dq_vk.append(读(DK))
            dk_vk.append(读(DK))
            dv_vk.append(读(DV))
        dKo_vk = 读(M * DK).reshape(M, DK)
        dVo_vk = 读(M * DV).reshape(M, DV)
        dBo_vk = 读(M)
        out_vk, Ka_vk, Va_vk, Ba_vk = [], [], [], []
        for _ in range(T):
            out_vk.append(读(DV))
            Ka_vk.append(读(M * DK).reshape(M, DK))
            Va_vk.append(读(M * DV).reshape(M, DV))
            Ba_vk.append(读(M))
        dbg0 = 读(3 * R)

    # ── t=0 写路径诊断：从 V_after 反解内核实际用的 p ──
    s0 = qs[0].detach() @ K.T / TAU + beta
    idx0 = torch.topk(s0, R).indices.tolist()
    k0, v0, e0 = ks[0].detach(), vs[0].detach(), etas[0].item()
    p_ref = torch.softmax(K[idx0] @ k0 / TAU_W + beta[idx0], -1)
    print("[t=0 write diag]")
    s_host = (K[idx0] @ k0 / TAU_W + beta[idx0])
    print("  kernel sel_q =", [int(x) for x in dbg0[:R].tolist()])
    print("  kernel s_kw  =", [round(x, 5) for x in dbg0[R:2 * R].tolist()])
    print("  host   s_kw  =", [round(x, 5) for x in s_host.tolist()])
    print("  kernel p     =", [round(x, 4) for x in dbg0[2 * R:].tolist()])
    print("  host   p_ref =", [round(x, 4) for x in p_ref.tolist()])
    print("  host   K0[0] =", [round(x, 5) for x in K[0][:R].tolist()])
    print("  host   idx   =", idx0)
    print("  idx =", idx0)
    print("  p_ref =", [round(x, 4) for x in p_ref.tolist()])
    for j, m in enumerate(idx0):
        comps = [((Va_vk[0][m][i].item() - gamma[m].item() * V[m][i].item()) / (e0 * v0[i].item())
                  if abs(v0[i].item()) > 1e-6 else float('nan')) for i in range(DV)]
        pb = (Ba_vk[0][m].item() - RHO * beta[m].item()) / LAM if LAM > 0 else float('nan')
        gam = [((Va_vk[0][m][i].item() - e0 * p_ref[j].item() * v0[i].item()) / V[m][i].item()
                if abs(V[m][i].item()) > 1e-6 else float('nan')) for i in range(DV)]
        uref = gamma[m] * K[m] + e0 * p_ref[j] * k0
        knref = uref / uref.norm()
        print(f"  slot {m}: p_from_beta={pb:.4f}  p_ref={p_ref[j].item():.4f}")
        print(f"        gamma_implied={[round(c, 4) for c in gam]}  gamma_ref={gamma[m].item():.4f}")
        print(f"        K_err={ (knref - Ka_vk[0][m]).abs().max().item():.3e}")

    # ── 前向逐步对照（先定位语义分叉）──
    Kc, Vc, Bc = K.detach().clone(), V.detach().clone(), beta.detach().clone()
    print(f"T={T} M={M} R={R} DK={DK} DV={DV}")
    print("[forward step-by-step]")
    first_bad = None
    for t in range(T):
        s = qs[t].detach() @ Kc.T / TAU + Bc
        idx = torch.topk(s, R).indices
        a = torch.softmax(torch.cat([s[idx], torch.tensor([BNULL], dtype=torch.float64)]), -1)
        o = a[:R] @ Vc[idx]
        p = torch.softmax(Kc[idx] @ ks[t].detach() / TAU_W + Bc[idx], -1)
        u = gamma[idx].unsqueeze(-1) * Kc[idx] + etas[t] * p.unsqueeze(-1) * ks[t].detach()
        Kn = u / u.norm(dim=-1, keepdim=True)
        Vn = gamma[idx].unsqueeze(-1) * Vc[idx] + etas[t] * p.unsqueeze(-1) * vs[t].detach()
        Bn = RHO * Bc[idx] + LAM * p
        Kc = Kc.index_copy(0, idx, Kn)
        Vc = Vc.index_copy(0, idx, Vn)
        Bc = Bc.index_copy(0, idx, Bn)
        eo = (o - out_vk[t]).abs().max().item()
        ek = (Kc - Ka_vk[t]).abs().max().item()
        ev = (Vc - Va_vk[t]).abs().max().item()
        eb = (Bc - Ba_vk[t]).abs().max().item()
        print(f"  t={t}  out={eo:.3e}  K={ek:.3e}  V={ev:.3e}  beta={eb:.3e}")
        if first_bad is None and max(eo, ek, ev, eb) > 1e-5:
            first_bad = t
    print("首个发散 step:", first_bad if first_bad is not None else "无（前向一致）")

    loss = torch.zeros(())

    loss = torch.zeros(())
    outs = []
    K_leaf, V_leaf, B_leaf = K, V, beta   # 初始叶子（循环内会重新赋值 K/V/beta）
    Kr, Vr, Br = K_leaf, V_leaf, B_leaf
    for t in range(T):
        s = qs[t] @ Kr.T / TAU + Br
        idx = torch.topk(s, R).indices
        a = torch.softmax(torch.cat([s[idx], torch.tensor([BNULL], dtype=torch.float64)]), -1)
        o = a[:R] @ Vr[idx]
        outs.append(o)
        p = torch.softmax(Kr[idx] @ ks[t] / TAU_W + Br[idx], -1)
        u = gamma[idx].unsqueeze(-1) * Kr[idx] + etas[t] * p.unsqueeze(-1) * ks[t]
        Kn = u / u.norm(dim=-1, keepdim=True)
        Vn = gamma[idx].unsqueeze(-1) * Vr[idx] + etas[t] * p.unsqueeze(-1) * vs[t]
        Bn = RHO * Br[idx] + LAM * p
        Kr = Kr.index_copy(0, idx, Kn)
        Vr = Vr.index_copy(0, idx, Vn)
        Br = Br.index_copy(0, idx, Bn)
        loss = loss + (o * dOs[t]).sum()

    grads = torch.autograd.grad(loss, qs + ks + vs + [K_leaf, V_leaf, B_leaf], allow_unused=True)
    grads = [g if g is not None else torch.zeros_like(p)
             for g, p in zip(grads, qs + ks + vs + [K_leaf, V_leaf, B_leaf])]
    gq, gk, gv = grads[:T], grads[T:2 * T], grads[2 * T:3 * T]
    gK, gV, gB = grads[-3], grads[-2], grads[-1]

    ok = True
    print(f"T={T} M={M} R={R} DK={DK} DV={DV}")
    print("gk[%d] ref =" % (T - 1), [round(x, 6) for x in gk[T - 1].tolist()])
    print("dk_vk[%d]  =" % (T - 1), [round(x, 6) for x in dk_vk[T - 1].tolist()])
    for t in range(T):
        ok &= 对比(f"dq[{t}]", gq[t], dq_vk[t])
    for t in range(T):
        ok &= 对比(f"dk[{t}]", gk[t], dk_vk[t])
    for t in range(T):
        ok &= 对比(f"dv[{t}]", gv[t], dv_vk[t])
    ok &= 对比("dK0", gK, dKo_vk)
    ok &= 对比("dV0", gV, dVo_vk)
    ok &= 对比("dβ0", gB, dBo_vk)
    print("\n总判定:", "全部 PASS" if ok else "存在 FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
