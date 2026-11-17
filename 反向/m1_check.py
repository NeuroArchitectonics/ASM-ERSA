"""M1：用 PyTorch 复算 M0 同图，与 Vulkan 训练器的逐步 loss 对照。

用法：python 反向/m1_check.py <export.bin> [期望步数]
export.bin 由 train_m0.exe 生成：6 个参数张量(fp32) + 步数(u32) + loss[steps](fp32)
"""

import struct
import sys

import numpy as np
import torch

V, D, HX, N = 64, 32, 32, 16
LR, B1, B2, EPS, WD = 3e-3, 0.9, 0.999, 1e-8, 0.01


def main():
    path = sys.argv[1]
    with open(path, "rb") as f:
        def 读(t):
            buf = np.frombuffer(f.read(4 * t.numel()), dtype=np.float32).copy()
            return torch.from_numpy(buf).reshape(t.shape)

        table = 读(torch.empty(V, D)); W1 = 读(torch.empty(HX, D)); b1 = 读(torch.empty(HX))
        g = 读(torch.empty(HX)); W2 = 读(torch.empty(V, HX)); b2 = 读(torch.empty(V))
        (steps,) = struct.unpack("<I", f.read(4))
        ref = np.frombuffer(f.read(4 * steps), dtype=np.float32).copy()

    ids = torch.tensor([(i * 7 + 3) % V for i in range(N)], dtype=torch.long)
    targets = (ids + 1) % V

    参数 = {k: v.clone().requires_grad_(True) for k, v in
            dict(table=table, W1=W1, b1=b1, g=g, W2=W2, b2=b2).items()}
    矩 = {k: torch.zeros_like(v) for k, v in 参数.items()}
    二矩 = {k: torch.zeros_like(v) for k, v in 参数.items()}

    损失 = []
    for step in range(steps):
        h0 = 参数["table"][ids]
        h1 = h0 @ 参数["W1"].T + 参数["b1"]
        h2 = h1 * torch.rsqrt(h1.pow(2).mean(-1, keepdim=True) + 1e-6) * 参数["g"]
        logits = h2 @ 参数["W2"].T + 参数["b2"]
        loss = torch.nn.functional.cross_entropy(logits, targets, reduction="sum")
        损失.append(loss.item() / N)   # C++ 报的是逐行均值；梯度按 sum（与 shader 一致）

        grads = torch.autograd.grad(loss, list(参数.values()))
        b1t, b2t = B1 ** (step + 1), B2 ** (step + 1)
        with torch.no_grad():
            for (名, p), gr in zip(参数.items(), grads):
                m, v = 矩[名], 二矩[名]
                m.mul_(B1).add_(gr * (1 - B1))
                v.mul_(B2).add_(gr * gr * (1 - B2))
                m_hat = m / (1 - b1t)
                v_hat = v / (1 - b2t)
                p.sub_(LR * m_hat / (v_hat.sqrt() + EPS))
                p.sub_(LR * WD * p)   # 解耦权重衰减（shader 顺序：先 Adam 更新，再衰减）

    损失 = np.array(损失, dtype=np.float32)
    差 = np.abs(损失 - ref)
    print(f"步数 {steps} | 首步 loss: vk={ref[0]:.6f} torch={损失[0]:.6f} 差={差[0]:.2e}")
    print(f"逐步最大差: {差.max():.3e} | 末步: vk={ref[-1]:.6f} torch={损失[-1]:.6f} 差={差[-1]:.2e}")
    print(f"loss 曲线: " + " ".join(f"{ref[i]:.3f}/{损失[i]:.3f}" for i in range(0, steps, max(1, steps // 5))))
    阈值 = 5e-3
    ok = 差.max() <= 阈值
    print("判定:", f"PASS（≤{阈值}）" if ok else f"FAIL（>{阈值}）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
