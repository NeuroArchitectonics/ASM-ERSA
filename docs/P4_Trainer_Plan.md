# P4：Vulkan 训练器规划（2026-11-16）

> 目标：在 780M 上用 Vulkan 张量系统训练 tiny 模型，复现 CPU 版 loss 曲线（先小图，再扩到 ASM/ERSA）。
> 原则：**固定图硬编码反向顺序**，不做通用 autograd；前向/反向全部走已对拍的算子。

## 0. 已完成（P1–P3）

- 反向：linear+bias、rmsnorm、masked softmax、fused CE、ERSA attention、ASM 单步、embedding、MoE（token+权重）
- 优化器：`adamw_update`（融合 m/v + 偏差修正 + 解耦衰减 + 梯度清零）
- 对拍：29 项梯度检查全 PASS（PyTorch float64 autograd 为基准）

## 1. 训练器最小图（M0）

```
ids ──embedding_fwd──► h0 ──linear_fwd(W1,b1)──► h1 ──rmsnorm_fwd(g)──► h2
    ──linear_fwd(W2,b2)──► logits ──fused_ce_bwd──► loss, dlogits
反向（硬编码顺序）：
  dlogits → linear_bwd_dx/dwdb(W2,b2) → rmsnorm_bwd_dx/dw(g) → linear_bwd_dx/dwdb(W1,b1)
          → embedding_bwd → 各参数梯度 → adamw_update
```

- 前向还需新增 3 个算子：`embedding_fwd`（gather）、`linear_fwd`（xWᵀ+b）、`rmsnorm_fwd`（含输出，供反向重算）；
- 反向与优化器全部复用已对拍算子；
- 损失评估：`fused_ce_bwd` 同一次 dispatch 输出 loss，无需额外算子。

## 2. 里程碑

| 阶段 | 内容 | 验收 |
|---|---|---|
| M0 | 上述最小图 + AdamW，1 步训练 | ✅ loss 有限、60 步 4.402→0.054 |
| M1 | 训练 200 步，与 PyTorch CPU 同图对照 | ✅ 首步差 4.77e-07，逐步最大差 **1.669e-06**，末步 0.0075/0.0075 |
| M2 | 接入 ASM 层（单步反向 + BPTT 状态串联） | ✅ T=8：dq/dk/dv 逐步 ≤1.1e-8，dK0 2.26e-7 / dV0 7.06e-8 / dβ0 3.42e-8 |
| M3 | 接入 ERSA 层 + 多模态（冻结音频/视觉编码器） | ✅ 三件套全 PASS（见 §2.1） |
| M4 | 性能：融合 dispatch、chunk 扫描 | ✅ 首片完成：融合 5.2–8.7×，批式反传内核已验证（见 §2.2）；并行批式 / 缓存读仍是下一步 |

> M1 说明：C++ 侧 loss 报逐行均值、梯度按 sum（与 `fused_ce_bwd` 一致）；Python 参考必须用 `reduction='sum'` 再除以 N，否则梯度差 N 倍。

### 2.1 M3 明细（2026-11-16）

口径说明：CPU 版 `asm_ersa.ERSA` 是"参考版"语义（q 归一化、JL sketch 路由、无因果、out_norm），与 `ersa_attn_fused` 内核（mean-K 块摘要、`q·k·scale/tau`、逐 query 因果窗口、sink）不是同一套；M3 对拍按**内核精确语义**在 PyTorch float64 复算。

| 子步 | 内容 | 结果 |
|---|---|---|
| M3a | `train_m3.cpp`：`ersa_block_sum` + `ersa_attn_fused`（T=8 全 cache，含"选中块在 query 之后被因果过滤"）+ 逐 query `ersa_attn_bwd`（tau 折进 q、dq 再除回） | out/dq/dK/dV/dsink ≤ **7.5e-08** |
| M3b | `train_m3b.cpp`：KV cache 逐 token 增长（T=8），块摘要每步 GPU 全量重算 vs 主机增量式；逐 token 反传，dK/dV 在 cache 槽位跨步累加 | 块摘要 **0 err**；out/dq/dK/dV/dsink ≤ **8.1e-08** |
| M3c | `train_m3c.cpp`：冻结多模态输入（音频2+视觉2+文本2 token）+ Wq/Wk/Wv 投影(`linear_fwd`) + ERSA + MSE + `linear_bwd_dwdb` + `adamw_update` 内核 20 步 | loss 0.394→0.321，曲线差 **1.8e-07**，最终参数差 **4.2e-06** |

M3b 关键坑：逐 token 前向必须按**前缀长度**派发（`grid.x = Nt`）并取第 t 行输出；`grid.x=1` 时内核里 `t` 恒为 0，窗口/因果全按第 0 个 token 算——此时梯度（按主机掩码）仍与参考一致、前向输出却错，容易误判。
M3 未覆盖：`SemanticModalityEmbedding`、ERSA-Omni 的模态感知掩码（前缀内同模态双向）、ModalityMoE 尚未进 Vulkan 训练器；内核只支持标准因果。

### 2.2 M4 明细（2026-11-17，AMD Radeon 780M）

做法（只改调度，不改数学）：

- 训练环从「每算子一次提交+一次围栏等待」改为「每条算子串一次提交+一次围栏」。
  M3c 的 T=6 图从 **18 次同步点降到 3 次**（前向 5 op / 反传 / 更新 7 op 各一条串）；
  T=64 从 76 次降到 3 次。
- 逐 query 反传按块录制：`融合逐q(块=N)` 把 N 个 query 的反传录进同一条命令缓冲；
  这些 query 互不读写对方输出，串内可省掉逐步屏障（实测省屏障后 T=128 从 113 ms 降到 68 ms）。
- 新增设备侧批式反传内核 `ersa_attn_bwd_batch`：一次 dispatch 扫完全部 query，
  dK/dV/dsink 跨 query 累加，**不再回读 T 份逐 query 梯度**。
- 新增 `反向/train_m4.cpp`：同图跑「串行 / 融合逐q / 融合批式」三条路径，内建
  dispatch 开销、映射内存带宽、分段耗时与一致性检查。

实测（T=8/64/128，DX=DK=DV=32，BLOCK=8，TOPB=2，WINDOW=8）：

| 尺寸 | 串行 ms/步 | 融合逐q（块=T） | 融合批式 | 批式 token/s | 批式加速 |
|---|---|---|---|---|---|
| T=8 | 26.8 | 5.40 | **5.20** | 1540 | 5.2× |
| T=64 | 149.7 | 23.8 | **20.3** | 3146 | 7.4× |
| T=128 | 359.0 | 68.5 | **41.1** | 3117 | 8.7× |

关键测量（同一台 780M，iGPU 计时波动大，取 40/20/10 步中位）：

- **单次「提交+围栏等待」≈ 0.59–0.87 ms**，串行路径的成本基本就是同步点数 × 这个值。
- **映射内存读 67 MB/s、写 26 GB/s**：缓冲都要求 HOST_VISIBLE|COHERENT，780M 上落在非缓存类型，
  主机读极慢。T=64 时逐 query 回读 1 MB 就要 **15.5 ms**，是融合逐q 路径的最大单项。
- 反传分段（T=64）：准备 0.12 + 提交 3.3 + 回收 15.5 ms；批式把回收压到 0，但内核自身 17.5 ms。
- **链路本身有 ~1e-7 的 fp32 非确定性**：串行路径跑两遍，T=8/64 逐位相同，T=128 出现 8.9e-8 差异；
  因此路径间对拍按 1e-6 判据，不再宣称「逐位一致」。

正确性：

- T=6、20 步：串行 vs 融合逐q loss/参数差 0；串行 vs 融合批式 loss 1.49e-7、参数 6.1e-6。
- 批式路径导出给 `反向/m3c_check.py`（PyTorch float64）：loss 2.9e-7、参数 ≤6.4e-6，**PASS**。
- `ersa_attn_bwd_batch` 的 tau 口径：逐 query 内核由主机把 q 预除 tau，批式内核直接吃原始 q，
  因此把 1/tau 折进内核 scale（同一推送常量 24 B 的最后一格）。这一处漏掉时参数差 3e-2，是本次的坑。

M4 未完成：

- 批式内核仍是「1 个工作组、1 个线程扫全部 query」的 O(T²) 写法：T=128 时 39 ms。
  下一版应改成「每 query 一个工作组 + 按 query 分块的 dK/dV 暂存 + 归约内核」。
- 主机读慢的根因（内存类型）没有治：可给「读多」张量单独申请 HOST_CACHED 类型，或加 staging 回读路径。
- 前向算子串仍是 5 次 dispatch（linear×3 + block_sum + attn），可再融合；`ersa_block_sum` 与
  `ersa_attn_fused` 的块摘要可并进 attn 内核，省一次 dispatch 和一份中间量。


## 3. 关键约束

- 状态梯度串联：训练器负责把上一步 `dKnew/dVnew/dβnew` 与读端梯度相加后传给上一步（内核只做单步）；
- 先支持 B=1；多 batch 需要 float 原子加或按 batch 归约；
- 所有中间激活可重算的不存（pre/h 等），显存优先；
- 数据管线先在 CPU 准备 ids，再上传。

## 4. 风险

- 前向算子（linear/rmsnorm/embedding）需要先与 PyTorch 对拍，再进训练器；
- 逐 token 串行 + 每算子一次 dispatch，小图可行，扩展前必须做算子串（`录制算子串`）与融合；
- 780M 共享内存，M2 起要注意激活显存。
