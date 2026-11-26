# P1 融合反向（Vulkan）+ PyTorch 梯度对拍

> 2026-11-16：linear+bias / rms_norm / masked softmax / fused CE 四个反向的第一批实现。
> 定位：**正确性优先**的朴素实现（一线程一行/一元素、行内串行循环），性能优化另开。

## 文件

| 文件 | 说明 |
|---|---|
| `spv/linear_bwd_dx.spv` | `dX = dY @ W` |
| `spv/linear_bwd_dwdb.spv` | `dW = dYᵀ @ X`，`k==0` 线程顺带算 `db = Σ dY`（融合 bias） |
| `spv/rmsnorm_bwd_dx.spv` | 行内重算 `inv_rms`，不存归一化中间量 |
| `spv/rmsnorm_bwd_dw.spv` | `dW[d] = Σ_r dy·x·inv_rms` |
| `spv/masked_softmax_bwd.spv` | `dS = a·(dA − Σ dA·a)`（掩码位置 a=0，结果自然为 0） |
| `spv/fused_ce_bwd.spv` | 一次算完 `max / logsumexp / loss / dlogits = softmax − onehot` |
| `spv/ersa_attn_bwd.spv` | ERSA 注意力反向：`dq/dK/dV/dsink`（候选掩码 + sink 融合；B=1） |
| `spv/ersa_attn_bwd_batch.spv` | ERSA 注意力反向（批式）：一次 dispatch 扫全部 query，dK/dV/dsink 跨 query 累加 |
| `spv/asm_step_bwd.spv` | ASM 单步反向：读端 + 写端 + key 归一化切空间投影（B=1,H=1；22 个绑定） |
| `spv/embedding_bwd.spv` | embedding scatter-add：`dTable[v]=Σ dEmbed[n]`（线程 per 词表项，无原子） |
| `spv/moe_bwd_dx.spv` | MoE token 梯度：dScore（router）+ dX（专家 + shared + gate） |
| `spv/moe_bwd_dw_expert.spv` | dW1 / dW2（线程 per 元素遍历 token，无原子） |
| `spv/moe_bwd_dw_shared.spv` | dW1s / dWup / dW2s（同一套路） |
| `bwd_gradcheck.cpp` | 生成对拍数据：跑 kernel → 写 `输入+输出` 二进制 |
| `gradcheck.py` | 读二进制 → PyTorch autograd 复算 → 逐张量比对 |

## 构建与运行

> 着色器源码（`.comp`）不随仓库发布；`spv/` 只带编译产物（`.spv` + 登记元信息的 `.spv.hpp`）。
> 需要改内核时，在私有设计档案环境里重编，再把新产物同步回 `spv/`。

```bat
:: 1) 编译对拍程序（需 Vulkan SDK；spv/ 提供 .spv.hpp，ASM-ERSA-Design\vk 提供调度器头）
cl /std:c++20 /EHsc /utf-8 /O2 bwd_gradcheck.cpp /I spv /I ASM-ERSA-Design\vk /I %VULKAN_SDK%\Include ^
   /link vulkan-1.lib

:: 2) 在 spv 的上一级运行（spv 用相对路径），然后对拍
bwd_gradcheck.exe spv C:\temp\gradcheck.bin
python 反向\gradcheck.py C:\temp\gradcheck.bin
```

> 注意：每个算子必须在翻译单元里 `#include "<算子>.spv.hpp"` 才会登记元信息，否则调度器会以"没有 .spv.hpp 登记元信息"跳过。

## 对拍结果（2026-11-16，AMD Radeon 780M）

| 算子 | 张量 | max abs err | 相对误差 | 判定 |
|---|---|---|---|---|
| linear+bias | dx | 5.573e-07 | 1.45e-07 | PASS |
| | dw | 1.995e-07 | 6.99e-08 | PASS |
| | db | 1.192e-07 | 3.87e-08 | PASS |
| rms_norm | dx | 2.513e-07 | 1.61e-07 | PASS |
| | dw | 2.861e-07 | 9.03e-08 | PASS |
| masked softmax | ds | 2.099e-08 | 1.72e-07 | PASS |
| fused CE | dlogits | 2.924e-08 | 2.93e-08 | PASS |
| | loss | 3.938e-07 | 6.27e-08 | PASS |
| ERSA attention | dq | 7.360e-09 | 9.52e-08 | PASS |
| | dK | 5.403e-09 | 1.06e-07 | PASS |
| | dV | 1.850e-08 | 1.46e-07 | PASS |
| | dsink | 2.611e-09 | 2.91e-07 | PASS |
| ASM 单步 | dq | 3.599e-09 | 2.60e-07 | PASS |
| | dk | 9.842e-09 | 1.99e-07 | PASS |
| | dv | 7.051e-09 | 3.31e-07 | PASS |
| | deta | 6.134e-08 | 3.76e-07 | PASS |
| | dKold | 5.973e-08 | 6.14e-08 | PASS |
| | dVold | 4.085e-08 | 4.57e-08 | PASS |
| | dBold | 3.763e-09 | 4.19e-08 | PASS |
| | dgamma | 6.889e-08 | 5.23e-08 | PASS |
| embedding | dTable | 0.000e+00 | 0.00e+00 | PASS |
| MoE token | dx | 1.321e-07 | 7.80e-08 | PASS |
| | dscore | 5.425e-08 | 1.47e-07 | PASS |
| | dx（shared 隔离） | 2.704e-07 | 2.15e-07 | PASS |
| MoE 权重 | dW1 | 6.654e-08 | 9.86e-08 | PASS |
| | dW2 | 7.094e-08 | 9.34e-08 | PASS |
| | dW1s | 1.004e-07 | 1.28e-07 | PASS |
| | dWup | 4.584e-08 | 3.66e-08 | PASS |
| | dW2s | 4.887e-08 | 7.00e-08 | PASS |

容差：梯度 2e-4、loss 1e-5（fp32）；参考值用 float64 autograd。

## 测试规模

`N=8, K=16, M=24`（linear）；`R=8, D=32`（rmsnorm）；`R=8, C=16`（softmax）；`R=8, V=32`（CE）。

## 局限

- 朴素实现：`rmsnorm_bwd_dw` 每个线程重复算整行平方和，`linear` 三重循环；大模型需要向量化/分块/shared memory 版本。
- 只支持 fp32；bf16 需先扩展内核。
- **ERSA 反向目前只支持 B=1**：dK 按 `[T,H,DK]` 写回，多 batch 需要 float 原子加或按 batch 归约；块路由（候选选择）不可微，梯度不回流 router，这是设计有意为之。
- **ASM 反向只支持 B=1、H=1**（写 dK/dgamma 需原子加），且 `R<=16, M<=64, DK,DV<=64`；top-r 索引直通（不可微）。BPTT 需要训练器把上一步的 `dKnew/dVnew/dβnew` 与本步读端梯度累加后传回上一步，内核只负责单步。
- **改完着色器必须重编对拍程序**：`.spv` 指纹与 exe 内嵌元信息不一致时调度器会拒绝加载（本次踩过一次，报"产物不同源"）。
- **MoE 已完整**：token 梯度（dScore/dX）+ 权重梯度（dW1/dW2/dW1s/dWup/dW2s），均为无原子写法（线程 per 目标元素遍历 token）。
- MoE dx 的参考必须包含 gate 路径 `dScore @ gateW`；本次先漏后补，说明"参考漏项"也会伪装成 kernel bug。
- 验证规模小（8×32 级），未做数值稳定性压力测试（超大值、长累加）。
- 未做性能基准；当前只证明"数学与 PyTorch autograd 一致"。

## 下一步（M4 之后）

1. **并行批式反传**：现在 `ersa_attn_bwd_batch` 是单工作组单线程扫全部 query（T=128 反传 39 ms）；
   改成「每 query 一个工作组 + 按 query 暂存 dK/dV + 归约内核」，目标把 T=128 反传压到 5 ms 内。
2. **治映射内存读**：给「读多」张量申请 HOST_CACHED 类型（或 staging 回读），把 67 MB/s 的读拉到 GB/s 级；
   这同时会改善 loss/路由回读。
3. **前向再融合**：`ersa_block_sum` 并进 `ersa_attn_fused`，省一次 dispatch 和一份块摘要中间量；
   `linear_fwd` 三个投影可合成一个 kernel（同一份 X 读三遍）。
4. **覆盖缺口**：`SemanticModalityEmbedding`、NSA 模态感知掩码、ModalityMoE 进训练器。

## P4 训练器对拍（M0–M3）

| 子步 | 文件 | 验收 |
|---|---|---|
| M0/M1 | `train_m0.cpp` + `m1_check.py` | 200 步逐步最大差 1.669e-06 |
| M2 | `train_m2.cpp` + `m2_check.py` | ASM 单层 T=8：dq/dk/dv ≤1.1e-8，dK0 2.26e-7 / dV0 7.06e-8 / dβ0 3.42e-8 |
| M3a | `train_m3.cpp` + `m3_check.py` | ERSA 前向（块摘要+路由+sink）+ 逐 query 反传：≤7.5e-08 |
| M3b | `train_m3b.cpp` + `m3b_check.py` | KV cache 逐 token 增长训练环：块摘要 0 err，梯度/输出 ≤8.1e-08 |
| M3c | `train_m3c.cpp` + `m3c_check.py` | 冻结多模态输入 + 投影+ERSA+AdamW 20 步：loss 差 1.8e-07，参数差 4.2e-06 |

M3 口径：CPU 版 `asm_ersa.ERSA` 是参考版语义（q 归一化/JL sketch/无因果），与内核不同；对拍按内核精确语义（mean-K 块摘要、`q·k·scale/tau`、逐 query 因果窗口、sink）复算。
M3b 坑：逐 token 前向必须按前缀长度派发（`grid.x=Nt`）取第 t 行；`grid.x=1` 时内核 `t` 恒为 0，梯度能对、前向输出错。
构建：把 `train_mN.cpp` 拷到 `ASM-ERSA-Design/vk` 下 ASCII 文件名，用 vcvars64 + `cl /std:c++20 /EHsc /utf-8 /O2 /I . /I <仓库根>\spv /link vulkan-1.lib` 编到 `build/`，在 `ASM-ERSA-Design/vk` 下运行并传 `<仓库根>\spv` 相对路径。

## M4：融合 dispatch + 分块扫描（2026-11-17）

`train_m4.cpp` 用同一张 M3c 图跑三条路径、内建基准：

```bat
copy 反向\train_m4.cpp ASM-ERSA-Design\vk\train_m4.cpp
cd ASM-ERSA-Design\vk
cl /nologo /std:c++20 /EHsc /utf-8 /O2 /I . /I ..\..\spv /I %VULKAN_SDK%\Include ^
   /Fe:build\train_m4.exe /Fo:build\train_m4.obj train_m4.cpp ^
   /link /LIBPATH:%VULKAN_SDK%\Lib vulkan-1.lib
build\train_m4.exe ..\..\spv check C:\temp\m4.bin     :: 三条路径互相对拍 + 导出（给 m3c_check.py）
build\train_m4.exe ..\..\spv bench 64 20 64           :: T=64，20 步，每串 64 个 query
```

| 路径 | 调度 | 反传内核 |
|---|---|---|
| 串行 | 每算子提交 + 围栏等待（M3c 原样） | `ersa_attn_bwd`，每 query 一次 dispatch + 回读 |
| 融合逐q | 前向/反传/更新各一条算子串 | 同上，`T/chunk` 次提交 |
| 融合批式 | 同上 | `ersa_attn_bwd_batch`，一次 dispatch 扫全部 query |

结果（780M，DX=DK=DV=32，BLOCK=8；中位，40/20/10 步）：

| T | 串行 | 融合逐q(块=T) | 融合批式 | 批式 token/s |
|---|---|---|---|---|
| 8 | 26.8 ms | 5.40 ms | **5.20 ms** (5.2×) | 1540 |
| 64 | 149.7 ms | 23.8 ms | **20.3 ms** (7.4×) | 3146 |
| 128 | 359.0 ms | 68.5 ms | **41.1 ms** (8.7×) | 3117 |

实测数据：

- 单次「提交+围栏等待」**0.59–0.87 ms**；融合把 T=64 的 76 个同步点压到 3 个。
- **映射内存读 67 MB/s、写 26 GB/s**（同 16 KB 张量）。T=64 的逐 query 回读 1 MB → **15.5 ms**，
  这是融合逐q 路径的最大单项；批式内核靠设备侧累加把它消掉。
- 批式内核仍慢在「单工作组单线程扫全部 query」：T=128 时反传自身 39 ms。
- **链路本身有 ~1e-7 fp32 非确定性**：串行跑两遍，T=8/64 逐位相同，T=128 差 8.9e-8。
  因此路径间对拍用 1e-6 判据；`check` 在 T=6 上 A-vs-B 仍为 0。

坑：

- 批式内核直接吃原始 q，**必须把 1/tau 折进内核 scale**（逐 query 内核是靠主机预除 tau）。
  漏掉时 T=6 参数差 3e-2，看起来像内核算错。
- `提交算子串` 每串是独立命令缓冲+独立提交：串内首步之前没有屏障（本次已在调度器补了一道），
  串内相邻 dispatch 默认有屏障；不需要屏障的步骤要显式置 `不需前置屏障`，否则 T=128 会多 45 ms。
- 融合路径里不要用 vk张量 的运算符重载排依赖，直接用 `提交自定义算子/提交算子串` 并显式 `等待`。


## M4 + Apollo 优化器（2026-11-23）

`优化器/ApolloSolver.cpp`（对角拟牛顿，arXiv:2009.13586）已接进训练器：`启用Apollo(lr, sigma)`，
每个参数张量一套 VK 常驻状态 `m/B/d`，每步两次 dispatch（`apollo_reduce` 单工作组归约 + `apollo_update`），
归约结果经映射内存回读到主机算 `alpha`。

构建/运行（`反向/build_m4_apollo.bat`，产物 `反向/build/train_m4_apollo.exe`）：

```bat
build_m4_apollo.bat
train_m4_apollo.exe ASM-ERSA-Design\vk\spv apollo 0.0125 64 40 0.0001 31337
:: 参数：模式 apollo / lr / T / 步数 / sigma / seed；省略 lr 时扫 {0.02, 5e-3, 1e-3}
```

修复（原 `apollo` 模式不可用）：

- `串行步` 的 Apollo 分支漏了 3 条 `linear_bwd_dwdb`：solver 拿到的是从未由 dQ/dK/dV 计算的 `dW`，
  参数实际不动。现在先算 dW 再 `步骤()`。
- `重置()` 现在同步复位 4 个 solver 的 `m/B/d`，避免同一训练器二次跑时带残留状态。

数值对拍：`apollo_test`（N=1000，5 步，fp64 参考）max|Δ|=1.161e-06，PASS
（`ASM-ERSA-Design/vk/build/apollo_test.exe spv`）。

收敛对照（780M，T=64，DX=DK=DV=32，BLOCK=8，40 步；AdamW lr=0.02 wd=0.01；Apollo lr=0.0125 sigma=1e-4，4 seeds）：

| seed | AdamW 末步 | Apollo 末步 | Δ |
|---|---|---|---|
| 31337 | 0.176156 | 0.166571 | -0.009585 |
| 1 | 0.182039 | 0.198126 | +0.016087 |
| 2 | 0.180751 | 0.174253 | -0.006498 |
| 3 | 0.178047 | 0.161873 | -0.016174 |

Apollo 3/4 seed 优，末步差均值 -0.0040（约 -2%）。前期更快（step2 0.32 vs 0.38），后段被 AdamW 追上。

超参敏感性（T=64，40 步，seed 31337）：

| lr | sigma | 末步 loss | 现象 |
|---|---|---|---|
| 0.02 | 1e-3 | 0.2419 | 稳定但慢，末段 ~0.242 徘徊 |
| 0.005 | 1e-3 | 0.2871 | 更慢 |
| 0.001 | 1e-3 | 0.3423 | 最慢 |
| 0.02 | 1e-4 | 0.3293 | 第 3 步后完全冻结（步长 < fp32 ulp） |
| 0.005 | 1e-4 | 0.1947 | 平滑 |
| 0.0125 | 1e-4 | 0.1666 | 本组最优 |
| 0.005 | 5e-5 | 0.1673 | 与 (0.0125,1e-4) 同比值，曲线一致 |
| 0.005 | 3e-4 | 0.2491 | 偏慢 |
| 0.02 | 1e-6 | NaN/Inf | 初始步长 1e6·g 直接炸 |

观察：`|B| << sigma` 时更新正比于 lr/sigma，曲线由比值决定；(0.0125,1e-4) 附近是当前尺寸的稳定上界，
再大（(0.02,1e-4)）就跌入「B 被推大 → 步长被吞 → loss 冻结」或直接发散。T=6 小尺寸上 (0.02,1e-4) 同样会炸，
说明 sigma 必须跟着 lr 一起调，不能沿用 AdamW 的 lr。

结论：VK Apollo 数学实现可信（1.16e-6 对拍），但在这个训练器上只是与 AdamW 同量级、seed 依赖，
没有稳定优势；要用它需要 lr/sigma 联合调参（并补 warmup/裁剪），或换 ApolloLite/Adan 之类变体。

### 补充：warmup + 梯度裁剪与公平 lr 对照（2026-11-23）

新增两个 VK 内核（源码 `ASM-ERSA-Design/vk/glsl_dev/grad_norm_sq.comp` / `grad_scale.comp`，
产物在 `ASM-ERSA-Design/vk/spv/`）：

- `grad_norm_sq`：4 个梯度张量（dWq/dWk/dWv/dsink）单工作组归约 Σg²，输出 1 float；
- `grad_scale`：设备侧把 4 个梯度就地乘同一裁剪系数（梯度全程不上主机）。

`启用Apollo(lr, sigma, clip, warmup)`：clip>0 时每步多 1 次归约 dispatch + 1 float 回读 + 1 次缩放 dispatch；
warmup>0 时按 `lr_eff = lr·min(1,(step+1)/warmup)` 线性预热。`train_m4 apollo` 的参数扩为
`apollo [lr] [T] [步数] [sigma] [seed] [clip] [warmup] [跑AdamW] [AdamW_lr]`。

单 seed 扫参（T=64，40 步，seed 31337）：

| lr | sigma | clip | warmup | 末步 |
|---|---|---|---|---|
| 0.0125 | 1e-4 | 0 | 0 | 0.1666 |
| 0.0125 | 1e-4 | 1e-2 | 0 | 0.1554 |
| 0.03 | 1e-4 | 1e-2 | 0 | 0.1327 |
| 0.03 | 1e-4 | 1.2e-2 | 0 | 0.1310 |
| 0.03 | 1e-4 | 2e-2 | 0 | 0.2581 |
| 0.0125 | 1e-4 | 0 | 5 | 0.1727 |
| 0.02 | 1e-4 | 1e-2 | 10 | 0.2036 |

- 梯度裁剪是唯一真正有效的改动：clip 在 1e-2 附近（本模型梯度范数量级）把 Apollo 从 0.1666 拉到 0.1310；
  clip=0.1 与不开逐位相同（不触发），clip≥2e-2 又退回冻结（0.3293）。
- warmup 对 Apollo 有害（前期大步是它的优势），只在救冻结区间时有意义。
- 4 seed 复核（Apollo 0.03/1e-4/clip 1.2e-2 vs AdamW 原 lr=0.02）：Apollo 0/4 胜——但问题在基线：
  AdamW lr=0.02 本身不是最优。
- **AdamW lr 扫描（seed 31337）**：0.02→0.1762、0.06→0.1204、0.08→0.1215、**0.1→0.1041**、0.12→0.1073、0.2→0.1114。
- 公平对照（4 seeds，AdamW lr=0.1 vs Apollo 0.03/1e-4/clip 1.2e-2）：
  AdamW 0.1041/0.1060/0.1129/0.1072（均值 0.1075），Apollo 0.1310/0.1342/0.1425/0.1341（均值 0.1355），
  **Apollo 0/4 胜，平均差 +0.0279（约 +26%）**。
- 100 步 4 seed 复核（同配置）：步 40 AdamW 0.1075 / Apollo 0.1355（0/4 胜）；步 80 AdamW 0.0845 / Apollo 0.0922（1/4 胜）；
  步 100 AdamW 0.0791 / Apollo 0.0830（1/4 胜）。单 seed 看到的「100 步 Apollo 反超」不成立。

修正结论：**之前「Apollo 3/4 seed 优」是拿未调 lr 的 AdamW 当基线造成的，不是真实优势。**
公平调参后（40 步）AdamW 明显更好；100 步 Apollo 仍差约 5%。Apollo 在这个训练器上没有可复现的优势，
且对 clip/σ/lr 三参同时敏感，不建议作为默认优化器。若要继续，优先怀疑对角曲率估计 B 被噪声推大导致过阻尼，
需要给 B 或 alpha 加上界（论文实现之外的工程约束）。

### B 上界实验（2026-11-23）

改动：`apollo_update.comp` 的 push constant 从 16B 扩到 32B，新增 `b_max`，内核里 `bi = clamp(bi, -b_max, b_max)`；
`Apollo::配置` / `ApolloSolver::配置` 加 `b_max`（默认 FLT_MAX，行为不变）。重编后 `apollo_update` 指纹 561607857、
推送 32B；`Apollo.cpp` / `ApolloSolver.cpp` 同步 32B 结构；`apollo_test` 重编后对拍仍 1.161e-06 PASS。
`train_m4 apollo` 参数末尾追加 `[Bmax]`（第 13 位）。

扫描（T=64，40 步，seed 31337，lr=0.03，sigma=1e-4，clip=1.2e-2）：

| Bmax | 末步 |
|---|---|
| 不启用（FLT_MAX） | 0.1310 |
| 1e-2 / 3e-2 / 0.05 / 0.08 / 0.1 / 1 / 10 / 1e4 | 0.1310（逐位相同） |
| 3e-3 | 0.1278 |
| 1e-3 | 0.1285 |
| 3e-4 | 0.1304 |
| 1e-4 | 0.1274 |
| 0 | 0.1274 |

- Bmax ≥ 1e-2 与不启用逐位相同 ⇒ **|B| 全程 < 1e-2**；Bmax=1e-4 与 0 相同 ⇒ |B| 基本在 sigma(1e-4) 量级。
  对角曲率估计在本训练器上从未真正起作用，`D=max(|B|,σ)` 实际恒为 σ。
- 无裁剪时（clip=0），Bmax 取 0 / 0.1 / FLT_MAX 都冻结在 0.3293 ⇒ 冻结与 B 无关，是 m 变小后步长被 fp32 吞掉。
- Bmax=0（强制 D=σ，退化成「梯度 EMA / σ」的大步长法）4 seeds：
  40 步 AdamW 0.1075 / Apollo 0.1304（0/4）；100 步 AdamW 0.0791 / Apollo 0.0803（1/4，差 1.6%）。
- 结论：B 上界不是解。Apollo 在这个 toy 训练器上实际退化为 lr·m/σ；要让它体现拟牛顿优势，
  需要让 B 能收到曲率信号（当前 alpha 估计把 B 压在 sigma 附近），属于机制层面的问题，不是调参能救的。

### 假 Apollo 与 B 初始化修复（2026-11-23）

核对确认旧实现是「假 Apollo」：

- B 与 d 都初始化为 0。第一步 `alpha = (d·Δm + d²B)/(‖d‖₄+ε)⁴ = 0/ε⁴ = 0`；第二步 `d = m/σ = 1e4·m`，
  分母达到 1e13 量级，`alpha ~ 1e-11`。
- fp64 复算（N=1000，5 步）：旧实现 `|B|max` 恒在 2.0e-4、98% 元素的 `D=max(|B|,σ)` 取 σ，B 从未真正更新。
  GPU 侧 b_max 扫描（1e-4 **1e-3 … FLT_MAX 逐位相同**）与之吻合。
- 结论：旧实现退化为 `lr·m/σ` 的 SGD-with-momentum，不是对角拟牛顿。

修复：`apollo_reduce` 新增 `init_b`，第一步把 `B` 初始化为 `g²`（`apollo_update.comp` 的 b_max 保持 32B push）。
重编指纹 529995137、推送 32B；`apollo_test` 的 fp64 参考同步加 `b=g²`，对拍 **max|Δ|=1.406e-06 PASS**。

修复后（fp64 复算）：`|B|max ≈ 1.09e-1`，`D` 由 B 主导的元素从 0% 升到 94%，B 不再恒 0；
但 `alpha` 仍只有 1e-7~1e-12，B 近似静态（由初始 g² 决定）——仍是「半 Apollo」，
完整的曲率自适应还差 warmup 或 alpha 尺度修正（Apollo 官方实现带 500 步 warmup）。

实测（T=64，40 步，lr=0.03，σ=1e-4，clip=1.2e-2，seed 31337）：

| 版本 | 末步 loss |
|---|---|
| 旧（B 恒 0，假 Apollo） | 0.1310 |
| B=g² 修复后 | 0.1274 |
| AdamW lr=0.1（公平基线） | 0.1041 |

toy VK 训练器（`cpp/train_main.exe`，20 步）：Apollo（B=g²）4.275→3.685，AdamW（3e-4）4.157→3.842，无 NaN。

### Apollo 的状态与内存（6B-lite 口径）

当前 VK 实现每参数常驻：参数 fp32 4B + 梯度 fp32 4B + m/B/d 各 fp32 12B = **20 B/param**。
6B 参数 → **~120 GB**（常驻，不含激活值）。此前「90 GB」是按 AdamW 16 B/param 的口径，不适用于 Apollo。

降内存的路径（当前未实现，按实现难度排序）：

| 方案 | 每参数 | 6B 总 | 说明 |
|---|---|---|---|
| 全 fp32（现状） | 20 B | ~120 GB | 本机不可行 |
| bf16 参数/梯度 + fp32 状态 | 16 B | ~96 GB | 与 AdamW 同档 |
| bf16 全量（参数/梯度/状态） | 10 B | ~60 GB | 需 bf16 状态 kernel |
| int8 状态 + bf16 参数/梯度 | ~7 B | ~43 GB | 需量化状态 kernel |
| 逐层 offload + bf16 | 常驻 1 层状态 | ~3–5 GB | 受内存带宽限制，最接近「Apollo 低内存」 |

### APOLLO-Mini（低秩辅助缩放）VK 实现（2026-11-23）

用户澄清「低秩状态 Apollo」指 MLSys'25 的 APOLLO（SGD-like memory, AdamW-level performance），
不是 Ma 2020 的对角拟牛顿 Apollo。两者同名不同算法；此前 B/σ/clip 的调参只适用于对角版本。

新增 3 个内核（`ASM-ERSA-Design/vk/glsl_dev/apollo_mini_*.comp`）：

| 内核 | 作用 | 工作组/绑定/推送 |
|---|---|---|
| `apollo_mini_proj` | R = PᵀG（rank-1 随机投影，P[rows]） | 64 / 3 / 16B |
| `apollo_mini_stat` | M_R/V_R 更新 + 归约 ‖R‖²、‖R̃‖² | 256 / 4 / 32B |
| `apollo_mini_apply` | W -= lr·s·G（tensor-wise 缩放） | 256 / 2 / 16B |

主机类 `优化器/ApolloMini.{h,cpp}`：每参数矩阵一份状态 P[rows] + M/V/R[cols]；
rank-1、tensor-wise、scale=128（官方 APOLLO-Mini 配置），每 200 步换投影。

接入：`cpp/train_main.cpp` 读 `ASM_OPT=apollo_mini`（2D+ 参数走低秩，1D 参数回退 AdamW）。

修复与对拍（2026-11-23，对齐官方 apollo-torch）：

- 投影方向：官方 `proj_type='std'` 在 rows>=cols 时用 right（R=G·Pᵀ，R 长 rows），否则 left；
  状态长度 = min(rows, cols)，不是固定按列。
- **scale 乘的是 sqrt(scale)**（官方 `norm_grad *= sqrt(scale)`），此前直接用 scale，步长偏大 11.3×。
- **Norm-Growth Limiter**（官方默认开）：每参数保存上一步 ‖scaled_grad‖，limiter=max(增长比,1.01)/1.01，
  超限时缩放；新增 `apollo_mini_gnorm` 内核归约 ‖G‖²（256/2/16B）。
- 投影 P 用标准正态（不乘 1/sqrt(rank)）；小矩阵 rows<8 回退 AdamW。

对拍：`apollo_mini_test.exe` → [64,32]、10 步、fp64 参考，**max|Δ|=8.047e-07**（判据 3e-5）。

收敛（toy，200 步，同初始化）：

| 配置 | @20 | @50 | @100 | @200 |
|---|---|---|---|---|
| AdamW lr=3e-2 | 2.8791 | 2.1082 | 1.1551 | 0.3898 |
| **APOLLO-Mini lr=3e-2 scale=128** | 3.1777 | 2.1231 | 1.0809 | **0.3307** |
| APOLLO-Mini lr=1e-2 scale=128 | 3.1220 | 2.8332 | 2.3968 | 1.6621 |

对齐官方后 APOLLO-Mini 在 200 步比同 lr AdamW 低 15%（0.3307 vs 0.3898），与论文「AdamW 级」一致；
lr 仍需与 AdamW 同档（1e-2 偏小）。6B-lite 按官方 rank-1/scale=128 配置。

6B-lite 状态账：24 层 / 138 专家 / D=2048 / F=384，状态约 20–70 MB（P 占大头，视 W1/W2 的行列布局）；
AdamW 的同参数状态是 72 GB（fp32 m/v）。训练态（bf16 参数+梯度+状态）约 27–30 GB，本机 96 GB 可训。

### 6B-lite 内存实测（2026-11-23）

`cpp/build/asm_ersa_model.exe cpp/spv 6bmem`（本机 96 GB，780M）。**780M 是 iGPU：GTT 与主机是同一块物理内存，
主机占用 + GPU 占用不能相加成两份**——第一次把主机 72 GB 与 GPU 48 GB 叠加，总需求 120 GB，直接把系统内存打爆。
现在改为单阶段分配（每项测完立即释放）：

| 项 | 分配 | 结果 |
|---|---|---|
| 主机 fp32 参数 | 23.96 GB | RSS 24.02 GB（测完释放） |
| 主机 bf16 参数 | 11.98 GB | RSS 12.04 GB |
| 主机 int8 权重 | 5.99 GB | RSS 6.05 GB |
| GPU int8 权重 | 5.99 GB | 分配成功 0.5 s |
| GPU fp32 梯度 | 23.96 GB | 分配成功 2.0 s（与 int8 合计 30 GB） |

真实预算（共享 96 GB 池）：6B 推理 int8 ≈ 12 GB（主机 6 + GTT 6）；训练 fp32 ≈ 48 GB（参数 24 + 梯度 24 + 状态 ~80 MB）；
bf16 ≈ 26 GB。结论：96 GB 内可装，瓶颈是 780M 带宽（~130 GB/s）。

### APOLLO-Mini INT8 直写（2026-11-23）

目标：参数只留一份 INT8（+组 scale）、梯度只留一份 fp32，更新直接写回 INT8，不存 fp32 master。

- 内核 `apollo_mini_apply_i8.comp`：`w_i8 = clamp(round_rs((w_i8*scale - lr*s*g)/scale))`，
  scale 形状 `[rows, G]`（G=ceil(cols/128)），随机舍入 `floor(x+U[0,1))`。
- 随机舍入是必需的：确定性 round 下每步 0.5 量化步长的更新，200 步后仍是 0（冻结）；
  随机舍入 200 步均值 -100.1（期望 -100）。
- 对拍 `apollo_mini_i8_test.exe`：确定性 0/2048 不一致；随机模式（同 hash）0/2048 不一致。
- 3B 接入（`ASM_PARAM=i8 ASM_OPT=apollo_mini`，`3b_cache`）：
  `i8 更新一步: loss=11.990150 | 参数 803 | Wq(L0) 变化 79239/4194304 mean|dq|=0.0190`，`3B CACHE PASS`。
- 第一版限制：MoE 的 W1/W2 前向用独立转置副本（moe_W1_i8），仍保留 fp32 master；
  其余量化矩阵（Wq/Wk/Wv/Wo/shW1/shWup/shW2/gate/embed/lmhead）已只存 INT8。
- 3B 内存：i8 参数 1.25GB（fp32 5GB）+ W1/W2 fp32 12.8GB + 梯度 11.4GB ≈ 20GB（全 fp32 约 45GB）。
- 6B 全量 i8 还差：W1/W2 的转置同步 kernel（或让前向直接读优化器的 i8 布局）。

- 3B 10 步对照（`3b_train`，同 lr=1e-4，单 batch 单 seed）：
  - fp32 master：11.99 / 10.12 / 10.13 / 8.02 / 8.66 / 7.81 / 6.66 / 7.56 / 6.84 / 6.87
  - i8 直写：   11.99 / 11.40 / 10.06 / 11.03 / 10.00 / 8.70 / 8.81 / 9.09 / 9.67 / 6.85
  - 末步 6.845 vs 6.870（基本齐平）；i8 过程波动更大（随机舍入方差），单 seed/单 batch 不足以定论。

- **W1/W2 也走 i8 直写**：`moe_i8_transpose.comp` 把优化器布局 `[E*D,F]`/`[E*F,D]` 转置到前向布局
  `[E*F*D]`/`[E*D*F]`，单测两个方向均 0/64 不一致 PASS；更新后自动同步。
  3B 全量 i8 4 步：11.99 / 12.17 / 11.00 / 10.19（fp32 master 同 seed：11.99 / 11.35 / 11.31 / 9.52），
  内存再省 W1/W2 的 fp32 master ≈ 6.4 GB（3B 落到 ~13 GB）。
