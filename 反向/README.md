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

