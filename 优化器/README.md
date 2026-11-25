# ASM-ERSA 专属优化器（AdamW · Vulkan）

> 2026-11-16 移植：从 SpectraSSM 拷贝件改为本仓库公开技术版张量系统（`ASM-ERSA-Design/vk`）的专属优化器。
> 依赖 `vk张量系统::vk张量` 与 `vk张量系统::流水线调度器`，无 `模块` / `算子集` 依赖。

## 文件

| 文件 | 作用 |
|---|---|
| `AdamW.h` / `AdamW.cpp` | 主机侧：参数注册、m/v 生命周期、单步更新、检查点 |
| `spv/adamw_update.spv` | 设备侧融合内核（**已编译并登记**；着色器源码不随仓库发布） |
| `spv/Adam参数更新.spv` | 原始拷贝件（仅作参考，算子名不同） |

## 融合到什么程度

`adamw_update.comp` 一次 dispatch 完成：

1. 一阶/二阶矩 `m/v` 更新；
2. 偏差修正（`m_corr`/`v_corr` 主机按步数预计算）；
3. 参数更新 `param -= lr · m̂ / (√v̂ + ε)`；
4. **解耦权重衰减**（norm/bias 参数由主机传 0）；
5. **梯度清零**（`zero_grad=1`，省掉每步一次主机上传）。

优化器自身零临时张量：每个参数只有 `m`、`v` 两块 fp32 状态，原地更新，无主机往返。

## 已编译登记（实测）

```
shader_tool.exe 优化器 <私有档案>/spv
[ok] adamw_update: 工作组=256 绑定数=4 推送=36B 指纹=3176240966
成功 2 / 共 2，失败 0
```

- 描述符约定：第 i 个张量 `layout(set = i, binding = 0)`，绑定顺序
  `0=梯度, 1=参数, 2=m 状态, 3=v 状态`；
- 推送常量 36 B：`lr, beta1, beta2, eps, m_corr, v_corr, size(u32), weight_decay, zero_grad(u32)`；
- 生成物：仓库 `spv/adamw_update.spv` 与 `.spv.hpp`（静态登记元信息）。

## 主机侧编译验证

用 MSVC 19.51（VS18）对 `AdamW.cpp` 做过编译检查：

```
cl -nologo -std:c++20 -EHsc -utf-8 -c -I ASM-ERSA-Design/vk -I %VULKAN_SDK%/Include AdamW.cpp
→ AdamW.obj（无错误）
```

## 接口

```cpp
std::vector<AdamW::参数项> 参数项;
参数项.push_back({"layers.0.wq", &权重张量, &梯度张量});
AdamW 优化器(std::move(参数项), {.学习率 = 3e-4f, .权重衰减 = 0.1f});

优化器.步骤();                    // 原地更新 + 清零梯度
优化器.保存检查点("opt.bin");      // 魔数 ADMW、版本 2
```

- 仅支持 fp32 参数（内核按 `float*` 解释；bf16 需先扩展内核）；
- `norm` / `Norm` / `bias` 命名的参数不做权重衰减；
- 检查点 v2 为显式字段序列化，**与 SpectraSSM 拷贝件的 v1 格式不兼容**。

## 未做（后续）

- 梯度裁剪：需要跨张量的全参数范数规约，建议单独 `grad_norm` 内核，把 `clip_coef` 乘进 `lr`；
- 多张量合并 dispatch：当前每参数一次 dispatch，小模型可接受；
- bf16 参数支持。

## APOLLO-Mini（低秩辅助缩放，MLSys'25）

| 文件 | 作用 |
|---|---|
| `ApolloMini.h` / `ApolloMini.cpp` | 每参数矩阵 rank-1 状态（P/M/V/R），`W -= lr·s·G` |
| `spv/apollo_mini_{proj,stat,apply}.spv` | 投影 / 低秩状态+归约 / 缩放更新（源码在 `ASM-ERSA-Design/vk/glsl_dev/`） |

配置：rank=1、tensor-wise、scale=128（APOLLO-Mini 官方默认），每 200 步换随机投影。
状态是 O(rows + cols) 而非 O(rows×cols)：6B-lite 约 20–70 MB（AdamW 的 m/v 是 72 GB）。
注意：这是与下面「对角拟牛顿 Apollo」同名但不同的算法，B/σ/clip 的调参不适用。

对拍：`apollo_mini_test.exe ASM-ERSA-Design/vk/spv` → [64,32]、10 步、fp64 参考，
max|Δ|=8.047e-07（判据 3e-5）。收敛（toy 200 步）：APOLLO-Mini lr=3e-2 scale=128 末步 0.3307，
同 lr AdamW 0.3898（低 15%），与论文「AdamW 级」一致。

## Apollo（对角拟牛顿，arXiv:2009.13586）

| 文件 | 作用 |
|---|---|
| `Apollo.h` / `Apollo.cpp` | 拥有型：主机上传参数/梯度，m/B/d 常驻 |
| `ApolloSolver.h` / `ApolloSolver.cpp` | 非拥有型：绑定外部参数/梯度张量，训练器用 |
| `apollo_test.cpp` | N=1000、5 步 fp64 对拍（实测 max abs err = 1.161e-06） |

两步 dispatch：`apollo_reduce`（单工作组 256 线程扫整个张量，更新 m 并归约 `s1=d·Δm`、`s2=d²B`、`s3=Σd⁴`）
→ 主机算 `alpha=(s1+s2)/(‖d‖₄+ε)⁴` → `apollo_update`（`B-=alpha·d²`；`D=max(|B|,σ)`；`d=m/D`；`θ-=lr·d`）。
状态全在 GPU，只回读 3 个 float；每个参数 4 块 fp32 状态（m/B/d + 归约输出）。

B 初始化：第一步（`init_b=1`）把 B 置为 g²。B 从 0 起步会让 d=m/σ 过大、alpha 被压到 ~1e-11，
B 永远不动（实测 |B|<σ，退化成 lr·m/σ）；这是「假 Apollo」的根因，已修（对拍 1.406e-06 PASS）。

内存口径：参数 fp32 4B + 梯度 fp32 4B + m/B/d 12B = 20 B/param；6B 参数 ~120 GB。
「Apollo 低内存」要靠 bf16/int8 状态或逐层 offload（bf16 全量 ~10 B/param = 60 GB，
int8 状态 ~7 B/param = 43 GB，逐层 offload ~3–5 GB），当前 VK 实现尚未做这些；
账见 `反向/README.md` 的「Apollo 的状态与内存」节。

实测（M4 训练器，见 `反向/README.md` 的「M4 + Apollo 优化器」节）：对拍 PASS；加入 VK 侧全局梯度裁剪
（`grad_norm_sq` + `grad_scale`）后 40 步 loss 从 0.1666 降到 0.1310，但仍不及同尺寸调过 lr 的 AdamW
（lr=0.1，4 seed 均值 0.1075）；100 步 4 seed 均值 0.0803 vs AdamW 0.0791，仍差 1.6%。`sigma`、`lr`、`clip` 三参必须联合调，
沿用 AdamW 的 lr 会冻结或发散。`b_max`（|B| 上界，默认 FLT_MAX）已加：实测 |B| 全程小于 1e-2，
Bmax≥1e-2 与不启用逐位相同——B 上界不是瓶颈，冻结与 B 无关。

### ApolloMiniI8（INT8 直写）

- `ApolloMiniI8` 复用 ApolloMini 的 proj/stat/gnorm 与 sqrt(scale)+NL，参数/尺度换成 `int8 + [rows,G]`，
  更新走 `apollo_mini_apply_i8`（随机舍入）。
- 对拍：`apollo_mini_i8_test.exe` → 确定性 0/2048、随机 0/2048、小更新 200 步 -100.1（期望 -100），PASS。
- 3B 训练器接入：`ASM_PARAM=i8`；Wq(L0) 一步更新变化 79239/4194304，mean|dq|=0.019。

### W1/W2 前向布局 INT8 直写（2026-11-25）

- 3B + `ASM_PARAM=i8` 默认不再保留 W1/W2 的 fp32 master 与 m/v；前向仍读取原来的转置 INT8/scale，
  ApolloMiniI8 的低秩投影仍读 master 布局梯度，apply 阶段直接按 W1/W2 映射写前向布局并随机舍入。
- 新内核：`apollo_mini_apply_moe_i8{,_bf16}`；`ASM_MOE_I8_MASTER=1` 可恢复旧 master + `moe_quant_w` 路径。
- 对拍：`apollo_mini_moe_i8_test.exe` 覆盖 W1/W2、fp32/bf16、跨 128 组边界、确定性/随机舍入、尾部哨兵和下降方向，
  `MOE_DIRECT_I8 PASS`，不一致/越界 0。
- 3B seed=7 实测：VK live **20.66 → 14.66 GiB**，减少 **6.00 GiB（约 29%）**；`p(权重)` 6.03 → 0.03 GiB。
  4 步 loss `12.800043 → 11.925464 → 12.199209 → 12.039916`，无 NaN/Inf；W1/W2 每步均有实际 INT8 变化。
  `3b_cache` 单步 `3B CACHE PASS`，Wq(L0) 变化 93536/4194304；780M fwd 1943.83 ms / bwd 1059.78 ms。
- 证据：`archives/2026-11-25/3b_moe_i8_direct/日志/`；旧路径复跑同为 20.66 GiB，回退开关有效。

### 激活缓存优化（2026-11-25）

- 删除从未被消费的反向临时区：`dTable`、`dLmHead`、`dW1V/dW2V`、融合后遗留的 `dShW*`、`dWq/dWk/dWv/dWo` 等。
  这些缓冲只分配、不参与 dispatch 或累加，删除无数值语义变化；3B VK live **14.66 → 11.87 GiB**。
- ASM 逐 token 旧 K/V 快照默认改 bf16 存储，新增 `asm_step_eta_bf16` / `asm_step_bwd_seq_wg_bf16`；
  前向状态更新与新颖性门控仍为 fp32，只有反向快照量化为 bf16。`ASM_OLD_STATE_F32=1` 恢复 fp32 快照。
- bf16 快照再省 **1.13 GiB**，3B VK live **11.87 → 10.74 GiB**；相对 W1/W2 直写前累计省 **3.92 GiB（约 27%）**。
- 数值边界：死缓冲删除档梯度全局范数 `2.72418e7`，与删除前 `2.72417e7` 一致；bf16 快照档为 `3.01368e7`（约 +10.6%），
  288 个 bf16 梯度张量 / 2,222,981,120 元素 `nonfinite=0`。4 步 loss `12.800043 → 13.536366 → 12.309326 → 11.735956`。
- `3b_cache`：bf16 快照 `3B CACHE PASS`，Wq(L0) 变化 99781/4194304；fp32 回退档 11.87 GiB，同样 `3B CACHE PASS`。
  780M 串提交实测 fwd 2026.76 ms / bwd 1030.56 ms；cache 分项计时受同期争用影响，不作为性能回归结论。
- 证据：`archives/2026-11-25/3b_activation_cache/日志/`。

### 词表梯度 bf16（2026-11-25）

- `lm_head` 梯度改由 `linear_bwd_dwdb_bf16` 直接写 bf16；`embed` 梯度先用 `embedding_bwd_scatter_partial`
  在 `U×D` 的小 fp32 partial 上原子累加，再由 `embedding_bwd_partial_to_bf16` 一次 cast 到 bf16 全词表，避免逐原子舍入。
- 3B VK live **10.74 → 9.58 GiB**，梯度缓冲 **6.49 → 5.33 GiB**；`ASM_WORD_GRAD_F32=1` 恢复两份 fp32 词表梯度。
- 对拍：`embed_lm_grad_bf16_test.exe` → lm_head fp32 最大差 1.19e-7、embed partial 差 0、不一致/越界 0，`EMBED_LM_GRAD_BF16 PASS`。
- 3B seed=7：bf16 梯度张量 **290 个 / 2,845,835,264 元素**，`nonfinite=0`；4 步 loss
  `12.800043 → 12.350849 → 12.210994 → 11.265177`，`3b_cache` 单步 `3B CACHE PASS`，Wq(L0) 变化 99778/4194304。
  780M 串提交实测 fwd 1359.55 ms / bwd 876.53 ms；cache 分项计时受同期争用影响，不作为性能回归结论。
- 证据：`archives/2026-11-25/3b_cache_stage3/日志/`。

### bf16 梯度（只存不累加）

- 目标：梯度缓冲从 fp32（4 B/param）降到 bf16（2 B/param），参数本身仍是 int8。
- 现状：bf16 梯度已覆盖 **Wo/Wq/Wk/Wv + MoE 的 W1/W2/shW1/shWup/shW2**；`ASM_BF16_SCOPE=linear|moe|all` 可分段隔离（默认 all）。
  W1/W2 默认走上面的前向布局 ApolloMiniI8 直写；旧 master 路径用 `adamw_update_bf16`（梯度 uint16，master/m/v 仍 fp32），sh* 走 ApolloMiniI8 bf16 变体。
- 对拍：`linear_bwd_dwdb5_bf16_test.exe`（small N=8 + pow2 N=32，5 组不同形状，256 词守卫哨兵）
  fp32 版 vs CPU max|Δ| 1.19e-07 / 1.91e-06；bf16 解包后超差(>1%) 0、守卫越界 0、db 不一致 0 → PASS。
- 3B 实测：`3b_cache` PASS（`linear` / `moe` / `all` 三种 scope 都 PASS），
  `ASM_DBG_BF16=1` 显示 `W1(L0) n=25165824 nan=0`、`shW1(L0) n=786432 nan=0`、`Wq(L0) nan=0`；
  一步更新 `Wq(L0) 124737/4194304 mean|Δq|=0.0318`。
  4 步（seed 7，all）：`12.800 12.343 11.396 9.671`；同 seed fp32：`12.800 12.018 12.310 11.735`，无 NaN/Inf。
- 2026-11-25 隔离构建复验（seed=7，`ASM_PARAM=i8 ASM_OPT=apollo_mini ASM_BF16_SCOPE=all`）：
  `ASM_DBG_BF16=1` 全量扫描 **288 个 bf16 梯度张量 / 2,222,981,120 元素**，`nonfinite=0`，全局范数 `2.72417e7`；
  4 步 loss `12.800043 → 12.994665 → 12.622510 → 11.759311`，均完成更新并退出 0；
  `3b_cache` 单步更新 `Wq(L0) 变化 93541/4194304 mean|Δq|=0.0223`，`3B CACHE PASS`。
  780M 实测 `fwd 2117.45 ms / bwd 1087.33 ms / VK live 20.66 GiB`。
  首个 4 步臂在 `s=1 upd` 出现一次 `VK_ERROR_DEVICE_LOST`；同配置 1 步与 4 步复跑均 PASS，失败日志保留，尚未定位为确定性内核错误。
- 内存（按缓冲字节数，3B/32 层）：W1/W2 `32×2×25,165,824` 元素 ×2 B 省 3.22 GB；线性四权省 1.07 GB；
  合计 **≈4.3 GB**（i8 训练态 13 → 8.7 GB）。6B-lite 专家 5.2 B 参数，估算再省 ≈10 GB（30 → ≈20 GB）。
- 稳定性：`all` 首次跑出现过一次 `VK_ERROR_DEVICE_LOST`，连续重跑两次均 PASS，判为偶发驱动/残留状态，未定位。
- 踩过的坑（重要）：`gate` 未量化（fp32 梯度），而 `linear_bwd_dwdb5` 一次写 5 组；若整层切 bf16 又用 fp32 内核，
  会把 fp32 值（4 B/元素）写进 2 B/元素的缓冲 → 该参数 NaN，并破坏同层其它 bf16 梯度。
  修法：组 0（gate）单独用 fp32 内核 dispatch（M1..M4=0），组 1..4 用 `_bf16` 内核（M0=K0=0）。
  同理 `moe_bwd_dw_all_i8_bf16` 的 5 个输出必须全为 bf16，否则不能切（W1/W2 与 sh* 必须一起切）。
- 使用前提：所有按「4 B/元素」清零/读取梯度的辅助函数（`填`、`范数`、抓梯度）必须按 `张量类型::浮点bf16` 走 2 B 分支。
- 仍未切 bf16：`asm_step_bwd_*`、`ersa_*_bwd`、`gr_*`、`embedding_bwd_scatter`、`rmsnorm_bwd_dw`（累加型，需 fp32 局部累加后 cast）；
  对应参数（ASM 槽位、GR、gamma、embed）仍是 fp32 梯度，量级远小于 MoE。

### MoE 前向 i8 副本的更新（此前是死的）

- 现象：`刷新前向量化()` 在 `用I8参数` 时直接 return，而 W1/W2 的转置同步被 `W1->miniI8` 门控，
  但 W1/W2 是 `量化=false`（没有 i8/miniI8）⇒ **`moe_W1_i8`/`moe_W2_i8` 在 i8 训练里从不更新，routed 专家等于冻结**。
- 修法：新内核 `moe_quant_w`（push 24 B `{E,F,D,mode,G,base}`，128 线程/组，共享归约求 max|w|）：
  读 fp32 master，直接写前向布局 int8 + 组 scale（mode 0: `p[e*D+d,f]→q[e*F+f,d]`；mode 1: `p[e*F+f,d]→q[e*D+d,f]`）。
  更新后对每层 W1/W2 各派发一次（按 `base` 分块 32768 组，避开单维 dispatch 上限）。
- 单测 `moe_quant_w_test.exe`：E=2/D=256/F=40，`mode0 q 不一致 0/20480、s 0/160`；`mode1 q 0/20480、s 0/512` → PASS。
- 端到端（`ASM_DBG_MOE=1`）：一步后 `L0.moe_W1_i8 前4096和` 由 `-2545 → -2568`，证明前向副本真的在变。
- 4 步（seed 7，bf16 梯度 + i8 + ApolloMini）：`12.800 12.405 12.881 11.097`，无 NaN/Inf。
- 顺带：`vkCreateShaderModule` 报 `Capability Int16 … shaderInt16` 告警 —— 该特性属于 **`VkPhysicalDeviceFeatures`**，
  不在 `VkPhysicalDeviceVulkan11Features` 里；已在 `流水线调度器.hpp` 加 `特性.shaderInt16 = VK_TRUE`，告警归零。

### `adamw_update_bf16` 单测与「累加型内核为何留在 fp32」

- `优化器/adamw_update_bf16_test.cpp`：N=4000（非 256 倍数）、缓冲 4096、5 步、lr=1e-3、wd=0.01，
  CPU fp64 参考 + 尾部哨兵 + `zero_grad` 检查：
  `max|Δp|=9.528e-08  max|Δm|=1.688e-10  max|Δv|=4.925e-13  非有限=0`
  `越界写=0  g 未清零=0  g 越界写=0` → **ADAMW_BF16 PASS**。
- 剩下的 fp32 梯度是设计选择，不是漏项：
  - `embedding_bwd_scatter` 用 **atomic CAS** 累加（同 id 多 token 并发写同一行）。Vulkan core 没有 16 位原子加，
    改 bf16 只能拆成两段（fp32 临时 + 二次 scatter），省下的只有 152064×2048×2 B ≈ **0.62 GB（20.66 GiB 的 3%）**，
    却给梯度最稠密的参数引入二次舍入，收益不成立。
  - `asm_step_bwd_*` / `ersa_*_bwd` / `gr_*` 的 dK/dV/dsink 等是**跨工作组累加**的小缓冲（每层几百 KB 量级），
    寄存器和显存都没压力，不改。
- 结论：bf16 梯度只用于「一次写定」的参数（线性四权、MoE W1/W2/sh*），累加型保持 fp32。

### MoE 的 W1/W2 改用 ApolloMini（去掉 AdamW 的 m/v）

- 之前 W1/W2 是 `量化=false` → p + m + v（≈12 B/param），1.61 B 参数就是 19 GB 量级。
  现在 `用I8参数` 时对 MoE 也建 `ApolloMini`（fp32 master + rank-1 状态），并在建好后释放 m/v：
  ```cpp
  if (q->mini || q->miniI8) { q->m = vk张量(); q->v = vk张量(); }
  ```
- bf16 梯度支持：新增 `apollo_mini_apply_bf16`（fp32 w，uint16 g），`ApolloMini` 构造加 `梯度bf16` 标志，
  按标志派发 `apollo_mini_{proj,gnorm,apply}_bf16`。
- 实测 `VK live`（3b_cache，逐项叠加）：
  | 配置 | VK live |
  |---|---|
  | bf16 梯度 + AdamW(m/v) | 36.80 GiB → 32.66 GiB |
  | fp32 梯度 + ApolloMini | 24.80 GiB |
  | bf16 梯度 + ApolloMini | **20.66 GiB** |
  合计比最初省 **16.14 GiB（−44%）**。
- 回归：`apollo_mini_test` 仍 PASS（`max|Δ|=8.046931e-07 @i=526`）；`3b_cache` PASS，
  `ASM_DBG_MOE` 显示 `L0.moe_W1_i8 前4096和 -2512 → -2509 → -2507 → -2491`（专家仍在学）。
- 4 步（seed 7）：bf16 g + mini `12.800 12.663 12.357 11.733`；fp32 g + mini `12.800 13.090 12.590 11.526`，均无 NaN/Inf。
