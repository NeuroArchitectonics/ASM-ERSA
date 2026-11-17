# spv/ —— 预编译 Vulkan 着色器二进制

本目录只放**编译产物**，不放着色器源码（`.comp`）：

- `.spv`：SPIR-V 二进制，运行时由流水线调度器加载；
- `.spv.hpp`：由着色编译器生成的登记头，内含算子名、工作组大小、绑定数、
  推送常量字节与 SPIR-V 指纹（FNV-1a，32 位小端字）；
- `register_all.hpp`：一次性 include 全部登记头。

## 来源与一致性

- 二进制与同名 `.spv.hpp` 严格同源：头部 `产物指纹` 等于对应 `.spv` 的
  FNV-1a（按 32 位小端字计算），可用脚本复核；
- 着色器源码（`反向/glsl/*.comp`、`优化器/*.comp`）按商业秘密处理，**不随仓库发布**；
  修改内核需在私有设计档案环境重编，再把新的 `.spv` + `.spv.hpp` 同步回本目录。

## 清单

| 算子 | 用途 |
|---|---|
| `asm_step_bwd` | ASM 单步反向 |
| `ersa_attn_bwd` / `ersa_attn_bwd_batch` | ERSA 注意力反向（单 query / 批式） |
| `embedding_fwd` / `embedding_bwd` | embedding 前向 / scatter-add 反向 |
| `linear_fwd` / `linear_bwd_dx` / `linear_bwd_dwdb` | 线性层前向 / dX / dW+db |
| `rmsnorm_fwd` / `rmsnorm_bwd_dx` / `rmsnorm_bwd_dw` | RMSNorm 前向 / 反向 |
| `masked_softmax_bwd` / `fused_ce_bwd` | masked softmax 反向 / 融合交叉熵反向 |
| `moe_bwd_dx` / `moe_bwd_dw_expert` / `moe_bwd_dw_shared` | MoE 反向三件套 |
| `adamw_update` / `Adam参数更新` | AdamW 融合更新 / 原始拷贝件 |
| `adamw_update_fp16` | AdamW 混合精度（fp16 参数/梯度 + fp32 master/m/v） |
| `apollo_reduce` / `apollo_update` | Apollo 优化器：整张量归约（m/B/d 统计）+ 对角拟牛顿更新 |
| `linear_fwd_int8` / `linear_fwd_int8_f32` | 融合 INT8 GEMM（int8×int8→int32→scale），输出 fp16 / fp32 |
