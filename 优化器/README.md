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
