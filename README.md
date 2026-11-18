# ASM-ERSA: Addressable Slot Memory + Exact-Recall Sparse Attention

**可寻址槽位记忆 + 精确回召稀疏注意力，面向超长上下文**

**Addressable Slot Memory + Exact-Recall Sparse Attention for Long-Context Understanding**

> **状态（2026-11-18）**：主运行路径为 `cpp/` 的 C++ + Vulkan INT8 融合实现（`ASM_ERSAModel`，INT8×INT8，ASM8BIN 权重）。
> `asm_ersa` Python 包保留为机制参考/数值 oracle，不作为运行时主路径。
> 当前没有训练过的 checkpoint，也没有设备训练；ASM8BIN 可由未训练/随机权重生成，用于架构与 kernel 验证。
> Vulkan 反向/优化器 kernel 见 `反向/`、`优化器/`；当前没有完整设备训练器。
> 下文性能表与部分机制描述仍沿用旧口径（标 ⚠️ 处未复核），正在按 ASM-ERSA 设计重写，请以代码为准。

---

## 中文介绍

ASM-ERSA 是一种混合 Transformer 架构：用可寻址槽位记忆（ASM）替代标量门控线性记忆，用精确回召稀疏注意力（ERSA）替代全注意力，实现有界状态下的长上下文建模。

### 核心特性

- **可寻址槽位记忆（ASM）**：每头 M 个槽位，top-r 精确读 + 凸组合写 + 槽位级标量衰减
- **精确回召稀疏注意力（ERSA）**：块路由 + 因果窗口 + sink，只对候选块做精确打分
- **动态稀疏嵌入**：N-gram 嵌入 10.7× 压缩（51B → 4.8B）
- **长上下文高效**：状态有界，不随上下文长度增长
- **多模态 Omni**：支持文本 + 音频 + 视觉联合训练（3B 配置）
- **层比例**：每 4 层为 3 层 ASM 槽位记忆 + 1 层 ERSA 精确回召（48 层 / 32 层 Omni 同构）

### 3B Omni 模型

| 组件 | 配置 |
|------|------|
| d_model | 2048 |
| n_heads | 16 |
| n_layers | 32（24 ASM-ERSA + 8 ERSA-Omni） |
| MoE experts | 32（8 shared + 12 text + 6 audio + 6 vision） |
| 总参数 | 3.22B（`build_model(TrainConfig())` 实测 3,219,116,392） |
| 训练内存 | ~61 GB（bf16，含优化器状态，未实测） |

### Vulkan 张量计算系统

**主运行路径**：`cpp/`（C++ + Vulkan INT8 融合 shader，ASM8BIN 权重）。Python `asm_ersa` 仅保留为参考/oracle。

本项目基于 **Vulkan 张量计算系统**，用于高效部署 ASM-ERSA 模型。该系统基于 Vulkan Compute API，支持：

- 跨平台 GPU 推理（Windows / Linux / Android）
- 自定义算子开发（ASM 槽位读/写、ERSA 回召、反向算子等）
- INT8/FP16 混合精度推理
- 内存高效的流式推理

**如果您有移动端 AI 部署需求，或需要定制 Vulkan 张量系统开发，欢迎联系合作！**

---

## English

ASM-ERSA is a hybrid Transformer architecture: addressable slot memory (ASM) replaces scalar-gated linear memory, and exact-recall sparse attention (ERSA) replaces full attention, enabling long-context modeling with bounded state.

### Key Features

- **Addressable Slot Memory (ASM)**: M slots per head, top-r exact read + convex-combination write + slot-level scalar decay
- **Exact-Recall Sparse Attention (ERSA)**: block routing + causal window + sink, exact rescoring only on candidate blocks
- **Dynamic Sparse Embedding**: 10.7× compression of N-gram embeddings (51B → 4.8B)
- **Long-Context Efficient**: Process 1M tokens with 99.995% less compute than full attention
- **Multimodal Omni**: Text + audio + vision joint training (3B config)
- **Layer Layout**: 3 ASM slot-memory layers to 1 ERSA exact-recall layer, repeated over 48 layers (32 for Omni)

### 3B Omni Model

| Component | Config |
|-----------|--------|
| d_model | 2048 |
| n_heads | 16 |
| n_layers | 32 (24 ASM-ERSA + 8 ERSA-Omni) |
| MoE experts | 32 (8 shared + 12 text + 6 audio + 6 vision) |
| Total params | 3.22B (measured 3,219,116,392 via `build_model(TrainConfig())`) |
| Training memory | ~61 GB (bf16, incl. optimizer states; not measured) |

### Vulkan Tensor System

ASM-ERSA is built on the **Vulkan Tensor Computing System** for efficient deployment:

- Cross-platform GPU inference (Windows / Linux / Android)
- Custom operator development (ASM slot read/write, ERSA recall, backward kernels, etc.)
- INT8/FP16 mixed-precision inference
- Memory-efficient streaming inference

**If you need mobile AI deployment or custom Vulkan tensor system development, please contact us for collaboration!**

---

## 📊 Performance

| Architecture | Attention FLOPs (1M context) | Relative to Full Attention | State / KV Cache (1M context, per sample) |
|--------------|----------------------|---------------------------|----------------------|
| Full Attention Transformer | 1.23 × 10¹⁶ ⚠️ | 1.0× | 16 GB ✓ |
| 同类混合注意力基线 | 5.92 × 10¹¹ ⚠️ | 4.8 × 10⁻⁵ ⚠️ | 2.5 TB ⚠️ |
| **ASM-ERSA (Ours)** | **5.96 × 10¹¹ ⚠️** | **4.8 × 10⁻⁵ ⚠️** | **96 MiB (fp32, 恒定) ✓** |

⚠️ = 该数字**未能从本仓库代码复现**，引用前请自行复核；✓ = 已复核（见下方说明）。

> **数据来源说明（2026-09-25 复核）**
>
> **已由代码确认的部分**：ASM-ERSA 的 state 大小是本仓库可实测的。ASM-ERSA的状态形状为
> `[B, n_scales=4, n_heads=16, d_v=128, d_k=128]`（`asm_ersa/asm_ersa.py:165` `initial_state`），
> fp32 即每层 4 MiB，3B 配置下 24 个 ASM-ERSA层共 **96 MiB，且与上下文长度无关**（实测输出
> `单层 state 形状 (1, 4, 16, 128, 128) float32 → 24 层共 96.00 MiB`）；8 个 ERSA 层的候选块
> （`block_size=64`、`top_blocks=4`、`window=64`）也均为有界缓存。
> **原表把 ASM-ERSA 的 KV cache 写作 2.5 TB，与架构直接矛盾**（那等于放弃了线性注意力的全部收益），
> 已按代码更正。全注意力一行的 16 GB 也可复核：1M × 2048 × 2(K,V) × 4 B(fp32) = 16.4 GB。
>
> **未能复核的部分**：FLOPs 与"相对全注意力"两列无法由代码推出。表中 1M 上下文下每 token
> 的注意力 FLOPs 上界为 `4·L·d`（48 层 × 4 × 10⁶ × 2048 ≈ 3.9 × 10¹¹），原值 `1.23 × 10¹⁶`
> 与该量级差约 3 万倍，疑为整条序列的总量口径；`5.92 × 10¹¹` 与 `5.96 × 10¹¹` 两个不同架构的
> 数字仅差 0.7%，来源不明。故这些数字**保留原值并标 ⚠️，未作替换**，待补充可复核的测量方法后再定。

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🎯 Quick Start

### Basic Usage

```python
import torch
from asm_ersa import ASM_ERSA

# Tiny config：CPU 上秒级跑完，便于验证前向/反向
model = ASM_ERSA(
    d_model=64, n_heads=4, d_k=16, d_v=16, d_ff=128,
    n_layers=4, vocab_size=256, n_experts=16, moe_top_k=4,
)

input_ids = torch.randint(0, 256, (2, 8))  # [batch, seq_len]
outputs = model(input_ids)
print(outputs.shape)
```

> 3B Omni 配置（`d_model=2048`、`n_layers=32`、32 专家）需要约 61 GB 训练内存（bf16 + 优化器状态），不适合在本机直接实例化。
> 随便验几步训练闭环可以用：`python train_omni_3b.py --smoke`（随机数据，不需要数据集/分词器）。

### Omni 3B Training

```bash
# Single node, 8 GPUs
torchrun --nproc_per_node=8 train_omni_3b.py \
    --data-path /data/omni_train \
    --output-dir /checkpoints/asm_ersa-omni-3b
```

## 🏗️ Architecture

ASM-ERSA consists of 48 layers with a 3:1 hybrid structure:

```
[ASM-ERSA] → [ASM-ERSA] → [ASM-ERSA] → [ERSA]  (repeat 12 times)
```

### ASM: Addressable Slot Memory

每个头维护 M 个槽位 `(κ_m, υ_m, β_m)`：

- **读**：`q` 与槽位 key 的相似度取 top-r，与可学习空槽位 ∅ 一起做 softmax，`o = Σ a_m υ_m`
- **写**：选中槽位上的凸组合（`η_t` 为 token 级标量），K/V/beta 按槽位衰减 `γ_m` 更新，K 重新归一化
- **衰减**：槽位级标量 `γ_m`，按 `n_scales` 组初始化（0.99–0.99999）

状态 `(K, V, beta)` 大小为 `B×H×M×(d_k+d_v+1)`，与序列长度无关。

### ERSA: Exact-Recall Sparse Attention

1. **块摘要路由**：块级摘要只负责选 top-B 候选块（不可微，梯度不回流 router）
2. **精确重打分**：候选块 ∪ 局部因果窗口用**原始 K/V** 精确打分（+ sink）
3. **输出**：RMSNorm

Python 参考实现为掩码后全序列精确打分；稀疏 gather 版见保密档案 `ASM-ERSA-Design/vk` 的 `ersa_attn_fused`（未入库）。

## 📚 Documentation

- [Architecture Details](docs/ASM_ERSA_Architecture.md)
- [License Agreement（中文）](LICENSE.txt) / [License Agreement (English)](LICENSE_EN.txt)

## 🔬 Research / 研究引用

If you use ASM-ERSA in your research, please cite:

```bibtex
@article{asm_ersa-2026,
  title={ASM-ERSA: Addressable Slot Memory + Exact-Recall Sparse Attention},
  author={NeuroArchitectonics},
  year={2026},
  url={https://github.com/NeuroArchitectonics/ASM-ERSA}
}
```

## 🤝 Contributing / 贡献

We welcome contributions! Please read our contributing guidelines before submitting PRs.

欢迎贡献代码！提交 PR 前请先阅读贡献指南。

## 📄 License / 许可

This project is licensed under the **ASM-ERSA Custom Community License Agreement V1.0** (not an OSI-approved open-source license) - see [LICENSE_EN.txt](LICENSE_EN.txt) for details. Individuals, academic institutions, and non-profit organizations may use it free of charge; commercial organizations whose consolidated revenue over any consecutive 12 months does not exceed RMB 5,000,000 may also use it free of charge, while those exceeding this threshold must obtain prior written commercial authorization.

本项目基于 **《ASM-ERSA 自定义社区许可协议》V1.0**（非 OSI 开源许可）。个人、学术机构、非营利组织免费使用；商业组织连续十二个月合并总营业额不超过 500 万元人民币可免费使用，超过阈值须事先取得书面商业授权。详见 [LICENSE.txt](LICENSE.txt)。

## 📧 Contact / 联系方式

- **Email**: 1393475440@qq.com
- **GitHub**: https://github.com/NeuroArchitectonics/ASM-ERSA
- **Vulkan 张量系统定制开发**：移动端 AI 部署合作

---

**Note**: This is a research project. Licensed under the ASM-ERSA Custom Community License V1.0 (not an OSI-approved open-source license).

**说明**：本项目为研究项目，基于《ASM-ERSA 自定义社区许可协议》V1.0（非 OSI 开源许可）。
