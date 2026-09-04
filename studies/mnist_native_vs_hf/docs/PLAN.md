# Study Plan: mnist_native_vs_hf

> 状态：已对照。`max_grad_norm: 0` + 共用 DataLoader 后 500/2000 逐点对齐。

## 1. 研究问题

在 **同一套算法超参**（SGD lr=0.1、cosine、20 epoch、MNIST linear）下，`algorithm.source: custom_torch`（native 循环）和 `transformers_trainer`（HuggingFace `Trainer`）的 test accuracy 差多少？

## 2. 变量轴

| 轴 | 字段 | 取值 |
|----|------|------|
| 执行实现 | `algorithm.source` | `custom_torch`, `transformers_trainer` |
| 训练样本量 | `data.config.train_size` | `500`, `2000`, `8000` |

每个格子 × seeds `0, 1, 2` = **18 次 train Run**。不做独立 eval 轴（本题比的是两种 train 实现）。

baseline：`custom_torch` × `train_size=500`。

## 3. 固定条件（与 `mnist_train_size` 的 train 合同对齐）

| 项 | 取值 |
|----|------|
| `algorithm.mode` | `train` |
| `num_epochs` | `20` |
| `progress_unit` | `epoch` |
| `eval_period` | `1` |
| `optimizer` | `SGD`（无 momentum） |
| `max_grad_norm` | `0`（不裁；关掉 HF Trainer 默认 1.0） |
| `lr` | `0.1` |
| `scheduler` | `cosine`，`eta_min: 0` |
| `batch_size` | `64` |
| `save_best` | `true` |

HF 侧：同一套键映射到 `TrainingArguments`；优化器 / cosine / **max_grad_norm** 走算法层。Train DataLoader 复用 native 的 shuffle Generator。

预期：同 seed、同 `max_grad_norm: 0` 时应对得很近（仍可能因 Trainer 内部 RNG 有极小差）。

## 4. 成功标准

- 18 Run `succeeded`
- `process.paired` 或按 source 分组的表能并排看 accuracy / best_accuracy
- `STUDY_REPORT.md` 有对照表 + learning curve

## 5. 刻意不做什么

- 不把独立 eval 做成 Flow 阶段
- 不做 TensorBoard
- 不为对齐数字去改 native 循环（clip 是共享超参；缺省仍不裁）
