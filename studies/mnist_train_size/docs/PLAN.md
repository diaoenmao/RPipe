# Study Plan: mnist_train_size

> 状态：计划（执行前契约）。跑完后结果见同目录 `STUDY_REPORT.md`。

## 1. 研究问题

在固定模型与训练预算下，**MNIST 训练集样本量**如何影响 **测试集准确率**（及训练 loss）？

## 2. Study / Experiment

本 Study 名叫 `mnist_train_size`。`experiment_config` 里的 `experiment: mnist_linear` 只是基底名字（给人看、进 config），**不是**磁盘上的 Experiment 目录。

真正的 Experiment 由 `axes` 展开：三个 `train_size` 取值 = **三个 Experiment**；每个再 × 3 个 seed = 9 次 Run。index 按 `factors` 分组。

## 3. 变量轴（有意变化）

| 轴 | 字段 | 取值 |
|----|------|------|
| 训练样本量 | `data.config.train_size` | `500`, `2000`, `8000` |

其余固定（见下）。样本量进入 Config → **参与 Run `id` hash**。

**子集口径（嵌套前 N 条）：** `train_size` 取训练集编号 `0 .. N-1`，因此 **500 ⊂ 2000 ⊂ 8000**。加大样本量是「多给前面那些图」，不是每个格子重新抽一袋。`seed` 只影响初始化与 DataLoader shuffle，不换图。格子之间比的是「同一批图变多了」加上随机性，不是独立抽样方差。

## 4. 固定条件

| 项 | 取值 |
|----|------|
| `seeds` | `0, 1, 2`（每个 Experiment 三次 Run） |
| `data.name` | `MNIST` |
| `data.source` | `torch`（缓存到 `shared/data/`） |
| `model.name` | `linear`（784→10） |
| `algorithm.mode` | `train`（循环内 algorithm hook，不是第二个 Flow mode） |
| `algorithm.num_epochs` | `20` |
| `algorithm.eval_period` | `1`（每个 epoch 末 `on_eval_period` 评完整 test；`0` = 只在训完评一次） |
| `data.config.batch_size` | `64` |
| `algorithm.lr` | `0.1`（SGD 初始 lr） |
| `algorithm.scheduler` | `cosine`（`CosineAnnealingLR`，`T_max=num_epochs`） |
| `algorithm.eta_min` | `0.0` |
| `system.device` | `cpu` |
| `system.deterministic` | `false`（写在 `experiment_config` / `study.yaml` 的 `system`；prepare 最先落地） |
| `system.cudnn_benchmark` | `true`（跟 main；`deterministic: true` 时会关掉） |
| 测试集 | 完整 MNIST test（或固定子集，实现里写明） |

## 5. Tags

| Run | tags |
|-----|------|
| `train_size=500` | `baseline`（最小数据量作为对照） |
| 其余 | （无，或后续可加 `sweep`） |

## 6. 编排顺序（对齐 CONCEPT §9）

1. 写 / 确认 Study 根 `experiment_config.yaml`（真实训练默认）
2. `python -m rpipe run studies/mnist_train_size` → config + index + Flow
3. 读 `process.json`、`docs/figures/learning_curves.png`（及各 Run `result.json`）写 `STUDY_REPORT.md`（人 / agent；必须嵌图；Flow 不改 markdown）

## 7. 成功标准

- 3 Experiment × 3 seed = 9 次 Run 均 `status: succeeded`
- 各 Result 含 `metrics.train_loss`（AlgorithmTracker 最后一段 train mean）与 `accuracy`（全 test）
- 每条 Run 有 `assets/logs/run.log`（含 Loss）和 `assets/tracker/`（state / jsonl）
- index 按 `train_size` 分组，每组 3 条 Run；`train_size=500` 带 `baseline`
- `STUDY_REPORT.md` 嵌 learning curve（`docs/figures/`），不能只有表格

## 8. 刻意不做什么

- 不引入独立 baseline 对象 / Findings / MCP
- 不为「好看」加 UI
