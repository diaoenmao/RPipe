# Study Plan: mnist_train_size

> 状态：计划（执行前契约）。跑完后结果见同目录 `STUDY_REPORT.md`。

## 1. 研究问题

在固定模型与训练预算下，**MNIST 训练集样本量**如何影响 **测试集准确率**（及训练 loss）？独立 `mode=eval` Run 用 train 的 `best.pt` 再评一次，作为最终口径。

## 2. Study / Experiment

本 Study 名叫 `mnist_train_size`。`experiment_config` 里的 `experiment: mnist_linear` 只是基底名字（给人看、进 config），**不是**磁盘上的 Experiment 目录。

真正的 Experiment 由 `axes` 展开：三个 `train_size` × `{train, eval}` = **六个 Experiment**；每个再 × 3 个 seed = 18 次 Run。index 按 `factors` 分组。`process.paired` 把同一 `train_size` 的 train / eval 拼回一行，供报告表格。

## 3. 变量轴（有意变化）

| 轴 | 字段 | 取值 |
|----|------|------|
| 训练样本量 | `data.config.train_size` | `500`, `2000`, `8000` |
| 算法 mode | `algorithm.mode` | `train`, `eval` |

其余固定（见下）。两轴都进入 Config → **参与 Run `id` hash**。yaml 里 `train_size` 在前、`mode` 在后，因此每个 size 先跑完 3 个 train，再跑 3 个 eval（eval 才能找到 sibling `best.pt`）。

**子集口径（嵌套前 N 条）：** `train_size` 取训练集编号 `0 .. N-1`，因此 **500 ⊂ 2000 ⊂ 8000**。加大样本量是「多给前面那些图」，不是每个格子重新抽一袋。`seed` 只影响初始化与 DataLoader shuffle，不换图。

## 4. 固定条件

| 项 | 取值 |
|----|------|
| `seeds` | `0, 1, 2` |
| `data.name` | `MNIST` |
| `data.source` | `torch`（缓存到 `shared/data/`） |
| `model.name` | `linear`（784→10） |
| `algorithm.source` | `custom_torch`（HF Trainer 走同一套键，本 Study 不切 source） |
| `algorithm.num_epochs` | `20`（eval Run 忽略预算，只 resume + 评 test） |
| `algorithm.progress_unit` | `epoch` |
| `algorithm.eval_period` | `1`（仅 train 循环内 hook） |
| `algorithm.checkpoint` | `latest` |
| `algorithm.checkpoint_period` | `1` |
| `algorithm.save_best` | `true` |
| `algorithm.resume` | 不写：train 默认 `latest`（无文件从头）；eval 默认 `best`（从 sibling train Run 读） |
| `algorithm.optimizer` | `SGD` |
| `algorithm.max_grad_norm` | `0`（不裁；算法层超参，HF 必须写 0 才能关默认 1.0） |
| `data.config.batch_size` | `64` |
| `algorithm.lr` | `0.1` |
| `algorithm.scheduler` | `cosine` |
| `algorithm.eta_min` | `0.0` |
| `system.device` | `cpu` |
| `system.deterministic` | `false` |
| `system.cudnn_benchmark` | `true` |
| 测试集 | 完整 MNIST test |

## 5. Tags

| Run | tags |
|-----|------|
| `train_size=500` 且 `mode=train` | `baseline` |
| 其余 | （无） |

## 6. 编排顺序（对齐 CONCEPT §9）

1. 写 / 确认 Study 根 `experiment_config.yaml`
2. `python -m rpipe run studies/mnist_train_size` → config + index + Flow
3. 读 `process.json`（含 `paired`）、`docs/figures/learning_curves.png` 写 `STUDY_REPORT.md`（必须有 train 表 + eval 表；Flow 不改 markdown）

## 7. 成功标准

- 6 Experiment × 3 seed = 18 次 Run 均 `status: succeeded`
- train Result：`train_loss`、`accuracy`、`best_accuracy`；`latest.pt` 与 `best.pt`
- eval Result：`accuracy` 与 `eval_accuracy`（独立评测，加载 sibling `best.pt`）
- index 按 `train_size` + `mode` 分组；仅 500×train 带 `baseline`
- `STUDY_REPORT.md` 嵌 learning curve，并列出独立 eval 表
- learning curve 只画 train Run（eval 没有 epoch 曲线）

## 8. 刻意不做什么

- 不把独立 eval 做成 Flow 第二阶段
- 本 Study 不切换 `algorithm.source: transformers_trainer`
- 不做 TensorBoard
- 不引入独立 baseline 对象 / Findings / MCP
