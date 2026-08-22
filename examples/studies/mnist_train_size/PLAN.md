# Study Plan: mnist_train_size

> 状态：计划（执行前契约）。跑完后结果见同目录 `RESULTS.md`。

## 1. 研究问题

在固定模型与训练预算下，**MNIST 训练集样本量**如何影响 **测试集准确率**（及训练 loss）？

## 2. Study / Experiment

| 项 | 取值 |
|----|------|
| Study | `mnist_train_size` |
| Experiment | `mnist_linear`（在本 Study 下扩展真实训练能力） |
| 描述 | Sweep MNIST train subset size → test accuracy |

## 3. 变量轴（有意变化）

| 轴 | 字段 | 取值 |
|----|------|------|
| 训练样本量 | `data.config.train_size` | `500`, `2000`, `8000` |

其余固定（见下）。样本量进入 Config → **参与 Run `id` hash**。

## 4. 固定条件

| 项 | 取值 |
|----|------|
| `seed` | `0` |
| `data.name` | `MNIST` |
| `model.name` | `linear`（784→10） |
| `algorithm.mode` | `train`（同一次 Run 内训完并在 test 上评估） |
| `algorithm.num_epochs` | `2` |
| `algorithm.batch_size` | `64` |
| `algorithm.lr` | `0.1` |
| `system.device` | `cpu` |
| 测试集 | 完整 MNIST test（或固定子集，实现里写明） |

## 5. Tags

| Run | tags |
|-----|------|
| `train_size=500` | `baseline`（最小数据量作为对照） |
| 其余 | （无，或后续可加 `sweep`） |

## 6. 编排顺序（对齐 CONCEPT §5.6）

1. 写 / 确认 Experiment 基底 `experiment_config.yaml`（真实训练默认）
2. Study `grid` 按 `train_size` 展开 → 各 Run Artifact `config.yaml`（含 `description` / `tags`）
3. 写 Study `index.json`（**launch 之前**）
4. `launch` 跑 Flow：prepare → execute → collect → summarize → index
5. 汇总 Result → `RESULTS.md`（含卡点）

## 7. 成功标准

- 三次 Run 均 `status: succeeded`
- 各 Result 含真实 `loss` 与 `accuracy`（非 stub 0.0 占位）
- Study `index.json` 能指向三次 Config；带 `baseline` 的 Run 可识别
- 本文流程走通；**卡点记入 RESULTS.md**，用于反哺壳子设计

## 8. 刻意不做什么

- 不引入独立 baseline 对象 / Findings / MCP
- 不扩成多 seed 方差分析（可后续 Study）
- 不为「好看」加 UI
