# Study Plan: mnist_native_vs_hf

> 状态：2026-09-19 从空 Run 目录重跑。`max_grad_norm: 0` + 共用 DataLoader 后 500/2000 逐点对齐；8000 仍有约 0.0008 的 last accuracy mean 差。

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

## 5. 排班与执行

该 Study 全部是 `system.device: cpu`：`make` 不应探测或打印 GPU，生成的 job 不应包含 `gpu` / `CUDA_VISIBLE_DEVICES`。CPU Run 按默认并发上限分组。

```bash
python -m rpipe make studies/mnist_native_vs_hf
python -m rpipe launch studies/mnist_native_vs_hf --console shared
```

## 时长预估

launch 之前，按 make 的 conservative 秒数。同一 `wait` 里并行，组墙钟取该组最慢的一条；整轮是各组相加。不含显存。本 Study 在 CPU 上跑，预估表和 CUDA 用的是同一套毫秒/step。

| wait | factors | mode | seed | id | est |
|---:|---|---|---:|---|---:|
| 1 | source=custom_torch, train_size=500 | train | 0 | `7c48352c847b2096` | 5s |
| 1 | source=custom_torch, train_size=500 | train | 1 | `5662540f720e8797` | 5s |
| 1 | source=custom_torch, train_size=500 | train | 2 | `5bb24f8a4b518994` | 5s |
| 1 | source=custom_torch, train_size=2000 | train | 0 | `c96de76cbb910155` | 20s |
| 2 | source=custom_torch, train_size=2000 | train | 1 | `7d8d11439fd02689` | 20s |
| 2 | source=custom_torch, train_size=2000 | train | 2 | `71a8ba2c73de2c4f` | 20s |
| 2 | source=custom_torch, train_size=8000 | train | 0 | `e971cf5220878f98` | 1m15s |
| 2 | source=custom_torch, train_size=8000 | train | 1 | `a34e2f965d427746` | 1m15s |
| 3 | source=custom_torch, train_size=8000 | train | 2 | `8de79ab58f66b2bb` | 1m15s |
| 3 | source=transformers_trainer, train_size=500 | train | 0 | `6b3522139506555d` | 5s |
| 3 | source=transformers_trainer, train_size=500 | train | 1 | `7036783459b4dee3` | 5s |
| 3 | source=transformers_trainer, train_size=500 | train | 2 | `e006258b46e5e54c` | 5s |
| 4 | source=transformers_trainer, train_size=2000 | train | 0 | `529a5bf98413e51e` | 20s |
| 4 | source=transformers_trainer, train_size=2000 | train | 1 | `c5847d1f3f2da766` | 20s |
| 4 | source=transformers_trainer, train_size=2000 | train | 2 | `6ab372aeaa9bd64a` | 20s |
| 4 | source=transformers_trainer, train_size=8000 | train | 0 | `021d59955431eda9` | 1m15s |
| 5 | source=transformers_trainer, train_size=8000 | train | 1 | `38b06b38804eb60e` | 1m15s |
| 5 | source=transformers_trainer, train_size=8000 | train | 2 | `71be2847ca254e8e` | 1m15s |
| | 整轮 | | | | 5m20s |

## 6. 刻意不做什么

- 不把独立 eval 做成 Flow 阶段
- 不做 TensorBoard
- 不为对齐数字去改 native 循环（clip 是共享超参；缺省仍不裁）
