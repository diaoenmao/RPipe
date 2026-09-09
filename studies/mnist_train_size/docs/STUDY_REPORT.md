# Study Report: mnist_train_size

> Plan: [PLAN.md](PLAN.md)
> Date: 2026-09-10
> Recipe: native `custom_torch`；MNIST + linear；`num_epochs: 20`；`progress_unit: epoch`；SGD + cosine；`system.device: cuda`；axes = `train_size` × `{train, eval}`
> Location: Study `docs/`。产物在 `../index.json`、`../process.json`、`../runs/<id>/`。

## 1. 怎么做的实验

研究因素：`data.config.train_size ∈ {500, 2000, 8000}` × `algorithm.mode ∈ {train, eval}` × `seeds = [0, 1, 2]` → **6 个 Experiment、18 次 Run**。模型与预算固定（linear、20 epoch、SGD + cosine）。`train_size` 是训练集前 N 条（500 ⊂ 2000 ⊂ 8000）。独立 eval 加载同 size、同 seed 的 train `best.pt`。

**调度（这次实际执行）：**

```bash
python -m rpipe make studies/mnist_train_size --num-gpus 1 --init-gpu 0 --round 4
python -m rpipe launch studies/mnist_train_size --num-gpus 1 --init-gpu 0 --round 4 --include-done
```

机器：1 张 `NVIDIA GeForce RTX 5090 D v2`，全部 `CUDA_VISIBLE_DEVICES=0`。`--round 4` 表示同一时刻最多 4 个 `run-one` 进程（bash 里是 `&` + `wait`，Windows 上 `rpipe launch` 用 `Popen` 复现）。

波次：

1. **9 次 train**：4 + 4 + 1，每组 `wait` 后再开下一组。
2. **9 次 eval**：等 train 全部结束后才启动，同样 4 + 4 + 1。eval 才能读到 sibling `best.pt`。

2026-09-10 这次 launch 日志：先同时出现 4 个 train `+ gpu=0 …`，该组全部写出 `result.json` 后再打下一组；第 9 个 train 单独跑完后才出现 eval。eval 日志里 `resume_path` 指向对应 train 的 `best.pt`。墙钟约 **30 s**（格子里已有 `latest` 在 epoch 20，train 是续跑收口；数字与下面表格一致）。`process.complete: true`。

脚本形状见 [STUDY_GUIDE.md](../../../docs/STUDY_GUIDE.md) §3；本 Study 的 `scripts/launch.sh` 默认不入库。

## 2. Conclusion

18 次 Run 全部 `succeeded`（9 train + 9 独立 eval）。

独立 eval 加载 sibling train 的 `best.pt`，口径与 train 的 `best_accuracy` **一致**（不是 last-segment `accuracy`）。比大小仍按样本量：

| train_size | train last acc | train best | **eval (best.pt)** | vs eval 500 | train_loss |
|------------|----------------|------------|--------------------|-------------|------------|
| 500（baseline train） | 0.834 | 0.836 | **0.836** | — | 0.080 |
| 2000 | 0.883 | 0.884 | **0.884** | +0.049 | 0.123 |
| 8000 | 0.906 | 0.907 | **0.907** | +0.072 | 0.161 |

结论：数据越多 test 越好；`train_loss` 随样本量上升是过拟合，不是 bug。最终口径用 **eval** 列。

## 3. Learning curves

只画 **train** Run（eval 没有 epoch 曲线）。`progress_unit: epoch` + `eval_period: 1`。process 出图。

![learning curves](./figures/learning_curves.png)

[打开 learning_curves.png](./figures/learning_curves.png)

读图：500 过拟合；8000 一上来就在 0.89 附近。

## 4. Train Runs

数字来自 `process.json`（cuda，3 seed mean 见上表）。

| train_size | seed | tags | Run id | accuracy | best_accuracy | train_loss |
|------------|------|------|--------|----------|---------------|------------|
| 500 | 0 | baseline | 8176170127f91cf0 | 0.8328 | 0.8346 | 0.080 |
| 500 | 1 | baseline | fc81e18a5207a45d | 0.8341 | 0.8344 | 0.079 |
| 500 | 2 | baseline | 442977f37212ef68 | 0.8359 | 0.8375 | 0.080 |
| 2000 | 0 | — | d48fd72e60c7f6ac | 0.8818 | 0.8841 | 0.123 |
| 2000 | 1 | — | bb1c4bab027af782 | 0.8828 | 0.8835 | 0.124 |
| 2000 | 2 | — | 31c40cec0f6d9cd5 | 0.8846 | 0.8851 | 0.123 |
| 8000 | 0 | — | ae8cf11290ed6b54 | 0.9065 | 0.9070 | 0.161 |
| 8000 | 1 | — | 8217ff232db48b12 | 0.9056 | 0.9072 | 0.161 |
| 8000 | 2 | — | 778bd5b434d18762 | 0.9062 | 0.9078 | 0.161 |

## 5. Eval Runs（独立 Algorithm，加载 sibling `best.pt`）

| train_size | seed | Run id | eval_accuracy | matches train best |
|------------|------|--------|---------------|--------------------|
| 500 | 0 | e957d8a364952d1f | 0.8346 | 0.8346 |
| 500 | 1 | dff1613a7bc1fd5f | 0.8344 | 0.8344 |
| 500 | 2 | 56bc9bd79216fbba | 0.8375 | 0.8375 |
| 2000 | 0 | 478225eadad22061 | 0.8841 | 0.8841 |
| 2000 | 1 | b42b4fd9d969a3d6 | 0.8835 | 0.8835 |
| 2000 | 2 | 02214314ba92307e | 0.8851 | 0.8851 |
| 8000 | 0 | 92abeedb98737917 | 0.9070 | 0.9070 |
| 8000 | 1 | 9d9e6347dee60c86 | 0.9072 | 0.9072 |
| 8000 | 2 | a83a100692ba2930 | 0.9078 | 0.9078 |

数字来自 `process.paired`。eval 不是 Flow 第二阶段。

## 6. 这次钉死的合同

- **native 循环**不再看 `data.name == MNIST`；有 `module` + `iter_batches` 就训。线性层才 flatten。
- **独立 eval Run**：`algorithm.mode: eval`，默认 resume `best`；本 Run 没有文件则按 index 找同 seed、同 `train_size` 的 train `best.pt`。
- **并行**：make / launch 用 `--round` 与 `&`/`wait`；eval 单独第二波。
- **HF Trainer**：`algorithm.source: transformers_trainer` 把同一套 `optimizer` / `scheduler` / `resume` 映射到 `TrainingArguments`（optional extra `nlp`）。本 Study 仍是 `custom_torch`。
- **不做 TensorBoard**。

## 7. Reproduce

从头训（跳过已 succeeded 的格子）：

```bash
python -m rpipe make studies/mnist_train_size --num-gpus 1 --init-gpu 0 --round 4
python -m rpipe launch studies/mnist_train_size --num-gpus 1 --init-gpu 0 --round 4
```

把调度再走一遍（含已完成 Run）：加上 `--include-done`。
