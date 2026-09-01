# Study Report: mnist_train_size

> Plan: [PLAN.md](PLAN.md)
> Date: 2026-09-02
> Recipe: MNIST + linear，2 epoch，`eval_period: 1`，SGD lr=0.1
> Location: Study `docs/`（读 `process.json` / result 后写）。产物在 `../index.json`、`../process.json`、`../runs/<id>/`。

## 1. Conclusion

9 次 Run 全部 `succeeded`。`process.complete: true`。`metrics.train_loss` 是 AlgorithmTracker **最后一段 train mean**；`accuracy` 是最后一次 `on_eval_period` 的 test mean（本 Study 每 epoch 评一次，所以是 epoch 2）。按 Experiment 聚合后，测试准确率随样本量上升，训练 loss 下降：

| train_size | mean accuracy | std | vs baseline 500 | mean train_loss |
|------------|---------------|-----|-----------------|-----------------|
| 500（baseline） | 0.793 | 0.009 | — | 0.474 |
| 2000 | 0.842 | 0.033 | +0.049 | 0.386 |
| 8000 | 0.889 | 0.007 | +0.096 | 0.312 |

中等样本量 seed 方差仍大；结论写 mean±std。

## 2. Runs

| train_size | seed | tags | Run id | status | accuracy | train_loss |
|------------|------|------|--------|--------|----------|------------|
| 500 | 0 | baseline | 6c8710bcc61d7f2b | succeeded | 0.7872 | 0.481 |
| 500 | 1 | baseline | 3f9177028b932f45 | succeeded | 0.7892 | 0.443 |
| 500 | 2 | baseline | 78686bef8a1535f7 | succeeded | 0.8029 | 0.498 |
| 2000 | 0 | — | e559aa5210bcf7e4 | succeeded | 0.8550 | 0.399 |
| 2000 | 1 | — | c3613618c6d97163 | succeeded | 0.8039 | 0.385 |
| 2000 | 2 | — | 12dbabca8e550d18 | succeeded | 0.8663 | 0.373 |
| 8000 | 0 | — | 87413b8d655d6726 | succeeded | 0.8968 | 0.312 |
| 8000 | 1 | — | ac5fbebe86cd9620 | succeeded | 0.8833 | 0.314 |
| 8000 | 2 | — | 7dbe0d26ca04d265 | succeeded | 0.8877 | 0.308 |

每条 Run 均有 `assets/logs/run.log`（含 `train` / `test` 行）和 `assets/tracker/`（`tracker_state.json` + `scalars.jsonl`）。config 含 `algorithm.eval_period: 1`，因此 Run `id` 与上一轮（未写该字段）不同。

## 3. 这次跑出来的问题 / 注意

**还在的**

- **中等样本量仍和相邻格子重叠。** 2000 seed=1 accuracy 0.804，同组另外两次 0.855 / 0.866；也低于 500 seed=2 的 0.803。嵌套子集口径见 [PLAN.md](PLAN.md)。结论写 mean±std。
- **`result.accuracy` 只取最后一次 hook 的 test。** hook 每个 epoch 都评了完整 test，但进 `result.json` 的是最后一段。2000 seed=1：epoch 1 test 0.850 → epoch 2 0.804；8000 seed=1：0.893 → 0.883。曲线在 jsonl，摘要看不到中间更好的点。不是 Flow 问题；若要比「最好 test」需要另记，或以后接 `on_checkpoint`。
- **同 seed 重跑数字会漂。** 与 2026-09-01 同配方（未改 epoch / lr / 子集）比，500 mean acc 从 0.795 到 0.793，2000 从 0.861 到 0.842。DataLoader shuffle 尚未绑到 Run seed，初始化以外的随机源不稳。

**已经收掉的**

- **algorithm hook 合同。** 包内 `Algorithm.on_eval_period`（基类 no-op；`TrainAlgorithm` 评 test + 可选 early stop）。循环用 `eval_period` / `due_eval_period` 点名调用。本 Study 写明 `eval_period: 1`。不是新的 Flow 阶段。
- **`NameError: self`（epoch 末）。** 已传入 `TrainAlgorithm`。
- **`train_loss` 口径。** 最后一段 train mean。
- **同 id 重跑叠日志。** prepare 时截断 `run.log` 和 `scalars.jsonl`。
- **`process` 空操作。** 按 Experiment 写 mean / std / Δ；本次 `complete: true`，2000 / 8000 相对 500 的 accuracy Δ 约 +0.049 / +0.096。不改各 Run `result.json` 正文。

## 4. Reproduce

```bash
python -m rpipe run studies/mnist_train_size
```

`study.yaml`：`axes.train_size = [500, 2000, 8000]`，`seeds = [0, 1, 2]`。`experiment_config.yaml` 现含 `data.source: torch`，`num_epochs: 2`，`eval_period: 1`。
