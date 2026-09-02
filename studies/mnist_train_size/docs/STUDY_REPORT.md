# Study Report: mnist_train_size

> Plan: [PLAN.md](PLAN.md)
> Date: 2026-09-02
> Recipe: MNIST + linear，20 epoch，cosine（`T_max=20`，`eta_min=0`），SGD lr=0.1，`eval_period: 1`
> Location: Study `docs/`（读 `process.json` / result 后写）。产物在 `../index.json`、`../process.json`、`../runs/<id>/`。

## 1. Conclusion

9 次 Run 全部 `succeeded`。`process.complete: true`。20 epoch + cosine 之后，三个样本量**不再重叠**：

| train_size | mean accuracy | std | vs baseline 500 | mean train_loss |
|------------|---------------|-----|-----------------|-----------------|
| 500（baseline） | 0.834 | 0.002 | — | 0.080 |
| 2000 | 0.883 | 0.001 | +0.049 | 0.123 |
| 8000 | 0.906 | 0.000 | +0.072 | 0.161 |

seed 方差已经很小。`train_loss` 随样本量**上升**：500 训到 train acc≈1.0（过拟合），8000 还在 0.16。比大小仍以 **accuracy** 为准。

## 2. Learning curves

`eval_period: 1`：每个 epoch 训完走一次 `on_eval_period`，评完整 test。下图由 **process** 从各 Run `tracker_state.json` 的 `history` 画出（3 seed mean±std；密点仍在 `scalars.jsonl`）。以后每轮 Study 报告都要嵌这类图，不能只有表。重画：`python studies/mnist_train_size/docs/plot_curves.py`。

![learning curves](figures/learning_curves.png)

读图：

- **test acc** 三条分开，大约 epoch 6–10 就到平台；8000 一上来就在 0.89 附近。
- **500 train acc → 1.0**，test 停在 ~0.83：过拟合。2000 / 8000 的 train–test 缝更小。
- **test loss** 和 acc 同序；500 的带子最宽。
- **train loss** 样本量越小降得越狠，所以最后一段 train mean 不能当「越大越好」。

## 3. Runs

| train_size | seed | tags | Run id | status | accuracy | train_loss |
|------------|------|------|--------|--------|----------|------------|
| 500 | 0 | baseline | 971b9a9e29d8a0a8 | succeeded | 0.8329 | 0.080 |
| 500 | 1 | baseline | e8a938db5fd01f2c | succeeded | 0.8343 | 0.079 |
| 500 | 2 | baseline | 4efda7a0ef6326f9 | succeeded | 0.8359 | 0.080 |
| 2000 | 0 | — | df8522ff324c59f5 | succeeded | 0.8819 | 0.123 |
| 2000 | 1 | — | 3c8752e61e05371b | succeeded | 0.8827 | 0.124 |
| 2000 | 2 | — | a44954c8197b24d0 | succeeded | 0.8845 | 0.123 |
| 8000 | 0 | — | 1bdd13de9d21ffd0 | succeeded | 0.9065 | 0.161 |
| 8000 | 1 | — | 088ef91a47d936de | succeeded | 0.9056 | 0.161 |
| 8000 | 2 | — | 5d1f3dbbdebec565 | succeeded | 0.9062 | 0.161 |

config 含 `num_epochs: 20`、`scheduler: cosine`、`eta_min: 0.0`（进 `id` hash）。日志里 lr 从 0.1 降到约 6e-4。

## 4. 这次跑出来的问题 / 注意

**还在的**

- **`result.accuracy` 仍是最后一次 hook。** 曲线上 500 大约 epoch 12 到顶（~0.835），后面 test 几乎不动。要比「最好 test」仍得另记。
- **`train_loss` 不能当「越大越好」的轴。** 样本量越小越容易把训练集拟合掉，最后一段 train mean 反而更低。process 里 2000 / 8000 相对 500 的 train_loss Δ 是正的（+0.043 / +0.081）。

**已经收掉的**

- **2 epoch 太短、中等样本量叠格子。** 20 epoch + cosine 后 500 / 2000 / 8000 分开，std 都在 0.002 以内。
- **scheduler 只写死 SGD。** `algorithm.scheduler: cosine` → `CosineAnnealingLR`（`T_max=num_epochs`）；`constant` / 缺省 = 固定 lr。
- **同 seed 重跑会漂。** 上一轮已绑 DataLoader generator；本轮未再复跑。
- **runtime 开关 / hook / process。** 仍按上一轮合同。

## 5. Reproduce

```bash
python -m rpipe run studies/mnist_train_size
```

`algorithm`：`num_epochs: 20`，`scheduler: cosine`，`eta_min: 0.0`，`lr: 0.1`，`eval_period: 1`。`system`：`deterministic: false`，`cudnn_benchmark: true`。
