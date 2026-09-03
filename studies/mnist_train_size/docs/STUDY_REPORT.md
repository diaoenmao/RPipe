# Study Report: mnist_train_size

> Plan: [PLAN.md](PLAN.md)
> Date: 2026-09-04
> Recipe: MNIST + linear，`num_epochs: 20` 推导步数，`progress_unit: epoch`，`optimizer: SGD` lr=0.1，cosine（`T_max=20`，`eta_min=0`），`resume: latest`，`eval_period: 1`，`checkpoint: latest` + `save_best: true`
> Location: Study `docs/`（读 `process.json` / result 后写）。产物在 `../index.json`、`../process.json`、`../runs/<id>/`。

## 1. Conclusion

9 次 Run 全部 `succeeded`。`process.complete: true`。三个样本量仍分开：

| train_size | mean accuracy | std | vs baseline 500 | mean best_accuracy | mean train_loss |
|------------|---------------|-----|-----------------|--------------------|-----------------|
| 500（baseline） | 0.834 | 0.002 | — | 0.836 | 0.080 |
| 2000 | 0.883 | 0.001 | +0.049 | 0.884 | 0.123 |
| 8000 | 0.906 | 0.000 | +0.072 | 0.907 | 0.161 |

`accuracy` 是最后一段 test；`best_accuracy` 是过往最好 test（`best.pt`）。两者差约 0.001，结论不变。`train_loss` 随样本量上升仍是过拟合，不是 bug。比大小以 **accuracy** 为准。

预算：`num_epochs: 20` 覆盖步数。`ceil(train_size / 64) * 20` → 500=160、2000=640、8000=2500 step。周期按 epoch，不是每步评 test。

这次把文档里的算法接口落到 native loop：`make_optimizer` / `make_scheduler`、`resume`（缺 latest 则从头）、独立 `mode=eval` 走 `best`。数值与上一轮（同样 SGD+cosine）一致，hash 因 yaml 显式写了 `optimizer` / `resume` 而更新。

## 2. Learning curves

`progress_unit: epoch` + `eval_period: 1`：每个 epoch 训完走一次 `on_eval_period`，评完整 test，并覆盖 `latest.pt`；test Accuracy 创新高时写 `best.pt`。下图由 **process** 从各 Run `tracker_state.json` 的 `history` 画出（3 seed mean±std；密点仍在 `scalars.jsonl`）。不做 TensorBoard。重画：`python studies/mnist_train_size/docs/plot_curves.py`。

![learning curves](./figures/learning_curves.png)

[打开 learning_curves.png](./figures/learning_curves.png)

读图：

- **test acc** 三条分开，大约 epoch 6–10 到平台；8000 一上来就在 0.89 附近。
- **500 train acc → 1.0**，test 停在 ~0.83：过拟合。2000 / 8000 的 train–test 缝更小。
- **test loss** 和 acc 同序；500 的带子最宽。
- **train loss** 样本量越小降得越狠，所以最后一段 train mean 不能当「越大越好」。

## 3. Runs

| train_size | seed | tags | Run id | status | accuracy | best_accuracy | train_loss |
|------------|------|------|--------|--------|----------|---------------|------------|
| 500 | 0 | baseline | f61860256fcaaaab | succeeded | 0.8329 | 0.8346 | 0.080 |
| 500 | 1 | baseline | 12b6927f005ec604 | succeeded | 0.8343 | 0.8347 | 0.079 |
| 500 | 2 | baseline | 465a8f85039fb6c4 | succeeded | 0.8359 | 0.8375 | 0.080 |
| 2000 | 0 | — | aa38cd8edc39ff8e | succeeded | 0.8819 | 0.8841 | 0.123 |
| 2000 | 1 | — | ac6ebe86536786ae | succeeded | 0.8827 | 0.8835 | 0.124 |
| 2000 | 2 | — | 9be3d4b1f8a66ef6 | succeeded | 0.8845 | 0.8851 | 0.123 |
| 8000 | 0 | — | aa46a8f2daf0311f | succeeded | 0.9065 | 0.9069 | 0.161 |
| 8000 | 1 | — | dc28d50579b31bce | succeeded | 0.9056 | 0.9072 | 0.161 |
| 8000 | 2 | — | 071c2da9901c5513 | succeeded | 0.9062 | 0.9078 | 0.161 |

每条 Run 有 `assets/checkpoints/latest.pt` 与 `best.pt`。日志开头有 `resume skip (no latest)`，说明 train 默认走 resume 接口、缺文件则从头。config 含 `progress_unit: epoch`、`optimizer: SGD`、`resume: latest`、`save_best: true`（进 `id` hash）。日志里 lr 从 0.1 降到约 6e-4。

## 4. 这次钉死的合同 / 注意

- **预算**：`num_steps` 是基础单位；本 Study 用 `num_epochs` 从 `train_size` / `batch_size` 推导并覆盖步数。
- **周期**：显式 `progress_unit: epoch`。库默认是 `step`，若只写 `eval_period: 1` 会每步评完整 test。
- **optimizer / scheduler**：算法层接口；本 Study `SGD` + `cosine`。system 不建优化器。
- **resume**：train `resume: latest`；没有文件则从头。同一 hash 再跑会加载 `latest.pt`，预算已满则只再评 test。独立评测是另一次 `mode: eval`（默认 `best`），不是 train 循环里的 hook。
- **checkpoint**：默认只覆盖 latest；本 Study `save_best: true` 另存过往最好。`percent` 未开。
- **`result.accuracy` 仍是最后一次 hook**；最好 test 现在在 `metrics.best_accuracy` 和 `best.pt`。
- **`train_loss` 不能当「越大越好」的轴。** 样本量越小越容易把训练集拟合掉。
- **不做 TensorBoard**；图从 history / jsonl 出。

## 5. Reproduce

```bash
python -m rpipe run studies/mnist_train_size
```

`algorithm`：`num_epochs: 20`，`progress_unit: epoch`，`eval_period: 1`，`checkpoint: latest`，`checkpoint_period: 1`，`save_best: true`，`resume: latest`，`optimizer: SGD`，`scheduler: cosine`，`eta_min: 0.0`，`lr: 0.1`。`system`：`deterministic: false`，`cudnn_benchmark: true`。
