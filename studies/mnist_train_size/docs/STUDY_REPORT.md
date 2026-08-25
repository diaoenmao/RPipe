# Study Report: mnist_train_size

> Plan: [PLAN.md](PLAN.md)
> Date: 2026-08-25
> Recipe: MNIST + linear，2 epoch，SGD lr=0.1
> Location: Study `docs/`（人写）。产物在 `../index.json`、`../runs/<id>/result.json`、`../shared/data/`。

## 1. Conclusion

每个 Experiment（`train_size`）跑 3 个 seed（0/1/2），共 9 次 Run，全部 `succeeded`。按 Experiment 聚合后，测试准确率随样本量上升：

| train_size | mean accuracy | std | vs baseline 500 |
|------------|---------------|-----|-----------------|
| 500（baseline） | 0.790 | 0.018 | — |
| 2000 | 0.847 | 0.023 | +0.057 |
| 8000 | 0.895 | 0.002 | +0.105 |

方向与单 seed 时一致；中等样本量的 seed 方差更大（2000 的 seed=2 只有 0.817，接近 500 的最好一次）。

## 2. Runs

| train_size | seed | tags | Run id | status | test accuracy | last-batch loss |
|------------|------|------|--------|--------|---------------|-----------------|
| 500 | 0 | baseline | 228ece2a968b8f05 | succeeded | 0.7642 | 0.415 |
| 500 | 1 | baseline | b648203b55dcf487 | succeeded | 0.7980 | 0.263 |
| 500 | 2 | baseline | 3793d088ea40b881 | succeeded | 0.8066 | 0.448 |
| 2000 | 0 | — | 4c82d5b4d0a82b82 | succeeded | 0.8495 | 0.384 |
| 2000 | 1 | — | 05a4f03d1481d38d | succeeded | 0.8734 | 0.120 |
| 2000 | 2 | — | 283afa30aa4d1835 | succeeded | 0.8171 | 0.545 |
| 8000 | 0 | — | da2d5c6a9c84db58 | succeeded | 0.8928 | 0.275 |
| 8000 | 1 | — | 7bcaa94659790d61 | succeeded | 0.8980 | 0.424 |
| 8000 | 2 | — | 4cedd36f86e6fbb6 | succeeded | 0.8947 | 0.224 |

## 3. 这次跑出来的问题 / 注意

- **`metrics.loss` 是最后一个 batch 的 CE，不是 epoch 均值。** 同一 `train_size` 下 loss 可差到 0.12–0.55，不能当比较轴。比大小请看 `accuracy`（全 test set）。
- **`train_size` 子集是前 N 条，不是按 seed 再抽样。** seed 只影响权重初始化和 DataLoader shuffle，不改变「哪 500 张图」。若研究的是数据抽样方差，现在的 control 还没覆盖到。
- **3 seed 时中等样本量会和相邻格子重叠。** 单看 seed=2：2000 (0.817) ≈ 500 最好的一次 (0.807)。结论要写 mean±std，不要只报 seed=0。
- **`process` 仍是空的。** index 已按 Experiment 分组，但均值 / Δ baseline 是人在本报告里算的，Flow 不会写聚合表。
- 三次 `train_size` 共用 `shared/data/mnist`（本机这次因 `shared/` 未入库，开头重新下了一次数据；之后的 Run 走同一缓存）。

## 4. Reproduce

```bash
python -m rpipe run studies/mnist_train_size
```

`study.yaml`：`axes.train_size = [500, 2000, 8000]`，`seeds = [0, 1, 2]`。
