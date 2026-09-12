# Study Report: mnist_train_size

> Plan: [PLAN.md](PLAN.md)
> Date: 2026-09-13
> Recipe: native `custom_torch`；MNIST + linear；20 epoch；SGD + cosine；cuda。axes = `train_size` × `{train, eval}`。数字读 `../process.json`。排班标准 [STUDY_GUIDE.md](../../../docs/STUDY_GUIDE.md) §3。

## 1. 怎么跑的

研究因素：`train_size ∈ {500, 2000, 8000}` × `{train, eval}` × seeds `0,1,2` → **6 Experiment、18 Run**。`train_size` 是前 N 条（500 ⊂ 2000 ⊂ 8000）。eval 加载 sibling `best.pt`。基底不写 `resume`：train 默认 `latest`，eval 默认 `best`。metric 用默认 Loss + Accuracy。

```bash
python -m rpipe make studies/mnist_train_size --num-gpus 1 --init-gpu 0
python -m rpipe launch studies/mnist_train_size --num-gpus 1 --init-gpu 0 --console shared
```

机器：1× RTX 5090 D v2。make 打印：

`pack 2 waits: 9[linear×9], 9[linear×9] est wall 1m34s` — 先 9 次 train 同一组，`wait` 完再 9 次 eval。实测整轮约 48s；conservative 墙钟只供排班。本 Study 很短，用 `--console shared`。

18 次全部一次 `succeeded`，没有 retry。eval 日志里的 Accuracy 与 `result.json` 一致（不是 0）。

## 2. Conclusion

独立 eval 与 train `best_accuracy` 一致。最终口径用 **eval**：

| train_size | train last acc | train best | **eval (best.pt)** | vs eval 500 | train_loss |
|------------|----------------|------------|--------------------|-------------|------------|
| 500（baseline） | 0.834 | 0.836 | **0.836** | — | 0.080 |
| 2000 | 0.883 | 0.884 | **0.884** | +0.049 | 0.123 |
| 8000 | 0.906 | 0.907 | **0.907** | +0.072 | 0.161 |

数据越多 test 越好；`train_loss` 随样本量上升是过拟合。

## 3. Learning curves

只画 train Run。

![learning curves](./figures/learning_curves.png)

[打开 learning_curves.png](./figures/learning_curves.png)

500 过拟合；8000 一上来在 0.89 附近。

## 4. Train Runs

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

## 5. Eval Runs

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

## 6. Reproduce

```bash
python -m rpipe make studies/mnist_train_size --num-gpus 1 --init-gpu 0
python -m rpipe launch studies/mnist_train_size --num-gpus 1 --init-gpu 0 --console shared
```

默认跳过已 succeeded。要把装箱再走一遍：加 `--include-done`。
