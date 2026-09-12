# Study Report: mnist_train_size

> Plan: [PLAN.md](PLAN.md)
> Date: 2026-09-12
> Recipe: native `custom_torch`；MNIST + linear；20 epoch；SGD + cosine；cuda；`algorithm.metric` = Loss + Accuracy；axes = `train_size` × `{train, eval}`
> Location: Study `docs/`。数字读 `../process.json`。排班标准 [STUDY_GUIDE.md](../../../docs/STUDY_GUIDE.md) §3。

## 1. 怎么跑的

研究因素：`train_size ∈ {500, 2000, 8000}` × `{train, eval}` × seeds `0,1,2` → **6 Experiment、18 Run**。`train_size` 是前 N 条（500 ⊂ 2000 ⊂ 8000）。eval 加载 sibling `best.pt`。`resume` 不写进 `fixed`：train 默认 `latest`，eval 默认 `best`。

```bash
python -m rpipe make studies/mnist_train_size --num-gpus 1 --init-gpu 0
python -m rpipe launch studies/mnist_train_size --num-gpus 1 --init-gpu 0 --console shared
```

机器：1× RTX 5090 D v2。make 打印：

`pack 2 waits: 9[linear+linear+…] , 9[linear+linear+…] est wall 1m34s` — 先 9 次 train 同一组，`wait` 完再 9 次 eval（打印把 9 个 `linear` 用 `+` 拼起来，没有写成 `linear×9`）。实测整轮约 35s；conservative 墙钟只供排班。本 Study 很短，用 `--console shared` 把 printout 收在一个窗口。

18 次全部一次 `succeeded`，没有 retry。

## 2. Conclusion

18 次 Run 全部 `succeeded`。独立 eval 与 train `best_accuracy` 一致。最终口径用 **eval**：

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
| 500 | 0 | baseline | 38d30d5ce0995fd2 | 0.8328 | 0.8346 | 0.080 |
| 500 | 1 | baseline | 3136ab140a6f9427 | 0.8341 | 0.8344 | 0.079 |
| 500 | 2 | baseline | 0568b0bcf435267c | 0.8359 | 0.8375 | 0.080 |
| 2000 | 0 | — | b2e77ac4b1ae4fde | 0.8818 | 0.8841 | 0.123 |
| 2000 | 1 | — | f553417234b151ea | 0.8828 | 0.8835 | 0.124 |
| 2000 | 2 | — | 24cff5673f69e307 | 0.8846 | 0.8851 | 0.123 |
| 8000 | 0 | — | 579b0430eac14cee | 0.9065 | 0.9070 | 0.161 |
| 8000 | 1 | — | d595c9125c7c18bb | 0.9056 | 0.9072 | 0.161 |
| 8000 | 2 | — | 74a2c146872b9fad | 0.9062 | 0.9078 | 0.161 |

## 5. Eval Runs

| train_size | seed | Run id | eval_accuracy | matches train best |
|------------|------|--------|---------------|--------------------|
| 500 | 0 | e8b53056a1c7bacc | 0.8346 | 0.8346 |
| 500 | 1 | 8a23559c423a51e4 | 0.8344 | 0.8344 |
| 500 | 2 | 04ec24e82f75c070 | 0.8375 | 0.8375 |
| 2000 | 0 | a127ce36d137819d | 0.8841 | 0.8841 |
| 2000 | 1 | 2e0fa8441c03dec5 | 0.8835 | 0.8835 |
| 2000 | 2 | 85fcb2fc33546c32 | 0.8851 | 0.8851 |
| 8000 | 0 | d16cb5b430f41ffe | 0.9070 | 0.9070 |
| 8000 | 1 | 523eff3e5b0e6de8 | 0.9072 | 0.9072 |
| 8000 | 2 | ff57d137fad29be2 | 0.9078 | 0.9078 |

## 6. Reproduce

```bash
python -m rpipe make studies/mnist_train_size --num-gpus 1 --init-gpu 0
python -m rpipe launch studies/mnist_train_size --num-gpus 1 --init-gpu 0
```

默认跳过已 succeeded。要把装箱再走一遍：加 `--include-done`。
