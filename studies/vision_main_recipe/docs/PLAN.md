# Study Plan: vision_main_recipe

> 状态：长训契约。**400 epoch + CUDA**，对齐 main README 的 CIFAR 横轴。`num_epochs` 推导 `num_steps = epochs × ceil(N/B)`，每 epoch 评一次 test。调度用 `rpipe make` / `launch`，`--round` 并行，不是顺序 `rpipe run`。

## 1. 研究问题

现在的 registry 能否用 **main 同一套超参**（`src/module/hyper.py`）把 **CIFAR10 / SVHN** 上的 **linear / mlp / cnn / resnet18** 训满 400 epoch？本轮 1 个 seed，先看通路与曲线形状。不要求 bit-identical 于 main 的长训图。

## 2. 变量轴

| 轴 | 字段 | 取值 |
|----|------|------|
| 模型 | `model.name` | `linear`, `mlp`, `cnn`, `resnet18` |
| 数据 | `data.name` | `CIFAR10`, `SVHN` |

**8 个 Experiment × seed 0 = 8 次 Run。** 无独立 eval 轴（只 train；`save_best` 按 test Loss min）。main `process.py` 画图用 4 seed；多种子放到下一轮。

`axes` 里 **model 写在 data 前面**：`itertools.product` 会先铺两个数据集的同一模型。`--round 2` 时每一组 wait 是：

| 组 | 并行的两个 Run | 为什么这样切 |
|----|----------------|--------------|
| 1 | linear CIFAR10 + linear SVHN | 都小，墙钟接近 |
| 2 | mlp CIFAR10 + mlp SVHN | 同上 |
| 3 | cnn CIFAR10 + cnn SVHN | 同上 |
| 4 | resnet18 CIFAR10 + resnet18 SVHN | 最重的一对单独占一组 |

不要用 `--round 4` 把 cnn 和两个 resnet 塞进同一 `wait`（一组墙钟被最慢的拖死，显存也挤）。

## 3. 固定条件（对 `hyper.py`）

| 项 | 取值 |
|----|------|
| `batch_size` | `250` |
| `test_batch_ratio` | `4` |
| `num_epochs` | `400`（不再用 main 默认 60 step 短预算） |
| `progress_unit` | `epoch` |
| `eval_period` / `checkpoint_period` | `1` |
| SGD | `lr=0.1`, momentum 0.9, nesterov, `weight_decay=5e-4` |
| cosine | `T_max` = 推导出的 `num_steps` |
| `best_metric` | test Loss min |
| `resume` | 默认 `latest`（中断后续跑同一 `runs/<id>/`） |
| 增强 | CIFAR10: flip + pad-4 crop；SVHN: pad-4 crop（torchvision，不是 kornia） |
| Normalize | 常用 torchvision 均值方差（main 用预计算 stats 文件） |
| 形状 | prepare 把 `Data.meta.data_size` 交给 model factory |
| `system.device` | `cuda` |

## 4. 调度（并行）

机器默认 **1 张 GPU**。`make` / `launch` 按 **同类、相近耗时** 装箱：linear 一组、mlp 一组、cnn 一组、resnet 一组；组内再按空闲×50% 能叠几个就叠几个。resnet 不和 linear 同一 `wait`。

```bash
python -m rpipe make studies/vision_main_recipe --num-gpus 1 --init-gpu 0
python -m rpipe launch studies/vision_main_recipe --num-gpus 1 --init-gpu 0
```

make 会打印 `pack N waits: …`。中断后续跑同一条 `launch`（不要 `--include-done`）。长训放独立终端。

## 5. 成功标准

- 8 次 Run `status: succeeded`，`process.complete: true`
- cnn/resnet 前向保持 NCHW
- `docs/figures/learning_curves.png` 按 epoch 画满（不是 60 step 两个点）
- `STUDY_REPORT.md` 用本轮 400 epoch 数字，并写清 make/launch 与实际 `round`（auto 或手写）

## 6. 刻意不做什么

- 本轮不加 seed、不加 wresnet / CIFAR100 / FashionMNIST
- 不接 TensorBoard
- 不对 bit-identical（kornia vs torchvision、stats 文件）
- 不把独立 eval 做成第二波（本 Study 没有 `algorithm.mode: eval`）
