# Study Plan: vision_main_recipe

> 状态：改为 **400 epoch + CUDA**（对齐 main README CIFAR 横轴；`num_epochs` 推导 `num_steps = epochs × ceil(N/B)`，每 epoch 评一次）。正在重跑。

## 1. 研究问题

现在的 registry 能否用 **main 同一套超参**（`src/module/hyper.py`）训 **CIFAR10 / SVHN** 上的 **linear / mlp / cnn / resnet18**？不要求 60 step 打到旧仓库长训精度。

## 2. 变量轴

| 轴 | 字段 | 取值 |
|----|------|------|
| 数据 | `data.name` | `CIFAR10`, `SVHN` |
| 模型 | `model.name` | `linear`, `mlp`, `cnn`, `resnet18` |

1 seed（`0`）。main `process.py` 画图用 4 seed；本 Study 先验证通路。MNIST × 四模型已由 `mnist_main_recipe`（linear）覆盖；mlp/cnn 在 MNIST 上形状由 `data.meta.data_size` 自动对齐。

## 3. 固定条件（对 `hyper.py`）

| 项 | 取值 |
|----|------|
| `batch_size` | `250` |
| `test_batch_ratio` | `4` |
| `num_steps` | `60` |
| `eval_period` / `checkpoint_period` | `30` |
| SGD | `lr=0.1`, momentum 0.9, nesterov, `weight_decay=5e-4` |
| cosine | `T_max` = 60 step |
| `best_metric` | test Loss min |
| 增强 | CIFAR10: flip + pad-4 crop；SVHN: pad-4 crop（对齐 main `model/base.py`，用 torchvision 而非 kornia） |
| Normalize | 常用 torchvision 均值方差（main 用预计算 stats 文件，缺文件时本仓库不依赖 kornia） |
| 形状 | prepare 把 `Data.meta.data_size` 交给 model factory（不跨层 import） |

## 4. 成功标准

- 8 个 Experiment 的 Run `succeeded`
- cnn/resnet 前向保持 NCHW（不被当成 linear 展平）
- 有 `docs/figures/learning_curves.png`

## 5. 刻意不做什么

- 不做 wresnet / CIFAR100 / FashionMNIST
- 不接 TensorBoard、不接 dict collate
- 不对 bit-identical（kornia vs torchvision、stats 文件）
