# Study Plan: main_base

> 对照 [STUDY_GUIDE.md](../../../docs/STUDY_GUIDE.md) §3。跑完后写同目录 `STUDY_REPORT.md`。
> 对照物是 git `main` 的 `make.sh`：`make.py --mode base` 先 train 再 test，`--num_experiments 1`，`--round 1`。

## 1. 研究问题

这一轮能不能按 `main` 的 base 格子跑完：MNIST 与 CIFAR10 × linear / mlp / cnn / resnet18，train 再 test，seed 0。

先只跑探针。探针用同一格子、全量图片、`batch_size=250`，每条 train 只走 4 个 step。用这几步的墙钟改预估，再展开 `num_steps=60` 的正式网格。正式网格现在不进 `axes`：`launch` 只会按 `algorithm.mode` 过滤，两种规模写在同一份 `jobs.json` 里会一起启动。

## 2. Study / Experiment / Run

- Study 名：`main_base`
- 探针 Experiment：`data.name` × `model.name`，只有 train
- Run：每个 Experiment × `seeds: [0]`，共 8 条
- 正式网格（尚未展开）：同一 8 格再加 `algorithm.mode: eval`，`num_steps: 60`，共 16 条

## 3. 变量轴

| 轴 | 字段 | 探针现在 | 正式网格（探针之后） |
|----|------|----------|----------------------|
| 数据 | `data.name` | MNIST, CIFAR10 | 同左 |
| 模型 | `model.name` | linear, mlp, cnn, resnet18 | 同左 |
| 阶段 | `algorithm.mode` | train | train, eval |
| 步数 | `algorithm.num_steps` | 4 | 60 |
| 每次 test 的 batch 数 | `algorithm.eval_num_steps` | 4 | -1（整个 test 集） |

## 4. 固定条件

对齐 `main` 的 `src/module/hyper.py` 与 `src/config.yml`。

| 项 | 取值 |
|----|------|
| `seeds` | `0`（`num_experiments 1`） |
| 数据量 | 全量。不设 `train_size` |
| `batch_size` | 250 |
| `test_batch_ratio` | 4（test batch 1000） |
| `pin_memory` / `num_workers` | true / 0 |
| `augment` | false |
| `progress_unit` | step。不设 `num_epochs` |
| `eval_period` / `eval_num_steps` / `checkpoint_period` | 探针 `0` / `4` / `30`。`eval_period: 0` 是 4 个 step 跑完再测一次；这一次只测 4 个 test batch。正式网格写成 `eval_period: 30`、`eval_num_steps: -1`（每 30 step 测一次，一次走完 test 集） |
| optimizer | SGD，`lr=0.1`，`momentum=0.9`，`weight_decay=5e-4`，`nesterov=true` |
| scheduler | cosine，`eta_min=0` |
| `best_metric` | Loss（越小越好），`best_split=test` |
| `system.device` | cuda |
| `origin` | domestic。只换下载地址，不改算法 |

## 和 `main` 不一样的地方

下面是还对不上的地方。探针的 cosine 就是 4 个 step，不是 epoch。

| 项 | `main` | 这边 |
|----|--------|------|
| 均值从哪来 | 读完数据后自己算，交给模型里的 Kornia `Normalize` | 同样交给模型里的 Kornia `Normalize`。数来自 `python -m rpipe data studies/main_base` 写的 `shared/data/<数据集>/stats.yaml`（train 的 mean / std，另有张数、形状、类别数量、像素最小最大）。还没跑这个命令时，用代码里的常数 |
| CIFAR 训练增强 | 模型里 Kornia 随机翻转和 padding crop | 增强也在模型里、用 Kornia。探针 `augment: false`，所以这 4 步没有翻转和 crop |
| 探针的 test | 每 30 step 测一次，每次整个 test 集 | `eval_period: 0` 且 `eval_num_steps: 4`：4 个 step 结束后测 4 个 test batch。正式网格是 `eval_period: 30`、`eval_num_steps: -1` |
| 曲线和权重隔多久写盘 | 日志大约每 7–8 step 一行；checkpoint 每 30 step | 见下面「中间结果」。探针 4 step，权重只在结束时写 `latest` |
| 调度 | 一条进程结束再开下一条 | 见下面「调度」 |

采用这边、不再改回去的：Accuracy 是 0–100。种子额外设了 Python、NumPy 和全部 CUDA 设备。数据用 torchvision，`origin: domestic` 只换下载地址。不做 TensorBoard。

已经按 `main` 写上的：格子、`batch_size=250`、test batch 4 倍、SGD、cosine、最好指标 test Loss、`RandomSampler` 的 `num_samples`、`cross_entropy`、`init_param`。

### cosine

探针是 `progress_unit: step` 且 `num_steps: 4`，没有 `num_epochs`。cosine 的 `T_max` 跟这次的 step 预算走，所以是 4。正式网格改成 `num_steps: 60` 之后，`T_max` 才是 60。

### 调度

`main` 的 `--round 1` 是：启动 1 个训练进程，等它退出，再启动下一个。8 条 train 就是 8 段串行。

这边 `make` 先看每条大概要多少显存。同一种模型、放得下的几条放进同一个 `wait`，一起跑；这一组里最慢的那条结束、显存还回来，才开下一组。所以墙钟会比 `main` 短。每条自己的 step 时间不受这个影响。

### 中间结果

yaml 没写 `log_period`。它是整数 step，和 `checkpoint_period` 一样：写了 `8` 就每 8 个 optimizer step 在 `run.log` 打一行，并往曲线 jsonl 追加当前均值。不写就没有这些中间行。4 个 step 跑完时仍会打一次收尾。旧键 `log_interval` 仍能读。

曲线在 `runs/<id>/assets/tracker/`。这次 step 预算把 sampler 收成正好 4 个 batch，循环结束时写一个点，不是每个 step 一个点。

权重看 `checkpoint_period`。现在是 30，4 小于 30，循环里不存；结束时仍写 `latest`。`save_best: true` 时，结束那一次 test 若更好，再写 `best`。

## 5. 高效率排班（§3）

- **同类一组：** 同一 `model.name` 一起并行。
- **`wait` 闸门：** 组末必须等本组进程退出、显存释放完，才开下一组。
- **吃满 GPU：** 下面的 make 用 `--num-gpus 1`。看 make 打印的 `pack N waits`。
- **依赖：** 探针的 test 在每条 train 的 4 个 step 结束时做，没有单独的 eval 任务。正式网格若再加 `algorithm.mode: eval`，先全部 train 结束再 eval。
- **error：** 单条失败只记 `run_id`；整轮结束后对未 succeeded 的格子再跑。
- **seed ≠ 并发。**

```bash
python -m rpipe make studies/main_base --num-gpus 1 --init-gpu 0
python -m rpipe launch studies/main_base --num-gpus 1 --init-gpu 0
```

探针的 `est wall` 只作打包参考。2026-10-01 这次探针时显卡还在跑别的任务，日志里的 step 时间不能拿来填正式网格。等卡空下来再测一次，再改 yaml。

## 时长预估

`make` 之后、`launch` 之前填。探针 8 条。同一 `wait` 并行，组墙钟取该组最慢的一条；整轮是各组相加。不含显存。

| wait | factors | mode | seed | id | est |
|---:|---|---|---:|---|---:|
| 1 | data=MNIST model=linear | train | 0 | `71b3d24433e5023e` | 4s |
| 1 | data=CIFAR10 model=linear | train | 0 | `3276244bd3d96af8` | 4s |
| 2 | data=MNIST model=mlp | train | 0 | `547140b42c28c062` | 4s |
| 2 | data=CIFAR10 model=mlp | train | 0 | `ebd3b36e28633347` | 4s |
| 3 | data=MNIST model=cnn | train | 0 | `57a8d9eabbb2680f` | 4s |
| 3 | data=CIFAR10 model=cnn | train | 0 | `d8072c0ee4b2e48c` | 4s |
| 4 | data=MNIST model=resnet18 | train | 0 | `0aee6c175b9371ac` | 5s |
| 4 | data=CIFAR10 model=resnet18 | train | 0 | `fb2a5aecbbfd7a6e` | 5s |
| | 整轮 | | | | 17s |

## 6. 成功标准

探针：

- 8 条 train 均 `status: succeeded`（2026-10-01 已做到）
- 报告里每条有 `est` 和从 `[flow] start` 到 `[flow] succeeded` 的 `actual`
- 卡空闲时再测 step 时间，写出正式网格的预估，再改 yaml

正式网格（本文件改完、再 launch 之后才算）：

- 16 条均 succeeded
- `docs/figures/learning_curves.png`
- `STUDY_REPORT.md` 写整轮预估和实际

## 7. 刻意不做什么

- 探针阶段不跑 60 step，也不另开 eval 任务。test 只在 4 个 step 结束时做一次
- 不做 TensorBoard
- 不把 `cifar_grid`（`train_size=1024`、5 epoch）当作这一轮
