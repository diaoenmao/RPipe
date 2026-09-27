# Study Plan: cifar_grid

> 对照 [STUDY_GUIDE.md](../../../docs/STUDY_GUIDE.md) §3，以及 git `main` 的 `make.py` 网格（CIFAR10 × linear / mlp / cnn / resnet18）。跑完后写同目录 `STUDY_REPORT.md`。

## 1. 研究问题

NCHW、CIFAR 增广默认打开时，同一小训练子集上，四种 native 骨干的测试准确率差多少？独立 `mode=eval` 加载 sibling train 的 `best.pt`。这是对照切片，不是全量 CIFAR 论文表。

## 2. Study / Experiment / Run

- Study 名：`cifar_grid`
- Experiment：`model.name` × `{train, eval}` → **8 组**（不含 seed）
- Run：每组 × seed `0` → **8 次**

## 3. 变量轴

| 轴 | 字段 | 取值 |
|----|------|------|
| 骨干 | `model.name` | `linear`, `mlp`, `cnn`, `resnet18` |
| 算法 mode | `algorithm.mode` | `train`, `eval` |

## 4. 固定条件

| 项 | 取值 |
|----|------|
| `seeds` | `0`（小网格；要复测再加 seed） |
| `data.name` / `source` | `CIFAR10` / `torch` |
| `origin` | `domestic`（Study 级：数据走百度镜像，模型 hub 走 `hf-mirror.com`） |
| `data.config.train_size` | `1024`（训练集前 1024 张，不是 50000） |
| `data.config.batch_size` | `64` |
| 增广 | 不写 `augment`。CIFAR 默认开 RandomCrop + Flip |
| `algorithm.source` | `custom_torch` |
| `algorithm.num_epochs` | `5`，`progress_unit: epoch` |
| `algorithm.resume` | 不写。train 默认 `latest`；eval 默认 `best` |
| `system.device` | `cuda` |

## 5. 高效率排班（§3）

- **同类一组：** linear / mlp 轻，cnn 中，resnet18 重。默认 `--round auto` 按模型装箱，不要把 resnet 和 linear 塞进同一 `wait`。
- **`wait` 闸门：** 组末必须等本组退出再开下一组。
- **依赖：** 先全部 train `wait` 完，再 eval。只重跑评测：`python -m rpipe launch studies/cifar_grid --mode eval`。
- **error：** 单条失败记 `run_id`；整轮后再看 `rpipe status`。
- **seed ≠ 并发。**

```bash
python -m rpipe make studies/cifar_grid --num-gpus 1 --init-gpu 0
python -m rpipe launch studies/cifar_grid --num-gpus 1 --init-gpu 0 --console shared
python -m rpipe status studies/cifar_grid
```

## 6. 成功标准

- 8 次 Run `status: succeeded`
- eval 能读到 sibling train 的 `best`
- `docs/figures/learning_curves.png`
- `STUDY_REPORT.md` 按 Experiment（骨干）写，不把 8 行当主结论

## 7. 刻意不做什么

- 不做全量 50000、不多 seed、不切 HF Trainer
- 不做 TensorBoard / DDP
