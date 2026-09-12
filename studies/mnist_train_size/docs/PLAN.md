# Study Plan: mnist_train_size

> 对照 [STUDY_GUIDE.md](../../../docs/STUDY_GUIDE.md) §3。跑完后写同目录 `STUDY_REPORT.md`。

## 1. 研究问题

固定模型与训练预算时，MNIST **训练集样本量**如何影响 **测试准确率**（及 train loss）？独立 `mode=eval` Run 加载 sibling train 的 `best.pt`，作为最终口径。

## 2. Study / Experiment / Run

- Study 名：`mnist_train_size`
- Experiment：`train_size` × `{train, eval}` → **6 组**（不含 seed）；index 按 `factors` 分组
- Run：每组 × seeds `0,1,2` → **18 次**

`train_size` 取训练集编号 `0 .. N-1`，因此 **500 ⊂ 2000 ⊂ 8000**。`seed` 只影响初始化与 shuffle。

## 3. 变量轴

| 轴 | 字段 | 取值 |
|----|------|------|
| 训练样本量 | `data.config.train_size` | `500`, `2000`, `8000` |
| 算法 mode | `algorithm.mode` | `train`, `eval` |

## 4. 固定条件

| 项 | 取值 |
|----|------|
| `seeds` | `0, 1, 2`（复测，不是并发） |
| `data.name` / `source` | `MNIST` / `torch` |
| `model.name` | `linear`（784→10） |
| `algorithm.source` | `custom_torch` |
| `algorithm.num_epochs` | `20`，`progress_unit: epoch` |
| `algorithm.eval_period` / `checkpoint` | `1` / `latest`；`save_best: true` |
| `algorithm.resume` | 不写进基底。train 默认 `latest`；eval 默认 `best`（sibling train） |
| `algorithm.optimizer` | `SGD`，`lr=0.1`，cosine；`max_grad_norm: 0` |
| `data.config.batch_size` | `64` |
| `system.device` | `cuda` |

## 5. 高效率排班（§3）

- **同类一组：** 全部 train 都是 linear、相近耗时；不要把 train 和 eval 塞进同一 `wait`。
- **`wait` 闸门：** 组末必须等本组进程退出、显存释放完，才开下一组。
- **吃满 GPU：** `python -m rpipe make studies/mnist_train_size --num-gpus 1` 默认 `--round auto`；看打印的 `pack N waits`。conservative 墙钟是各组最慢条加总，只供排班。
- **依赖：** 先全部 train `wait` 完，再 eval（要读 sibling `best.pt`）。
- **error：** 单条失败只记 `run_id`；整轮结束后对未 succeeded 的格子 `resume: latest` 再跑。
- **seed ≠ 并发。** 本 Study 很短，可用 `--console shared`。

```bash
python -m rpipe make studies/mnist_train_size --num-gpus 1 --init-gpu 0
python -m rpipe launch studies/mnist_train_size --num-gpus 1 --init-gpu 0 --console shared
```

## 6. 成功标准

- 18 次 Run `status: succeeded`
- train：`train_loss`、`accuracy`、`best_accuracy`；`latest.pt` 与 `best.pt`
- eval：`eval_accuracy` 对齐对应 train 的 `best_accuracy`
- `docs/figures/learning_curves.png`
- `STUDY_REPORT.md` 嵌图，并写清实际 `pack`

## 7. 刻意不做什么

- 独立 eval 不是 Flow 第二阶段
- 本 Study 不切 `transformers_trainer`
- 不做 TensorBoard
