# Study Plan: mnist_train_size

> 对照 [STUDY_GUIDE.md](../../../docs/STUDY_GUIDE.md) §3。跑完后结果见 `STUDY_REPORT.md`。

## 1. 研究问题

在固定模型与训练预算下，**MNIST 训练集样本量**如何影响 **测试集准确率**（及训练 loss）？独立 `mode=eval` Run 用 train 的 `best.pt` 再评一次，作为最终口径。

## 2. Study / Experiment / Run

Study 名 `mnist_train_size`。`experiment: mnist_linear` 只是基底名字。

`axes`：三个 `train_size` × `{train, eval}` = **6 个 Experiment**；每个 × 3 个 seed = **18 次 Run**。index 按 `factors` 分组。`process.paired` 把同一 `train_size` 的 train / eval 拼回一行。

**子集口径：** `train_size` 取训练集编号 `0 .. N-1`，因此 **500 ⊂ 2000 ⊂ 8000**。`seed` 只影响初始化与 shuffle，不换图。

## 3. 变量轴

| 轴 | 字段 | 取值 |
|----|------|------|
| 训练样本量 | `data.config.train_size` | `500`, `2000`, `8000` |
| 算法 mode | `algorithm.mode` | `train`, `eval` |

## 4. 固定条件

| 项 | 取值 |
|----|------|
| `seeds` | `0, 1, 2`（复测，不是并发） |
| `data.name` | `MNIST` |
| `data.source` | `torch` |
| `model.name` | `linear`（784→10） |
| `algorithm.source` | `custom_torch` |
| `algorithm.num_epochs` | `20` |
| `algorithm.progress_unit` | `epoch` |
| `algorithm.eval_period` | `1` |
| `algorithm.checkpoint` | `latest` |
| `algorithm.save_best` | `true` |
| `algorithm.resume` | train 默认 `latest`；eval 默认 `best`（sibling train） |
| `algorithm.optimizer` | `SGD`，`lr=0.1`，cosine |
| `data.config.batch_size` | `64` |
| `system.device` | `cuda` |

## 5. 高效率排班（§3）

- **同类一组：** 全部 train 都是同一 linear、相近耗时（只是子集大小不同），一组里尽量叠满 GPU。不要把 train 和 eval 塞进同一 `wait`。
- **`wait` 闸门：** 组末必须等本组进程退出、显存释放完，才开下一组。不 wait 下一波会挤进来，容易 OOM。
- **吃满 GPU：** 默认 `--round auto`；看打印的 `pack N waits`。conservative 墙钟是各组最慢条加总，只供排班。
- **依赖：** eval 必须等 **全部 train `wait` 完** 再开，因为要读 sibling `best.pt`。
- **error：** 单条失败只记 `run_id`；整轮结束后对未 succeeded 的格子 `resume: latest` 再跑。
- **seed ≠ 并发。** 长训才丢独立终端；本 Study 20 epoch linear 很短，可用 `--console shared`。

```bash
python -m rpipe make studies/mnist_train_size --num-gpus 1 --init-gpu 0
python -m rpipe launch studies/mnist_train_size --num-gpus 1 --init-gpu 0
```

## 6. 成功标准

- 18 次 Run `succeeded`
- train：`train_loss`、`accuracy`、`best_accuracy`；`latest.pt` 与 `best.pt`
- eval：`eval_accuracy` 对齐 train `best_accuracy`
- `STUDY_REPORT.md` 嵌 learning curve，写清实际 pack

## 7. 刻意不做什么

- 独立 eval 不是 Flow 第二阶段
- 本 Study 不切 `transformers_trainer`
- 不做 TensorBoard
