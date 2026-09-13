# Study Plan: mnist_train_size

> 对照 [STUDY_GUIDE.md](../../../docs/STUDY_GUIDE.md) §3。跑完后写同目录 `STUDY_REPORT.md`。

## 1. 研究问题

固定模型与训练预算时，MNIST **训练集样本量**如何影响 **测试准确率**？独立 `mode=eval` Run 加载 sibling train 的 `best.pt`，作为最终口径。

## 2. Study / Experiment / Run

- Study 名：`mnist_train_size`
- Experiment：`train_size` × `{train, eval}` → **6 组**（不含 seed）
- Run：每组 × seeds `0,1,2` → **18 次**

`train_size` 取训练集编号 `0 .. N-1`（500 ⊂ 2000 ⊂ 8000）。

## 3. 变量轴

| 轴 | 字段 | 取值 |
|----|------|------|
| 训练样本量 | `data.config.train_size` | `500`, `2000`, `8000` |
| 算法 mode | `algorithm.mode` | `train`, `eval` |

## 4. 固定条件

| 项 | 取值 |
|----|------|
| `seeds` | `0, 1, 2` |
| `data.name` / `source` | `MNIST` / `torch` |
| `model.name` | `linear` |
| `algorithm.source` | `custom_torch` |
| `algorithm.num_epochs` | `20`，`progress_unit: epoch` |
| `algorithm.resume` | 不写进基底。train 默认 `latest`；eval 默认 `best` |
| `system.device` | `cuda` |

## 5. 高效率排班（§3）

- **同类一组：** 全部 train 都是 linear；不要把 train 和 eval 塞进同一 `wait`。
- **`wait` 闸门：** 组末必须等本组进程退出、显存释放完，才开下一组。
- **吃满 GPU：** `python -m rpipe make studies/mnist_train_size --num-gpus 1` 默认 `--round auto`。
- **依赖：** 先全部 train `wait` 完，再 eval。
- **error：** 单条失败只记 `run_id`；整轮后再 `resume: latest`。
- **seed ≠ 并发。** 本 Study 很短，用 `--console shared`。

```bash
python -m rpipe make studies/mnist_train_size --num-gpus 1 --init-gpu 0
python -m rpipe launch studies/mnist_train_size --num-gpus 1 --init-gpu 0 --console shared
```

## 6. 成功标准

- 18 次 Run `status: succeeded`
- eval 对齐 train `best_accuracy`
- `docs/figures/learning_curves.png`
- `STUDY_REPORT.md` 嵌图，写清实际 `pack`

## 7. 刻意不做什么

- 不做 TensorBoard
- 本轮不切 `transformers_trainer`
