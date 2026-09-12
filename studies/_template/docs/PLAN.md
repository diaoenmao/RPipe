# Study Plan: my_study

> 对照 [STUDY_GUIDE.md](../../../docs/STUDY_GUIDE.md) §3。跑完后写同目录 `STUDY_REPORT.md`。

## 1. 研究问题

（这一轮要比什么？成功长什么样？）

## 2. Study / Experiment / Run

- Study 名：`my_study`
- Experiment：由 `axes` 展开（不含 seed）
- Run：每个 Experiment × `seeds`

## 3. 变量轴

| 轴 | 字段 | 取值 |
|----|------|------|
| （因素） | | |

## 4. 固定条件

| 项 | 取值 |
|----|------|
| `seeds` | |
| `system.device` | `cuda` |

## 5. 高效率排班（§3）

- **同类一组：** 同一 `model.name`、相近耗时的 Run 一起并行；不要把 resnet 和 linear 塞进同一 `wait`。
- **`wait` 闸门：** 组末必须等本组进程退出、显存释放完，才开下一组。不 wait 下一波会挤进来，容易 OOM。
- **吃满 GPU：** `python -m rpipe make studies/<name> --num-gpus 1` 默认 `--round auto`；看打印的 `pack N waits`。conservative 墙钟是各组最慢条加总，只供排班。
- **依赖：** 有 `algorithm.mode: eval` 时先全部 train `wait` 完，再 eval。
- **error：** 单条失败只记 `run_id`；整轮结束后对未 succeeded 的格子 `resume: latest` 再跑。
- **seed ≠ 并发。**

```bash
python -m rpipe make studies/<name> --num-gpus 1 --init-gpu 0
python -m rpipe launch studies/<name> --num-gpus 1 --init-gpu 0
```

## 6. 成功标准

- 计划的 Run 均 `status: succeeded`
- `docs/figures/learning_curves.png`
- `STUDY_REPORT.md` 嵌图，并写清实际 `pack` 排班

## 7. 刻意不做什么

- 不做 TensorBoard
