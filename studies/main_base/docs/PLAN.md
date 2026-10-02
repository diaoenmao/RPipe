# Study Plan: main_base

> 对照 [STUDY_GUIDE.md](../../../docs/STUDY_GUIDE.md) §3。跑完后写同目录 `STUDY_REPORT.md`。
> 对照固定为旧 `main` 提交 `98648f3a5c7db7dccf3ca806410d5b6fdee9484c`，不随分支移动。入口是该提交的 `src/make.sh`：`make.py --mode base` 先 train 再 test，`--num_experiments 1`，`--round 1`。
> 2026-10-02 本机 60-step 可视化验证轮已执行完成，`version: local-curves-20261002`，16/16 succeeded。下文保留执行前设计与初始估时；实际结果、验收和 99.827s flow 墙钟见 [STUDY_REPORT.md](STUDY_REPORT.md)。历史 4-step 探针证据保留。

## 1. 研究问题

这一轮在本机 RTX 5090 D v2 上跑 MNIST 与 CIFAR10 × linear / mlp / cnn / resnet18，seed 0，8 train + 8 eval。每条 train 走 60 个 optimizer step，每 5 step 记录训练指标并完整评测一次，验证多点曲线、周期评测、checkpoint 与独立 eval 的对应关系。

探针已完成：同一格子、全量数据集、`batch_size=250`，每条 train 只走 4 个 step，结束后测 4 个 test batch。它只验证短链路。本轮 60×250=15000 次训练采样，不是 60 epoch，也不表示走遍训练集或达到收敛。

本轮为可视化验证，有意将旧 `main` 的 `eval_period: 30` 改为 5：训练中完整 test 从 2 次增加到 12 次，会增加耗时，也会改变 best 的候选时刻与选择结果，因此不能称为旧 `main` 的严格复现。不能把探针墙钟线性放大为本轮预估；也不把 4 / 60 同时写进 `axes`，因为 `launch` 只按 `algorithm.mode` 过滤，两种规模会一起启动。

## 2. Study / Experiment / Run

- Study 名：`main_base`
- 历史探针：`data.name` × `model.name`，只有 train，seed 0，共 8 条
- 本轮 Experiment：`data.name` × `model.name` × `algorithm.mode: [train, eval]`
- 本轮 Run：每个 Experiment × `seeds: [0]`，共 16 条（8 train + 8 eval）
- 本轮 `fixed.version`：`local-curves-20261002`，train / eval 共用；生成新 Run，保留旧探针目录

## 3. 变量轴

| 轴 / 预算 | 字段 | 历史探针 | 本轮可视化验证 |
|----|------|----------|----------------------|
| 数据 | `data.name` | MNIST, CIFAR10 | 同左 |
| 模型 | `model.name` | linear, mlp, cnn, resnet18 | 同左 |
| 阶段 | `algorithm.mode` | train | train, eval |
| 步数 | `algorithm.num_steps` | 4 | 60 |
| 训练记录周期 | `algorithm.log_period` | 未设置，仅收尾 | 5 |
| 训练中 test 周期 | `algorithm.eval_period` | 0（仅收尾） | 5 |
| 每次 test 的 batch 数 | `algorithm.eval_num_steps` | 4 | -1（整个 test 集） |

## 4. 固定条件

参照固定对照提交的 `src/module/hyper.py`、`src/config.yml`、`src/model/base.py` 与 `src/test_model.py`，但本轮评测频率有意不同。下表是待在本轮展开清单中核对的执行条件。

| 项 | 取值 |
|----|------|
| `seeds` | `0`（`num_experiments 1`） |
| 数据量 | 全量。不设 `train_size` |
| `batch_size` | 250 |
| `test_batch_ratio` | 4（test batch 1000） |
| `pin_memory` / `num_workers` | true / 0 |
| `augment` | 本轮 true，开启 CIFAR10 的训练翻转 / crop；MNIST 仍仅 Normalize，test 均仅 Normalize，见下节。历史探针为 false |
| `progress_unit` | step。不设 `num_epochs` |
| `log_period` / `eval_period` / `eval_num_steps` / `checkpoint_period` | `5` / `5` / `-1` / `30`。每 5 个 optimizer step 记录一次训练指标并评完整 test；latest 在 step 30 / 60 更新，best 随评测改善保存 |
| optimizer | SGD，`lr=0.1`，`momentum=0.9`，`weight_decay=5e-4`，`nesterov=true` |
| scheduler | cosine，`eta_min=0` |
| `best_metric` | Loss（越小越好），`best_split=test` |
| `system.device` | cuda，本机单张 RTX 5090 D v2，运行前核对空闲显存与其他负载 |
| `origin` | domestic。只换下载地址，不改算法 |

### 数据、增强与对照边界

数据集、模型结构 / 初始化、采样预算、优化器、cosine、增强与评测条件均需可核对。以下差异不能被“16 条都成功”掩盖：

| 项 | `main` | 这边 |
|----|--------|------|
| 归一化统计 | `src/module/stats.py` 加载预先计算的统计，模型使用 Kornia `Normalize` | 复用并核对本机已有 `shared/data/<数据集>/stats.yaml` 的 train mean / std 和原始数据可读性；缺失或失效时再执行 `rpipe data`。不能静默使用回退常数 |
| CIFAR 训练增强 | Kornia 翻转（p=0.5）→ 32×32 crop（padding=4，reflect）→ Normalize | 历史探针关闭；本轮开启同一顺序。基底与 fixed 的 `data.config.augment` 一起改为 true；当前 MNIST 的 `train_aug` 为空，因此仍只 Normalize，eval 也不做随机增强 |
| 训练中 test | 每 30 step 测一次，每次整个 test 集 | 本轮每 5 step 完整 test，共 12 次；历史探针则仅收尾测 4 个 test batch。频率不同带来的耗时和 best 选择差异单独说明 |
| 曲线和权重隔多久写盘 | 日志大约每 7–8 step 一行；checkpoint 每 30 step | 本轮训练记录每 5 step，latest 每 30 step，best 按评测改善保存。密曲线读取 scalars.jsonl，缺密记录时才回退稀疏 history，不改变 tracker 的 epoch 收口语义 |
| 调度 | 一条进程结束再开下一条 | 见下面「调度」 |

有意保留的差异包括更密的评测与记录，以及 torchvision 数据接口、domestic 下载源、额外设置 Python / NumPy / 全部 CUDA 的 seed、RPipe 日志与 artifact 布局、不做 TensorBoard 和按模型装箱并发。记录 Python、torch、torchvision、Kornia、GPU / CUDA 环境和实际并发条件；相同 seed 不保证跨实现、库版本和设备逐位一致。本轮不宣称严格配方复现。

数据核对以完整 train / test split 为准，不用四步采样结果估算统计：

| 数据 | train / test 张数 | 本机已记录的 train mean | 本机已记录的 train std |
|---|---|---|---|
| MNIST | 60000 / 10000 | 0.130661 | 0.308108 |
| CIFAR10 | 50000 / 10000 | 0.491400, 0.482158, 0.446531 | 0.247032, 0.243485, 0.261588 |

这些数是本轮核对参考，不是覆盖统计的指令。CIFAR10 无 stats 时的回退 std 为 0.2023 / 0.1994 / 0.2010，与上述不同。报告记录实际使用的统计与来源；若和对照统计有差异，先解释再比较，不因 YAML 相同就视为输入一致。shared 不进 Git，clone 不会带走这些文件。

### 评测与指标

- 本轮 train 在 step 5 / 10 / … / 60 各评完整 10000 张 test 图片（每次 10 个 batch），共 12 次；独立 eval 再走一次完整 test split，不沿用探针的 4-batch 截断
- best 由 test Loss 最低选择，不是 Accuracy 最高；独立 eval 读取同一因素、seed、version 的 sibling train best，记录来源 Run ID 和 checkpoint step
- 报告分开列 step 60 的 last test、被选中 best 的 Loss / Accuracy，以及独立 eval 的结果。eval 应与所加载 best 对照，不能强求它等于 last；发现差异需检查 checkpoint、评测条件与数值误差
- Accuracy 一律标为 0–100，差值用百分点；历史 0–1 结果若参与比较，显式转换并保留来源。Loss 保留原量纲
- seed 0 的一轮用于有限预算下的多点曲线和执行行为验证，不据此宣称最终精度、收敛或跨 seed 统计等价。较密 test 也增加了用 test Loss 选 best 的机会，不把其 best 分数优势当作模型改进

已经按 `main` 写上的：格子、`batch_size=250`、test batch 4 倍、SGD、cosine、最好指标 test Loss、`RandomSampler` 的 `num_samples`、`cross_entropy`、`init_param`。

### cosine

历史探针是 `progress_unit: step` 且 `num_steps: 4`，没有 `num_epochs`，cosine 的 `T_max=4`。本轮改成 `num_steps: 60`，`T_max=60`；评测频率改变不增加 optimizer step。

### 调度

`main` 的 `--round 1` 是：启动 1 个训练进程，等它退出，再启动下一个。8 条 train 就是 8 段串行。

这边 `make` 先看每条大概要多少显存。同一种模型、放得下的几条放进同一个 `wait`，一起跑；这一组里最慢的那条结束、显存还回来，才开下一组。并发的目标是改善整轮吞吐，但共享 GPU 会改变单条 step 延迟，整轮也不保证更快。不能把并发墙钟差异当成相对串行 `main` 的单 Run 算法加速。

耗时分开记录：算法返回的 `elapsed_seconds`、每个 Run 从 `[flow] start` 到 `[flow] succeeded` 的实际耗时、整轮最早 start 到最晚 succeeded 的墙钟。若另外记录 launch 命令的外部计时，单独标注；它包括父进程与聚合等额外开销。跨实现比较单 Run 延迟时应使用相同硬件、负载和并发条件，不混用本轮并行结果与 `main --round 1`。

### 中间结果

历史探针没有 `log_period`，4 个 step 收尾只写一个 history 点。本轮 `log_period: 5` 与 `eval_period: 5`：每条 train Run 预期有 12 个训练周期记录（optimizer step 5 / 10 / … / 60），再加 epoch 收尾的一次真实快照，共 13 个 train 观测；test 为 12 次完整评测记录。独立 eval 仅有一次评测，不混入训练曲线。

曲线源在 `runs/<id>/assets/tracker/`，按 [structure.md §6.9.2](../../../docs/code_structure/structure.md) 优先使用密记录 scalars.jsonl，缺失时才回退 history。train 每次记录的是当前 epoch 开始至该报告点的累积均值，不是最近 5 step 的窗口均值；稀疏 train history 仍可只有一个 epoch 收尾点。本轮不为画密曲线改变训练或 tracker 语义。

横轴是报告观测序号，不是 optimizer step 或 epoch。前 12 个 train 周期观测对应 step 5 / 10 / … / 60，第 13 个是同一预算结束后的收尾快照；不能一律将序号乘 5。test 的 12 个观测分别对应上述 12 个评测时刻。

本轮 `checkpoint_period: 30`，latest 在 step 30 / 60 更新；`save_best: true`，每 5 step 的 test Loss 若改善就保存 best。best 可以来自非 30 整倍数的时刻，报告须记录所选 step。

## 5. 高效率排班（§3）

- **同类一组：** 同一 `model.name` 一起并行。
- **`wait` 闸门：** 组末必须等本组进程退出、显存释放完，才开下一组。
- **吃满 GPU：** 下面的 make 用 `--num-gpus 1`。看 make 打印的 `pack N waits`。
- **依赖：** 本轮先全部 8 条 train 结束，再启动 8 条 eval；每条 train 自身还会每 5 step 完整 test。
- **error：** 单条失败只记 `run_id`；整轮结束后对未 succeeded 的格子再跑。
- **seed ≠ 并发。**

本机执行使用以下入口，先完成 §7 的配置与清单核对。

```bash
python -m rpipe make studies/main_base --num-gpus 1 --init-gpu 0 --console shared
python -m rpipe launch studies/main_base --num-gpus 1 --init-gpu 0 --console shared
```

历史探针的 `est wall` 只作打包参考。2026-10-02 显卡空闲时重跑过，整轮实际 33s，见 `STUDY_REPORT.md` 的探针记录。4 个 step 里启动和第一次 test 占了大头，不能按这个比例填 60 step。本轮已改为本机执行，且完整 test 次数更多，估时需重新核对。

## 时长预估

2026-10-02 本机 launch 前 make 已展开 16 条、8 个 wait，每组同模型的 MNIST / CIFAR10 两条。模型初始整轮预估为 **117s**；这是装箱估计，尚无本轮每 5 step 全量 test 的实测校准，不能当作完成承诺。首组结束再用实际耗时检查估计偏差，记录观测时间，不追写成预先已知的实际值。同一 `wait` 的估计墙钟取组内最慢的一条，整轮按组求和；不使用 33s × 15。

| wait | mode | model（两数据集） | 每条初始 est / 组 est |
|---:|---|---|---:|
| 1 | train | linear | 16s |
| 2 | train | mlp | 16s |
| 3 | train | cnn | 25s |
| 4 | train | resnet18 | 41s |
| 5 | eval | linear | 4s |
| 6 | eval | mlp | 4s |
| 7 | eval | cnn | 5s |
| 8 | eval | resnet18 | 6s |

以下为保留的历史四步探针估时，不能当作本轮 60-step 估时：

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
- 空闲显卡上的 actual 已写入报告，历史整轮为 33s

本轮可视化验证（执行结束后逐项核对）：

- 当前 index 中准确包含 8 train + 8 eval，均 succeeded；每条 config 的 version 为 `local-curves-20261002`，采样 / 评测预算与本计划一致
- 核对 stats、CIFAR 增强、step 5 / 10 / … / 60 的 full test 和独立 eval best 来源，逐格区分 last / best / eval；不以成功状态替代条件核对
- 每条 train Run 的密记录预期为 train 13 个观测（12 个周期记录 + 收尾）、test 12 个观测；train 稀 history 仍为一个 epoch 点。`docs/figures/learning_curves.png` 使用密记录显示多点曲线，横轴标明观测序号、Accuracy 标明百分制，保留原始 tracker 与 run.log
- `STUDY_REPORT.md` 写对照与本轮 commit、环境、有意差异、错误 / retry、各 Run 的预估与实际以及整轮耗时口径；有对照结果才作数值差异结论

## 7. 本机执行检查

1. 记录本机代码提交及未提交实现、Python / 库 / GPU 环境，核对相关测试与 torch/torchvision 导入；先验证绘图优先读取密记录、缺失时回退 history，历史 152 passed 不替代当前变更验证
2. 核对 MNIST / CIFAR10 原始数据可读、完整 split 与已有 train stats；缺失或失效才重新执行 `rpipe data`，有异常不静默回退常数
3. 维护 `experiment_config.yaml` 与 `study.yaml` 的 fixed：60 step、log period 5、eval period 5、eval num steps -1、checkpoint period 30，开启上述增强；axes 增加 `algorithm.mode: [train, eval]`，seed 仍为 0
4. 在 `fixed.version` 声明 `local-curves-20261002`（train / eval 共用），重新 make，再核对 16 条 config、index 和 jobs 中的预算、Run ID、mode、依赖与 wait 分组。不要只改 YAML 后复用旧 jobs；`--include-done` 不是 fresh，不能靠它清除已完成 latest
5. 将 make 的初始估计写入本计划或报告，再 launch；首组实际只用于校准剩余任务的预估。记录每 5 step 全量 test 的额外成本，不用 33s × 15，不覆盖历史探针证据
6. 全部结束后核对 Study process 与原始日志，生成数字表并写人工报告，按上节逐项验收。新实测保留旧 Run，不通过删除 checkpoint / log 覆盖旧证据

## 8. 刻意不做什么

- 本轮不扩成 3 seed / 48 Run，不增加训练预算或新模型；先验证单 seed 的 16 Run 与多点曲线
- 不把每 5 step 评测的结果称为旧 main 每 30 step 配方的严格复现，也不据此宣称收敛或模型排名稳定
- 不做 TensorBoard
- 不把 `cifar_grid`（`train_size=1024`、5 epoch）当作这一轮
