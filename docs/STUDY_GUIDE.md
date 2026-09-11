# Study 使用指南

怎么用**现在的代码**开一轮可复现实验。概念以 [CONCEPT.md](CONCEPT.md) 为准，目录以 [LAYOUT.md](LAYOUT.md) 为准。高效率排班见 **§3**，命令与脚本见 **§4**。

库内两柱：`structure` 与 `flow`。你要写的是包外的 **Study 目录**。入口是 `python -m rpipe`，即 flow 的 cli。

---

## 1. 三个词（决定你写什么）

| | **Study** | **Experiment** | **Run** |
|--|-----------|----------------|---------|
| 是什么 | 这一轮研究的壳 + 磁盘根 | 研究因素的一个取值点 | 该点 × 一个 seed 的一次实测 |
| 含 seed？ | 声明 `seeds` | **不含** | **至少**一个 |
| 例子 | `studies/mnist_train_size/` | `train_size=500` | `train_size=500, seed=0` |
| 磁盘 | `studies/<name>/` | 只在 **index** 里分组，无独立文件夹 | `runs/<id>/` |

展开：`study.yaml` 的 **`axes`** → 多个 Experiment；每个 × **`seeds`** → 多次 Run。

`experiment_config.yaml` 是 Study 的**基底默认**（怎么训、用什么数据/模型），不是「一个 Experiment 实例」。

---

## 2. 你要准备的文件

最少两份声明 + 一份计划（建议）：

```text
studies/<name>/
  study.yaml                 # 比什么：axes / seeds / tags
  experiment_config.yaml     # 基底：data / model / algorithm / system
  docs/
    PLAN.md                  # 研究问题（人写，可入库）
    STUDY_REPORT.md          # 跑完后写结论（人写）
```

跑完后会补上，默认不入库：

```text
  index.json                 # 按 Experiment 列 Run（make 写出）
  shared/{data,model}/       # Study 级 asset，多次 Run 共用
  runs/<id>/
    config.yaml              # 这一次 Run 的完整 config
    result.json              # Flow write：摘要（status / metrics / paths）
    assets/
      tracker/               # AlgorithmTracker 数字曲线
      logs/                  # Logger 文本（与终端同款，必写）
      checkpoints/           # latest.pt + latest/（分件）；save_best 时另有 best
```

现成例子：

| Study | 用来学什么 |
|-------|------------|
| `studies/mnist_train_size/` | 扫研究因素（三个 `train_size`，train + 独立 eval） |
| `studies/mnist_native_vs_hf/` | 同一超参：`custom_torch` vs `transformers_trainer` |
| `studies/mnist_main_recipe/` | 复现 git `main` 的 MNIST_linear（60 step + SGD extras） |
| `studies/vision_main_recipe/` | 复现 main 的 CIFAR10/SVHN × linear/mlp/cnn/resnet18（同一 60-step recipe） |
| `studies/_template/study.yaml` | 字段模版 |

新 Study 只要 `study.yaml`、`experiment_config.yaml` 和 `docs/`；格子与脚本由 **structure.make** 生成，入口是 `python -m rpipe`。

---

## 3. 高效率实验设计

追求的是**这一轮 Study 的实验效率**：把 GPU **算力和显存都吃满**，墙钟更短，同时尽量不把进程打爆。并行是排班手段，写进 `docs/PLAN.md`，和「比什么、几个 seed」一起定。

| 先分清 | 是什么 | 不是什么 |
|--------|--------|----------|
| `axes` | 研究因素，决定有几个 Experiment | 并发数 |
| `seeds` | 每个点要复测几次 | 同时开几个进程 |
| 并行 / `--round` | 同一时刻叠几个 `run-one` | seed 个数、模型个数 |

一组 `wait` 的墙钟等于组里**最慢**的那条。linear 和 resnet 放一起，linear 早就结束，整组还在等 resnet，卡上还互相抢；这是在浪费算力。

**标准（按优先级）：**

1. **正确性先于速度。** 有依赖就分波：全部 train `wait` 完再 eval。已 `succeeded` 的默认跳过；中断后续 `latest`。一次 `FlowRunner` 只跑一个 Run。
2. **吃满 GPU：显存用好、计算跑满、尽量不 error。** 同类、相近耗时的格子一起并行（CIFAR linear 和 SVHN linear 一组；resnet 和 linear 分开）。组内按当前空闲显存（约 50% 安全系数）能叠几个就叠几个，把 SM / 显存占住。估得太满会 OOM，所以保守叠，而不是按空卡理想值打穿。
3. **error 不中断整轮。** 某条 `run-one` 失败：记下 `run_id` 和退出码（日志在该 Run 的 `assets/logs/`），**同组其余进程和后面的组继续跑完**。全部命令结束后，对未 `succeeded` 的格子再排一次，用 `resume: latest` 续跑。不要一组一挂就停掉整张卡。
4. **按本轮格子排班。** 轻的同类型可以叠很多；重的 resnet 可能一组 1～2 个。不要用一个全局 `--round` 把轻重砍齐。默认 `auto` 按类型装箱；`--round N` 是均匀切块。长训丢独立终端。
5. **PLAN 里写清排班。** 几个 seed、哪类一组、error 后怎么续。报告里复述实际怎么跑的。

机制（`&` / `wait`、脚本形状）见下一节。

---

## 4. 一条命令怎么跑

```bash
pip install -e ".[dev]"

# 只写出格子：config + index
python -m rpipe run studies/<name> --skip-launch

# 写出格子并按顺序跑每个 Run
python -m rpipe run studies/<name>

# 写出格子与调度脚本，再按 §3 同类装箱并行
python -m rpipe make studies/<name> --num-gpus 1 --init-gpu 0
python -m rpipe launch studies/<name> --num-gpus 1 --init-gpu 0
# 默认 --round auto（装箱）。手写均匀切块：--round 4
# 也可在独立终端跑：
#   studies/<name>/scripts/launch.ps1
#   bash studies/<name>/scripts/launch.sh
```

排班标准见 §3。下面是脚本形状（跟 git `main` 一样：`&` + `wait`）。

进程级并行。默认 `auto`：同类一组、显存吃满但留安全系数。某条失败只打印 `error <id>`，整轮 `wait` 完再对失败格子 `resume` 重跑一次。手写 `--round N` 仍是均匀切块。

有 `algorithm.mode: eval` 时拆成两波：全部 train `wait` 完再启动 eval。

`mnist_train_size`：18 次 Run、`--round 4`、`--num-gpus 1` 时，`studies/mnist_train_size/scripts/launch.sh` 形状如下（路径已缩短）：

```bash
#!/bin/bash
cd "<repo>"
export KMP_DUPLICATE_LIB_OK=TRUE
# 第一波：9 次 train
CUDA_VISIBLE_DEVICES="0" python -m rpipe run-one "<study>" "<train>" &
CUDA_VISIBLE_DEVICES="0" python -m rpipe run-one "<study>" "<train>" &
CUDA_VISIBLE_DEVICES="0" python -m rpipe run-one "<study>" "<train>" &
CUDA_VISIBLE_DEVICES="0" python -m rpipe run-one "<study>" "<train>"
wait
CUDA_VISIBLE_DEVICES="0" python -m rpipe run-one "<study>" "<train>" &
CUDA_VISIBLE_DEVICES="0" python -m rpipe run-one "<study>" "<train>" &
CUDA_VISIBLE_DEVICES="0" python -m rpipe run-one "<study>" "<train>" &
CUDA_VISIBLE_DEVICES="0" python -m rpipe run-one "<study>" "<train>"
wait
CUDA_VISIBLE_DEVICES="0" python -m rpipe run-one "<study>" "<train>" &
wait
# 第二波：9 次 eval（上一波全部 wait 完才到这里）
CUDA_VISIBLE_DEVICES="0" python -m rpipe run-one "<study>" "<eval>" &
CUDA_VISIBLE_DEVICES="0" python -m rpipe run-one "<study>" "<eval>" &
CUDA_VISIBLE_DEVICES="0" python -m rpipe run-one "<study>" "<eval>" &
CUDA_VISIBLE_DEVICES="0" python -m rpipe run-one "<study>" "<eval>"
wait
# …再两组 eval，最后一条同样是 cmd & + wait
```

Windows 上 `launch.ps1` 只转调 `python -m rpipe launch`，波次在 Python 里用 `Popen` 复现。多卡时同一组里会看到 `CUDA_VISIBLE_DEVICES="0"`、`"1"`、… 轮转。`scripts/` 默认 gitignore。

1. **make** 读 `study.yaml`，按 `axes` × `seeds` 展开  
2. 每个补丁 ⊕ `experiment_config.yaml` → `runs/<id>/config.yaml`  
3. 写 `index.json`  
4. 按参数对每个 Run 跑：prepare → execute → collect → summarize → **write** → process  

常用参数：

| 开关 | 作用 |
|------|------|
| `--skip-launch` | 只写出 config 与 index |
| `--phases prepare,execute,...` | 只跑列出的阶段；相对顺序不变 |
| `rpipe make` | 写出 config、index 与 `scripts/` |
| `rpipe launch` | make 之后按装箱（或 `--round N`）跑未完成 Run |
| `--round` / `--num-gpus` / `--init-gpu` | `auto` = §3 装箱；`N` = 均匀切块；卡号轮转 |
| `--include-done` | 脚本里包含已经 succeeded 的 Run |

`python -m rpipe study run …` 与上面等价，只是旧别名。

---

## 5. 写 `experiment_config.yaml`

基底字段对应 structure 四层。native 真训通路：有 `model.module` 且 `Data.iter_batches`（不再要求 `data.name: MNIST`）。`algorithm.source` 默认 `custom_torch`；`transformers_trainer` 用同一套 `optimizer` / `scheduler` / `resume` 键映射到 `TrainingArguments`。

```yaml
experiment: mnist_linear          # 给人看的名字，不是磁盘路径
description: MNIST linear classifier
data:
  name: MNIST
  source: torch                   # torch：真下 MNIST 到 shared/data；stub/Toy：不下载
  config:
    train_size: 1000              # 可被 study.yaml 的 axes 覆盖
    batch_size: 64
model:
  name: linear                    # 默认 MNIST 784→10；CIFAR/SVHN 由 Data.meta.data_size 对齐
algorithm:
  source: custom_torch            # 或 transformers_trainer（同一套键 → TrainingArguments）
  mode: train
  num_epochs: 20              # 有 epoch 概念时：推导并覆盖 num_steps
  progress_unit: epoch        # 本例按 epoch 评 test / 存 latest；默认 step（LLM 只写 num_steps）
  eval_period: 1              # 每 N 个进度单位评 test；0 = 只在训完评一次
  checkpoint: latest          # latest = 覆盖 latest 这一份（.pt 整包 + 目录分件）；percent = 再按总预算百分比留快照
  checkpoint_period: 1        # 每 N 个单位更新 latest；0 = 只在训完写一次
  save_best: true             # 默认：test Accuracy 最好时另写 best；可用 best_metric / best_mode 改口径
  resume: latest              # train 的 resume 接口；没有 latest 则从头。eval Run 用 resume: best
  optimizer: SGD              # 名字；momentum / nesterov / weight_decay 等同层 extras，按构造函数 signature 过滤
  max_grad_norm: 0            # 缺省 / 0 = 不裁。HF Trainer 自带 1.0，必须映射此键
  lr: 0.1
  scheduler: cosine            # 算法层接口；无 / constant = 固定 lr；HF 映射 lr_scheduler_type
  eta_min: 0.0
system:
  device: cpu
  deterministic: false          # prepare 最先落地；true 则 cudnn.deterministic + use_deterministic_algorithms
  cudnn_benchmark: true         # 跟旧 main；deterministic 开时默认关
```

合并规则：`study.yaml` 的 `fixed` 与 `axes` 补丁 **覆盖** 基底同名字段（深层 dict 合并，list 整段替换）。

---

## 6. 写 `study.yaml`

### 扫研究因素（一个因素一个 Experiment）

```yaml
study: mnist_train_size
description: MNIST train_size sweep → test accuracy

experiment:
  name: mnist_linear

fixed:
  data:
    name: MNIST
    source: torch
    config:
      batch_size: 64
  model:
    name: linear
  algorithm:
    source: custom_torch
    num_epochs: 20
    progress_unit: epoch      # 每个 epoch 评一次 / 更新 latest；不写则默认 step（eval_period: 1 会每步评 test）
    eval_period: 1
    checkpoint: latest
    checkpoint_period: 1
    save_best: true
    optimizer: SGD
    max_grad_norm: 0
    lr: 0.1
    scheduler: cosine
    eta_min: 0.0
  system:
    device: cpu
    deterministic: false
    cudnn_benchmark: true

axes:
  data.config.train_size: [500, 2000, 8000]
  algorithm.mode: [train, eval]

seeds: [0, 1, 2]

tags:
  - when:
      data.config.train_size: 500
      algorithm.mode: train
    tags: [baseline]

run_description: "train_size={train_size} mode={mode} seed={seed}"
```

这会得到 **6 个 Experiment**（size × mode），每个下面 **3 个 Run**。同一 `train_size` 先 train 再 eval；eval 默认 resume `best`，从 sibling train Run 读 `best`。`process.paired` 把 train/eval 拼成报告表。九次训练共用 `shared/data` 里的 MNIST。

### 只扫 seed（一个 Experiment，多次 Run）

`axes: {}`，因素全部放进 `fixed`，再写 `seeds: [0, 1]`。index 里会是 **一组** `factors: {}`，下面两条 Run（只差 seed）。不需要单独再开一个 Study 目录。

### 字段约定

| 字段 | 含义 |
|------|------|
| `fixed` | 每次 Run 都带上的补丁（不要把研究因素只写在这里却期望它变成多个 Experiment） |
| `axes` | 研究因素；每个取值组合 = 一个 Experiment。**不要把 seed 放这里** |
| `seeds` | 每个 Experiment 下的随机复测 |
| `tags` | `when` 匹配当前格子（可含 seed）则打标签；`baseline` 只是 tag |
| `run_description` | 写入 config 的说明；**不进** `id` hash。占位符可用轴的末段名（如 `{train_size}`）以及 `{seed}`、`{experiment}` |

`id` hash **包含** 实验变量、seed、tags；**不含** `id`、`description`。同内容再跑会落到同一 `runs/<id>/`。

---

## 7. 跑完看什么

**index**（按 Experiment 分组，不是扁平 run 列表）：

```json
{
  "study": "mnist_train_size",
  "experiments": [
    {
      "factors": { "data.config.train_size": 500 },
      "runs": [
        { "id": "…", "seed": 0, "tags": ["baseline"], "config": "runs/…/config.yaml" },
        { "id": "…", "seed": 1, "tags": ["baseline"] },
        { "id": "…", "seed": 2, "tags": ["baseline"] }
      ]
    }
  ]
}
```

**result**（`runs/<id>/result.json`）：`status`、`metrics`（如 `train_loss` / `accuracy`）、`control`、`paths`。成功则 `status: succeeded`。

`metrics.train_loss` 应是 AlgorithmTracker **最后一段 train mean**，不是最后一个 batch 的 CE。完整曲线在 `runs/<id>/assets/tracker/`；终端同款文本**必写** `assets/logs/`。

`process` 读 sibling result 写 `process.json`（mean / std / Δ），并据各 Run `tracker_state.json` 的 `history` 画出 **`docs/figures/learning_curves.png`**。**不**改 `STUDY_REPORT.md`。

`docs/STUDY_REPORT.md` **必须有图**（至少嵌上 learning curve），不能只有表格和文字。图从 `docs/figures/` 引用；数字读 `process.json`。

---

## 8. 新 Study 最小步骤

1. 对照 `_template/study.yaml` 或 `studies/mnist_train_size/` 的 yaml 与 `docs/`，改目录名。  
2. 改 `experiment_config.yaml` 的基底；改 `study.yaml` 的 `axes` / `seeds` / `tags`。  
3. 在 `docs/PLAN.md` 写清：比什么、什么固定、成功标准，以及 **§3 高效率排班**（同类一组、吃满 GPU、error 记下来整轮后再 resume）。  
4. `python -m rpipe run studies/<name> --skip-launch`，核对 index。  
5. `python -m rpipe make studies/<name>`，看打印的 `pack N waits`，再 `python -m rpipe launch studies/<name>`。  
6. 读 `process.json` + `docs/figures/learning_curves.png`，按 Experiment 写 `docs/STUDY_REPORT.md`。

检查清单：

- [ ] 目录最终有 `docs/`、`shared/`、`runs/`  
- [ ] `axes` 与 `seeds` 分开  
- [ ] 每个 Run 的 config 含 `seed`  
- [ ] 结论按 Experiment 聚合，而不是按扁平 run 列表  
- [ ] `PLAN.md` / `STUDY_REPORT.md` 写清本轮怎么并行（同类一组、error 后续跑）  
- [ ] `STUDY_REPORT.md` 有 learning curve（或同等图），不是只有表格  

---

## 9. 现成能力 vs 要改库

| 你想做的 | 怎么做 |
|----------|--------|
| 扫已有字段（样本量、lr、seed…） | 只改 Study 的 yaml |
| 换 MNIST 子集大小 / epoch | yaml 即可 |
| 新数据集、新模型、新训练循环 | 改 `structure.data` / `model` / `algorithm`，再在 yaml 里点名 |
| 换 HF Trainer / Accelerate | 改 `algorithm.source`；`optimizer` / `scheduler` / `resume` 键不变（structure.md §6.11） |
| 独立评测（加载 best） | 另一次 Run：`algorithm.mode: eval`，`resume: best`；不是 Flow 多一个阶段 |
| 断点续训 | train 的 `resume: latest`（算法接口；system 只读文件） |
| 改一次 Run 的阶段顺序 | 不要改；最多 `--phases` 裁剪，相对顺序不变 |
| 自动出报告 / 跨 Run 对比表 | 人写 `STUDY_REPORT.md`（必须嵌图）；`process` 出 mean/std/Δ 和 `docs/figures/learning_curves.png` |
| 训练曲线 | process 画 epoch `history` → `docs/figures/`；密点仍在 `scalars.jsonl`。不做 TensorBoard |
| 终端 + 硬盘日志 | **Logger**（system）必写 `assets/logs/`，每次 report **flush** |

脚本里若要编程调用：`from rpipe.flow.cli import run_study`。读写路径用 `rpipe.structure.artifact`；造格子用 `rpipe.structure.make`。
