# Study 使用指南

怎么用**现在的代码**开一轮可复现实验。概念以 [CONCEPT.md](../docs/code/concept.md) 为准，目录以 [LAYOUT.md](../docs/code/layout.md) 为准。高效率排班见 **§3**，命令与脚本见 **§4**。

库内两柱：`structure` 与 `flow`。你要写的是包外的 **Study 目录**。入口是 `python -m rpipe`，即 flow 的 cli。

---

## 1. 三个词（决定你写什么）

| | **Study** | **Experiment** | **Run** |
|--|-----------|----------------|---------|
| 是什么 | 这一轮研究的壳 + 磁盘根 | 研究因素的一个取值点 | 该点 × 一个 seed 的一次实测 |
| 含 seed？ | 声明 `seeds` | **不含** | **至少**一个 |
| 例子 | `mnist_train_size`（声明在 `tests/_data/studies/`） | `train_size=500` | `train_size=500, seed=0` |
| 磁盘 | `studies/<name>/` | **无文件夹** | `runs/<id>/` |
| 这一级看什么 | 声明、shared、index、process 信封、报告 | 跨 seed 的 **mean / std / min / max** | 这一次的 config / result / `run.log` |

展开：`study.yaml` 的 **`axes`** → 多个 Experiment；每个 × **`seeds`** → 多次 Run。

`experiment_config.yaml` 是 Study 的**基底默认**（怎么训、用什么数据/模型），不是「一个 Experiment 实例」。

---

## 2. 你要准备的文件

最少两份声明 + 一份计划（建议）：

正式研究或可复用的验收案例放 `studies/`。一次性环境排错（例如文件占用、杀毒软件开关复测）放 `.tmp/diagnostics/`，把条件、结论及必要数字合并进关联 Study 报告；原始日志、缓存和临时配置保留本地，不为每次排错增加一个正式 Study。

```text
studies/<name>/
  study.yaml                 # 比什么：axes / seeds / tags
  experiment_config.yaml     # 基底：data / model / algorithm / system
  docs/
    PLAN.md                  # 研究问题（人写，可入库）
    STUDY_REPORT.md          # 跑完后写结论（人写）
    figures/                 # process 画出的 learning_curves.png
```

`docs/NUMBERS.md` 不在这份清单里：`rpipe report` 从 `process.json` 生成，可以入库，但它不是结论。

跑完后会补上，默认不入库：

```text
  index.json                 # 按 Experiment 列 Run；每条含 log → 该 Run 的 run.log
  process.json               # Study 信封；experiments[] 里是跨 seed 的 mean / std / min / max
  shared/{data,model}/       # Study 级 asset：make/launch 先准备共享数据，再 spawn
  runs/<id>/
    config.yaml              # 这一次 Run 的完整 config
    result.json              # Flow write：摘要（status / metrics / paths）
    assets/
      tracker/               # AlgorithmTracker 数字曲线
      logs/run.log           # 这一次 Run 的 Logger 文本（无 Study 级总 log）
      checkpoints/           # latest.pt + latest/（分件）；save_best 时另有 best
```

现成 Study：

| **Study** | **研究或验收范围** | **计划 / 报告** | **本机原始产物** |
|---|---|---|---|
| `_template` | 新 Study 的配置与报告起点 | [计划](_template/docs/PLAN.md) / [报告模板](_template/docs/STUDY_REPORT.md) | 尚未执行 |
| `main_reproduction` | 固定现代 main 的 60-step 计算对照、历史来源审计与 200-step 前缀探针；新机结果见 CURRENT_DEVICE | [计划](main_reproduction/docs/PLAN.md) / [原报告](main_reproduction/docs/STUDY_REPORT.md) / [本机结果](main_reproduction/docs/CURRENT_DEVICE_RESULT.md) | 原始 Run 与对照目录不随 Git 提供；产出结果的脚本在 `code/` |
| `main_historical` | 历史 PNG 候选配方的四 seed、80000-step 完整曲线复现；使用专用入口 | [入口](main_historical/README.md) / [计划](main_historical/docs/PLAN.md) / [完整报告](main_historical/docs/STUDY_REPORT.md) | 完整 Run / data / preflight / 诊断证据留在执行长实验的机器上，不随 Git 提供 |

阅读顺序：先 [main_historical 专用入口](main_historical/README.md)，再 [历史完整曲线报告](main_historical/docs/STUDY_REPORT.md)，再 [当前设备现代 main 结果](main_reproduction/docs/CURRENT_DEVICE_RESULT.md)。

2026-10-10 只保留以上三个 Study。其余 Study（`mnist_train_size`、`mnist_native_vs_hf`、`cifar_grid`、`main_base`、`checkpoint_recovery`、`mnist_cnn_lr`、`mnist_cnn_budget`、`mnist_cnn_budget_repeat`、`local_model_matrix`、`support_data_smoke`、`support_model_smoke`）已删除，计划与报告见提交 [`71143ab`](https://github.com/diaoenmao/RPipe/tree/71143ab/studies)。测试要用的 `mnist_train_size`、`mnist_native_vs_hf`、`cifar_grid` 声明在 [`tests/_data/studies/`](../tests/_data/studies/)。

Study 代码：

1. `main_historical/` 根目录：`run.py`（专用入口，子进程 prepare 前注册 `historical_4ccb28d`）、`recipe.py`、`prepare_data.py`、`verify_preflight.py`、`verify_group.py`、`compare.py`、`publish_figures.py`，以及 `docs/` 下的两份执行辅助脚本。报告里提到的部分审计脚本（如 `historical_group_audit.py`、`historical_controller_guard.py`、`final_historical_execution_audit.py`）当时留在执行机的 `.tmp/`，这台机器上没有，尚未入库。
2. `main_reproduction/code/`：从 `.tmp/main-reproduction-20261003/` 原样复制的 24 个脚本，包括数据准备、原始 main 与当前 RPipe 运行、CPU/GPU parity、B-017/B-018 验证、历史桥接与前缀探针、比较、作图和写报告。部分脚本会校验自己的 SHA-256，因此不改内容。它们假设脚本目录下有 `reference/src`（固定 main 源码）、`deps/` 和 `original-evidence/`，并用 `parents[1]` / `parents[2]` 定位仓库根，所以不能在 `code/` 里直接运行。`prepare_real_data.py`、`prepare_stats.py` 和三个 parity 脚本读取原 `studies/main_base/shared/data` 与 `studies/support_model_smoke/shared/data/cifar10` 的缓存。这两个 Study 已删除，复跑前需要先把同样的原始数据和 `stats.yaml` 放回这些路径，或改成从官方源重新下载并核对 SHA-256。复跑时先复制到 `.tmp/main-reproduction-<date>/`，再按 [计划](main_reproduction/docs/PLAN.md) 与 [报告](main_reproduction/docs/STUDY_REPORT.md) 的顺序准备依赖和执行。`current_device.py` 是本机对照入口。

声明、研究计划、报告、图、正式数字与来源清单随 Git 提供。`runs/`、`shared/`、`scripts/`、`index.json`、`process.json` 和 `.tmp/` 按仓库忽略规则留在本机；clone 只有报告，不自动拥有报告内引用的原始 checkpoint、日志和数据。旧报告及正式 JSON 保留自己的实际日期、来源和判定。不能用本机新结果补写旧实验的原始证据。

`main_historical` 的 `historical_4ccb28d` Registry 必须由专用 `run.py` 在每个子进程 prepare 前注册；直接运行通用 `python -m rpipe run/launch studies/main_historical` 不会完成该注册。准备依赖、数据、同设备 preflight 和长矩阵的顺序以该 Study 专用入口为准。

### 模版目录（`studies/_template/`）

```text
studies/_template/
  study.yaml
  experiment_config.yaml
  docs/
    PLAN.md          # 必须含 §3 高效率排班
    STUDY_REPORT.md  # 跑完再填；必须有图
```

PLAN 里写清：比什么、固定什么、几个 seed、同类怎么一组、error 后 resume。复制后把 `my_study` 改成新目录名。

新 Study 复制 `_template/`，改 yaml 与 `docs/`；格子与脚本由 **structure.make** 生成，入口是 `python -m rpipe`。

---

## 3. 高效率实验设计

追求的是**这一轮 Study 的实验效率**：把 GPU **算力和显存都吃满**，墙钟更短，同时尽量不把进程打爆。并行是排班手段，写进 `docs/PLAN.md`，和「比什么、几个 seed」一起定。

| 先分清 | 是什么 | 不是什么 |
|--------|--------|----------|
| `axes` | 研究因素，决定有几个 Experiment | 并发数 |
| `seeds` | 每个点要复测几次 | 同时开几个进程 |
| 并行 / `--round` | 同一时刻叠几个 `run-one` | seed 个数、模型个数 |

一组并发叫一个 **wait 组**。组内用 `&` 叠满当前估得下的显存；组末必须 **`wait`**：本组进程全部退出、显存释放完，才启动下一组。不 `wait` 的话下一组会挤进还在跑的进程，显存叠加，容易 OOM。所以 `wait` 不是可选项，是并行化的内存闸门。

一组 `wait` 的墙钟等于组里**最慢**的那条。`make` 打的 conservative 墙钟是各组这个 max **再加总**，只供排班参考，不是实测。linear 和 resnet 放一起，linear 早就结束，整组还在等 resnet，卡上还互相抢；这是在浪费算力。

**标准（按优先级）：**

1. **正确性先于速度。** 默认整轮 launch：有依赖就分波，全部 train `wait` 完再 eval。也可以 `--mode eval` 单独重跑评测。已 `succeeded` 的默认跳过；中断后续 `latest`。一次 `FlowRunner` 只跑一个 Run。
2. **吃满 GPU：显存用好、计算跑满、尽量不 error。** 同类、相近耗时的格子一起并行（CIFAR linear 和 SVHN linear 一组；resnet 和 linear 分开）。组内按当前空闲显存（约 50% 安全系数）能叠几个就叠几个，把 SM / 显存占住。估得太满会 OOM，所以保守叠，而不是按空卡理想值打穿。
3. **error 不阻断无关任务。** 某条 `run-one` 失败：记下 `run_id` 和退出码（日志在该 Run 的 `assets/logs/`），同组其余进程继续。本波结束后对未 `succeeded` 的格子重试一次，用 `resume: latest` 续跑；依赖该 train 的 eval 只有在它成功后才运行，不把局部故障扩散到其他 seed。
4. **按本轮格子排班。** 轻的同类型可以叠很多；重的 resnet 可能一组 1～2 个。不要用一个全局 `--round` 把轻重砍齐。默认 `auto` 按类型装箱；`--round N` 是均匀切块。
5. **PLAN 里写清排班。** 几个 seed、哪类一组、error 后怎么续。报告里复述实际怎么跑的。

`system.device` 决定资源队列：`cpu` Run 不绑定 GPU、不设置 `CUDA_VISIBLE_DEVICES`，按 CPU 并发上限分组；`cuda` Run 才探测 GPU 并按显存装箱。混合 Study 中两类 Run 分组执行。默认整轮仍是 train 全部完成后再进入 eval；`--mode` 可以只发其中一波。

机制（进程并行 / `wait`、脚本入口）见下一节。

---

## 4. 一条命令怎么跑

```bash
pip install -e ".[dev]"
# transformers_trainer 通路：pip install -e ".[dev,hf]"

# 只写出格子：config + index
python -m rpipe run studies/<name> --skip-launch

# 写出格子并按顺序跑每个 Run
python -m rpipe run studies/<name>

# 写出格子与调度脚本，再按 §3 同类装箱并行
# pack 只出现在 make；launch 复用 scripts/jobs.json
python -m rpipe data studies/<name>
python -m rpipe make studies/<name> --num-gpus 1 --init-gpu 0
python -m rpipe launch studies/<name> --num-gpus 1 --init-gpu 0
# launch 结束会跑 Study process；也可单独再跑：
python -m rpipe process studies/<name>
# 默认 --round auto（装箱）。手写均匀切块：--round 4
# 也可在独立终端跑：
#   studies/<name>/scripts/launch.ps1
#   bash studies/<name>/scripts/launch.sh
```

`data` 先给每个数据集写一份概况：`studies/<name>/shared/data/<数据集>/stats.yaml`（张数、形状、类别数量、像素范围，以及 train 的 mean / std）。Normalize 优先读这份文件。

排班标准见 §3。保持与 git `main` 一致的组内进程并行 / 组末 wait；生成脚本复用同一调度器，以免裸 `run-one` 绕过失败重试和依赖检查。

进程级并行。默认 `auto`：同类一组、显存吃满但留安全系数。每组末尾的 `wait` 挡住下一组，避免还在占显存时下一波挤进来。某条失败打印 `error <id>`，在本波结束后对失败格子 `resume` 重跑一次；train 的重试必须在 eval 波之前完成。手写 `--round N` 仍是均匀切块。

有 `algorithm.mode: eval` 时，**默认**一次 `launch` 拆成两波：全部 train `wait` 完再启动 eval。Study 可以按这个写 PLAN。要单独重跑 eval（或只发 train）：`python -m rpipe launch studies/<name> --mode eval`。已成功的格子默认 skip，加上 `--include-done`。单条仍可用 `run-one`。eval 找不到 sibling `best` 照样失败。

**失败恢复验收约定：** `launch` 按当前 index 的同因素（仅替换 mode）与同 seed 识别 sibling train；它最终未成功时，不启动依赖它的 eval，并将该 eval 记为非成功，其他独立任务仍可继续。开始重跑 train 前，已成功的相关 sibling eval 必须失效，防止 `--mode train` 后聚合旧评估。后续 launch 还会将父 train 未成功、或 train result 修改时间晚于 eval result 的旧成功 eval 视为待重评；不因此重写 jobs 清单，清单漏掉的 Run 仍需 `--remake`。`resume: false`、明确的外部权重或 eval 自己的 checkpoint 不应被误当作 sibling 依赖。

这是本地 `launch` 的依赖保护，不是通用 DAG 或文件来源追踪系统。手动修改/复制权重、回拨文件时间，或绕过调度器直接 `run-one`，不在自动失效承诺内；研究结论仍需核对实际 checkpoint 来源。`--include-done` 只是重执行，不清 checkpoint；需要独立无中断对照时换新 `version`。

以 `tests/_data/studies/mnist_train_size/` 复制成 `studies/mnist_train_size/` 为例：18 次 Run、`--round auto`、1 张卡时，linear 很轻，通常 **2 个 wait 组**（9 train，再 9 eval）。推荐入口：

```bash
python -m rpipe launch studies/mnist_train_size --num-gpus 1 --console shared
# 或使用 make 生成的脚本
bash studies/mnist_train_size/scripts/launch.sh
```

`launch.sh` 用 Python heredoc 调用同一 `launch_jobs`，保留 make 生成的 wait 组及 GPU 分配；`--split-round` 仍按 wait 组切脚本，按顺序执行各片段，最后一个片段执行 Study process。每组子进程全部退出后下一组才启动。旧版已生成的裸 `run-one` 脚本不会自动变更，需要重新 make 后使用新脚本。

训练 Logger 每一行是 `时间 级别 Run id [事件] 内容`，写入该 Run 的 `run.log`（与终端同一套）。时间是本地 RFC 3339（毫秒和时区）。`--console shared` 时终端会混，靠 Run id 这一格分辨；文件仍是每 Run 一份。`[time]` 里的 `elapsed` / `eta` 是这一轮的进度时钟。Windows 上 `python -m rpipe launch` 默认 **每个 run-one 一个新控制台窗口**。`launch.ps1` 只转调 `rpipe launch`（有 `jobs.json` 就不再 make）。多卡时仍设 `CUDA_VISIBLE_DEVICES`。`scripts/` 默认 gitignore。

Windows / Conda 若报 `OMP: Error #15`，说明环境里加载了多份 OpenMP runtime。先用 `where.exe libiomp5md.dll` 检查来源，并在同一个包管理器中重装 PyTorch / NumPy，或改用干净虚拟环境。`KMP_DUPLICATE_LIB_OK=TRUE` 只能由使用者临时显式设置用于诊断；RPipe 不默认注入它。

1. **make** 读 `study.yaml`，按 `axes` × `seeds` 展开  
2. 每个补丁 ⊕ `experiment_config.yaml` → `runs/<id>/config.yaml`  
3. 写 `index.json`（每条 Run 带 `log`）  
4. 按 `data.name` + `source` 各准备一次共享数据到 `shared/data/`（忽略 `train_size`；只有成功后的 `.ready` 才跳过；半截压缩包会重下；下载不刷 tqdm）。数据地址和模型 hub 都看 Study 的 `origin`。  
5. **launch** 读 `scripts/jobs.json` 跑未完成 Run（缺清单或 `--remake` 才再 make）；每个 Run：prepare → execute → collect → summarize → **write** → process  

常用参数：

| 开关 | 作用 |
|------|------|
| `--skip-launch` | 只写出 config 与 index（不下载共享数据、不 spawn） |
| `--phases prepare,execute,...` | 只跑列出的阶段；相对顺序不变 |
| `rpipe make` | 写出 config、index、`scripts/jobs.json`，并把共享数据落到 `shared/data/`；这里打印 `pack N waits` |
| `rpipe launch` | 已有 `scripts/jobs.json` 且 GPU/`round` 一致则直接跑未完成 Run，**不**再 make、**不**重印 `pack`；缺清单、参数变了或 `--remake` 才 make。Windows 默认每条 Run 新窗口；全部 wait 完再跑 Study `process` |
| `--mode` | 仅 `launch`：只发该 `algorithm.mode`（可重复，如 `eval`）。**不**改写 `jobs.json`。已成功的默认 skip，重跑加 `--include-done` |
| `--remake` | 仅 `launch`：忽略已有 `jobs.json`，重新 make 再跑 |
| `--console` | `auto`（Windows=`new` 窗口 / 其它=`shared`）；`new`；`shared` |
| `rpipe process` | 只跑 Study 级聚合（信封 + Experiment 的 mean/std/min/max + 图） |
| `rpipe status` | 只读，实现在 `structure/artifact/readout/`。把 `index.json` 和各条 `result.json` 列成表。`pending` / `failed` 才填 `note`（见 §7）。`--mode` 可滤。不写文件 |
| `rpipe logs` | 同一份 readout。各 Run 的事件行按时间打到终端。不写 Study 级总 log |
| `rpipe report` | 同一份 readout。从 `process.json` 写 `docs/NUMBERS.md`。不改 `STUDY_REPORT.md` |
| `--round` / `--num-gpus` / `--init-gpu` | `auto` = §3 按 `system.device` 分流，CUDA 按显存装箱、CPU 按进程上限分组；`N` = 均匀切块；GPU 卡号轮转 |
| `--include-done` | 把 `jobs.json` 里已经 succeeded 的也排进去（仍复用清单；清单里没有的格子才要 `--remake --include-done`） |

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
    # pin_memory: true            # DataLoader；缺省 false
    # num_workers: 0
model:
  name: linear                    # 默认 MNIST 784→10；CIFAR/SVHN 由 Data.meta.data_size 对齐
algorithm:
  source: custom_torch            # 或 transformers_trainer（同一套键 → TrainingArguments）
  mode: train
  num_epochs: 20              # 有 epoch 概念时：推导并覆盖 num_steps
  progress_unit: epoch        # 本例按 epoch 评 test / 存 latest；默认 step（LLM 只写 num_steps）
  eval_period: 1              # 每 N 个进度单位评 test；0 = 只在训完评一次
  # eval_num_steps: -1        # 正整数限test batch数；缺省 / <0 = 完整test；0报错
  checkpoint: latest          # latest = 覆盖 latest 这一份（.pt 整包 + 目录分件）；percent = 再按总预算百分比留快照
  checkpoint_period: 1        # 每 N 个单位更新 latest；0 = 只在训完写一次
  save_best: true             # 默认：test Accuracy 最好时另写 best；可用 best_metric / best_mode 改口径
  metric:                     # 可选；默认 train/test 都是 Loss + Accuracy（structure.md §6.9）
    train: [Loss, Accuracy]   # 还认 MSE（batch）；RMSE / GLUE 是 full，在 save() 时收口
    test: [Loss, Accuracy]
  # 不要在会覆盖 eval 的基底里写 resume。缺省：train=latest（没有则从头），eval=best（sibling train）
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
origin: foreign                 # Study 级。domestic 时数据走国内镜像，模型 hub 为 hf-mirror.com
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
| `recipe` | 可选，Study 内的 Python 文件（如 `recipe.py`），定义 `register(ctx)`，注册本 Study 自己的 data / model / algorithm `source`。prepare 在每条 Run 建构前调用；**不进** `id` hash。见 [flow.md](../docs/code/flow.md) §14.1 |
| `freeze` | `true` 时，make 之后源码、声明、recipe 或计划有任何变化，`launch` / `run-one` 都拒绝运行，重新 make 才接受 |
| `provenance.include` | 额外纳入来源清单的 Study 内文件（glob 列表），例如固定的参考数据或清单 |

### Study 里只写声明

调度、阶段链、来源记录和 Run 对比都由库提供，Study 不再自写：

| 需要 | 用库里的 |
|------|----------|
| 注册 Study 特有的数据、模型或算法 | `recipe` + `register(ctx)`，不要另写 `run.py one` / `launch` |
| 记录源码、计划和环境，防止中途改代码 | make 写的 `provenance.json`，加 `freeze: true` |
| 两条 Run 的指标、曲线、checkpoint 是否一致 | `python -m rpipe compare <run_a> <run_b> [--atol --rtol --checkpoint NAME --out FILE]` |
| 聚合、数字表、曲线图 | `process` / `report` |

Study 目录内的代码只剩 recipe 和必要的一次性数据准备；与外部实现对比时，先把外部结果写成同样的 Run 目录，再用 `compare`。

`id` hash **包含** 实验变量、seed、tags，以及可选 `version`；**不含** `id`、`description`。Run 是最底层的一次实测。同内容、同 `version` 再跑仍落到同一 `runs/<id>/`，用于 skip / resume；需要避免相同实验参数与 seed 的不同实测发生 ID 冲突时，换一个 `version` 生成新 Run。timestamp 只是可选内容之一。

version 可写在 `experiment_config.yaml` 顶层，或在 `study.yaml` 的 `fixed` 下覆盖，例如：

```yaml
fixed:
  version: "baseline-repeat-2"
```

这段合并到已有 fixed，不替换其余配置。不要写成 Study 顶层的 `version`，它不会被展开。省略字段保持原有 ID；改变它后重新 make，旧 Run 文件保留，新 index 只列本轮。launch 会复用 jobs，`--include-done` 不清 checkpoint，不代表从头训练。需要保留一次独立实测时使用新 version，不靠删除旧日志或权重实现。

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
        { "id": "…", "seed": 0, "tags": ["baseline"], "config": "runs/…/config.yaml", "log": "runs/…/assets/logs/run.log" },
        { "id": "…", "seed": 1, "tags": ["baseline"] },
        { "id": "…", "seed": 2, "tags": ["baseline"] }
      ]
    }
  ]
}
```

**result**（`runs/<id>/result.json`）：`status`、`metrics`（如 `train_loss` / `accuracy`）、`control`、`paths`。成功则 `status: succeeded`。

**`rpipe status`** 把上面两份拼成一张表，只读，不写文件、不跑训练。实现在 `structure/artifact/readout/`，`flow/cli.py` 只转发：

```bash
python -m rpipe status studies/<name>
python -m rpipe status studies/<name> --mode eval
```

先打一行计数：`planned` / `succeeded` / `failed` / `pending`。然后每条 Run 一行（tab）：`status`、`mode`、`seed`、`id`、因素、一个 metric、失败时的 `error`、`note`、`log`。顺序跟 index。没有 `result.json` 的格子是 `pending`。metric 只在 `succeeded` 时出现，优先 `accuracy`，否则 `best_accuracy` 或 `train_loss`。`pending` 的 `note` 读该 Run 的 `run.log`：最后一条 `[epoch]` 或 `[error]` 摘要，跳过 `Traceback` / `File ` 续行；都没有就用最后一条 `[flow]`。没有日志是 `-`。`failed` 的 `error` 仍是 result 里的消息，`note` 是日志里最后一条 `[error]` 摘要。`succeeded` 的 `note` 是 `-`。`--mode` 只滤显示。缺 `index.json` 则退出码 2。

`launch` 每一组开始前打 `launch: wait i/n mode=train`（失败再试是 `launch: retry`）。结束时再打一行和 `rpipe status` 相同的 `planned` / `succeeded` / `failed` / `pending`。不改 `jobs.json`。

`python -m rpipe logs studies/<name>` 把各 Run 的事件行按时间打到终端，不写 Study 级总 log。`python -m rpipe report studies/<name>` 从 `process.json` 写 `docs/NUMBERS.md`（Experiment 的 mean / std / min / max，以及 Run 表）。不改 `STUDY_REPORT.md` 里的结论。

`make` 进行中会立刻打出阶段（flush），并写 study 根上的 `activity.json`（不进 Git）：`make: origin <foreign|domestic> model <hub>`、`make: expand`、`make: shared <name> download <origin> <url>` / `ready` / `cached`、`make: pack`。另开一个终端跑上面的 `rpipe status`，有这份文件时第一行就是当前阶段；index 还没写出时只打这一行也退出 0。make 成功结束后删掉它。下载仍不刷 tqdm。`origin` 写在 `study.yaml` 顶层，不写在 `data` 下。`foreign` 用 torchvision 官方地址和 `https://huggingface.co`；`domestic` 用国内数据镜像和 `https://hf-mirror.com`。缺省 `foreign`，且不改本机已有的 `HF_ENDPOINT`。只有 `.ready` 才跳过；半截包会删掉再下。

`metrics.train_loss` 应是 AlgorithmTracker **最后一段 train mean**，不是最后一个 batch 的 CE。完整曲线在 `runs/<id>/assets/tracker/`；终端同款文本**必写** `assets/logs/run.log`（`时间 级别 Run id [事件] 内容`）。

`process` 分三层含义，对应 CONCEPT §2.1：每条 Run 写 `runs/<id>/process.json`。整轮结束后 `rpipe process` 写根 `process.json`（Study **信封**）。信封里每个 Experiment 才是跨 seed 的 metrics / history **mean / std / min / max**。图在 `docs/figures/learning_curves.png`。**不**改 `STUDY_REPORT.md`。

Study process 只聚合当前 index 列出的 Run，不扫描旧 version 目录。index 缺失或不可读取时会失败并提示 make，已有 process、图与 result 保留。重建前先确认当前 YAML 就是要分析的那一轮，再执行 make 和 process。

`docs/STUDY_REPORT.md` **必须有图**，并且图和各次 `run.log` **可点开**（Markdown 预览，或源码里 Ctrl+点击）。图链到 `docs/figures/learning_curves.png`；log 链到 index 里的 `log`（`../runs/<id>/assets/logs/run.log`）。数字读 `process.json`。按 Experiment 写结论，不要把 18 行 Run 表当主结论。

---

## 8. 新 Study 最小步骤

1. 复制 `studies/_template/` 为 `studies/<name>/`，对照本指南 §1–§3。  
2. 改 `experiment_config.yaml` 的基底；改 `study.yaml` 的 `axes` / `seeds` / `tags`。  
3. 在 `docs/PLAN.md` 写清：比什么、什么固定、成功标准，以及 **§3 高效率排班**（同类一组、吃满 GPU、error 后在本波重试，成功后才放行依赖 eval）。`make` 之后、`launch` 之前，把每条 Run 的预估秒数写进 **时长预估** 表。同一 `wait` 的墙钟是组内最慢的一条，整轮是各组相加。不含显存。
4. `python -m rpipe run studies/<name> --skip-launch`，核对 index。  
5. `python -m rpipe make studies/<name>`，看打印的 `pack N waits`，再 `python -m rpipe launch studies/<name>`（launch 不应再印 pack）。  
6. `python -m rpipe status studies/<name>` 看谁 `succeeded` / `failed` / `pending`。`python -m rpipe report studies/<name>` 把数字表写到 `docs/NUMBERS.md`。读 `process.json` + `docs/figures/learning_curves.png`，按 Experiment 写 `docs/STUDY_REPORT.md`：图做成可点链接，Run 表带各 `run.log` 链接。结论仍由人写。

检查清单：

- [ ] 目录最终有 `docs/`、`shared/`、`runs/`  
- [ ] `axes` 与 `seeds` 分开  
- [ ] 每个 Run 的 config 含 `seed`  
- [ ] 结论按 Experiment 聚合，而不是按扁平 run 列表  
- [ ] `PLAN.md` / `STUDY_REPORT.md` 写清本轮怎么并行（同类一组、error 后续跑）
- [ ] `STUDY_REPORT.md` 的「怎么跑的」写整轮预估和实际；Runs 表每行有 `est` 和 `actual`。不另开时长记录，不含显存  
- [ ] `STUDY_REPORT.md` 有可点开的 learning curve，以及各 Run 的 `run.log` 链接  

---

## 9. 现成能力 vs 要改库

| 你想做的 | 怎么做 |
|----------|--------|
| 扫已有字段（样本量、lr、seed…） | 只改 Study 的 yaml |
| 换 MNIST 子集大小 / epoch | yaml 即可 |
| 新数据集、新模型、新训练循环 | 改 `structure.data` / `model` / `algorithm`，再在 yaml 里点名 |
| 换 HF Trainer / Accelerate | 改 `algorithm.source`；`optimizer` / `scheduler` / `resume` 键不变（structure.md §6.11） |
| 独立评测（加载 best） | 另一次 Run：`algorithm.mode: eval`，`resume: best`；不是 Flow 多一个阶段。整波重跑：`rpipe launch --mode eval`（已成功加 `--include-done`） |
| 断点续训 | train 的 `resume: latest`（算法接口；system 只读文件） |
| 改一次 Run 的阶段顺序 | 不要改；最多 `--phases` 裁剪，相对顺序不变 |
| 自动出报告 / 跨 Run 对比表 | 人写 `STUDY_REPORT.md`（必须嵌图）。Study `process` 出 mean/std/min/max 和 `docs/figures/learning_curves.png`。`rpipe report` 把同一份数字写成 `docs/NUMBERS.md` |
| 训练曲线 | process 优先读 JSONL 的有效训练轨迹：显式 optimizer_step（优先）或 epoch；旧记录 / state history 回退仍用 observation。跨 seed 按坐标并集统计，缺失点不插值，逐点 n 见 process / 图；不同单位分开。原始日志保留回滚分支，恢复规则见 structure §6.9.2。不做 TensorBoard |
| 终端 + 硬盘日志 | **Logger** 必写 `runs/<id>/assets/logs/run.log`（`时间 级别 Run id [事件]`）；失败时 traceback 每一行都是 `[error]`。`index.json` 的 `log` 指向它。没有 Study 级总 log |
| 并行时日志挤在一起 | 文件按 Run 分开；终端每行带 Run `id`。Windows：`rpipe launch` 默认 `--console new`；`--console shared` 只混终端 |
| 看这轮谁好了谁挂了 | `python -m rpipe status studies/<name>`（只读；`--mode eval` 可滤） |
| 下一组挤进还在跑的实验、显存爆 | 组末必须 `wait`（make 脚本 / `rpipe launch` 都这样）；不要手改脚本去掉 `wait` |

脚本里若要编程调用：`from rpipe.flow.cli import run_study`。读写路径用 `rpipe.structure.artifact`；造格子用 `rpipe.structure.make`。
