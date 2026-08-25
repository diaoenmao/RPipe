# Study 使用指南

怎么用**现在的代码**开一轮可复现实验。概念以 [CONCEPT.md](CONCEPT.md) 为准，目录以 [LAYOUT.md](LAYOUT.md) 为准。

库内两柱：`structure`（含 artifact）+ `flow`。你要写的是包外的 **Study 目录**；展开 config、写 index、调 Flow 由薄 CLI 完成，不必自己调 `FlowRunner`。

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

跑完后 CLI 会补上（默认不入库）：

```text
  index.json                 # 按 Experiment 列 Run（launch 前就写好）
  shared/{data,model}/       # Study 级 asset，多次 Run 共用
  runs/<id>/
    config.yaml              # 这一次 Run 的完整 config
    result.json              # Flow write 落盘
    assets/                  # 本 Run 日志等
```

现成例子：

| Study | 用来学什么 |
|-------|------------|
| `studies/mnist_train_size/` | 扫研究因素（三个 `train_size`，每个 3 个 seed） |
| `studies/_template/study.yaml` | 字段模版 |

仓库里若还有 `grid/`、`launch/`、`run.py`，那是历史薄包装。**新 Study 不必抄**，用下一节的 CLI 即可。

---

## 3. 一条命令怎么跑

```bash
pip install -e ".[dev]"

# 只展开：写 config + index，不跑训练（先看格子对不对）
python -m rpipe run studies/<name> --skip-launch

# 展开并跑完全部 Flow
python -m rpipe run studies/<name>
```

CLI 内部顺序（`src/rpipe/cli.py`）：

1. 读 `study.yaml`，按 `axes` × `seeds` 展开补丁  
2. 每个补丁 ⊕ `experiment_config.yaml` → 写出 `runs/<id>/config.yaml`（`id` 是内容 hash）  
3. 写 `index.json`：按 Experiment 的 **factors** 分组，下面列各 seed 的 Run  
4. 对每个 Run 调 `FlowRunner`：prepare → execute → collect → summarize → **write** → process  

常用开关：

| 开关 | 作用 |
|------|------|
| `--skip-launch` | 停在 config + index |
| `--phases prepare,execute,...` | 只跑列出的阶段（默认全链） |

`python -m rpipe study run …` 与上面等价，只是旧别名。

---

## 4. 写 `experiment_config.yaml`

基底字段对应 structure 四层。当前代码里**真训通路**是：`data.name: MNIST` + `model.name: linear` + `algorithm.mode: train`。其它组合会走 stub（能跑通 Flow，指标可能是占位）。

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
  name: linear                    # 784→10；可在 model.config 里改 in/out
algorithm:
  mode: train
  num_epochs: 1
  lr: 0.1
system:
  device: cpu
```

合并规则：`study.yaml` 的 `fixed` 与 `axes` 补丁 **覆盖** 基底同名字段（深层 dict 合并，list 整段替换）。

---

## 5. 写 `study.yaml`

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
    mode: train
    num_epochs: 2
    lr: 0.1
  system:
    device: cpu

axes:
  data.config.train_size: [500, 2000, 8000]

seeds: [0, 1, 2]

tags:
  - when:
      data.config.train_size: 500
    tags: [baseline]

run_description: "train_size={train_size} seed={seed}"
```

这会得到 **3 个 Experiment**，每个下面 **3 个 Run**。九次训练共用 `shared/data` 里的 MNIST。

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

## 6. 跑完看什么

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

**result**（`runs/<id>/result.json`）：`status`、`metrics`（如 `accuracy` / `loss`）、`control`、`paths`。成功则 `status: succeeded`。

结论按 **Experiment** 聚合后写进 `docs/STUDY_REPORT.md`（现在不会自动生成）。`process` 阶段是预留钩子（相对 baseline 的 Δ 等），目前是空操作。

---

## 7. 新 Study 最小步骤

1. 复制 `studies/mnist_train_size/` 或对照 `_template/study.yaml`，改目录名。  
2. 改 `experiment_config.yaml` 的基底；改 `study.yaml` 的 `axes` / `seeds` / `tags`。  
3. 在 `docs/PLAN.md` 写清：比什么、什么固定、成功标准。  
4. `--skip-launch`，核对 index 里 Experiment 个数、factors、seed、baseline tag。  
5. `python -m rpipe run studies/<name>`。  
6. 读各 `result.json`，按 Experiment 写 `docs/STUDY_REPORT.md`。

检查清单：

- [ ] 目录最终有 `docs/`、`shared/`、`runs/`  
- [ ] `axes` 与 `seeds` 分开  
- [ ] 每个 Run 的 config 含 `seed`  
- [ ] 结论按 Experiment 聚合，而不是按扁平 run 列表  

---

## 8. 现成能力 vs 要改库

| 你想做的 | 怎么做 |
|----------|--------|
| 扫已有字段（样本量、lr、seed…） | 只改 Study 的 yaml |
| 换 MNIST 子集大小 / epoch | yaml 即可 |
| 新数据集、新模型、新训练循环 | 改 `structure.data` / `model` / `algorithm`，再在 yaml 里点名 |
| 改一次 Run 的阶段顺序 | 不要改；最多 `--phases` 裁剪，相对顺序不变 |
| 自动出报告 / 跨 Run 对比表 | 尚未做；人写 `STUDY_REPORT.md`，或以后填 `flow.process` |

脚本里若要编程调用：`from rpipe.cli import run_study`（这是 CLI 辅助，不是第三柱）。读写路径用 `rpipe.structure.artifact`。
