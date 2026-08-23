# Study 使用指南

如何用 RPipe 做一轮可复现实验。权威概念见 [CONCEPT.md](CONCEPT.md)；目录约定见 [LAYOUT.md](LAYOUT.md)。

---

## 1. Study 和 Experiment 是什么（先分清）

| | **Study** | **Experiment（配方）** |
|--|-----------|------------------------|
| 问的是 | **这一轮研究比什么**（变量轴、对照、结论） | **这一类任务怎么跑**（数据/模型/算法能力 + Flow） |
| 例子 | 「train_size 变大，准确率是否升」 | 「MNIST + linear + train」 |
| 磁盘 | **`studies/<name>/` 一整棵树** | **住在 Study 里**：`experiment_config.yaml`（+ 可选 grid/launch） |
| 产物 | `docs/`、`shared/`、`runs/`、`index.json` | 不单独占顶层文件夹 |

**没有** `examples/experiments/` 这一层了。以前那是历史遗留：Artifact 还挂在「实验类型」下，和 Study 抢地盘。

---

## 2. 推荐流程

1. 在 Study 内写好 `experiment_config.yaml`（基底配方）  
2. 填 **`study.yaml`**（变量轴、tags、描述）  
3. 展开 → 写各 Run Config 到 `runs/<id>/` → 写 **`index.json`**（launch 前）  
4. launch Flow：`prepare → execute → collect → summarize → persist → process`  
5. 读 `runs/<id>/result.json`；写 **`docs/STUDY_REPORT.md`**

---

## 3. Study 目录（LAYOUT 定死）

```text
studies/<study>/
  study.yaml                 # 编排声明
  index.json                 # launch 前清单
  experiment_config.yaml     # 基底配方（Experiment 概念的落盘）
  run.py                     # 本 Study 入口
  docs/
    PLAN.md                  # 执行前计划（可选）
    STUDY_REPORT.md          # 结论 / 对照 / 卡点（给人看）
  shared/
    data/                    # 数据集缓存（Study 内各 Run 共用）
    model/                   # 可复用权重
  runs/
    <run_id>/
      config.yaml
      result.json
      assets/
  grid/  launch/             # 可选：展开与启动脚本（也在 Study 内）
```

仓库根下只有 **`studies/`**，不再用 `examples/`。

---

## 4. `study.yaml` 模版

```yaml
study: mnist_train_size
description: MNIST train_size sweep → test accuracy

# 配方就在本 Study 目录；不必再写外部 experiments 路径
experiment:
  name: mnist_linear

fixed:
  seed: 0
  data:
    name: MNIST
    source: torch                 # stub | torch
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

tags:
  - when:
      data.config.train_size: 500
    tags: [baseline]

run_description: "mnist_linear train_size={data.config.train_size}"
```

---

## 5. Config 约束（摘要）

| 字段 | 进 Run id hash？ | 说明 |
|------|------------------|------|
| 四层 + seed + **tags** | 是 | 改 tag 换目录 |
| `description` | 否 | 给人看 |
| `id` | 否（结果字段） | 由内容算出 |

| `data.source` | 行为 |
|---------------|------|
| `stub` / Toy | 不下载 |
| `torch` | 真数据 / 真训 |

---

## 6. Flow 阶段名（persist / process 白话）

代码里 Flow 最后一步曾经叫 **`index`**，和 Study 的 **`index.json`** 重名，容易混。

| 阶段 | 干什么 |
|------|--------|
| **persist** | 把 Result **写成** `runs/<id>/result.json`（定稿落盘）。就是改名后的旧 `index` 阶段。 |
| **process** | Result **已经写好之后**再干的事：相对 baseline 算 Δ、填报告骨架、回填 index 里的 metrics。可先空着 / 跳过。 |

Study 的 **`index.json`** =「打算跑哪些 Run」的清单，**不是** Flow 的 persist。

「persist 改名 + 空 process」=：把阶段名改清楚，并挂一个暂时什么都不做的 `process` 钩子，方便以后自动写对照表。

---

## 7. 最小检查清单

- [ ] Study 树符合 §3（有 `docs/`、`shared/`、`runs/`）  
- [ ] 已写 `study.yaml` / `index.json` 再 launch  
- [ ] 真数据 `source: torch`  
- [ ] Result 可 JSON（无 Loader/Module）  
- [ ] 结论在 `docs/STUDY_REPORT.md`
