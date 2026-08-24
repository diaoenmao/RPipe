# Study 使用指南

如何用 RPipe 做一轮可复现实验。权威概念见 [CONCEPT.md](CONCEPT.md)；目录约定见 [LAYOUT.md](LAYOUT.md)。

---

## 1. Study / Experiment / Run（先分清）

| | **Study** | **Experiment** | **Run** |
|--|-----------|----------------|---------|
| 是什么 | 编排壳：索引、文档、grid、shared | **同一组实验变量**的一个取值点 | 该点下的**一次实测** |
| 含 seed？ | 声明 `seeds` 列表 | **不含** seed | **至少**有一个 random seed |
| 例子 | `mnist_train_size` 整棵目录 | `train_size=500` | `train_size=500, seed=0` |
| 磁盘 | `studies/<name>/` | 逻辑分组（见 `index.json`） | `runs/<id>/` |

展开关系：**Study** 的 `axes` → 多个 Experiment；每个 Experiment × `seeds` → 多次 Run。

`experiment_config.yaml` = Study 的**基底默认值**（怎么训），不是「一个 Experiment 实例」。

---

## 2. 推荐流程

1. 写好 Study 基底 `experiment_config.yaml`  
2. 填 **`study.yaml`**：`axes`（实验变量）+ **`seeds`**（复测）+ tags  
3. `python -m rpipe study run studies/<name>` → Config + `index.json` + Flow  
4. 读 `runs/<id>/result.json`；按 Experiment 聚合后写 **`docs/STUDY_REPORT.md`**

---

## 3. Study 目录（LAYOUT 定死）

```text
studies/<study>/
  study.yaml
  index.json                 # 按 Experiment 列出 Runs
  experiment_config.yaml     # 基底默认值
  run.py                     # 可选薄包装
  docs/
    PLAN.md
    STUDY_REPORT.md
  shared/{data,model}/
  runs/<run_id>/{config.yaml,result.json,assets/}
  grid/  launch/             # 可选；优先用 CLI
```

---

## 4. `study.yaml` 模版

```yaml
study: mnist_train_size
description: MNIST train_size sweep → test accuracy

# 基底默认（合并进每次 Run；非「一个 Experiment」）
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

# 实验变量轴 → 每个组合是一个 Experiment（不含 seed）
axes:
  data.config.train_size: [500, 2000, 8000]

# Run 复测轴：同一 Experiment 下至少要有 seed
seeds: [0]

tags:
  - when:
      data.config.train_size: 500
    tags: [baseline]

run_description: "train_size={train_size} seed={seed}"
```

若暂时把 seed 写在 `axes` 里也能跑，但语义上应把 **seed 视为 Run 轴**，不要当成实验因素。

---

## 5. Config 约束（摘要）

| 字段 | 进 Run id hash？ | 说明 |
|------|------------------|------|
| 实验变量 + **seed** + tags | 是 | 同 Experiment 不同 seed → 不同 id |
| `description` | 否 | 给人看 |

| `data.source` | 行为 |
|---------------|------|
| `stub` / Toy | 不下载 |
| `torch` | 真数据 / 真训 |

---

## 6. Flow 阶段名（persist / process）

| 阶段 | 干什么 |
|------|--------|
| **persist** | 写入 `runs/<id>/result.json`（旧名 `index` 阶段） |
| **process** | 定稿后派生：按 Experiment 聚合、相对 baseline Δ 等（可先空） |

Study 的 **`index.json`** = 编排清单，≠ Flow persist。

---

## 7. 最小检查清单

- [ ] Study 树有 `docs/`、`shared/`、`runs/`  
- [ ] `axes` 与 `seeds` 语义分开  
- [ ] 每个 Run Config 含 seed  
- [ ] 结论按 Experiment 聚合写在 `docs/STUDY_REPORT.md`
