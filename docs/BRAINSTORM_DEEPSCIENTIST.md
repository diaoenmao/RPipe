# Brainstorm: 借鉴 DeepScientist，不改变 RPipe 定位

> **状态**：对照笔记 + 路线图（**不是** CONCEPT / LAYOUT 权威）。  
> **参照**：[ResearAI/DeepScientist](https://github.com/ResearAI/DeepScientist)

权威以 [CONCEPT.md](CONCEPT.md)、[LAYOUT.md](LAYOUT.md)、[STUDY_GUIDE.md](STUDY_GUIDE.md) 为准。

---

## 1. 白话对照

| | DeepScientist | RPipe |
|--|---------------|--------|
| **定位** | 研究 OS | 研究执行底座 |
| **主角** | Agent + Quest | Study →（配方）→ Run |
| **磁盘** | Quest 账本 | `studies/<name>/{docs,shared,runs}` |
| **入口** | bash_exec 等 | **`python -m rpipe study run <study>`** |

---

## 2. Study vs Experiment（配方）

| | Study | Experiment（配方） |
|--|-------|-------------------|
| 问什么 | **比什么** | **怎么跑** |
| 磁盘 | `studies/<name>/` 整棵 | `experiment_config.yaml`（在 Study 内） |
| 顶层文件夹 | 有 | **没有**（已删 `examples/experiments/`） |

**建议**：保留**软区分**（文档/字段里仍可说「配方」），不要再恢复独立 Experiment 目录。是否把词表收成 Study→Run 见交付说明。

---

## 3. persist / process

| 名字 | 含义 |
|------|------|
| Study `index.json` | launch 前清单 |
| Flow **persist** | 写 `result.json`（原阶段名 index） |
| Flow **process** | persist 后钩子（现为空实现） |

---

## 4. 已落地（本轮）

| 项 | 状态 |
|----|------|
| 顶层 `studies/`；无 `examples/` / `experiments/` | 已做 |
| Study layout：`docs/` + `shared/` + `runs/` | 已做 |
| Flow `persist` + 空 `process` | 已做 |
| `python -m rpipe study run` | 已做 |
| 删根目录 `data/`、`output/` | 已做 |
| prepare MNIST → `shared/data` | 已做 |

---

## 5. 下一步

1. process：`compare_to_tag(baseline)` → 填 `docs/` 对照表  
2. 强制 `data.source: stub|torch`  
3. 可选：淡化 Experiment 用词，统一「配方 / recipe」  
4. 暂缓：daemon / Web / Findings Memory
