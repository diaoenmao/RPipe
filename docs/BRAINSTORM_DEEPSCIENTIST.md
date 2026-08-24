# Brainstorm: 借鉴 DeepScientist，不改变 RPipe 定位

> **状态**：对照笔记 + 路线图（**不是** CONCEPT / LAYOUT 权威）。  
> 权威：[CONCEPT.md](CONCEPT.md)、[LAYOUT.md](LAYOUT.md)、[STUDY_GUIDE.md](STUDY_GUIDE.md)

---

## 1. 拍板：Study / Experiment / Run

| | Study | Experiment | Run |
|--|-------|------------|-----|
| 是什么 | 编排壳（index、docs、grid、shared） | **同一组实验变量**的一个点 | 该点下一次实测 |
| seed | 声明 `seeds` | **不含** | **至少**一个 random seed |
| 磁盘 | `studies/<name>/` | 逻辑分组（index） | `runs/<id>/` |

`experiment_config.yaml` = Study **基底默认值**，不是「一个 Experiment 实例」。

展开：`axes` → Experiments；× `seeds` → Runs。

---

## 2. 已落地

顶层 `studies/`；layout `docs/shared/runs`；persist + 空 process；`rpipe study run`；根 `data/`/`output/` 已删；expand 支持独立 `seeds`。

---

## 3. 下一步

1. `index.json` 按 Experiment 显式分组（含 factors，下列 runs）  
2. process：按 Experiment 聚合 Runs（均值/Δ baseline）→ `docs/`  
3. 强制 `data.source`  
4. 暂缓 daemon / Web
