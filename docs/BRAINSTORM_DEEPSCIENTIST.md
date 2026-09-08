# Brainstorm: 借鉴 DeepScientist，不改变 RPipe 定位

> **状态**：对照笔记 + 路线图（**不是** CONCEPT / LAYOUT 权威）。  
> 权威：[CONCEPT.md](CONCEPT.md)、[LAYOUT.md](LAYOUT.md)、[STUDY_GUIDE.md](STUDY_GUIDE.md)

---

## 1. 拍板：Study / Experiment / Run

与 CONCEPT 一致：Study = 编排壳 + artifact 根；Experiment = 研究因素的一个点（不含 seed，无顶层目录）；Run = 该点 × seed。`experiment_config.yaml` = Study **基底默认值**。展开：`axes` → Experiments；× `seeds` → Runs。

相对 DeepScientist：学 durable 契约与编排纪律，不学 OS / UI / 决策器。

---

## 2. 已落地（相对本笔记起草时）

- 库内两柱 `structure` + `flow`；artifact IO 在 `structure.artifact`
- 顶层 `studies/`：`docs/` + `shared/` + `runs/<id>/`
- cli：`python -m rpipe`，`study run` 为别名
- Flow 阶段：prepare → execute → collect → summarize → **write** → process（process 仍可空）
- `index.json` 按 Experiment 的 factors 分组列 Run
- 真数据须显式 `data.source`（如 `torch`）
- 数字 / 文本拆分已写入结构文档：**AlgorithmTracker**（algorithm）+ **Logger**（system）；实现按文档跟进

---

## 3. 下一步（仍非权威）

1. 实现 AlgorithmTracker 与 system.Logger；`metrics.train_loss` 用段均值，曲线进 `assets/tracker/`，日志进 `assets/logs/`
2. `flow.process`：按 Experiment 聚合 Runs（均值 / Δ baseline），不覆盖各 Run 的 `result.json` 正文
3. 暂缓 daemon / Web / 决策器
