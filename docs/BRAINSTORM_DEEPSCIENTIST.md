# Brainstorm: 借鉴 DeepScientist，不改变 RPipe 定位

> **状态**：对照笔记。权威是 [CONCEPT.md](CONCEPT.md)、[LAYOUT.md](LAYOUT.md)、[STUDY_GUIDE.md](STUDY_GUIDE.md)。

---

## 1. 拍板：Study / Experiment / Run

与 CONCEPT 一致：Study 是编排壳与 artifact 根；Experiment 是研究因素的一个点，不含 seed，无顶层目录；Run 是该点 × seed。`experiment_config.yaml` 是 Study 基底默认。展开：`axes` → Experiments；× `seeds` → Runs。

相对 DeepScientist：学 durable 契约与编排纪律。OS / UI / 决策器放在本库之外。

---

## 2. 已落地

- 两柱：`structure` + `flow`。artifact IO 在 `structure.artifact`。
- **make** 在 `structure/make/`：写出 N 份 config、index，以及 `&` / `wait` 调度脚本。train 与 eval 分成两波，eval 等全部 train `wait` 完再开。
- **cli** 在 `flow/cli.py`。入口：`python -m rpipe`。
- Flow 阶段：prepare → execute → collect → summarize → **write** → **process**。
- **AlgorithmTracker**（algorithm）：数字；`assets/tracker/`。`metrics.train_loss` 用段均值。
- **Logger**（system）：终端与 `assets/logs/` 同一套，report 后 flush。
- **process**：按 Experiment 写 `process.json`（mean / std / Δ），画 `docs/figures/learning_curves.png`；不改 `STUDY_REPORT.md`。
- `index.json` 按 Experiment 的 factors 分组列 Run。
- 真数据须显式 `data.source`。

---

## 3. 下一步

1. 跑 `vision_main_recipe` 400 epoch：`make` / `launch` 按同类装箱（linear 与 resnet 不同组）。
2. 对照 git `main` 的长训图：Normalize 统计量、多种子、CIFAR100 / wresnet。
3. daemon / Web / 决策器仍放在本库之外。
