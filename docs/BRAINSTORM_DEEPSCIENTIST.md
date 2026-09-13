# Brainstorm: 借鉴 DeepScientist，不改变 RPipe 定位

> **状态**：对照笔记。权威是 [CONCEPT.md](CONCEPT.md)、[LAYOUT.md](LAYOUT.md)、[STUDY_GUIDE.md](STUDY_GUIDE.md)。

---

## 1. 拍板：Study / Experiment / Run

与 CONCEPT 一致：Study 是编排壳与 artifact 根；Experiment 是研究因素的一个点，不含 seed，无顶层目录；Run 是该点 × seed。`experiment_config.yaml` 是 Study 基底默认。展开：`axes` → Experiments；× `seeds` → Runs。

相对 DeepScientist：学 durable 契约与编排纪律。OS / UI / 决策器放在本库之外。

---

## 2. 已落地

- 两柱：`structure` + `flow`。artifact IO 在 `structure.artifact`。
- **make** 在 `structure/make/`：写出 N 份 config、index，以及 `&` / `wait` 调度脚本。`wait` 是内存闸门：一组结束才开下一组，避免显存叠加。train 与 eval 分成两波，eval 等全部 train `wait` 完再开。conservative 墙钟 = 各组 max 再加总，只供排班。
- **cli** 在 `flow/cli.py`。入口：`python -m rpipe`。
- Flow 阶段：prepare → execute → collect → summarize → **write** → **process**。
- **AlgorithmTracker**（algorithm）：数字；`assets/tracker/`。metric 名 Loss / Accuracy / MSE / RMSE / GLUE。`metrics.train_loss` 用段均值。
- **Logger**（system）：终端与 `assets/logs/run.log` 同一套，行首 Run `id`，report 后 flush；行含 `elapsed` / `eta`。没有 Study 级总 log。Windows `launch` 默认每条 Run 一个控制台；`--console shared` 靠行首 id 分辨。
- **process**：Run 阶段只写该次 `runs/<id>/process.json`；Study 级 `rpipe process` 写根信封 `process.json`，`experiments[]` 才是跨 seed 的 mean / std / min / max，并画 `docs/figures/learning_curves.png`；不改 `STUDY_REPORT.md`。
- `index.json` 按 Experiment 的 factors 分组列 Run，每条含 `config` 与 `log`。
- **make / launch**：`make` 写 `scripts/jobs.json` 并打印 pack；`launch` 复用清单，不重印 pack（`--remake` 除外）。共享数据在 make 时准备一次（缓存则跳过，下载静音）。
- 真数据须显式 `data.source`。

---

## 3. 下一步

1. daemon / Web / 决策器仍放在本库之外。
