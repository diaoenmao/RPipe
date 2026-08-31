> 历史交接备忘。权威文档是 [CONCEPT.md](CONCEPT.md) → [LAYOUT.md](LAYOUT.md) → [CODE_STRUCTURE.md](CODE_STRUCTURE.md)。

当前库内只有两柱：`src/rpipe/structure/`（含 `artifact/`）与 `src/rpipe/flow/`。Study 编排在包外 `studies/`，入口是薄 CLI `python -m rpipe run <study_dir>`。

不要再按旧树理解：根目录没有独立的 `rpipe.artifact` / `rpipe.study` 包；Flow 写 result 的阶段叫 **write**（不是 persist / 不是 Study index）；数字观测是 **AlgorithmTracker**（algorithm），文本日志是 **Logger**（system）。
