> 历史交接备忘。权威文档是 [CONCEPT.md](CONCEPT.md) → [LAYOUT.md](LAYOUT.md) → [CODE_STRUCTURE.md](CODE_STRUCTURE.md)。

当前库内两柱：`structure/` 与 `flow/`。make 在 `structure/make/`。cli 在 `flow/cli.py`，入口是 `python -m rpipe`。Study 声明在包外 `studies/`。

旧树里的 `rpipe.artifact` / `rpipe.study` 已并入 structure 与 flow。写 result 的阶段叫 **write**。数字观测是 **AlgorithmTracker**（algorithm），文本日志是 **Logger**（system）。
