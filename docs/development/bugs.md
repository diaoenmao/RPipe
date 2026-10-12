# Bugs

> 这里只保留尚未解决的缺陷。编号只增不复用；修复并验证后移除条目，开发证据留在 [record.md](record.md)，实验依据留在对应 Study 报告。正常差异或决定不修的事项不列为开放缺陷。权威仍是 [concept.md](../code/concept.md) → [layout.md](../code/layout.md) → 代码。

更新日期：2026-10-13。

## 开放

当前没有已复现且未修复的开放软件缺陷。B-002、B-021 / B-024及整仓检查的B-025至B-029已按支持合同修复和回归；修复条件、红测与失败保留在[开发记录](record.md#2026-10-13)。B-022的代表性容量 / 并发已测并采用显式预算；B-023的benchmark覆盖和精度合同已修复，严格FP32复验通过，见[main_exp报告](../../studies/main_exp/docs/speed_calibration.md)。

没有开放条目不等于支持所有后端或不存在未知问题：私有RNG / multi-worker / HF完整恢复、未测设备的容量、旧B-023权重的具体kernel及当前源码长矩阵仍属明确验证边界，见[结构合同](../code/structure.md)、[brainstorm](brainstorm.md#2026-10-13-当前收口与后续边界)和Study报告。旧失败不删除，不把这些未实测范围标为通过。

下次新增问题从 **B-030** 起编号。

新增条目一条一事，记录编号、open / investigating状态、复现条件、期望 / 实际、影响落点、方案和实际验证证据。
