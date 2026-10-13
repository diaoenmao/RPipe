# Bugs

> 这里只保留尚未解决的缺陷。编号只增不复用；修复并验证后移除条目，开发证据留在 [record.md](record.md)，实验依据留在对应 Study 报告。正常差异或决定不修的事项不列为开放缺陷。权威仍是 [concept.md](../code/concept.md) → [layout.md](../code/layout.md) → 代码。

更新日期：2026-10-13。

## 开放

### B-030 — 新 index 建立后 report 仍可导出旧聚合（open）

- 发现日期与源码：2026-10-13，发布 v0.3.0 / `804b3823`。main_exp 设置新 version 并 fresh make 后，index 已含新的64个 Run；原根 `process.json` 保留旧矩阵的64个 Run 和 complete=true。
- 复现：将实际新 index 与原保留 process 逐字节复制到独立 `.tmp/v030-20261013/stale-report-repro/`，调用公开 `rpipe report`；没有训练或改动正式运行目录。命令 exit0，生成的 NUMBERS 含64个旧 ID、0个当前 ID，两份清单交集为0。正式证据见 [Study 复现 JSON](../../studies/main_exp/docs/v030_stale_report_repro.json)，临时完整生成表只保留在本机。
- 期望 / 实际：当前 Study 报告应核对聚合对应的 index，来源过期时拒绝或明确标明旧轮次；目前 report 直接读取 process 数字，index 仅用于补 log 链接，未检查身份，表头也未标出来源已过期。make 保留旧证据本身不应变成删除历史结果的理由。
- 影响：换 version / 清单后、再次 process 完成前，旧成功统计可被当成当前进度或结果；风险在报告与来源识别，不是已运行训练参数的数值错误。涉及 `structure.make.expand_study`、`structure.artifact.readout.report` 与 Study 聚合身份合同。
- 处理边界：本轮长矩阵已完成固定804b3823源码的实际终验，逐Run及index / process / COMPARISON身份均核对后再导出正式数字；这项辅助核对不代表B-030已修复。运行期间不修改被冻结的库代码或旧provenance。
- 后续修复方向：明确process绑定index身份 / 生成来源，并让report在来源不匹配时拒绝或显式展示历史状态；先核对保留历史聚合与缺失index的既有合同，再最小修改。回归覆盖换version后的过期聚合、当前有效聚合、缺失 / 损坏index、合法部分聚合与原证据保全。尚未修复，不宣称回归通过。

B-002、B-021 / B-024及B-025至B-029的既有修复证据仍在 [record.md](record.md)。B-022 / B-023 的代表性容量、精度合同与探索边界保留各自报告；本轮新的设备 / 并发结果见 [本轮执行记录](../../studies/main_exp/docs/v030_execution.md)。未实测支持范围不是已通过，缺环境依赖的导入失败也不自动列为库缺陷。

下次新增问题从 **B-031** 起编号。

新增条目一条一事，记录编号、open / investigating状态、复现条件、期望 / 实际、影响落点、方案和实际验证证据。
