# Bugs

> 这里只保留尚未解决的缺陷。编号只增不复用；修复并验证后移除条目，开发证据留在 [record.md](record.md)，实验依据留在对应 Study 报告。正常差异或决定不修的事项不列为开放缺陷。权威仍是 [concept.md](../code/concept.md) → [layout.md](../code/layout.md) → 代码。

更新日期：2026-10-15。

## 开放

### B-070：status 提前强转 metrics，绕过已有单元格类型回退

- 状态：open；在实际 dev b2390b0 上真实 public write_result→list_runs/CLI 复现，[Issue #50](https://github.com/DreamSoul-AI/RPipe/issues/50) 已先创建，随后失败回归提交 f15e067；生产代码仍未修改。
- 期望 / 实际：非 object 指标沿用 _metric_cell 的已有 '-' 回退、保留 Run 状态和正常 peer，不改输入。实际 metrics=[1] /标量/字符串使整张列表异常，CLI退出2；列表对还能被强转为accuracy=0.9000，误展示未按object记录的值。
- 来源与范围：list_runs 是 _metric_cell 的唯一调用方，其 dict(...) 提前执行使非mapping guard不可达。公共 validate_result/write_result 仅要求成功记录含metrics键，接受上述值；此次不加强写入合同或重新签名旧结果。
- 证据：新10项回归7 failed /3 passed，覆盖None/空mapping/有效mapping控制、非法list/scalar、列表对、原生eval过滤和输入逐字节保全。.tmp/audit070/REPRO.json、before.zip/BEFORE.json、red.xml/red.log保留原反例，不随clone。
- 最小方案：取消调用方的dict强转，复用既有_metric_cell mapping/numeric guard；不新增schema/parser、异常吞错或格式迁移。见record §63。

B-069（[#48](https://github.com/DreamSoul-AI/RPipe/issues/48)）已在当前工作分支修复并通过本机 81 项回归，按本文件规则将原始条目转入历史；详细期望、实际、红例与最小方案见 [record §62](record.md#62-readout-存在性查询与内容读取的错误边界b-069--48) 和 [修复前条目](https://github.com/DreamSoul-AI/RPipe/blob/1046452/docs/development/bugs.md)。实际 dev 集成尚须核对 [PR #49](https://github.com/DreamSoul-AI/RPipe/pull/49) 的最终 head、mergeCommit 和对应 CI，不以本机通过代替；Issue 保持 open。除上列 B-070 外，当前工作代码暂无缺少修复的已确认软件条目。

B-056～B-064、B-066～B-068 已通过 [PR #46](https://github.com/DreamSoul-AI/RPipe/pull/46) 合入 dev `5ee1214b7ce0dc28b65139ac7b3f9ca16eb8b2ac`；原始期望、实际、来源和失败证据保留在 [合入前条目快照](https://github.com/DreamSoul-AI/RPipe/blob/90b7d75323d266c4d272f786c28f4c00959d0432/docs/development/bugs.md) 与 [record §43–60](record.md)。B-065 性能去重已经撤回，不算本轮修复。

GitHub #36 / #37 / #38 / #41 仍保持 open，不自动关闭。它们在 dev 已有修复，本轮逐项补边界与复验，没有新根因反例；已发布 main/v0.3.1 尚不含这些修复。当前 CPU 验证不代表 GPU 或全部 Study 的数值验收。

## 暂缓（不进入正确性修复队列）

### B-065：单次来源清单采集对重叠输入反复解析、读取和哈希

- 状态：deferred；按最终仅做正确性的要求已用新普通提交撤回ad67f74次数去重优化，原复现与性能记录仅作历史保留，不进入本轮修复队列。见[范围收缩记录](record.md#57-按最终范围暂缓-b-065-性能去重)。
- 来源：source_files 经 study_phase_files、flow/structure 自动扫描、recipe 与 provenance.include 收集同一路径，最终 mapping 覆盖重复键，但此前每次均执行 _inside/读取/哈希。生产调用方为 write_provenance 和 provenance_changes，后者供 launch/prepare/Runner/Study process 冻结检查。
- 期望：同一次清单采集每个解析后 Study 文件只读取/哈希一次；重复原始字符串不反复解析；新一次调用仍完整发现文件并重读，保持变化检测、路径边界和读取错误。
- 实际：两项真实读取计数用例失败，flow 开关关闭时两个 flow 文件各读2次、helper4次，开启时 flow各3次、helper4次。32个合成源文件、2个声明及1个隔离库文件共35个唯一输入，被读取/哈希100次；三次小场景结果相同。
- 证据：.tmp/phase065/red.log/red.xml 为 2 failed /1 passed，0.20 s；before.json/日志逐层记录 total/resolve/glob/open/read_bytes/hash，源码原字节在before.zip；BEFORE.json/before.json大小写别名覆盖的原hash索引缺口及事后重建SNAPSHOT_FILES.json已在record §52如实记录，不随clone。当前量化冻结 helper 的162文件单调用仅诊断其内部阶段，不等于原480s现场，也不是本问题的同一函数。
- 最小方案：仅在一次 source_files 调用内按首次出现去重原始声明，再按已验证的解析 Path 去重 Study 文件；不缓存跨调用内容/清单，不减少任何独立 freeze 检查，不改库文件集合、格式、Run或下游业务。补修改后下一次调用、读取拒绝、空匹配和越界回归，保留 add/edit/delete 旧门。

## 集成与历史说明

[归档快照](maintenance_archive_2026-10-15.md)明确区分合入前与后续状态；PR #46 的最终工作提交 `90b7d75323d266c4d272f786c28f4c00959d0432` 与实际 dev `5ee1214` 分别验证，不能互相替代。归档前条目中的 open / 草案文字不再代表当前代码状态。B-053（跨 Study 输入共享）和 B-054（关闭扩展时的初始冻结拒绝保全）此前经 [PR #39](https://github.com/DreamSoul-AI/RPipe/pull/39) 合入 dev `614d0a6`，证据见 [集成记录](record.md#39-共享与三个反馈修复合入-dev)。

B-055 的原生 launch 重试关闭入口已通过 [PR #42](https://github.com/DreamSoul-AI/RPipe/pull/42) 合入 dev `8c15cd84f8a47d5bfeb80edfa69bc5c370998e74`。最终工作提交 `8269bff` 的三项必需检查及实际 dev 的 [Unit Tests](https://github.com/DreamSoul-AI/RPipe/actions/runs/38557561151) / [Package Check](https://github.com/DreamSoul-AI/RPipe/actions/runs/38557561212) 均通过，包含 Windows / Python 3.13；开发证据见 [记录 §40](record.md#40-原生-launch-显式关闭失败重试41验证通过)。默认重试仍保留，#41 不自动关闭。

GitHub #36—#38及#41仍保持open，原v0.3.1/main尚未发布这些修复；当前dev修复与已发布版本必须区分。没有开放软件条目不代表所有后端、设备或研究条件均已验收。

此前撤回的范围外反例保留历史事实，未自动恢复为本目标的实施队列；接续以新的实际反馈和已确认设计为准。旧收缩依据见[当时记录](record.md#32-按用户要求收缩到本机适配与三个下游-issue)。

B-002、B-021 / B-024及B-025至B-029的既有修复证据仍在 [record.md](record.md)。B-022 / B-023 的代表性容量、精度合同与探索边界保留各自报告；本轮新的设备 / 并发结果见 [本轮执行记录](../../studies/main_exp/docs/v030_execution.md)。未实测支持范围不是已通过，缺环境依赖的导入失败也不自动列为库缺陷。

B-031 至 B-070 已使用，编号不复用。下次新增问题从 **B-071** 起编号。

新增条目一条一事，记录编号、open / investigating状态、复现条件、期望 / 实际、影响落点、方案和实际验证证据。
