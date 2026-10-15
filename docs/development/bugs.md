# Bugs

> 这里只保留尚未解决的缺陷。编号只增不复用；修复并验证后移除条目，开发证据留在 [record.md](record.md)，实验依据留在对应 Study 报告。正常差异或决定不修的事项不列为开放缺陷。权威仍是 [concept.md](../code/concept.md) → [layout.md](../code/layout.md) → 代码。

更新日期：2026-10-16。

## 开放

### B-087：可选 tracker_state 坏UTF8阻断有效主轨迹和process（[#99](https://github.com/DreamSoul-AI/RPipe/issues/99)）

- 来源：第二类读取合同批次，真实坏UTF8原文件经load_curve_series抛UnicodeDecodeError；缺失/损坏JSON/IO原已空history。3条循环symlink实际正常降级，不登记；普通控制正常。REPRO/原码ZIP/hash/真实导入路径/红日志在`.tmp/audit087`，不随clone。
- 期望：现有共享_state_history将可选UTF8解码失败纳入同一空history边界，所有series/history/Run/Study/curves调用方复用；有效JSONL/peer/数值/原字节保留，不改磁盘布局或checkpoint恢复，不吞RuntimeError。
- 先Issue99，冻结原字节与正式同字节测试24项18 failed/6 passed、0error/skip，红回归先提交；设计docs/code/flow.md先于生产。f5f85b7仅1行共享边界修复；24合同全绿、相关966/core1163各1平台skip、Flow181通过；待最终PR/实际dev验收，见record §113。

B-086/#97已由PR98按最终head全部实际CI普通合dev8dc6a91，实际dev Unit38790569174/Package38790569148验收中；未计全完成，Issue等下版，见record §112。

B-085/#95已由PR96普通merge devbf57c88，最终head与实际dev全部实际矩阵/聚合成功并清结束remote；Issue留至下版，见record §110。

B-083/#91与B-084/#93已由PR94合dev2b7c769，最终head/实际dev全部实际CI成功并清结束remote，已从未解决队列移除；Issue等下版。完整红绿、117新增节点、实际验收与八条远程清单见record §103—107。

B-081/#85已由PR87合dev37d0262，最终head/实际dev全部实际CI通过并删除结束远程，Issue保留open，见record §101。

B-080/#82已由[PR84](https://github.com/DreamSoul-AI/RPipe/pull/84)普通merge dev6863f7f，最终head/实际dev全部实际矩阵/聚合成功，已从未修队列移除；Issue留待下版。已删结束远程codex/saved-plan-read-boundary，详record §99。

B-079/#81已由[PR83](https://github.com/DreamSoul-AI/RPipe/pull/83)普通merge dev4536ec4，最终head与实际dev全部实际矩阵/聚合成功，已从未修队列移除；Issue仍open到下版。已删结束远程codex/declared-batch-contract，错误查询EOF与恢复证据见record §97。本轮当前未完成验收项为B-080/#82，下一新编号B-081。

B-078/#79已由[PR80](https://github.com/DreamSoul-AI/RPipe/pull/80)普通merge devf4acb93，最终head与实际dev所有实际矩阵/聚合成功，从未修队列移除；Issue仍open到下版。YAML远程结束分支已删除，详细证据见record §93/95。

B-077/#77已由[PR78](https://github.com/DreamSoul-AI/RPipe/pull/78)普通合dev1667f51，最终head及实际dev全部实际矩阵/聚合成功；从未修队列移除，Issue77留待下版。红绿及分支实际清理见record §92。本轮当前未修验证项为上方B-078，下一新编号B-079；下文“下一B-077”是历史批次快照。

本轮B-074/B-075/B-076已分别通过PR69/71/70普通合dev，最终工作提交与各实际dev所有实际CPU/包矩阵及聚合门均成功；本轮没有尚未修复的已确认缺陷。原始条目见[三类修复前快照](https://github.com/DreamSoul-AI/RPipe/blob/5d2e174/docs/development/bugs.md)，逐项清单、红绿、专用协议与实际CI见[record §83—87](record.md)。GitHub #53/#62/#66/#67/#68已在v0.4.1正式发布后按明确授权逐项留言引用Release并completed关闭；当前main为00e70b1f83ac23d113b9582a4270a41cfadb1444，验收见[record §90](record.md)。该历史批次当时下一编号B-077；当前已使用至B-080，下一新编号B-081，绿色合同控制不记新缺陷。

上一批B-072/B-073分别经PR61/63合dev，修复与证据保留于record §79—81；#53原始index反例已随v0.4.0发布，其后Study发布扩展仅在dev，#62也仅在dev。B-065性能去重仍撤回不恢复。

B-071（[Issue #53](https://github.com/DreamSoul-AI/RPipe/issues/53)）已通过[PR #52](https://github.com/DreamSoul-AI/RPipe/pull/52)合入 dev ea2210bc27e5456062bdef0ffb331e208c64f838；最终 head eb52c65 与实际 dev 的全部实际 CI 通过。真实旧 index 截断与红绿合同见 [record §69 / §73](record.md)。修复已随v0.4.0进入main54f8d31；Issue53不在本次关闭清单，仍保持open。

B-070（[#50](https://github.com/DreamSoul-AI/RPipe/issues/50)）已通过[PR #51](https://github.com/DreamSoul-AI/RPipe/pull/51)合入 dev d294c1b5aa99b4d7148f991f9a4ac8903fe23374，最终工作提交及实际 dev 的必需 CI 全部实际矩阵和聚合通过；原始条目保留在[修复前记录](https://github.com/DreamSoul-AI/RPipe/blob/3a198ff4e04d25999a895b131f0c01ab964b5e0a/docs/development/bugs.md)和[record §63–64](record.md)。Issue50已在v0.4.0发布后按明确授权留言并关闭。三类[产物一致性设计](artifact_consistency_design.md)已按PR52/54/55完成，不再处于待审或未实施状态。

B-069（[#48](https://github.com/DreamSoul-AI/RPipe/issues/48)）已经由[PR #49](https://github.com/DreamSoul-AI/RPipe/pull/49)合入dev b2390b0，最终head与实际dev Unit38629909667 / Package38629909697全部实际矩阵和聚合通过；来源/红例和包门未完成拒绝保留于record §62–63与Issue正文。该条不再属于待修缺陷，Issue48已在v0.4.0发布后留言并关闭。

B-056～B-064、B-066～B-068 已通过 [PR #46](https://github.com/DreamSoul-AI/RPipe/pull/46) 合入 dev `5ee1214b7ce0dc28b65139ac7b3f9ca16eb8b2ac`；原始期望、实际、来源和失败证据保留在 [合入前条目快照](https://github.com/DreamSoul-AI/RPipe/blob/90b7d75323d266c4d272f786c28f4c00959d0432/docs/development/bugs.md) 与 [record §43–60](record.md)。B-065 性能去重已经撤回，不算本轮修复。

GitHub #36 / #37 / #38 / #41 / #48 / #50 的修复已随v0.4.0发布到main54f8d31，并按Enmao本次明确授权逐项注明修复版本后completed关闭；#53仍open。发布与评论证据见[record §78](record.md#78-v040-正式发布完成与六项-issue-关闭)。当前CPU验证不代表GPU或全部Study的数值验收。

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

B-055 的原生 launch 重试关闭入口已通过 [PR #42](https://github.com/DreamSoul-AI/RPipe/pull/42) 合入 dev `8c15cd84f8a47d5bfeb80edfa69bc5c370998e74`。最终工作提交 `8269bff` 的三项必需检查及实际 dev 的 [Unit Tests](https://github.com/DreamSoul-AI/RPipe/actions/runs/38557561151) / [Package Check](https://github.com/DreamSoul-AI/RPipe/actions/runs/38557561212) 均通过，包含 Windows / Python 3.13；开发证据见 [记录 §40](record.md#40-原生-launch-显式关闭失败重试41验证通过)。默认重试仍保留；#41在v0.4.0发布后经本次明确授权关闭，而非默认自动关闭。

原v0.3.1未包含这些修复；截至v0.4.0，#36—#38及#41已在main发布并经明确授权关闭。历史dev验证与各自发布版本仍分开记录。没有开放软件条目不代表所有后端、设备或研究条件均已验收。

此前撤回的范围外反例保留历史事实，未自动恢复为本目标的实施队列；接续以新的实际反馈和已确认设计为准。旧收缩依据见[当时记录](record.md#32-按用户要求收缩到本机适配与三个下游-issue)。

B-002、B-021 / B-024及B-025至B-029的既有修复证据仍在 [record.md](record.md)。B-022 / B-023 的代表性容量、精度合同与探索边界保留各自报告；本轮新的设备 / 并发结果见 [本轮执行记录](../../studies/main_exp/docs/v030_execution.md)。未实测支持范围不是已通过，缺环境依赖的导入失败也不自动列为库缺陷。

B-031 至 B-072 已使用，编号不复用。下次新增问题从 **B-073** 起编号。

新增条目一条一事，记录编号、open / investigating状态、复现条件、期望 / 实际、影响落点、方案和实际验证证据。
