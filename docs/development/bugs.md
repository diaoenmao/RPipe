# Bugs

> 这里只保留尚未解决的缺陷。编号只增不复用；修复并验证后移除条目，开发证据留在 [record.md](record.md)，实验依据留在对应 Study 报告。正常差异或决定不修的事项不列为开放缺陷。权威仍是 [concept.md](../code/concept.md) → [layout.md](../code/layout.md) → 代码。

更新日期：2026-10-16。

## 开放

### B-080：可选jobs缓存的stat/编码/数值元数据失败越过重建边界（[#82](https://github.com/DreamSoul-AI/RPipe/issues/82)）

- 来源：持续第二类可选读取复查。load_launch_plan合同为不可用返回None请求make重建；但stat在try外、UTF-8及缓存int转换未在同一边界。
- 复现：8小探针中坏UTF8、坏init_gpu文本、null num_gpus、坏round文本、Infinity round、注入stat PermissionError共6异常逃逸；坏JSON已返回None、合法空计划可用为2控制。7个物理缓存原字节不改；没有启动进程/GPU或更改ACL。
- 期望：只在现有函数内把缓存stat/read/decode/已存元数据转换归入明确文件/类型/值/溢出边界；保留合法/缺省参数和缓存不匹配逻辑。调用方非法参数、意外RuntimeError继续上抛，不改缓存字节/格式、实际分组、retry、资源策略或布局。
- 已先创建Issue82；便携红测试与生产修复尚未执行，待当前#81批次验收后在单一新分支接续。源码/测试原字节ZIP与REPRO.json在.tmp/audit080，不随clone。该问题不同于B-079的jobs/batches一致性，不混写未做验收。

### B-079：显式batches越过声明jobs，可能先失效未列入任务的旧成功（[#81](https://github.com/DreamSoul-AI/RPipe/issues/81)）

- 来源：三类合同全仓接续，schedule.wait_groups直接返回batches；native与render_bash均信任它，缓存loader也未核对jobs/batches一致性。
- 真实隔离复现：jobs仅planned、batches仅peer，Popen为内存替身；实际mock启动peer，旧peer result从succeeded变failed，原SHA256 c0cf5aa5350fbf1628583326b8a9b4b9cea22d6d0064886e38cdadc2f071f37b→c8d12e2aa95bc93159dfc5987230ab16b992f2c32fe649fb09c1c10799565da0。未启动真实子进程/GPU，未触碰正式数据。生成Bash仅嵌入peer并从groups重建jobs，原planned被丢弃；脚本未执行。
- 期望：既有wait_groups验证batches与jobs完整多重集/字段相同，允许合法重排；writer在发布前验证，缓存在过滤成功项前验证，不一致请求重建，native/script在失效旧结果/启动/覆盖前拒绝。不修改合法rerun保全范围、默认retry/依赖顺序、格式或布局。
- 先Issue，再红测试5597727：26节点，23 failed/3 passed，0error/skip；覆盖foreign/missing/metadata/duplicate的共享分组、native旧字节/peer、脚本发布、渲染、两种include_done缓存读取，以及合法/重排控制。原字节ZIP、native REPRO、旧peer bytes/hash、脚本、RED.xml/log/命令在.tmp/audit079，不随clone。首次补充脚本探针少传两个必需参数的调用错误另存，已纠正，不当库失败。

B-078/#79已由[PR80](https://github.com/DreamSoul-AI/RPipe/pull/80)普通merge devf4acb93，最终head与实际dev所有实际矩阵/聚合成功，从未修队列移除；Issue仍open到下版。YAML远程结束分支已删除，详细证据见record §93/95。

B-077/#77已由[PR78](https://github.com/DreamSoul-AI/RPipe/pull/78)普通合dev1667f51，最终head及实际dev全部实际矩阵/聚合成功；从未修队列移除，Issue77留待下版。红绿及分支实际清理见record §92。本轮当前未修验证项为上方B-078，下一新编号B-079；下文“下一B-077”是历史批次快照。

本轮B-074/B-075/B-076已分别通过PR69/71/70普通合dev，最终工作提交与各实际dev所有实际CPU/包矩阵及聚合门均成功；本轮没有尚未修复的已确认缺陷。原始条目见[三类修复前快照](https://github.com/DreamSoul-AI/RPipe/blob/5d2e174/docs/development/bugs.md)，逐项清单、红绿、专用协议与实际CI见[record §83—87](record.md)。GitHub #53/#62/#66/#67/#68已在v0.4.1正式发布后按明确授权逐项留言引用Release并completed关闭；当前main为00e70b1f83ac23d113b9582a4270a41cfadb1444，验收见[record §90](record.md)。本轮下一编号B-077，近期绿色合同控制不记新缺陷。

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
