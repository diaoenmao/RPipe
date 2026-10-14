# Flow 与 Study 重构交接

## 2026-10-15 v0.4.0 已正式发布

发布[PR #59](https://github.com/DreamSoul-AI/RPipe/pull/59)已dev→main正常merge为54f8d31c0f5505742677dd2ce6dec323ee114c61，实际main全部CPU/包CI成功后建立annotated [tag v0.4.0](https://github.com/DreamSoul-AI/RPipe/tree/v0.4.0)并发布[同名Release](https://github.com/DreamSoul-AI/RPipe/releases/tag/v0.4.0)。正文取自该提交发布报告，转换相对链接并附验收。#36/#37/#38/#41/#48/#50均已分别留言版本后completed关闭，#53保持open；准确链接和hash见[record §78](record.md#78-v040-正式发布完成与六项-issue-关闭)。

不要重复发版、关闭评论或移动标签；下方“待发布/仅合dev/交监督助手”为当时状态。复用原codex/v040-layout-release提交发布后文档补记并PR合dev，当前以实际PR状态为准，不另建同质分支。全部分支/数据/Run/provenance保留，仍不测速、跑GPU、改产物布局或主动相邻审计。

## 2026-10-15 v0.4.0 已授权继续发布

以Enmao最新明确指令为准：本任务继续完成dev→main普通merge、实际main CI验收、annotated tag v0.4.0、GitHub Release，再留言并关闭#36/#37/#38/#41/#48/#50。先检查实际PR/tag/Release与Issue状态，复用codex/v040-layout-release，不重复发布或移动已有标签。下方“仅合dev后停止/由监督助手发布”是当时状态，已被本节取代。#53不在关闭清单，分支和数据不删，不squash/强推，仍不测速、不跑GPU。完成事实见[record](record.md)，精确PR/commit/CI以Release附录为准。

## 2026-10-15 v0.4.0 目录规范与发布准备

本轮唯一工作分支 codex/v040-layout-release 从已验收dev5cff920建立，按Enmao授权准备0.4.0并经PR合dev后停止；实际PR/head/mergeCommit与CI以最新状态为准，避免重复建分支或用下方历史pending状态重派任务。项目根 shared/data、shared/model 不搬，layout.md补位置/内容/写入方与“先文档后路径”合同，新增真实默认路径与文档对照；生产仅版本号更新。pyproject、__version__、README为0.4.0候选，[发布材料](../releases/v0.4.0.md)包括0.3.x迁移命令、私有设置、三类正确性与已修Issue，见[record §76](record.md#76-v040-目录合同迁移材料与版本准备)。

本机36项目录/版本、core662通过/1平台跳过、CPU流程157通过；初轮临时目录setup错误和两项漂移负控留证。最终head与实际dev CI另绑定，不拿旧结果替代。main仍已发布0.3.1，本任务不合main、不打tag、不发布；这些由监督助手接续。七个已修Issue保持open，全部分支/正式数据/Run/provenance保留，不测速、不跑GPU、不改磁盘布局或调查480秒。

## 2026-10-15 B-070 指标类型审计与收尾

新用户要求审计readout/_atomic/launch参数与make/run-one边界；基于已验收dev b2390b0确认[B-070 / Issue50](https://github.com/DreamSoul-AI/RPipe/issues/50)，先开Issue、提交红色回归f15e067（10项7 failed/3 passed），记录设计3a198ff，再做一行修复0b51ee6：取消status提前dict强转，复用已有metric cell的非object占位，保留状态/模式/输入字节。候选与相邻91项通过，见[record §63](record.md#63-相邻正确性审计与-status-指标类型回退b-070--50)。

当前唯一新分支codex/status-metric-shape；先核对该分支PR最新head/state/mergeCommit，若已合dev则接续实际集成CI，不重复修复/新建分支。最终head的三门及所有实际矩阵、实际dev的两门分别验收；没有新验证前不拿旧B-069绿灯替代。原三个分支和全部数据保留，main7a15c46未变，六个Issue均不自动关闭，等Enmao决定main。

B-069/PR49已在dev b2390b0完成全部实际CI（含此前未完成包门的拒绝与最终全绿保全），下方“待验收”为历史阶段；B-065性能优化仍撤回。不测速、不改性能、不跑GPU或查480s，不覆盖下游vendor/实验，原科学结果不重签。

## 2026-10-15 四个 Issue 评论与 B-069 正确性接续

四个旧 Issue 已按新授权各补一条真实修复提交、测试位置与 dev 验收评论，保持 open，链接见 [record §61](record.md#61-四个开放-issue-补齐修复与验收评论)。新缺陷 [#48 / B-069](https://github.com/DreamSoul-AI/RPipe/issues/48) 来自相邻 readout I/O 查询：红色测试提交31d1511为5 failed /8 passed，最小候选与相邻81项通过，实际PR/集成CI必须另核对。

唯一新工作分支codex/status-readout-errors从已验收dev f9c6e90建立，旧两分支保留。生产只移动status/logs三个存在性查询到各自已有I/O边界；不恢复B-065性能去重，不测速/查超时，不改格式、算法或下游。接续先核对该分支PR的state/head/mergeCommit，若已经合入则不重复修复或派发；实际dev CI须与对应提交绑定，保留原5个失败。Issue48与旧四项均不自动关闭，main不变，数据/Run/provenance不删除。

## 2026-10-15 PR #46 已合 dev，四个 Issue 正确性接续

PR #46 已 normal merge，实际 dev `5ee1214b7ce0dc28b65139ac7b3f9ca16eb8b2ac`；最终 head `90b7d75` 三门及实际矩阵全通过，实际 dev 的 Unit38622291540 / Package38622291406 全部实际矩阵和聚合门通过，见 [record §59–60](record.md#59-pr-46-实际合入-dev-与归档后状态)。不要重复合并 PR #46 或把下方历史草案限制当作当前指令。

后续分支 codex/open-issue-correctness 从该 dev 建立，四个 Issue 已逐项补回归，本机 75 项通过；没有新增生产改动或新的根因反例。合入后仍需核对该 PR 最新 head 和实际 dev CI。#36 / #37 / #38 / #41 不自动关闭，main 不变，两个工作分支和全部数据保留。只做正确性，不测速、改性能、查超时或干预下游实验；B-065 优化已撤回，不恢复。

## 2026-10-15 归档、dev集成与四个Issue正确性接续

最终授权以本节为准：PR #46可转ready并在最终head全部必需检查/实际矩阵通过后合dev；main不改，不强推，不删除任何分支/数据，不发布或花钱。此前“保持草案/不合dev”为历史阶段要求。先核对[PR #46](https://github.com/DreamSoul-AI/RPipe/pull/46)当前state/mergeCommit和origin/dev；若已经合入，直接核对实际dev CI，不重复派发合并。

[归档状态](maintenance_archive_2026-10-15.md)列出已完成与未完成。B-056～B-068修复/测试和记录均已主题提交；B-065/ad67f74次数去重优化按最终只做正确性要求单独撤回，历史源码/失败/记录保留。927c36e三门与实际矩阵已通过；后续归档提交和真实dev必须各自验收。已完成Study结果保持原来源，未重训或重签。

合入后逐项核对开放#36/#37/#38/#41，现有修复已分别在更早PR中，先复验并补有意义的回归缺口，不伪称未修复或重复改代码，不自动关闭Issue。只做代码正确性，不测速、不做性能改动、不查超时；不干预三个下游对话的GPU工作。四任务共用24GB GPU，计时受外部负载影响的历史材料不作为新性能结论。


## 2026-10-14 当前 dev 集成与持续维护

共享及#36—#38修复已通过[PR #39](https://github.com/DreamSoul-AI/RPipe/pull/39)合入dev `614d0a6`，最终工作提交07bcf2e的三项必需检查及全部实际矩阵通过。B-053 / B-054从bugs开放区移除，详细依据见[集成记录](record.md#39-共享与三个反馈修复合入-dev)。dev集成CI独立核对；main仍为v0.3.1，不把dev修复描述成已发布。

已合并的codex/powershell-launch-boundaries按规则清理；后续工作分支codex/shared-inputs-followup从该dev建立，只保留一个未合并工作分支。继续维护时先fetch并核对实际基线 / 工作区，读当前Issues；其他三个对话的vendor和运行中的实验不自动覆盖，新反馈依据实际最小复现处理。#36—#38暂仍open，不在缺少新问题时恢复早前撤回的人工探索。

共享根默认项目shared/data与shared/model，raw及同版本processed / 固定Stats实际复用；Run模型对象、sampler、RNG、checkpoint和可变状态独立。已完成历史Run及来源保持；重新make / 实测必须新version，不能复用旧成功当新来源验收。下文待CI、未合并等记录保留其当时状态，以本节为接续入口。

## 2026-10-14 跨 Study 共享与持续缺陷目标

当前工作分支 `codex/powershell-launch-boundaries`，基线6619edc已含 #36—#38 修复；新增项目级共享输入、私有例外、进程文件锁与显式迁移，见 [当前记录](record.md#35-跨-study-共享输入与本机缓存迁移b-053) / [恢复身份复查](record.md#36-共享统计路径与恢复身份复查) / [共享合同](../code/layout.md#跨-study-共享资产2026-10-14)。core456 passed / 2 skipped、CPU integration66 passed、两个分发包各9 CLI / 4 Toy Run通过；跳过的是本机目录符号链接权限。[草稿PR #39](https://github.com/DreamSoul-AI/RPipe/pull/39)首批ac909ec检查全绿，profile_path后续提交的最终CI须单独核对。

本机公共 raw 已放在项目 shared/data，旧路径仍指向同一文件。历史result/index/process/provenance和数值来源保留；不能重新make这些已完成Study后冒充旧验收。新独立实测设置新version；probe仍需同时更新fixed.version与基底version。当前小验证只有MNIST/linear单格CPU及CUDA，详细来源和配置层覆盖错误保留在Study报告。

用户授权持续处理其他三个对话反馈的bug、关注Issues。当前#36—#38仍open；不得仅因本机回归就宣称已集成到dev/main或已发布。其他项目的vendored来源和正在运行的实验不由本目标自动覆盖；新反馈先核对实际Issue、源码及最小复现，再定向修复。下文收缩 / 仅待设计等文字属于早前状态，本节为当前接续入口。

最新接续：#36的未扩展prepare初始冻结拒绝也补齐成功保全，见 [B-054记录](record.md#37-issue-36-未开启扩展的初始冻结拒绝b-054)。定向32、core456 / 2 skip、CPU integration68通过，新CPU单格Run44492c0d8f5cf190通过。PR #39首批ac909ec全绿；后续最终提交的必需CI以GitHub当前head为准，不能据旧一次全绿合并。

当前还将同版本processed / 固定Stats纳入项目级派生缓存，区别4ccb28d与98648f3格式，见[最新缓存验证](record.md#38-固定格式-processed--stats-也跨-run-复用)。当前core463 / 2 skip、CPU integration68及两分发包基本CLI通过；两个linear数据格只是新来源小门，不是新完整Study。共享公共输入与独立Dataset / 模型 / sampler状态不可混为一谈。最终PR检查须绑定最新head。

## 2026-10-14 工作分支异机交接

用户已授权将当前收缩后的代码、测试和文档提交推送到 `codex/powershell-launch-boundaries`，在另一台机器继续；不合并 dev / main，不发布，不关闭 Issue #36 / #37 / #38。此前“未提交 / 不推送”是当时状态，以本节最新授权为准。实际交接提交以远端分支和 `git log -1` 为准。

新机器先保留当地改动，再执行 `git fetch origin`；尚无本地分支时执行 `git switch --track origin/codex/powershell-launch-boundaries`，已有分支则切换后用 `git pull --ff-only`。阅读[当前收缩记录](record.md#32-按用户要求收缩到本机适配与三个下游-issue)和[共享下载待设计要求](brainstorm.md#2026-10-14-数据与模型下载共享约定用户要求待设计)。共享下载仅记录要求，尚未实现。

`.tmp/` 的 venv、MKL 激活脚本、缓存、备份和本机验证日志不随 Git 推送；正式 Study 生成的 Run / 数据 / checkpoint 也不作为本次异机传输内容。目标机器需检查自己的 Python / Conda / OpenMP 环境并按 README 的源码接入方式准备，再做小验证；不能把本机 SEQUENTIAL 绕过或本机通过当作异机环境已验收。

## 2026-10-14 当前收缩范围（未提交）

当前 Issue #36 / #37 / #38 均为 open / reopened：用户要求暂不关闭、后续再修，已撤回本轮关闭操作并复查状态。现有本机补丁保留但不继续修复 / 提交 / 发布，见[状态记录](record.md#33-issue-状态按用户最新要求恢复开放)。

仅保留本机迁移 / MKL 和 Issue #36、#37、#38。公共源码差异为 runner.py、readout/status.py、_atomic.py 三个文件；四个对应测试文件保留。其他主动探索的公共改动已撤回，历史证据和完整撤回备份保留，见[收缩记录](record.md#32-按用户要求收缩到本机适配与三个下游-issue)。

当前定向75 passed、core440 passed / 2 skipped；新PowerShell下两seed四个CPU Toy/Stub Run全链和重复跳过通过。正式Study / 数据保全，八调度只做清单与语法核对。未commit / push / 发布、未更新Issue；此前扩展边界不属于当前已修复范围，后续不再自动扩展。

## 2026-10-14 v0.3.1 文档与发布准备

当前发布准备版本为0.3.1：B-030修复、源码模板接入与旧804b3823的实验材料均进入版本区间；新增研究 / knowledge skills与下游Issue反馈约定，入口见[README](../../README.md)和[发布报告](../releases/v0.3.1.md)。正式科学报告、数字、配方、旧失败与快照保持原来源；本次不训练或扩大矩阵。core420 passed / 2 skipped、CPU流程52 passed、两种分发包各9条CLI / 4个Toy/Stub Run通过，环境失败、skip、文档扫描与本机包边界见[开发记录](record.md#10-研究工作下游反馈与-v031-发布准备)。

维护者已授权工作PR进入dev、dev发布PR进入main和手工v0.3.1标签 / Release；所有实际提交仍须通过必需检查，不直接push长期分支。下一阶段按[发布合同](cicd.md#六维护者授权后的版本发布)完成，最终状态以GitHub Release及其commit / PR / CI链接为准。下方按日期保留旧交接，不把其中“main不改动”或“尚未提交”视为本次授权边界。

## 2026-10-14 B-030 修复接续

report 已补充当前 index 与 process 的 Study / Run ID / seed / 因素归属校验；过期或不可用来源拒绝导出，旧数字表通过原子替换保全，不新增持久化字段、不删除旧聚合。下文及正式 Study 报告中的“B-030仍open”描述当时804b3823实验基线；当前修复与验证见[开发记录](record.md#9-b-030-报告来源校验与分支清理)。旧实验来源不变，新独立实验仍须新version与前置门，本次不跑大规模实验。

PR #32 的实验 / 本地模板文档已合入 dev。已删除 codex/v030-local-studies 远端与本地分支，以及已核对完成的旧本地工作分支；后续工作仅保留当前单个工作分支，合并完成后再清理，main不改动。

## 2026-10-13 v0.3.0 本机完整实验接续

维护者新增授权本机环境、完整数据、小门、受控并发校准及门通过后的完整 Study。固定软件基线为发布标签 `v0.3.0` / `804b3823ab6d2af46cbda1a65688ca8b76c60b22`，工作分支 `codex/v030-local-studies`；没有跟随更新后的 main，也没有改库 / Study 计算源码或数值门。下面发布准备阶段的“只做短实验”及旧未完成状态保留其历史语境。

隔离环境位于 `.tmp/v030-20261013/venv`，不继承系统 site-packages；旧 `.tmp/runtime` 原样保留为 `.tmp/runtime-before-v030-20261013`。完整 raw 16/16 hash、真实 CUDA FP32、397项core、51项CPU integration均通过。首次缺归档 transformers 的导入失败及初始 collection skip 保留；最终依赖门无 skip。实验与校准数字只在对应 Study 维护，见 [main_exp 本轮记录](../../studies/main_exp/docs/v030_execution.md)、[main_probe 本轮记录](../../studies/main_probe/docs/v030_execution.md)。

两项 Study 的并发资源独立实测：main_exp 27个800-step原生容量Run通过；main_probe串行 / 手动round2及原生自动分组通过，round3因保护未执行。正式预算分别4.5 / 5GiB，原生自动两并发、safety=0.5。正式版本为 `main-exp-v030-20261013-final` / `main-probe-v030-20261013-final`，fresh make后的来源保持冻结，旧报告 / 声明 / 清单已逐字节归档，旧Run保留。

本轮完整验收已完成：main_probe八格60-step及Study终验、main_exp八格600-step前置門、32条各自连续80000-step train、32条own-best eval、400点完整性及八曲线 / 两模型次序固定門全部通过。实际软件来源仍是最新已核对main / v0.3.0的804b3823，未改库 / Study计算源码或数值门。

原调度20:59左右在30条train完成后终止，两条CIFAR10 / ResNet18 seeds2–3保留约24400步的中断证据，退出原因未取回。按[同源码接续方案](../../studies/main_exp/docs/v030_recovery.md)，仅为这两train及其配套eval新建recovery01身份；原30条结果hash / provenance保持原样，旧 / 新源码映射完全一致。21:14–23:11接续新两条从step0训练并执行全部32条评测，public launch exit0、source_unchanged=true。不能称一次连续launch，也不代表自定义sampler支持resume。

[64条最终审计](../../studies/main_exp/docs/v030_long_audit.json)验证actual FP32权重、committed scheduler进度、400点、当前index / process / COMPARISON身份、32个自身best来源及两份真实provenance；原中断 / 新重跑976个已保留指标前缀精确一致。正式结论及图在[main_exp报告](../../studies/main_exp/docs/STUDY_REPORT.md)，数字由原生report生成。未复原历史原始seed / 逐点指标 / 权重，结论限预置PNG估读门。

原会话97306与exec cell93已失效；接续使用隐藏后台观察器，现已正常结束。资源摘要为studies/main_exp/docs/v030_recovery_resources.json，原中断为v030_interruption.json；临时启动 / 观察输出和JSONL在.tmp/v030-20261013，仅本机。普通沙箱看不到外部后台进程，不能据此判定死亡；观察器可选磁盘计数缺失已容错并保留首次启动失败。当前接续index保留30个原train与4个新的身份槽位，直接rpipe make原声明会回到中断的旧ID；以后独立复跑使用新version / fresh make与新前置门。

这轮提交 / 推送 / PR状态以Git和GitHub为准，不自动同步main或发布新版本。

新发现B-030：新index与保留旧process不匹配时，report仍能exit0导出旧64个Run数字。独立复制元数据的公开CLI复现已保存，详见[bugs.md](bugs.md)与[Study证据](../../studies/main_exp/docs/v030_stale_report_repro.json)。尚未修复；本轮固定804b3823计算来源的长矩阵已终验，通过额外核对新index / process / COMPARISON的ID集合防止混用旧聚合。细分预算、耗时均衡与依赖就绪检查另记[brainstorm.md](brainstorm.md)，不在本轮自动实施。

## 2026-10-13 接续摘要

当前工作树以dev ce8b3e6为基线，已完成原生CPU/CUDA恢复、CPU设备分流及整仓检查发现的比较 / 写入 / 名称 / metric / 空训练边界修复。395项core、51项CPU integration、CUDA跨进程恢复及wheel / sdist公开CLI验证通过；详细失败与支持范围见[record.md](record.md)2026-10-13。

main_exp已完成27次真实短容量Flow、固定best的32次精度诊断，以及最终候选源码的600-step八格前置对照；使用显式4.5GiB预算，库安全系数不变。新的正式候选声明为main-exp-flow-20261013-04；旧长矩阵声明与结果在版本快照保留，不声称新候选已长训。前置报告只绑定其中的实际来源；旧PREFLIGHT不自动升级，异机仍须同设备fresh gate。

main_probe使用main-probe-flow-20261013-06执行八组合60-step短复验，原version05两组子集证据和生成清单逐字节保存。发布按工作PR→dev→main发布PR，main合并在用户要求的确认边界前不执行。下文2026-10-12条目保留当时状态，不作为本轮最新状态。

更新日期：2026-10-12。本轮基线为 dev `7eb16438` 加两柱重构：两个 Study 已拆为 `structure/` 建构 / 计算与 `flow/` 阶段检查，库加载和 provenance 自动来源发现已接通。已完成 main_probe version05 两组真实 60-step Run 与 main_exp 单组手动 200-step 前置对照；未运行新完整矩阵，历史 version03 不作为新链验收。本次提交前复核发现 collect 包入口仍有重复实现，再次清除并重新回归；实测来源 hash 保留实验时状态，不改写成最终源码。提交、PR 和远端必需检查状态以 Git / GitHub 为准。当前接续顺序见第五节。

最新接续：上述重构已通过 [PR #28](https://github.com/diaoenmao/RPipe/pull/28) 合入 dev `ce8b3e6`，PR 与 dev 提交后的必需测试 / 包检查均通过。随后本机修复 B-021 的 native checkpoint RNG / 数据位置恢复，专项 10 passed、core 335 passed / 2 skipped、CPU integration/e2e 49 passed，尚未提交；具体支持范围与证据见 [record.md](record.md) 2026-10-12 §6 和 [恢复合同](../code/structure.md#6113-resumetrain-与-eval-的特殊接口)。B-022 / B-023 仍开放；旧 Study 的拒绝续训门、旧来源和失败不改写。下文旧阶段表与旧“尚未提交”描述保留当时状态，不当作本轮最新状态。

继续修复：make 的嵌套配置 / 设备优先级、训练行数、累积 batch 估时及 test 显存遗漏已修正，旧计划设备不符时重建；新发现 B-024 的 epoch 越界 / 零更新循环已修复并关闭。最新 core 355 passed / 2 skipped、CPU integration/e2e 51 passed、CUDA 恢复1 passed，包含累积尾 batch 的跨进程恢复；尚未提交推送。B-022 的配置正确性修复不替代峰值 / 并发吞吐校准，B-023 kernel 调查未扩大。具体失败和验证见 [record.md](record.md) 2026-10-12 §7，旧 Study 拒绝续训门继续保留。

最新校准接续：显式 `system.config.vram_bytes` 预算已接入 make / 原装箱器，仍保留 safety=0.5；现有速度工具补齐 raw / 源码 / 实际 runtime 证据与 reserved 峰值，代表性 MNIST 短校准通过，失败保留在 [速度报告](../../studies/main_exp/docs/speed_calibration.md)。最终 core 363 passed / 2 skipped、CPU integration 51 passed，尚未提交。没有改正式预算或验证 CIFAR10 并发；原 B-023 checkpoint 缺失、数据不齐，kernel 反例不能在当前工作区重放。开发事实见 [record.md](record.md) 2026-10-12 §8，实验数字只见 Study 报告。

2026-10-11 已合入基线：代码修复及完整复跑证据已通过 [PR #26](https://github.com/diaoenmao/RPipe/pull/26) 合入 dev 51a4ce0，最终 PR HEAD 31acb5b 的三项必需检查全部通过。Flow 职责重构通过 [PR #25](https://github.com/diaoenmao/RPipe/pull/25) 合入 dev `d73b217`，version02 真实复跑源码为 `e682f25`。main_probe 八组合通过；main_exp 32 train / 32 eval 执行成功但 best 一致性终验失败，旧版完整报告已在各 Study docs/versions/ 下归档。B-023 在默认 cuDNN TF32 的并发及串行条件下均复现；FP32 诊断八次一致。已实现显式精度配置并通过 runtime / core 回归，声明两个 version03，main_probe 八格真实探针通过（源码 c9c782b、参数最大差 0）；main_exp 归档导入覆盖 benchmark 的修复基线 09e495b 八格前置门通过，完整矩阵于 05:03 启动，round=3，实际设备开关核验通过，32 train / 32 eval 与固定曲线终验全部通过（19:48，原生 exit 0）。后续先查看 [长实验记录](../../studies/main_exp/docs/RERUN.md)、[恢复方案](../../studies/main_exp/docs/eval_recovery.md) 和 [探针记录](../../studies/main_probe/docs/RERUN.md)，不要复用旧版前置门或训练冒充新精度验收。B-021 / B-022 / B-023 均开放，尚未发布或合入 main。

## 一、摘要

| **范围** | **接续前** | **当前状态** |
| --- | --- | --- |
| 库扩展 | Study 自写调度和比较入口 | PR #21 合入 recipe、来源清单、freeze 与库 compare |
| 阶段链 | Study 阶段目录未接入 Runner | 六阶段扩展已接入；本机加载路径改为 `flow/<phase>/`，两柱源码自动冻结，旧根阶段包须迁移 |
| main_exp | 根 compare.py 执行终验 | 历史建构放 `structure/`，`flow/prepare/` 检查来源、Study process 做四 seed 曲线终验；手动读取直接调用 `flow/process/curves.py`，根 compare.py 已删除 |
| main_probe | 独立 probe.py 与当前侧 16 Run 两个入口 | 统一 CLI 管理八个成对 Run；Algorithm 放 `structure/algorithm/`，六阶段检查与终验放 `flow/` |
| 本机目录 | 存在未跟踪的 main_historical 残留 | 已按维护者要求删除，正式 Study 为 main_exp / main_probe |
| 科学验收 | version03 与速度校准已有历史实测，version02 失败保留 | 2026-10-12 main_probe 两组 60-step Run 与 main_exp 单组 200-step 手动前置对照通过；未通过新完整矩阵验收，最终入口清理后的实测仍须新 version / make |

## 二、库与 Study 的分工

库负责声明展开、调度、Run 阶段链、artifact 合同、来源冻结及通用聚合。Study 专用建构 / 计算放 `structure/`，通过根 recipe 注册自己的 source，通过 `flow/<phase>/` 的同名阶段包补充本研究的检查、证据和终验。

```text
prepare → execute → collect → summarize → write → process
```

`flow.study_phases: true` 显式启用阶段包。prepare 初始化后调用 Study `before(ctx)`，再执行 recipe / Factory 构造，最后调用 Study `run(ctx)`；write 先执行 Study，再由库定稿；其他阶段先库后 Study。recipe 在 Data / Model 构造前注册。没有 `__init__.py` 的工具目录不自动执行；两柱全部 Python 源码自动纳入 provenance，额外输入才需 include。freeze 只核对来源与执行计划哈希，不检查环境一致，也不保存源码备份，详见 [来源清单](../code/flow.md#142-来源清单)。

库与两个 Study 的 Flow 阶段实现已移出 `__init__.py`：普通阶段在 `run.py`，库 process 沿用 `individual.py` / `study.py` 等职责模块，包入口只做导出。现有包级 `run` 调用和 Study hook 加载方式不变；目录展开到文件，见 [layout.md](../code/layout.md)。无 Study hook 时仍执行库默认阶段。

Run process 使用 `ctx.scope == 'run'`，读取本 Run 的结果。整轮 process 在通用聚合后另调用一次 Study hook，使用 `ctx.scope == 'study'`，没有单条 Run 的 layout/control。需要让本 Run 的失败反映在 result 中的数值门，必须在 write 成功前执行。process 派生失败保留已写成功的 Run result，并向调用方报错。

## 三、正式 Study 当前状态

### （一）main_exp

[main_exp](../../studies/main_exp/README.md) 声明 32 条训练与 32 条自身 best 独立评测。recipe 注册 `structure/` 的固定历史来源 `4ccb28d` 适配器，检查 preflight，并拒绝中断训练的 checkpoint 续跑。`flow/prepare_data.py` 负责 raw 下载及清单核对。声明中关闭通用共享 Data 预构造，专用 source 在 Run prepare 内注册。

Study `flow/prepare/` 检查构造对象的历史来源。`flow/process/curves.py` 保留曲线门限计算，Study process 在聚合后调用，也支持显式手动完整 / partial 读取；速度工具由 `recipe.py calibrate` 调用 `flow/execute/speed.py`，不自动进入阶段链。完整性与数值门见 [PLAN.md](../../studies/main_exp/docs/PLAN.md) 和 [TARGET.md](../../studies/main_exp/docs/TARGET.md)。

### （二）main_probe

[main_probe](../../studies/main_probe/README.md) 声明八个成对 Run，每个只处理一个 data/model 组合。Study prepare.before 完成该组合 CPU 准备，recipe 注册专用 Data / Algorithm，并恢复本 Run 的 seed。`flow.prepare_shared: false` 避免 recipe 注册前构造专用数据。

`structure/algorithm/paired.py` 的成对 Algorithm 在同一进程先执行固定原版 `98648f3`，再用 Flow 构造的当前侧 Data / Model / System / Tracker 执行当前版，并分别评测自身 best。初始化 RNG 在构造后记录，进入当前侧计算时恢复。

`flow/execute/` 确认一对计算完成；collect 比较输入、RNG、样本计数、逐段参数、optimizer、scheduler、best 与独立 eval；summarize 执行数值成功门；write 投影已有观测并调用库 compare，在库 result 成功定稿前验收。数值门失败交由 Flow 写 failed result，原始证据留在 Run 的 `assets/probe/`。Study process 只汇总当前 index，八组齐全、来源/设备/Torch 一致且各组通过才通过终验。

## 四、验证与证据边界

| **阶段** | **代码基线** | **本地验证** | **集成状态** |
| --- | --- | --- | --- |
| 库扩展 | `0e3fec5` | core 272 passed / 29 deselected，CPU integration/e2e 26 passed / 275 deselected | [PR #21](https://github.com/diaoenmao/RPipe/pull/21)，合入 dev `2917e3e` |
| 同构阶段链 | `8fbff78` | core 282 passed / 35 deselected，CPU integration/e2e 32 passed / 285 deselected | [PR #22](https://github.com/diaoenmao/RPipe/pull/22)，合入 dev `3fe8bb8` |
| 成对探针接入 | `818689e` | core 286 passed / 35 deselected，CPU integration/e2e 32 passed / 289 deselected | [PR #23](https://github.com/diaoenmao/RPipe/pull/23)，合入 dev `22442f9` |
| 2026-10-12 两柱加载与来源发现 | dev `51a4ce0` 加本机未提交改动 | 针对性 44 passed / 2 skipped；core 308 passed / 2 skipped / 40 deselected；CPU integration 37 passed / 313 deselected；真实 Study 的隔离构造与 loader 检查通过，未训练 | 本机验证，尚无对应提交的远端 CI；两个目录符号链接用例因本机权限跳过，证据见 [record.md](record.md) |
| 2026-10-12 阶段实现显式化 | 同一 dev 基线加本机未提交改动 | Flow 针对性 103 passed / 2 skipped；core 321 passed / 2 skipped；CPU integration 40 passed；13 个搬迁模块 AST 不变、15 个包入口仅导出，实际 Study 隔离自检通过 | 无正式训练或新科学验收；符号链接边界仍跳过，尚无新提交 CI |

上述已合入 PR 的 Unit tests、Build package、Branch flow 必需检查均通过，不覆盖本机未提交改动。前一阶段缺 Kornia、最新 basetemp 父目录缺失的失败及后续复验记录保留在 [record.md](record.md)。本机验证使用 `MKL_THREADING_LAYER=SEQUENTIAL`、`MPLBACKEND=Agg`，并复用隔离的 Kornia 依赖。

本轮真实短实验记录见 [main_probe 报告](../../studies/main_probe/docs/STUDY_REPORT.md) 和 [main_exp 报告](../../studies/main_exp/docs/STUDY_REPORT.md)。main_probe 的 Study process 正确拒绝剩余六组未运行的子集；main_exp 不是正式 Flow Run，也不解锁八格 600-step 前置门。首次缺 evaluate 的 prepare 失败及 version04 快照保留；MNIST raw 8/8 校验通过，CIFAR 下载未完成。本轮提交前再次检查发现 collect 入口恢复为旧实现，核心回归 319 passed / 2 failed / 2 skipped，CPU integration 40 passed；两项失败均由 collect 导出造成。后续清除重复实现并重新验证，不能把旧一次通过当作当前源码通过。

本次修正后核心回归 321 passed / 2 skipped，CPU integration/e2e 40 passed；15 个 Flow 包入口核查通过，collect 函数 AST 不变，原短实验 result / provenance 与历史快照保留。符号链接边界仍因本机权限跳过，跨平台及安装包验收以本次 PR 的 Unit tests / Build package 必需检查为准。

成对探针的隔离 CPU 检查使用伪计算和微小张量，覆盖单组合调度、真实 Flow 对象与 RNG 交接、artifact compare、保留既有证据、failed 状态、子集拒绝与八组聚合。本机检查在 `.tmp/probe-flow/`，不随 Git clone 提供。它没有执行正式训练，不能证明本版本的科学数值一致性。

先前正式实测保存在 [`18cd76c` 的 studies](https://github.com/diaoenmao/RPipe/tree/18cd76c/studies)，当时名称为 main_historical / main_reproduction。原交接快照见 [`0e3fec5` 的 handover.md](https://github.com/diaoenmao/RPipe/blob/0e3fec5/docs/development/handover.md)，其中未完成项描述当时状态。

## 五、后续工作与入口

### （一）2026-10-12 当前接续建议

1. 两柱设计、Study 文档和短实测已对齐。当前授权先修 collect 导出，重新回归，核对完整暂存范围，再通过工作分支提交、push 和 PR 合入 dev；不直接 push dev
2. 已完成的短数值结果不重新覆盖：main_probe 两组完整六阶段链通过，子集 process 拒绝八组 complete；main_exp 单组 200-step 手动前置对照通过。实测后源码变化需新 version / make，再执行真实复验
3. 后续八组合探针、main_exp 同设备 600-step 八格前置门及异机验收按计划另行授权。换机须重新 make 并核对依赖、数据、实际精度与设备；本机结果不能替代异机验收
4. 完整四 seed / 80000-step 矩阵另行授权。保留历史 version 与失败，不重用旧 Run / 前置报告；B-021 后续已按原生单进程恢复合同本机修复，待提交和跨平台验收；B-022 / 默认精度 kernel 调查仍开放，不因目录重构自动关闭

### （二）2026-10-10—11 历史接续

以下保留当时的实施顺序与已完成实验，不作为再次启动会话或扩大实验的授权。

PR #24 的文档审校已合入 dev `37b9cef`。维护者在 2026-10-10 追加了后续顺序：

1. Flow 阶段职责已通过 [PR #25](https://github.com/diaoenmao/RPipe/pull/25) 合入 dev `d73b217`：prepare 准备、execute 捕获、collect 比较、summarize 数值门、write 投影与定稿。CPU 核心 286 项、集成 36 项及三条探针隔离路径通过，远端三项必需检查通过
2. 当前持续目标覆盖 Flow 重构及实验复跑；已进入实验阶段，按现行计划复跑 main_probe 与 main_exp。原始数据 16/16 SHA-256 核对通过；新 version、环境、执行状态与失败分别见 [probe 复跑记录](../../studies/main_probe/docs/RERUN.md) 和 [exp 复跑记录](../../studies/main_exp/docs/RERUN.md)。先完成真实单组合和计划前置门，再运行完整矩阵
3. 复跑使用新的 version / Run ID，记录实际源码基线、环境、预算与失败。代码问题记 bugs.md，开发修复记 record.md，实验数值与结论记各 Study 报告

本轮实验工作分支 feature/study-rerun 已通过 PR #26 合入 dev 并删除。main_probe version03 在 c9c782 基线上通过；main_exp version02 已结束但终验 invalid，原始结果与报告保留。main_exp 专用归档导入修复提交为 09e495b，当时未执行的 version03 计划重新冻结后、round=3，新八格前置门已全部通过，完整矩阵于 05:03 启动（会话 13127，.tmp/main-exp-flow-20261011-03-launch.log，仅本机证据）；旧源码通过的前置报告会被实际 launch 检查拒绝，拒绝回归已通过。全部 64 个 Run 实际开关与 checkpoint / tracker 合同核查通过，整个连续实验源码、声明与提交保持不变。原生 launch 已于 19:48 正常退出，不再重启会话 13127。同条件串行历史速度校准已完成，准备优化的独立八格回归和 PR 最终提交检查也已通过。实验产物推送及发布按相应授权和仓库流程处理，发布仍按工作分支 → PR → dev → 发布 PR → main 流程。

| **内容** | **入口** |
| --- | --- |
| Flow 阶段与扩展 | [flow.md](../code/flow.md) |
| 目录约定 | [layout.md](../code/layout.md) |
| Study 使用 | [README 的 Study 指南](../../README.md#study-使用指南)，现有研究见 [Study 导航](../../studies/README.md) |
| CI 与分支 | [cicd.md](cicd.md) |
| 开发事实与历史失败 | [record.md](record.md) |

4. 维护者于 2026-10-11 要求在当前连续实验结束后，与早年版本做同条件速度校准。按 B-022 的历史问题核对旧版本提交和计时证据，拆分训练、完整评测、checkpoint 与调度耗时，分别比较串行和并发；保持数值条件可比，再据实测决定优化。该问题后续收口见 [开发记录](record.md#2-b-022--b-023-实测收口)，不再链接已移除的开放 bug 锚点。长训期间不启动竞争 GPU 的速度实验；该历史长训已结束，最新短校准另记。

19:57 已在正式长训结束后执行串行校准首轮，CNN 数值门通过；[测量报告](../../studies/main_exp/docs/speed_calibration.md) 保留原始 JSON 和对照条件。六组独立进程复测全部通过，原生会话 66360 已 exit 0，日志 `.tmp/speed-repeat-1.log` 至 `speed-repeat-5.log`（仅本机）。该批次已结束，不再轮询或重启。校准修改了 recipe 的命令入口，已完成正式 version03 的来源仍按 09e495b 与原 provenance 解释；后续正式复跑须新 version / make / preflight，不能在这套旧冻结来源上重新 launch。

后续提交 `8c17d5d` 保留上述完整报告并将归档 blob 读取合并为 Git batch。全部36个文件字节与原清单相同，拒绝不同内容及四种损坏响应，真实导入运行条件回归通过；核心295项、独立 basetemp 的集成37项通过，首次共享临时目录导致的验证错误已记入 record.md。到 dev 的 [PR #26](https://github.com/diaoenmao/RPipe/pull/26) 已在最终提交31acb5b的 Branch flow / Unit tests / Build package 全部通过后，于20:29合入 dev 51a4ce0。批量导出后的新源码八格600-step回归已于20:23全部通过（参数最大差0、Loss最大差4.44e-16），原生会话10327 exit0，输出独立目录 `.tmp/main-exp-batch-prefix-20261011-3e158abd/`，完成后写 `docs/BATCH_EXPORT_PREFLIGHT.json`；没有覆盖原 version03 的 PREFLIGHT 或64个 Run。会话10327已结束，不再轮询或重启；后续接续应读取 dev 的实际 HEAD 与对应 CI 状态。不能把旧矩阵当作优化后的完整矩阵。
