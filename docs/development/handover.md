# Flow 与 Study 重构交接

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

4. 维护者于 2026-10-11 要求在当前连续实验结束后，与早年版本做同条件速度校准。按 [B-022](bugs.md#b-022-重型任务并发容量与吞吐尚未校准) 核对旧版本提交和计时证据，拆分训练、完整评测、checkpoint 与调度耗时，分别比较串行和并发；保持数值条件可比，再据实测决定优化。长训期间不启动竞争 GPU 的速度实验；该历史长训已结束，最新短校准另记。

19:57 已在正式长训结束后执行串行校准首轮，CNN 数值门通过；[测量报告](../../studies/main_exp/docs/speed_calibration.md) 保留原始 JSON 和对照条件。六组独立进程复测全部通过，原生会话 66360 已 exit 0，日志 `.tmp/speed-repeat-1.log` 至 `speed-repeat-5.log`（仅本机）。该批次已结束，不再轮询或重启。校准修改了 recipe 的命令入口，已完成正式 version03 的来源仍按 09e495b 与原 provenance 解释；后续正式复跑须新 version / make / preflight，不能在这套旧冻结来源上重新 launch。

后续提交 `8c17d5d` 保留上述完整报告并将归档 blob 读取合并为 Git batch。全部36个文件字节与原清单相同，拒绝不同内容及四种损坏响应，真实导入运行条件回归通过；核心295项、独立 basetemp 的集成37项通过，首次共享临时目录导致的验证错误已记入 record.md。到 dev 的 [PR #26](https://github.com/diaoenmao/RPipe/pull/26) 已在最终提交31acb5b的 Branch flow / Unit tests / Build package 全部通过后，于20:29合入 dev 51a4ce0。批量导出后的新源码八格600-step回归已于20:23全部通过（参数最大差0、Loss最大差4.44e-16），原生会话10327 exit0，输出独立目录 `.tmp/main-exp-batch-prefix-20261011-3e158abd/`，完成后写 `docs/BATCH_EXPORT_PREFLIGHT.json`；没有覆盖原 version03 的 PREFLIGHT 或64个 Run。会话10327已结束，不再轮询或重启；后续接续应读取 dev 的实际 HEAD 与对应 CI 状态。不能把旧矩阵当作优化后的完整矩阵。
