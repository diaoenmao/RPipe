# Flow 与 Study 重构交接

更新日期：2026-10-11。Flow 职责重构通过 [PR #25](https://github.com/diaoenmao/RPipe/pull/25) 合入 dev `d73b217`，version02 真实复跑源码为 `e682f25`。main_probe 八组合通过；main_exp 32 train / 32 eval 执行成功但 best 一致性终验失败，旧版完整报告已在各 Study docs/versions/ 下归档。B-023 在默认 cuDNN TF32 的并发及串行条件下均复现；FP32 诊断八次一致。已实现显式精度配置并通过 runtime / core 回归，声明两个 version03，main_probe 八格真实探针通过（源码 c9c782b、参数最大差 0）；main_exp 归档导入覆盖 benchmark 的修复基线 09e495b 八格前置门通过，完整矩阵已于 05:03 启动，round=3，实际设备开关核验通过，32 train / 32 eval 与固定曲线终验已全部通过（19:48，原生 exit 0）。后续先查看 [长实验记录](../../studies/main_exp/docs/RERUN.md)、[恢复方案](../../studies/main_exp/docs/eval_recovery.md) 和 [探针记录](../../studies/main_probe/docs/RERUN.md)，不要复用旧版前置门或训练冒充新精度验收。B-021 / B-022 / B-023 均开放，尚未发布或合入 main。

## 一、摘要

| **范围** | **接续前** | **当前状态** |
| --- | --- | --- |
| 库扩展 | Study 自写调度和比较入口 | PR #21 合入 recipe、来源清单、freeze 与库 compare |
| 阶段链 | Study 阶段目录未接入 Runner | PR #22 合入六阶段扩展、模块隔离和阶段源码冻结 |
| main_exp | 根 compare.py 执行终验 | prepare 检查来源，Study process 做四 seed 曲线终验，根 compare.py 保留手动 partial 读取 |
| main_probe | 独立 probe.py 与当前侧 16 Run 两个入口 | PR #23 移除脚本入口，统一 CLI 管理八个成对 Run、失败结果与八组终验 |
| 本机目录 | 存在未跟踪的 main_historical 残留 | 已按维护者要求删除，正式 Study 为 main_exp / main_probe |
| 科学验收 | 已有先前版本实测，重构后尚未实测 | main_probe FP32 version03 八组合通过；main_exp version02 终验失败保留，version03 修复基线前置门通过、32 train / 32 eval、64 个 Run 完整性及固定曲线终验全部通过；CNN / ResNet18 各三次串行历史速度校准完成，容量调查继续 |

## 二、库与 Study 的分工

库负责声明展开、调度、Run 阶段链、artifact 合同、来源冻结及通用聚合。Study 通过 recipe 注册自己的 source，通过同名阶段包补充本研究的检查、证据和终验。

```text
prepare → execute → collect → summarize → write → process
```

`flow.study_phases: true` 显式启用阶段包。prepare 初始化后调用 Study `before(ctx)`，再执行 recipe / Factory 构造，最后调用 Study `run(ctx)`；write 先执行 Study，再由库定稿；其他阶段先库后 Study。recipe 在 Data / Model 构造前注册。阶段源码自动纳入 provenance。

Run process 使用 `ctx.scope == 'run'`，读取本 Run 的结果。整轮 process 在通用聚合后另调用一次 Study hook，使用 `ctx.scope == 'study'`，没有单条 Run 的 layout/control。需要让本 Run 的失败反映在 result 中的数值门，必须在 write 成功前执行。process 派生失败保留已写成功的 Run result，并向调用方报错。

## 三、正式 Study 当前状态

### （一）main_exp

[main_exp](../../studies/main_exp/README.md) 声明 32 条训练与 32 条自身 best 独立评测。recipe 使用固定历史来源 `4ccb28d`，检查 preflight，并拒绝中断训练的 checkpoint 续跑。`prepare_data.py` 负责 raw 下载及清单核对。声明中关闭通用共享 Data 预构造，专用 source 在 Run prepare 内注册。

Study prepare 检查构造对象的历史来源。`process/curves.py` 保留曲线门限计算，Study process 在聚合后调用。根 `compare.py` 提供显式手动读取。完整性与数值门见 [PLAN.md](../../studies/main_exp/docs/PLAN.md) 和 [TARGET.md](../../studies/main_exp/docs/TARGET.md)。

### （二）main_probe

[main_probe](../../studies/main_probe/README.md) 声明八个成对 Run，每个只处理一个 data/model 组合。Study prepare.before 完成该组合 CPU 准备，recipe 注册专用 Data / Algorithm，并恢复本 Run 的 seed。`flow.prepare_shared: false` 避免 recipe 注册前构造专用数据。

成对 Algorithm 在同一进程先执行固定原版 `98648f3`，再用 Flow 构造的当前侧 Data / Model / System / Tracker 执行当前版，并分别评测自身 best。初始化 RNG 在构造后记录，进入当前侧计算时恢复。

execute 内保留输入、RNG、样本计数、逐段参数、optimizer、scheduler、best 与独立 eval 门，并调用 write 模块投影已有观测、执行库 compare。数值门失败交由 Flow 写 failed result，原始证据留在 Run 的 `assets/probe/`。write 阶段检查证据存在，Study process 只汇总当前 index，八组齐全、来源/设备/Torch 一致且各组通过才通过终验。

## 四、验证与证据边界

| **阶段** | **代码基线** | **本地验证** | **集成状态** |
| --- | --- | --- | --- |
| 库扩展 | `0e3fec5` | core 272 passed / 29 deselected，CPU integration/e2e 26 passed / 275 deselected | [PR #21](https://github.com/diaoenmao/RPipe/pull/21)，合入 dev `2917e3e` |
| 同构阶段链 | `8fbff78` | core 282 passed / 35 deselected，CPU integration/e2e 32 passed / 285 deselected | [PR #22](https://github.com/diaoenmao/RPipe/pull/22)，合入 dev `3fe8bb8` |
| 成对探针接入 | `818689e` | core 286 passed / 35 deselected，CPU integration/e2e 32 passed / 289 deselected | [PR #23](https://github.com/diaoenmao/RPipe/pull/23)，合入 dev `22442f9` |

上述 PR 的 Unit tests、Build package、Branch flow 必需检查均通过。前一阶段缺 Kornia 的失败及后续复验记录保留在 [record.md](record.md)。本机验证使用 `MKL_THREADING_LAYER=SEQUENTIAL`、`MPLBACKEND=Agg`，并复用隔离的 Kornia 依赖。

成对探针的隔离 CPU 检查使用伪计算和微小张量，覆盖单组合调度、真实 Flow 对象与 RNG 交接、artifact compare、保留既有证据、failed 状态、子集拒绝与八组聚合。本机检查在 `.tmp/probe-flow/`，不随 Git clone 提供。它没有执行正式训练，不能证明本版本的科学数值一致性。

先前正式实测保存在 [`18cd76c` 的 studies](https://github.com/diaoenmao/RPipe/tree/18cd76c/studies)，当时名称为 main_historical / main_reproduction。原交接快照见 [`0e3fec5` 的 handover.md](https://github.com/diaoenmao/RPipe/blob/0e3fec5/docs/development/handover.md)，其中未完成项描述当时状态。

## 五、后续工作与入口

PR #24 的文档审校已合入 dev `37b9cef`。维护者在 2026-10-10 追加了后续顺序：

1. Flow 阶段职责已通过 [PR #25](https://github.com/diaoenmao/RPipe/pull/25) 合入 dev `d73b217`：prepare 准备、execute 捕获、collect 比较、summarize 数值门、write 投影与定稿。CPU 核心 286 项、集成 36 项及三条探针隔离路径通过，远端三项必需检查通过
2. 当前持续目标覆盖 Flow 重构及实验复跑；已进入实验阶段，按现行计划复跑 main_probe 与 main_exp。原始数据 16/16 SHA-256 核对通过；新 version、环境、执行状态与失败分别见 [probe 复跑记录](../../studies/main_probe/docs/RERUN.md) 和 [exp 复跑记录](../../studies/main_exp/docs/RERUN.md)。先完成真实单组合和计划前置门，再运行完整矩阵
3. 复跑使用新的 version / Run ID，记录实际源码基线、环境、预算与失败。代码问题记 bugs.md，开发修复记 record.md，实验数值与结论记各 Study 报告

当前工作分支 feature/study-rerun。main_probe version03 在 c9c782 基线上通过；main_exp version02 已结束但终验 invalid，原始结果与报告保留。main_exp 专用归档导入修复提交为 09e495b，未执行的 version03 计划已重新冻结、round=3，新八格前置门已全部通过，完整矩阵于 05:03 启动（会话 13127，.tmp/main-exp-flow-20261011-03-launch.log，仅本机证据）；旧源码通过的前置报告会被实际 launch 检查拒绝，拒绝回归已通过。全部 64 个 Run 实际开关与 checkpoint / tracker 合同核查通过，整个连续实验源码、声明与提交保持不变。原生 launch 已于 19:48 正常退出，不再重启会话 13127。同条件串行历史速度校准已完成，准备优化的独立八格回归与 PR 检查正在执行。实验产物推送及发布按相应授权和仓库流程处理，发布仍按工作分支 → PR → dev → 发布 PR → main 流程。

| **内容** | **入口** |
| --- | --- |
| Flow 阶段与扩展 | [flow.md](../code/flow.md) |
| 目录约定 | [layout.md](../code/layout.md) |
| Study 使用 | [studies/README.md](../../studies/README.md) |
| CI 与分支 | [cicd.md](cicd.md) |
| 开发事实与历史失败 | [record.md](record.md) |

4. 维护者于 2026-10-11 要求在当前连续实验结束后，与早年版本做同条件速度校准。按 [B-022](bugs.md#b-022-gpu-容量估算与调度过于保守重型训练被串行安排) 核对旧版本提交和计时证据，拆分训练、完整评测、checkpoint 与调度耗时，分别比较串行和并发；保持数值条件可比，再据实测决定优化。当前长训期间不启动竞争 GPU 的速度实验。

19:57 已在正式长训结束后执行串行校准首轮，CNN 数值门通过；[测量报告](../../studies/main_exp/docs/speed_calibration.md) 保留原始 JSON 和对照条件。六组独立进程复测全部通过，原生会话 66360 已 exit 0，日志 `.tmp/speed-repeat-1.log` 至 `speed-repeat-5.log`（仅本机）。该批次已结束，不再轮询或重启。校准修改了 recipe 的命令入口，已完成正式 version03 的来源仍按 09e495b 与原 provenance 解释；后续正式复跑须新 version / make / preflight，不能在这套旧冻结来源上重新 launch。

后续提交 `8c17d5d` 保留上述完整报告并将归档 blob 读取合并为 Git batch。全部36个文件字节与原清单相同，拒绝不同内容及四种损坏响应，真实导入运行条件回归通过；核心295项、独立 basetemp 的集成37项通过，首次共享临时目录导致的验证错误已记入 record.md。到 dev 的 [草稿 PR #26](https://github.com/diaoenmao/RPipe/pull/26) 已创建，首轮精确提交的 Branch flow / Unit tests / Build package 已全部通过；补入回归证据后的最终提交须重新核对检查。批量导出后的新源码八格600-step回归已于20:23全部通过（参数最大差0、Loss最大差4.44e-16），原生会话10327 exit0，输出独立目录 `.tmp/main-exp-batch-prefix-20261011-3e158abd/`，完成后写 `docs/BATCH_EXPORT_PREFLIGHT.json`；没有覆盖原 version03 的 PREFLIGHT 或64个 Run。会话10327已结束，不再轮询或重启；最终 PR HEAD 的 CI 仍须核对。不能把旧矩阵当作优化后的完整矩阵。
