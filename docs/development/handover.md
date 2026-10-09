# Flow 与 Study 重构交接

更新日期：2026-10-10。当前集成基线为 `dev` 的 `22442f9`，来自 [PR #23](https://github.com/diaoenmao/RPipe/pull/23)。本轮未重跑正式 `main_exp` / `main_probe`，未发布版本。

## 一、摘要

| **范围** | **接续前** | **当前状态** |
| --- | --- | --- |
| 库扩展 | Study 自写调度和比较入口 | PR #21 合入 recipe、来源清单、freeze 与库 compare |
| 阶段链 | Study 阶段目录未接入 Runner | PR #22 合入六阶段扩展、模块隔离和阶段源码冻结 |
| main_exp | 根 compare.py 执行终验 | prepare 检查来源，Study process 做四 seed 曲线终验，根 compare.py 保留手动 partial 读取 |
| main_probe | 独立 probe.py 与当前侧 16 Run 两个入口 | PR #23 移除脚本入口，统一 CLI 管理八个成对 Run、失败结果与八组终验 |
| 本机目录 | 存在未跟踪的 main_historical 残留 | 已按维护者要求删除，正式 Study 为 main_exp / main_probe |
| 科学验收 | 已有先前版本实测 | 当前重构版本未重跑，先前数值不能替代本版本验收 |

## 二、库与 Study 的分工

库负责声明展开、调度、Run 阶段链、artifact 合同、来源冻结及通用聚合。Study 通过 recipe 注册自己的 source，通过同名阶段包补充本研究的检查、证据和终验。

```text
prepare → execute → collect → summarize → write → process
```

`flow.study_phases: true` 显式启用阶段包。每阶段先执行库，再执行 Study 的 `run(ctx)`。recipe 在 Data / Model 构造前调用，Study prepare 在构造后调用。阶段源码自动纳入 provenance。

Run process 使用 `ctx.scope == 'run'`，读取本 Run 的结果。整轮 process 在通用聚合后另调用一次 Study hook，使用 `ctx.scope == 'study'`，没有单条 Run 的 layout/control。需要让本 Run 的失败反映在 result 中的数值门，必须在 write 成功前执行。process 派生失败保留已写成功的 Run result，并向调用方报错。

## 三、正式 Study 当前状态

### （一）main_exp

[main_exp](../../studies/main_exp/README.md) 声明 32 条训练与 32 条自身 best 独立评测。recipe 使用固定历史来源 `4ccb28d`，检查 preflight，并拒绝中断训练的 checkpoint 续跑。`prepare_data.py` 负责 raw 下载及清单核对。

Study prepare 检查构造对象的历史来源。`process/curves.py` 保留曲线门限计算，Study process 在聚合后调用。根 `compare.py` 提供显式手动读取。完整性与数值门见 [PLAN.md](../../studies/main_exp/docs/PLAN.md) 和 [TARGET.md](../../studies/main_exp/docs/TARGET.md)。

### （二）main_probe

[main_probe](../../studies/main_probe/README.md) 声明八个成对 Run，每个只处理一个 data/model 组合。recipe 在库 prepare 内完成该组合的 CPU 准备，注册专用 Data / Algorithm，并恢复本 Run 的 seed。`flow.prepare_shared: false` 避免 recipe 注册前构造专用数据。

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

当前没有待合并的 Flow 接续 PR。正式实验重跑、实验产物推送及发布需符合后续授权。重跑使用各 Study 的当前计划，区分执行成功、数值门通过与完整验收。发布仍按工作分支 → PR → dev → 发布 PR → main 流程。

| **内容** | **入口** |
| --- | --- |
| Flow 阶段与扩展 | [flow.md](../code/flow.md) |
| 目录约定 | [layout.md](../code/layout.md) |
| Study 使用 | [studies/README.md](../../studies/README.md) |
| CI 与分支 | [cicd.md](cicd.md) |
| 开发事实与历史失败 | [record.md](record.md) |
