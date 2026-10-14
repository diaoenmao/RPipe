# 一、摘要

RPipe 已从旧 main 的独立训练脚本演进为可安装的研究执行库。本节按内容归纳当前 dev 与本机新增成果的累计变化；具体实施日期、失败和阶段边界保留在第二节。

## （一）改前与改后对比

### 1. 结构与公共接口

本表以旧 main `98648f3` 对比当前 dev。旧版已经有 YAML、命令参数、调度、聚合和 checkpoint；变化在于组织方式、公共接口与产物合同。

| **范围** | **改前** | **改后** | **依据** |
| --- | --- | --- | --- |
| 调用方式 | 从 `src/` 运行 `train_model.py`、`test_model.py`、`make.py` 等独立脚本，共享全局 `cfg` | 源码集中到 `src/rpipe/`，提供参考 CLI；下游复制源码模板并使用自己的项目入口，包构建仍作分发合同验证 | [代码结构](../code/code.md)、[源码接入](../../README.md#在其他项目中使用-rpipe) |
| 配置与编排 | `config.yml` 配合 `control_name` 解析；按 seed 与配置拼 tag，make 生成 Bash / wait 调度 | Study YAML 声明 axes / seeds，展开 Experiment × seed 的 Run；配置内容生成 Run ID，launch 执行固定阶段链 | [概念](../code/concept.md)、[Study 指南](../../README.md#study-使用指南) |
| 运行产物 | 配置、脚本、实验、结果和图分别落到 `output/config`、`script`、`exp/<tag>`、`result`、`vis`；checkpoint / best 为旧分件格式 | 按 Study / Run 统一落盘 config、result、tracker、log 和新 checkpoint；index / process 支持 status / logs / report | [目录约定](../code/layout.md) |

### 2. 计算与恢复修复

本表的“改前”是 dev 中对应问题修复前的行为；这些问题不能统称为旧 main 的缺陷。

| **范围** | **改前** | **改后** | **依据** |
| --- | --- | --- | --- |
| 来源与依赖（B-014 / B-015） | 未知 name / source 可能落入默认或占位路径；缺正式输入仍可能产出结果；SVHN 所需 scipy 未列入基础依赖 | Factory 严格匹配，未知/空 source 报错，省略时保留正常默认；占位必须显式 stub，缺输入失败落盘；声明并核验 scipy | 第二节 §1、[数据验收](https://github.com/diaoenmao/RPipe/blob/71143ab/studies/support_data_smoke/docs/STUDY_REPORT.md) |
| train 均值（B-016） | step 模式跨评测段累计，反例最后一段应约10却报告约5 | 每个评测周期收口并重置 train 段，同一反例报告约10；空段不覆盖最后有效均值 | 第二节 §4 |
| topk 计数（B-017） | topk=2 将3个样本的候选展平为6个索引，触发形状/计数错误 | 保留候选轴，每个样本命中任一候选只计一次；修复既有函数，未新增 CLI topk 配置 | 第二节 §5 |
| 零评测预算（B-018） | `eval_num_steps=0` 仍读取并评测第一批 | 读取数据前报错；正数限批、缺省/负数完整 test 的语义保留 | 第二节 §7 |
| CLI 重试控制（B-055 / #41） | API 有关闭入口，CLI 始终默认重试一次；非恢复型任务的第二次错误可覆盖首个最终错误 | `launch --no-retry` 显式传递既有 API；默认重试和 train→eval 屏障保留，首个失败可由单次执行保留 | 第二节 2026-10-14 §40 |
| 原生恢复（B-021） | checkpoint 缺少 RNG 与数据位置，续训样本后缀和最终权重可偏离连续运行 | 在单进程、受支持 sampler / batch metric 合同下保存并恢复完整训练状态；不匹配配置拒绝，其他路径明确提示部分恢复 | 第二节 2026-10-12 §6 |
| epoch 停止（B-024） | 累积尾 batch 可导致多跑 epoch；整轮无 optimizer 更新时可能无限循环 | epoch 按声明轮数停止，已完成恢复不重训；整轮无法更新时报错，原尾 batch 丢弃语义保留 | 第二节 2026-10-12 §7 |
| 曲线进度与回滚 | 横轴按 history 记录序号；记录频率变化不能直接解释为训练步数，恢复后的回滚分支未单独排除 | 按 optimizer_step / epoch 对齐；保留原始日志，聚合时排除回滚分支、处理重复坐标并记录逐点 n | [曲线修复记录](brainstorm.md#4-已经做的) |
| Windows checkpoint 保存（B-013） | 文件占用时保存出现 WinError5，训练中断；原占用者未查明 | 原子替换对特定 Windows 错误有限重试，持续失败保留旧档并报错；真实句柄探针与训练补测通过，原占用者仍未查明 | [保存补测](https://github.com/diaoenmao/RPipe/blob/71143ab/studies/mnist_cnn_budget_repeat/docs/STUDY_REPORT.md) |

### 3. 本机复现与交付整理

本表对比本次接续工作前后的成果；新增验收与新增 CI 配置分别记录。

| **范围** | **改前** | **改后** | **依据** |
| --- | --- | --- | --- |
| 现代 main 本机对照 | 已有旧设备受控60-step 结果，当前 cu130 设备尚缺同条件执行与存档核验 | 本机8组合、60-step 受控对照通过，16段参数最大差0，并完成独立 CPU 重载；原默认条件失败仍保留 | [本机结果](https://github.com/diaoenmao/RPipe/blob/18cd76c/studies/main_reproduction/docs/CURRENT_DEVICE_RESULT.md) |
| 历史曲线复现 | 只完成 seed0、200-step 前缀探针；接入主要在 `.tmp/`，四 seed 完整曲线未运行 | 新增正式历史 Study 与复跑入口；32条80000-step 训练、32条自身 best 评测成功，固定图像估读门通过 | [历史复现](https://github.com/diaoenmao/RPipe/blob/18cd76c/studies/main_historical/README.md) |
| 统一测试入口 | 子进程隐式继承环境与输出；当前沙箱曾有3项 Kornia 导入失败，正常本机原入口为286项通过 | 显式传递环境和 stdout/stderr，补2个真实子进程回归；当前统一 CPU 门288项通过，原失败记录保留 | [测试入口](../../tests/README.md) |
| 包与 CI | Ubuntu / Python3.10 只跑 core；包门构建后仅对 wheel 做无依赖安装和轻量 import | Linux / Windows、Python3.10 / 3.13 的 CPU 及本地 integration 全部通过；两种 OS 全新环境中的 wheel / sdist 安装和公开 CLI 全部通过 | [远端测试](https://github.com/diaoenmao/RPipe/actions/runs/37210453756)、[远端包验收](https://github.com/diaoenmao/RPipe/actions/runs/37210453815) |
| 导航与清理 | Study 入口分散，设计文档与开发记录同在 `docs/` 根目录；正式 Study 与测试用声明混放 | 设计文档在 `docs/code/`，开发记录在 `docs/development/`；通用接入与 Study 指南在根 README，`studies/README.md` 只保留研究导航；正式 Study 与测试声明分开 | 第二节（一）、[Study 使用指南](../../README.md#study-使用指南)、[研究导航](../../studies/README.md) |
| Study 阶段接续 | Study 没有与库同名的阶段回调；main_exp 终验只由根脚本调用，probe 没有两侧标准 artifact 对比入口 | 显式 `flow.study_phases` 加载 `flow/<phase>/`；prepare 有构造前 / 后入口，write 先 Study 后库定稿，其余阶段先库后 Study。专用实现放 `structure/`，两柱源码自动冻结；区分 Run 与 Study process，probe 投影观测并调用库 compare，原数值门保留 | [flow §14.0](../code/flow.md#140-study-阶段目录)、第二节 2026-10-12 |

### 4. 成对探针接入 Flow（2026-10-10）

| **范围** | **改前** | **改后** | **依据** |
| --- | --- | --- | --- |
| 探针入口 | 独立 probe.py prepare/run，普通 CLI 只跑当前侧 16 Run | 删除脚本入口；统一 CLI 展开八个成对 Run，recipe + Algorithm 接入完整阶段链 | [main_probe](../../studies/main_probe/README.md) |
| 准备与执行 | 临时工作区、脚本自建当前侧对象 | Run assets 内隔离准备；成对执行复用 Flow 的 Data / Model / System / Tracker，恢复初始化 RNG，原版先、当前版后 | [计划](../../studies/main_probe/docs/PLAN.md) |
| 状态与终验 | 脚本退出码和矩阵 JSON | 数值失败由 Flow 写 failed，保留原始证据；process 只读当前 index，齐全八组且同来源、同设备、同 Torch 才完整通过 | [process](../../studies/main_probe/flow/process/) |

CPU 核心 286 项与集成 32 项通过。隔离 CPU 验证覆盖单组合调度、真实 Flow 对象与 RNG 交接、两侧 artifact compare、保留已有证据、数值失败落盘、子集拒绝和八组终验（使用伪计算与微小张量，未训练正式模型）。本轮不重跑正式 main_probe/main_exp，尚无本版本数值验收。临时证据在 `.tmp/probe-flow/`，不随 Git clone 提供。

### 5. Flow 阶段职责重构（2026-10-10）

| **范围** | **改前** | **改后** |
| --- | --- | --- |
| 构造前准备 | Study prepare 只能检查已构造对象，探针 CPU 准备放在 recipe | 库初始化 control / runtime / System 后调用 prepare.before；recipe 注册，Factory 构造，prepare.run 检查对象 |
| 探针计算与验收 | execute 同时准备比较、投影观测、写报告；其他阶段主要检查文件 | execute 保存 OBSERVATIONS.pt，collect 计算逐段与独立 eval 对比，summarize 执行数值门，write 投影和库 compare |
| 成功定稿 | Study write 在库 succeeded result 写入后运行，写入失败仍可能保留成功 | Study write 先完成证据与附加门，库最后原子定稿并登记全部 asset；前置阶段失败覆盖旧成功状态，process 失败保留成功结果 |

CPU 核心 286 项、集成 36 项通过。隔离探针验证真实阶段链的成功、数值失败、库 compare 失败三条路径；伪计算和微小张量不代表正式数值复现。首轮集成 35 项通过、1 项失败，原因是新增旧结果 fixture 缺少 control / paths，修正后复验通过，失败记录保留。临时证据位于 `.tmp/flow-stages/` 与测试报告目录，不随 Git clone 提供。当前工作分支尚待远端检查与集成，正式八组合及长实验尚未启动。

PR #25 已合入 dev `d73b217`，绑定 head `1fef2ff` 的 Unit Tests（37998120290）、Package Check（37998120348）、Branch flow（37998120295）均通过。已进入新 version 的实验复跑准备，状态与数值分别见 [probe 记录](../../studies/main_probe/docs/RERUN.md)、[exp 记录](../../studies/main_exp/docs/RERUN.md)。

## （二）已经验证到哪一步

| **验证对象与日期** | **实测结果与范围** |
| --- | --- |
| CPU unit / integration（2026-10-04 历史批次） | 288 passed / 3 deselected；c1/c2，排除 external / gpu / slow，包含两个新测试入口回归。后续阶段使用分开的 core / integration 门，结果见下方 |
| 2026-10-10 清理阶段本地门 | `tests/run.py --core` 为 268 passed / 23 deselected；CPU 的 integration 与 e2e（c1/c2，排除 external / gpu / slow）为 20 passed / 271 deselected。本机缺 `libiomp5md.dll`，两轮都设置了 `MKL_THREADING_LAYER=SEQUENTIAL` |
| 2026-10-10 Flow 接续后本地门 | PR #23 的代码基线 `818689e`：core 286 passed / 35 deselected，CPU integration/e2e 32 passed / 289 deselected；临时隔离合同检查未执行正式训练 |
| 2026-10-10 必需 CI | PR #21、#22、#23 的 Unit tests、Build package、Branch flow 均通过，已合入 dev；最新接续基线 `22442f9` |
| 安装后的 wheel / sdist | 各9条公开 CLI 命令退出0，各4个 Toy/Stub Run、2个 Experiment 完整；配置不变，重复 launch 跳过成功任务 |
| 远端发布准备 CI | PR #12 的四个 OS / Python 测试矩阵、两个 OS 包矩阵及 Unit tests / Build package 聚合门全部成功，绑定提交 `216e2b8`；实际运行链接见上表 |
| 现代 main 计算对照 | 8组合、seed0、60-step；step30/60共16段参数最大差0，optimizer / RNG / scheduler 与完整 test 一致，独立 CPU 重载通过 |
| 历史 main 完整曲线 | 32条四 seed 连续80000-step 训练、32条自身 best 独立评测成功；8组固定图像估读门通过，最大终点差0.207502个百分点 |
| 整理后的证据保全 | 772个唯一科学证据文件哈希一致，原始失败与旧快照保留；正式相对链接、代码语法和 CI YAML 检查通过 |

现代对照使用原 Stats 的全精度 profile 与统一确定性条件；原默认条件未通过的结果仍保留。历史图缺少原始逐点指标，长曲线结论限定于预先固定的图像估读门，前缀逐位对齐不扩称为全80000步权重逐位一致。

本机包验收复用了已有科学依赖，Toy/Stub 只验证安装与产物合同。发布准备提交 `216e2b8` 的新跨平台 CI 已实际通过；后续提交的状态以其对应 Actions 为准。2026-10-04 整理阶段的固定验收快照见 [机器可读结果](https://github.com/diaoenmao/RPipe/blob/71143ab/docs/REPOSITORY_CLEANUP_RESULT.json)，发布验证与环境修复见 [v0.2.0 说明](https://github.com/diaoenmao/RPipe/blob/71143ab/docs/releases/v0.2.0.md)。2026-10-10 的目录清理见第二节（一）。

# 二、开发记录

按日期保留阶段事实，最新日期在前。较早记录中的“尚未完成”只描述当时范围；当前累计状态以上方摘要为准。

## 2026-10-14

### 40. 原生 launch 显式关闭失败重试（#41，验证通过）

来源为 [Issue #41](https://github.com/DreamSoul-AI/RPipe/issues/41) 的 CPU 合成复现与当前 dev 调用链。改前：API 支持 retry_failed=False，但 CLI 只能采用默认重试；排他产物任务的第二次 FileExistsError 可成为最终错误。改后方案：launch --no-retry 仅投影现有 API，默认保持重试一次，不改变计算、Run ID、jobs 清单或依赖屏障；原始失败日志仍保留。更大的尝试历史设计只记 brainstorm。

新增真实 CLI 四格回归：默认 / 关闭重试 × 失败 / 成功，检查实际调用次数、最终错误、原日志、jobs 字节、成功后空计划跳过；另有显式关闭重试时的失败 train→eval 屏障回归。使用测试公共合成输入，不接触正式 Study 或下游 vendor。本机此时存在 dayan 性能校准，因此未并行运行本地回归。

工作提交 `b678802d54a509532cbca35e40223c889d1c369e` 的 [Unit Tests](https://github.com/DreamSoul-AI/RPipe/actions/runs/38556030997)、[Package Check](https://github.com/DreamSoul-AI/RPipe/actions/runs/38556030922) 和 [Branch flow](https://github.com/DreamSoul-AI/RPipe/actions/runs/38556031025) 全部成功。Windows 两个 Python 环境 core 各 464 passed / 2 skipped；Ubuntu 两个环境各 463 passed / 3 skipped；四个环境 CPU integration 各 72 passed / 2 skipped，均无失败。两项模块级 skip 对应未安装的 HF 可选依赖，Linux 另有 Windows 专用入口 skip，不计作通过。两平台 wheel/sdist 均完成独立安装验收。

最终文档提交 `8269bff580eeb22d5050846d2452bbbc9b8f2b4b` 的 [Unit Tests](https://github.com/DreamSoul-AI/RPipe/actions/runs/38556840524)、[Package Check](https://github.com/DreamSoul-AI/RPipe/actions/runs/38556840501)、[Branch flow](https://github.com/DreamSoul-AI/RPipe/actions/runs/38556840590) 全部通过，已单独核对 Windows / Python 3.13 为 success，没有将 pending 当作通过。生产代码及测试与上述首轮工作提交相同。

[PR #42](https://github.com/DreamSoul-AI/RPipe/pull/42) 已合入 dev `8c15cd84f8a47d5bfeb80edfa69bc5c370998e74`，实际集成提交的 [Unit Tests](https://github.com/DreamSoul-AI/RPipe/actions/runs/38557561151) / [Package Check](https://github.com/DreamSoul-AI/RPipe/actions/runs/38557561212) 也全部通过，覆盖四个 CPU 环境与两平台安装包。核对最终工作树与集成树一致、工作区干净后，已删除该修复的本地 / 远端 codex/launch-no-retry；其他活动分支保留。main 仍为 v0.3.1 / `7a15c46`，未进入发布流程。

下游关于排他产物与自动重试的报告已核对，公开 #41 引用、固定 v0.3.1 来源和首个错误仍在日志的事实均正确；它记录的是原基线，不能改写为采用新 dev 后的验收。“尚未获得上游修复”属于当时记录，上游当前修复状态以上述确切提交与 CI 为准；未修改下游报告、vendor、锁或原 Run。

本次再次逐项核对开放 #36 / #37 / #38 / #41 及评论，均为已在 dev 集成的修复，没有新增未处理条件。三个对话的新反馈没有提供新的 RPipe 反例；已完成的下游回归不因历史超时或领域数值门失败重标为库缺陷。未自动关闭 Issue、未运行正式长实验。独立尝试历史仍为 brainstorm 候选。

### 39. 共享与三个反馈修复合入 dev

[PR #39](https://github.com/DreamSoul-AI/RPipe/pull/39)在最终工作提交`07bcf2eac9632166c9063c1412f764f1a6d516e3`的Unit tests / Build package / Branch flow全绿后，于2026-10-14 07:00（Asia/Shanghai）合入dev `614d0a6bc6dc99f82445a1566d41a0b6150c6b85`。Ubuntu / Windows × Python3.10 / 3.13实际CPU矩阵与两平台wheel/sdist安装检查全部成功；已核对dev与该工作提交的生产源码及Study实现没有差异。

远端依据：[Unit Tests](https://github.com/DreamSoul-AI/RPipe/actions/runs/38480920352)、[Package Check](https://github.com/DreamSoul-AI/RPipe/actions/runs/38480920322)、[Branch flow](https://github.com/DreamSoul-AI/RPipe/actions/runs/38480920341)。dev集成后的[Unit Tests](https://github.com/DreamSoul-AI/RPipe/actions/runs/38481509649)和[Package Check](https://github.com/DreamSoul-AI/RPipe/actions/runs/38481509815)独立运行，状态须以实际任务为准，不能仅凭上述PR通过算已完成。

本轮B-053 / B-054实现与必要回归完成，从开放bugs移除，详细反例 / 修复事实及实验限制仍保留§35—38和各Study证据。#36—#38公开反馈暂不关闭；main仍是原v0.3.1，未进入发布流程、未创建tag/Release。未来下游升级须明确固定dev提交或发布来源，不把旧冻结实验就地覆盖为新版本。

合并后核对07bcf2e为dev祖先、工作区干净，再清理本任务已合并的本地 / 远端codex/powershell-launch-boundaries；本机dev快进后新开codex/shared-inputs-followup记录交接。未删除其他任务分支，原数据、Run、缓存和失败保持；没有重新执行长训练或改变数值门。持续目标继续关注三个下游对话与新Issues，没有新反馈时不扩大为无依据的边界改造。

### 38. 固定格式 processed / Stats 也跨 Run 复用

进一步核对两份固定归档：4ccb28d的processed是NumPy pickle格式，98648f3格式不同，不能覆盖为同一目录；但同一归档与raw版本的文件可以跨Run共享，不能把全部processed一概当私有例外。已在项目shared/data/derived按归档commit、数据hash、Torch / NumPy版本及固定统计方法分目录，加载对象仍各自独立。

复用artifact共享工具新增prepared_resource：版本定义生成key，准备回调在公共存储的唯一staging中构建，完整文件与manifest写齐后整目录发布，每次复用核对文件清单和hash。失败staging保留，损坏或缺项不自动重建；没有新增训练或调度循环。main_probe的processed及固定原版Stats / native YAML共用；main_exp也使用独立4ccb28d格式缓存，避免不同Study的格式碰撞。普通Study的自定义profile仍私有，不借用不同归一化约定。

共享专项26 passed；当前core463 passed / 2 skipped（`20261013T223858Z_dd103c`），CPU integration68 passed（`20261013T224233Z_6c377b`），8条原partial-resume警告保留。更新源码的wheel/sdist各9CLI / 4Toy Run通过，依赖复用边界不变。此前38e0a3f远端三门已通过；本次提交需核对自己的最终CI。

研究数字 / 失败留在Study：probe冷 / 热MNIST和CIFAR10单格对照及最终两数据集linear单格均通过，processed / Stats路径samefile证明实际复用；main_exp两数据集linear200-step前缀通过、参数最大差0，不替代正式八格600-step前置门。首轮临时前缀脚本提前创建输出目录被入口拒绝，修正调用后新目录运行，原拒绝保留。

迁移只改字节完全一致的引用：162个旧processed路径及100个固定Stats路径改为同文件，未发现必须保留的不同编码匹配候选；原848个raw引用和278份历史result/index/process/provenance保持。详细hash、source和数字分别见[main_exp迁移](../../studies/main_exp/docs/shared_inputs_migration.json) / [前缀诊断](../../studies/main_exp/docs/shared_inputs_prefix.json)、[probe证据](../../studies/main_probe/docs/shared_inputs_validation.json)。逻辑字节合计不当作实测磁盘释放或速度收益。

### 37. Issue #36 未开启扩展的初始冻结拒绝（B-054）

在现有真实CPU回归加入flow.study_phases=false对照，发现prepare初始check_frozen拒绝仍覆写成功result，开启扩展的对照通过；红测1 failed / 10 passed（`20261013T222059Z_35423a`）。先在bugs记录，再仅将ProvenanceChangedError的成功保全覆盖prepare阶段；冻结条件 / 时机不变，其他prepare异常和execute / summarize / write失败继续写failed，process保全规则保持。没有恢复此前撤回的额外冻结门或调度改造。

定向32 passed（`20261013T222126Z_36f53c`），真实run-one拒绝非零退出、旧成功和非日志资产字节保持；普通prepare失败仍覆写为failed。当前core456 passed / 2 skipped（`20261013T222158Z_4f621f`）、CPU integration68 passed（`20261013T222209Z_ce9510`），8条历史partial-resume警告保留。首批共享提交ac909ec远端三门通过，3f7bf18及本次后续提交的CI分别核对，不互相替代。

来源变化后新version `shared-inputs-20261014-cpu05` / Run `44492c0d8f5cf190` 的MNIST/linear CPU60-step完整test单格再次通过、执行源不变；依据追加在[Study证据](../../studies/main_probe/docs/shared_inputs_validation.json)。仍无新的八格或长矩阵验收，原红测和各CPU / CUDA重复结果保留。工作仍在同一草稿PR #39，#36保持open，未合并或发布。

### 36. 共享统计路径与恢复身份复查

共享首批提交ac909ec已推至[草稿PR #39](https://github.com/DreamSoul-AI/RPipe/pull/39)，其Unit tests、Build package、Branch flow全部通过。进一步复查发现隐式统计目录被注入data.config，会令仅路径搬迁也改变native sampler的恢复身份；改用可选profile_path IO参数，不改变用户原实验配置，显式stats_path仍优先。

新增实际Factory / BatchCursor回归：两个同内容profile目录下恢复同一cursor，后续两batch逐值相同，config身份保持。定向14 passed；当前core456 passed / 2 skipped（`20261013T221521Z_c4c197`），CPU integration66 passed（`20261013T221541Z_af5771`），原8条partial-resume警告与符号链接权限skip保留。最终源码的wheel/sdist重新构建安装，各9CLI / 4Toy Run通过，科学依赖复用限制不变。

来源改变后再用新version `shared-inputs-20261014-cpu04` / Run `25ae12161dec991a` 完成MNIST/linear的CPU60-step完整test成对门，源码不变且固定归档字节校验通过，证据追加在[Study记录](../../studies/main_probe/docs/shared_inputs_validation.json)，不覆盖前三轮。首批CPU单格的整轮process也实际拒绝不完整矩阵，成功Run结果保留。最终提交仍须通过自己的远端CI，不用ac909ec的检查代替。

### 35. 跨 Study 共享输入与本机缓存迁移（B-053）

用户在新持续目标中授权同步 `codex/powershell-launch-boundaries`、处理其他三个对话反馈的 Issues，并实现跨 Study data / model 共享。已 fetch 并切换到 `6619edc`，确认分支已有 #36—#38 三个修复；没有恢复此前撤回的广泛边界改造。本轮新增范围为 [共享路径合同](../code/layout.md#跨-study-共享资产2026-10-14)。

- 改前：各 Study 返回自己的 shared/data、shared/model，probe 每个 Run 为两侧 copyfile raw。改后：默认项目级共享输入，data / model 独立显式私有例外；Factory 与 recipe 接收统一根，模型对象 / checkpoint / RNG / tracker 仍独立。统计 profile 留在 Study 并显式传入，避免跨 Study 归一化污染。
- 公共准备受进程文件锁保护；终止进程后可重新取得锁。同卷旧缓存迁移先验证全部重名文件，再建立硬链接，原路径保持可读；冲突和链接失败保留原输入。raw 的归档 / torchvision 路径通过引用适配，不再复制。自定义 builder 仍须按版本 / revision 分目录；未新增 HF / ModelScope builder 或训练调度器。
- main_exp 手动准备、preflight、容量 / 精度入口解析同一共享根；历史 processed 构造也受锁保护。探针的 processed / 原版 Stats 因归档格式和状态隔离仍私有。临时复制 Study 的首次原始导出得到空树；Git ls-tree 改为 --full-tree 后，固定历史提交的完整文件字节校验保持通过。
- 当前 core **456 passed / 2 skipped**（`20261013T220345Z_df43db`，UTC），两项跳过为本机目录符号链接权限，8 条原有 partial-resume 警告保留；CPU integration **65 passed**（`20261013T220523Z_128a81`）。新增共享专项18 passed，覆盖跨 Study 文件 / 训练隔离、私有 profile、路径配置、冲突拒绝、链接失败与被终止写者。首次双 Study 测试误用 checkpoint 文件名 / key，按实际 latest.pt / model 合同修正，原失败留存。
- wheel / sdist 在 `.tmp/` 源码副本构建、分别安装到新环境，复用已有科学依赖而非干净依赖安装；各9条 CLI、4个Toy/Stub Run通过。首次临时构建脚本误写 LICENSE 文件名，改为实际 LICENSE.md 后新目录成功，失败未删除。包仍为0.3.1开发来源，不发布版本。
- 本机原始输入迁移和单格数值依据留在 [main_exp 迁移记录](../../studies/main_exp/docs/shared_inputs_migration.json)、[main_probe 小验证](../../studies/main_probe/docs/shared_inputs_validation.json)。不是新的八格 / 长矩阵验收，不改写旧来源。

本机原始 stdout / 失败、迁移逐文件 hash 和安装验收在 `.tmp/shared-inputs-20261014/`，不随 clone 提供。#36—#38 继续保持 open，当前未回复或关闭；正式集成与远端 CI 以对应工作 PR 为准。持续目标保留，后续关注新 Issue 及三个下游对话的实际反馈，不把“目前没有新增项”当作所有后端已验收。

### 34. 工作分支提交推送交接

用户授权 push 当前代码，以便在另一台机器继续。提交范围为收缩后保留的三个公共源码修复、四个对应测试文件、设计 / 历史开发 / 交接文档，以及跨 Study 共享下载的 brainstorm 要求；不恢复撤回的探索代码，不加入 `.tmp/`、环境、缓存、Run / 数据 / checkpoint。本次只推工作分支 `codex/powershell-launch-boundaries`，不合并 dev / main 或发布，Issue #36—#38 继续开放。

源码与此前收缩后的定向75 passed、core440 passed / 2 skipped及四Run小流程验收保持同一实现，后续仅文档更新，不重复运行无变化的测试。提交前核对差异、忽略项及正式Study保全。实际交接提交与推送完成状态以远端分支为准，不在文档中自引用尚未生成的 commit；异机执行前须重新检查环境并做核心小验证。

### 33. Issue 状态按用户最新要求恢复开放

用户先授权清掉 Issue #36 / #37 / #38，三条在 GitHub 被关闭；拟附的本机验证说明评论被自动审批拒绝，没有发布。随后用户认为反馈来自其他项目，并明确要求暂不关闭、后续再修；已将三条恢复 open / reopened，并重新读取确认全部 open。原正文及讨论保留，没有追加公开评论。

三条提交账号均为 diaoenmao；仅账号不能确定具体创建会话，当前没有独立定位全部创建来源，不能将其描述为三个外部用户反馈。现有三个本机修复及证据保留、未提交 / 推送 / 发布；本轮没有新源码修改、测试或研究任务。后续按用户新要求再处理，不因本机通过自动关单。

### 32. 按用户要求收缩到本机适配与三个下游 Issue

用户明确选择仅保留本机迁移 / MKL 和已收到的 Issue 修复。先将当时29个修改 / 新增文件、完整 tracked diff 与清单备份至 `.tmp/scope-reduction-20261014/`；七个范围外新测试移入该目录的 withdrawn-tests。没有删除历史反例、隔离环境、Run、数据或 checkpoint。

当前公共源码仅保留三处：Runner 对阶段开始前 ProvenanceChangedError 保留旧成功（Issue #36）；status 缺轴 / result 模式时读生成配置并保留必要诊断（Issue #37）；原子文本初始写入纳入既有 finally 清理（Issue #38）。原阶段启用 / freeze 时机保持基线，撤回额外的 disabled-hook 冻结门拓展；未修改全局 config 解码合同。保留四个对应测试文件，撤回其余主动探索的 CLI 参数、脚本生成、自动 remake、计划资源 / 行 / 分组校验及 Study 解析改动。README 和 brainstorm 回到 HEAD；设计文档只同步这三项行为。

本机 `.tmp/runtime/activate.ps1` 的 SEQUENTIAL 设置及 UTF-8 BOM 保留，源码 pth / 新项目路径保留；迁移后 main_probe 八项调度参数保全，main_exp 未创建新脚本或启动正式任务。撤回代码的历史记录保留为已发生事实，§11—31 的“已修复”描述对应当时源码，不能当当前交付状态；本节为当前范围依据。范围外已确认反例没有因此被证明不存在，只是不纳入本次实施。

收缩后 Issue / readout / atomic 定向75 passed（`20261013T212721Z_885522`，UTC），core440 passed / 2 skipped（`20261013T212754Z_0ffd5d`），均exit0；原8条partial-resume警告保留。新 Windows PowerShell -NoProfile 使用最终本机激活步骤，源码导入和 help 正常；全新 version `scope-reduced-20261014-v1` 的 CPU Toy/Stub两seed、train/eval四Run、仅两训练step，make→launch→process→status/logs/report及重复launch跳过保全全部通过，见 `smoke/SMOKE_RESULT.json`。没有将它当科学结果验收。

验证阶段627个受保护文件字节保持（包括427个正式Study文件），当前源码差异仅3个文件；diff --check通过。临时备份 / 证据不随clone提供。未 commit / push / 发布，未回复或关闭Issue；未验证远端CI、包安装、其他宿主、GPU、正式研究或完整CPU integration门。后续不再依据旧的宽泛探索目标自动扩展，按本次范围收口。

### 31. B-052 保存分组与主清单一致性修复及范围收口

补回归确认分组遗漏、增项、重复及 gpu / device / mode / config 漂移：默认成功过滤与 include-done 各覆盖一遍，红测 14 failed / 2 passed（`20261013T212214Z_2ca8af`，UTC），两个合法重排 / 成功过滤控制通过。先明确[保存分组合同](../code/structure.md#10-make)，在共享读取入口按 Run ID 核对完整任务 mapping 与一一对应，检查先于成功过滤，坏清单返回 None；合法组顺序、空 / null / 全空组自动分组保持。不新增调度器、资源参数或持久化格式。

首轮调度目录 / launch 验证 181 passed / 1 failed（`20261013T212235Z_b97d1b`）：失败为 Bash 临时 Python 执行时报 Device or resource busy、exit126，原证据保留。单独新临时目录复查该用例 1 passed（`20261013T212322Z_d1714c`），未修改断言 / 自动重试，实际占用者未确认；不能将首轮失败删除或把这次复查当启动稳定性证明。

补原生冻结拒绝后，B-052 最终定向 19 passed（`20261013T212349Z_433477`），包含全部一致性反例 / 合法顺序控制、来源漂移拒绝前字节保全，以及来源未变的坏分组重建；没有实际 GPU 使用。427 个正式 Study 文件哈希一致，git diff --check 通过。临时证据 `.tmp/b052-plan-consistency/` 不随 clone 提供，旧审查反例保留。

用户指出累计改动过多。本轮先收口，不再追加边界探索；多个 B 编号是相邻输入 / 读取 / 保存边界的人工反例，并不表示正式实验出现同等数量的算法故障。当前仍有未经提交的累计改动，不把此前授权解读为可无限扩大重构。后续范围需以用户最新要求为准；不 commit / push / 发布，不改正式研究或数据。

### 30. 累计审查：B-052 分组与主任务清单的一致性

按 ponytail-review 核对未提交生产 diff 与连接调用链，范围为单写者 Study / 多 Run wait 的现有使用方式，不以本机行为保证多用户并发或性能。检查 CLI 输入 / freeze 自动重建、Runner 成功保全、原子文本清理、config / Study 解码、status 回退与脚本路径 / 资源传递；已有测试与整链证据按实际源码版本读取，没有再次运行测试门或增加通过数量。

发现 B-052：合法形状的 jobs / batches 可以不一致，加载器验证 jobs 而 wait_groups 选择 batches。新只读探针三组：原始两 Run 清单 / 分组一致，待启动环境 GPU 0；分组删掉 r1 后仍加载成功，planned_ids 两项而 dispatch_ids 只有 r0；batch 副本 GPU 改 7 后仍加载成功，主清单 GPU 0 而待启动环境变 7。输入文件哈希不变，未调用 Popen 或训练，也未访问 GPU 设备。源码 plan_files 仅包含 index / config，已有 freeze 无法拦截单独修改 jobs.json 的分组。

建议只在共享读取边界核对主任务 / 分组任务的一一对应与执行字段一致，不一致返回不可用，复用既有冻结门后的重建。保留合法分组顺序和空分组兼容；不引入新调度器、资源开关或迁移格式。在修复及针对性拒绝 / 合法控制验收前，本轮审查结论为需先修 B-052，而非可以交付。

证据 `.tmp/review-plan-consistency/` 不随 clone 提供。仅记录确认缺陷、开发事实与交接，生产源码没有改变；原失败和当前用户改动保留，427 个正式 Study 文件哈希仍一致。完整门 / smoke 属 B-050 基线，B-051 仅有其定向结果；未核验当前整套远端 CI、包安装、其他宿主、GPU / 正式实验、并发写者或吞吐。仍不 commit / push / 发布。

### 29. B-051 Study 顶层非 mapping 的拒绝

复查与配置解码相邻的 Study 声明读取边界。load_study_yaml 先用 `safe_load(...) or {}`，使 [] / 0 / false / 空字符串绕过既有 mapping 检查；四项 loader 反例失败而三项空文件 / null / mapping 默认控制通过（`20261013T211658Z_0defd1`，UTC）。原生 make / run --skip-launch 的八项反例也失败（`20261013T211711Z_e957fc`）：均 exit 0 并实际生成默认单 Run，不是纯解析问题。红测只生成临时计划，没有 launch 或训练，原输出保留。

先明确[Study 声明合同](../code/structure.md#10-make)，仅 safe_load 返回 None 时用空 mapping，其他非 mapping 复用既有 ValueError。expand_study 先解析声明再创建布局，避免 run --skip-launch 在拒绝时留下 docs / shared / runs 空目录；正常映射的布局 / recipe / 展开顺序保持。影响所有复用 loader 的 make、Runner、prepare、process 与 data profile；没有改变轴、seed、Run ID、算法或容错成默认研究。

定向 make / phase validation / Study / 扩展 / run_study 90 passed（`20261013T211753Z_ed0a08`），含三项默认读取兼容及八项原生拒绝：非零退出、mapping 诊断，无标准输出，文件字节与目录集合保持。既有合法 Study 小 CPU 回归通过，本轮新增反例没有训练。427 个正式 Study 文件哈希一致，git diff --check 通过。

下游 Issue 实查仍为 #36—#38，无新增项，未回复 / 关闭。证据 `.tmp/b051-study-mapping/` 与对应 `.tmp/test-results/<run_id>/` 不随 clone 提供。未重复完整门；上节累计门和四 Run smoke 对应 B-050 源码，不能替代本次新解析 / 布局顺序的完整验收。未 commit / push / 发布，未运行 GPU、正式实验、真实数据或长训练。

### 28. B-050 本机 PowerShell 5.1 激活编码与当前整链验证

本轮收口实际使用链，未增加公共库逻辑。新建 `.tmp/dev-current-smoke/`，复用原迁移验证脚本但使用全新 `dev-boundaries-20261014-v1` / Study；第一次 Windows PowerShell 5.1 -NoProfile 启动在 import 阶段失败，尚未 make 或训练。独立诊断显示 MKL 变量为空、NumPy / RPipe 原生退出 -1066598274（0xC06D007E），Torch 单独导入成功。

B-050 根因为本机 `.tmp/runtime/activate.ps1` 中文注释以 UTF-8 无 BOM 保存；5.1 按 ANSI 解码，Parser token 的 Comment 吞入下一行 MKL_THREADING_LAYER 赋值。直接赋值和单独 venv 激活对照均保留 SEQUENTIAL，排除 venv deactivate 清除该变量。原 wrapper 字节和解析 / 导入输出已备份；诊断期间一次嵌套 -Command 引用出现解析错误，未执行环境检查，后续统一使用 -File。

最小修复仅为本机 wrapper 加 UTF-8 BOM，并更新本机 runtime README；内容、依赖、全局 Conda / 环境变量、公共库与算法均未因此改变。Windows PowerShell 5.1 / PowerShell 7 各两个独立 -NoProfile 子进程，先删除继承的 MKL 变量再点调用 wrapper，断言 SEQUENTIAL 并导入当前源码，全都 exit 0，见 ACTIVATION_RESULT.json。启动仍为进入仓库后 `. .\.tmp\runtime\activate.ps1`，后续 CLI 子进程继承；关闭终端结束进程级设置。

新 PowerShell 下 source import / CLI help、make→launch→process→status/logs/report 全部 exit 0。Toy/Stub、CPU、linear、两 seed、train/eval 四 Run 成功，训练仅两 step，两个 Experiment 聚合 n=n_planned=2；重复 launch 输出 0 jobs / nothing to launch，result / config / log / tracker 原字节保持。源码来自本仓库 src/rpipe，Python 为本机源码 venv；本轮没有安装 RPipe 分发包。全部 smoke 命令明确 shared，未开可见子控制台。

当前源码完整 core 476 passed / 2 skipped（`20261013T211343Z_94edda`，UTC），CPU 流程 136 passed（`20261013T211401Z_0a4acf`），test_id 去重 612 passed / 2 skipped，均 exit 0。8 条已有 partial-resume 警告保留，不放宽合同。基线 HEAD `a424be5dedac46d8f459ff8748f18387ebb05169` 加本机未提交改动，源码逐文件哈希留 SOURCE_BASELINE.json。

实查默认 cwd / Git 根目录与 Codex rpipe 项目登记均为 D:\GitHub\DreamSoul\RPipe，旧仓库路径不存在。活动源码 / 声明 / 调度 / 新环境没有旧根路径命中；main_probe 八 jobs 的非路径字段与迁移前备份一致，现存 PowerShell 调度仅检查语法，没有启动正式研究。验证阶段627个受保护文件字节保持（包括427个正式 Study 文件）；历史路径证据、原 Run / checkpoint 与八个隔离环境保留。

本机证据在 `.tmp/dev-current-smoke/`，不会随 clone 提供；原失败与 wrapper 备份不删除。SEQUENTIAL 仍是本机缺失 Conda OpenMP DLL 的范围内绕过，不是全局运行库修复或科学 / 性能验收。未验证其他宿主、远端 CI、包安装、GPU、真实数据、长实验或并发写者；未 commit / push / 发布，仍同一工作分支。

### 27. B-049 保存计划行与分组的结构拒绝

继续核对计划加载的 trust boundary。原先只检查 jobs 为 list、batches 的第一组为 list，非 mapping job 被 _keep 静默删除，非法 batches 容器 / 首组被忽略并退回自动分组，合法首组之后出现整数组则 TypeError。五种反例均失败（`20261013T211004Z_e7b8d8`，UTC），分别为顶层任务行、分组容器、首组、后续组和组内任务行损坏；本轮红测仅只读，无训练。

先明确[保存计划结构合同](../code/structure.md#10-make)：任务行 mapping，非 null batches 为 list，每组 list 且组内任务 mapping；不符合返回 None，沿已有 CLI 冻结门处理，而非丢任务或放弃保存分组。load_launch_plan 在资源 / config 查询之前完成形状检查，保留省略 / null / 空 batches 的自动分组兼容。不新增持久化字段或另写调度器，仍由现有 launch_jobs 处理合法任务的依赖、重复 Run 与阶段屏障。

定向 schedule / make / launch freeze / selectors 105 passed（`20261013T211030Z_950f66`），包括来源漂移且后续分组损坏时 exit 2、全文件字节保全，空 / null / 全空组兼容与原任务集合保持。只做既有小 CPU Toy 回归，新反例无真实执行；427 个正式 Study 文件哈希一致，git diff --check 通过。

本机证据 `.tmp/b049-plan-rows/` 及对应 `.tmp/test-results/<run_id>/` 不随 clone 提供。未重复完整门或远端 CI，未验证 GPU、正式研究、并发写者；没有将形状检查宣称为全部计划内容一致性验收。未 commit / push / 发布，仍同一工作分支。

### 26. B-048 console 选择在计划处理前校验

检查 CLI 参数到调度的拒绝边界，确认无效 console 原来在 launch_jobs 内才抛 ValueError，fresh / 缺计划 / 显式 remake 已先完成 make 并写入配置、脚本和来源。真实红测 3 failed / 1 passed（`20261013T210753Z_859cb5`，UTC），三项 console 反例均在生成后拒绝；筛选同时选中一项已有合法模式控制且通过，没有训练或子进程调度。

先补充[launch 输入合同](../code/flow.md#13-climake--launch-与-wait)，在既有入口 try 中调用调度器 resolve_console；无效选择沿同一 ValueError→exit 2 路径，在读清单 / make / process 之前拒绝。把解析后的值传回 launch_jobs，保留它对直接 API 调用的独立校验；没有另写别名规则、修改默认平台选择或新增参数。

定向 59 passed（`20261013T210832Z_c762ec`），覆盖 modes / console 原生错误拒绝与文件保全、fresh / 缺清单 / remake、空任务计划、大小写与 MIX 别名、冻结重建和 PowerShell 计划参数。测试用 MIX / shared 及假执行验证选择；未由本轮测试打开可见控制台。427 个正式 Study 文件哈希一致，git diff --check 通过。

本机证据为 `.tmp/b048-console-input/` 和对应 `.tmp/test-results/<run_id>/`，不随 clone 提供。未再次执行完整测试门、远端 CI、GPU、正式研究或负载验收；前述完整门属于此前源码。未 commit / push / 发布，工作分支及用户既有改动保留。

### 25. B-047 保存计划资源字段的读取容错

实查下游开放 Issue 仍为 #36—#38，无新增反馈，未回复 / 关闭。核对 load_launch_plan 的 None→重建合同，发现 JSON 结构可读但 init_gpu / num_gpus / round 无法转为整数时，异常逃出读取路径，CLI 未进入已有来源冻结门。13 项真实红测失败（`20261013T210534Z_70bb71`，UTC），覆盖非数字字符串、list、mapping、Infinity 及来源漂移下的 CLI 行为；没有训练。

先明确[计划读取合同](../code/structure.md#10-make)，只把保存资源元数据的整数转换纳入 TypeError / ValueError / OverflowError 容错，返回 None；不捕获调用方参数错误，不新增资源默认规则、参数或清单格式。CLI 继续沿既有冻结门后重建，漂移时 exit 2 并保全文件，未漂移时可以恢复清单。仍不把本次读取改动当并发写者协议。

调度目录及 launch freeze / mode 定向 139 passed（`20261013T210551Z_31a819`）；补充旧默认 / 整数字符串兼容、来源未变的损坏清单重建后，schedule / freeze 控制批次 60 passed（`20261013T210640Z_317a24`）。两批均 exit 0；只有既有小 CPU Toy 回归，新增反例只读或假执行。未再次运行完整门，上节 581 passed / 2 skipped 对应 B-046 源码，不冒充本轮完整验收。

证据在 `.tmp/b047-plan-metadata/`、对应 `.tmp/test-results/<run_id>/`，不随 clone 提供。427 个正式 Study 文件哈希仍一致；同一工作分支，未 commit / push / 发布、未改全局环境或研究算法，未验证 GPU、远端 CI、正式实验或负载性能。

### 24. B-046 显式空 mode 的只读诊断

沿上轮审查的反例补回归：初次临时父目录缺失，4 项 fixture setup 失败（`20261013T210153Z_55f057`，UTC），未执行行为断言；补齐父目录后真实红测 1 failed / 3 passed（`20261013T210200Z_63dbb4`），确认显式空字符串误报 train，而省略、空 algorithm mapping、显式 None 仍正常默认 train。原失败证据均保留。

先补充[status 合同](../code/flow.md#13-climake--launch-与-wait)，仅在 config 模式回退中去掉空字符串默认，并以 ValueError 进入已有 unknown 诊断；不改变模式轴 / result 优先级、Run 状态、过滤器或算法。定向 readout / config IO / 原生 make→status 共 89 passed（`20261013T210212Z_fcaaa0`），含未知模式不计入 train、输入字节不变与默认兼容。

当前源码重新执行 core：454 passed / 2 skipped（`20261013T210227Z_a15a02`），CPU integration / e2e 门：127 passed（`20261013T210243Z_2fcf23`）；按 test_id 去重共 581 passed / 2 skipped，均 exit 0。core 的 8 条 partial-resume 警告保留，未改变恢复合同。流程门含既有离线小 CPU 计算，不代表正式实验、科学精度、GPU、外网数据、远端 CI 或其他宿主验收。

本机证据位于 `.tmp/b046-status-empty/` 及上述 `.tmp/test-results/<run_id>/`，不随 clone 提供。保全核验确认原快照的 427 个正式 Study 文件哈希一致，当前源码哈希写入 `PRESERVATION.json`。仍在同一工作分支，未 commit / push / 发布，不更新下游 Issue 状态。

### 23. 累计改动审查与 B-046 显式空mode边界

按ponytail-review检查当前未提交生产diff与连接代码：CLI调用链、Runner失败边界、调度依赖 / 重试、来源冻结、配置解码及status格式化；对照既有回归和README的本机单写者、多Run wait调度使用方式，不把它当多用户 / 并发写者负载保证。本次为审查，生产代码没有修改，也未重复运行已有测试门。

发现B-046：新status配置回退把显式空algorithm.mode当train，而共享执行合同拒绝该值。单Run只读探针确认status_mode=train、无mode诊断；同一配置的validate_control报invalid algorithm.mode空字符串，输入字节保持，无训练。建议最小修复只保留省略 / None的train默认，空字符串使用现有unknown诊断；在修复及针对性验收前不判当前改动可直接交付。

实际探针和JSON在 `.tmp/dev-change-review/`，不随clone提供。仅记录到bugs和交接，不擅自改算法、来源或用户Run。尚未验证当前完整门、远端CI、安装包、其他宿主、GPU、并发写者或性能，本次审查不产生新的通过数量。所有现有源码改动、原证据与不commit / push / 发布边界保持。

### 22. B-045 公共阶段选择在展开和布局之前校验

复查现有下游Issue仍为#36—#38，没有新增反馈或对外改状态；继续核对run / run-one与公共API输入边界。run_study原先在expand之后才通过后续Runner构造校验phases，skip-launch还会忽略错误选择；launch_one先创建布局 / 读配置，非法阶段会先留下空Run目录并以缺config错误拒绝。首轮10 failed / 4 passed（`20261013T205120Z_e212de`，UTC），覆盖unknown、乱序、重复、别名规范化后重复，fresh / 既有Study及原生skip-launch；没有训练。

先明确[公共阶段合同](../code/flow.md#3-runner-与失败)，再在run_study / launch_runs对显式选择复用FlowRunner构造校验，在launch_one将原Runner构造移到布局 / config之前并复用对象。没有另写阶段规则、扩展参数或改变Runner执行；空Run列表也校验显式非法阶段。默认全链、合法子集、index / persist的write别名与skip-launch支持保持，非法输入仍抛ValueError，原生错误调用返回非零而不生成新产物。

定向63 passed / 1 deselected，8.83 s（`20261013T205243Z_e40645`），覆盖全部15项新增边界及既有Study / 扩展 / Runner回归。原生无效skip-launch不输出路径、声明字节不变；API拒绝时目录集合和文件字节保全。既有回归仍包含其原有小CPU Toy计算，新输入反例本身只用skip-launch / 输入拒绝，无真实数据或GPU。

证据在 `.tmp/b045-phase-validation/` 与上述test-results，不随clone提供。427个正式Study文件hash保持，原Run、checkpoint、数据与历史失败未改；仍未commit / push / PR、未回复或关闭Issue。上一项完整core / CPU流程对应此次入口顺序修改之前，不记为当前完整重跑；未验证整轮研究、GPU、跨宿主、并发文件变化、包安装或远端CI。

### 21. Issue #37 核验：B-043 模式读取与 B-044 配置结构拒绝

核验[Issue #37](https://github.com/DreamSoul-AI/RPipe/issues/37)：mode固定在base / fixed、不在axes、且无result时，status原来直接默认train，遗漏eval过滤；相邻inference同样误分类。首轮9 failed / 7 passed（`20261013T203520Z_a30162`，UTC），包含损坏模式配置仍猜train及非对象result未进入既有读取容错的反例。先明确[只读status合同](../code/flow.md#13-climake--launch-与-wait)，保留轴值→已记录结果的优先级，仅缺模式信息时用现有load_config读生成配置；不能读取时显示unknown及异常类型 / 路径，不输出配置正文或把pending改为failed。原failed错误保留，已知轴 / 结果模式仍兼容没有旧config的历史记录；没有为显示修复改axes、配置、Run ID或新结果字段。

扩展损坏配置时另确认B-044：decode_config用truthiness把空list、0、false、空字符串当 `{}`，使结构错误静默变成默认配置。status补查3 failed / 5 passed（`20261013T203836Z_ebd7ee`），直接load_config为4 failed / 3 passed / 22 deselected（`20261013T203946Z_e28101`）。共享解码改成只将None视为空配置，拒绝其他非mapping；空文件、YAML null和合法空mapping默认保留。此项为上游相邻核验，不声称下游已报告该解码路径；没有包装新配置协议。

早期status / IO定向72 passed与82 passed属于后续增加完整边界前的批次；最终定向86 passed，7.92 s（`20261013T204127Z_5e639b`），包括base与Study fixed两条真实原生make→未过滤 / eval / train status：index没有mode轴，生成config为eval、没有result，表正确显示pending eval，全部Study文件字节不变。新增只读诊断、真实失败信息保留、已成功状态不被诊断改变和损坏result回退均通过。本次没有launch、run-one、训练、GPU或真实数据下载。

当前完整core451 passed / 2 skipped / 116 deselected，12.69 s，8条既有checkpoint部分恢复warning（`20261013T204308Z_ed78d1`）；标准CPU流程112 passed / 457 deselected，111.41 s（`20261013T204428Z_a1ea7e`）。两项skip仍为Windows符号链接权限，不计通过。按test_id合并563 passed / 2 skipped，不叠加更早或定向批次，也不以通过数量证明所有研究条件稳定。

证据在 `.tmp/b043-issue37/` 及上述test-results，不随clone提供；原生make/status输出在独立basetemp，源与测试hash核对后归并。427个正式Study文件hash不变，未改旧Run、checkpoint、数据、失败或来源；仍未commit / push / PR、未向Issue评论或改状态。未验证并发文件修改、真实GPU、跨宿主、长实验、科学数值、包安装或远端CI；unknown诊断是读取状态，不是执行失败或新科学分类。

### 20. Issue #38 核验：B-042 原子文本部分写入清理

复查实际下游反馈，新增[Issue #37](https://github.com/DreamSoul-AI/RPipe/issues/37)的pending固定eval误分类和[Issue #38](https://github.com/DreamSoul-AI/RPipe/issues/38)的文本部分写入清理遗漏。#37已读并进入[具体待核验队列](brainstorm.md#四下游具体反馈待核验)，当前不宣称上游已复现、不占B编号；先完成范围较小的#38。

旧atomic_write_text将tmp.write_text放在清理try/finally之前，部分内容写出后再抛OSError会留下本次临时文件，目标和原异常都保留；未发现成功目标损坏。新增目标存在 / 不存在两项故障注入及正常替换对照，同时预放一个peer临时文件检查不误删。首次2 failed / 7 passed（`20261013T202248Z_547420`，UTC）；peer只是预置文件，不是并发写者实测。

先明确[原子文本合同](../code/structure.md#9-structureartifact)，再将初始write_text移入原try/finally；复用既有unlink清理和Windows有界replace重试，无新文件格式、重试策略或清理扫描。首次定向命令误用了不存在的test_activity / config / result测试路径，exit4、no tests ran（`20261013T202412Z_b24338`）；保留原日志后实查目录，改为整份artifact镜像，78 passed / 2 skipped，2.06 s（`20261013T202504Z_fa5784`）。两项skip为Windows符号链接权限，不计通过。

原计划的完整CPU核验在新反馈前结束为424 passed / 2 skipped；它是B-042修改之前的证据。修改后重新完成当前完整core：427 passed / 2 skipped / 114 deselected，11.98 s，8条原有checkpoint部分恢复warning（`20261013T202532Z_65c398`）；当前标准CPU流程110 passed / 433 deselected，111.71 s（`20261013T202658Z_224161`）。两批按test_id合并为537 passed / 2 skipped，所有当前生产文件hash一致；不重复相加artifact定向或更早批次，不把全量已选通过当所有支持范围已验证。

原始证据在 `.tmp/b042-issue38/`、`.tmp/dev-current-verification/` 和上述test-results，不随clone提供。427个正式Study文件hash保持，原Run / checkpoint / 数据与早期失败未修改；仍未commit / push / PR、未对Issue评论或改状态。未验证真实磁盘耗尽、断电、无法unlink时零残留、并发写者、GPU、跨宿主、包安装或远端CI。清理仍为尽力且只针对本次临时文件，原异常优先。

### 19. B-041 脚本计数在写入前校验

直接write_launch_scripts先创建scripts目录和写jobs，再调用render_bash校验有效round_size / split_round；零值会留下新目录 / jobs或覆盖旧计划。四项直接API反例全部失败（`20261013T201248Z_fd4ac2`，UTC）。原生CLI也允许非正split-round：make可能先展开 / 改写产物再报错，复用计划的launch则忽略它。原生首轮3 failed / 1 passed / 8 deselected（`20261013T201418Z_a986cf`）；两个复用计划反例意外完成两组临时单seed train/eval，CPU Toy/Stub、train预算2 step，均已退出。原始结果与日志保全，未执行正式Study；后续该测试先写占位成功结果，避免再次实际调度，不能把占位当真实训练证据。

先明确[计数合同](../code/structure.md#10-make)，再把已有render_bash调用移到目录与jobs写入之前，不另写重复校验或跨文件事务。CLI添加正整数split-round解析及帮助说明，在make / launch参数解析时拒绝零、负数及非整数，防止复用清单忽略非法值；round的0 / auto仍走原自动装箱规则。有效生成、分组、预算、训练算法与持久化格式不变。

定向82 passed，20.74 s（`20261013T201652Z_5c5ec7`），覆盖新增4项直接API拒绝和4项原生CLI split-round边界，以及原mode、make / schedule、Bash / PowerShell回归。非法API调用目录清单和文件字节均不变；原生CLI返回2、stdout空、stderr指向split-round，全部Study文件保全。正split-round=1与默认round=auto的make对照正常。随后只补帮助字符串，不改变执行逻辑；该文本修改另做CLI帮助核对，不把更早源码hash记为完全相同。

证据在 `.tmp/b041-script-validation/` 和上述test-results，不随clone提供；原生输出、原意外临时Run和占位回归分属独立basetemp。427个正式Study文件hash不变，未操作用户原Run、checkpoint、数据或旧迁移备份。仍未commit / push / PR；未完成当前整套core / CPU流程重跑、跨宿主、磁盘故障事务、真实GPU、长实验或远端CI。

### 18. B-040 launch mode 校验先于 make

检查CLI输入拒绝的副作用边界。原 `_execute_launch` 在自动 / 显式make和来源签署之后才调用 `_normalize_modes`；无效 `--mode train,invalid` 虽exit2，首次启动却已生成产物，缺清单或 --remake也会改写来源与计划。首次3 failed / 3 passed（`20261013T200458Z_044836`，UTC），三个失败场景对应首次启动、缺清单、显式重建，三个对照验证合法选择；仅临时Toy/Stub声明与假执行器，不启动训练。

先补[CLI合同](../code/flow.md#13-climake--launch-与-wait)，再将已有规范化 / ValueError处理块移到launch入口，在任何清单读取、make或共享输入准备之前完成；无新参数、helper或选项。逗号、重复参数、大小写、去重、默认整轮与合法过滤语义保留，status调用同一规范化函数的行为不变。

初次定向68 passed，11.53 s（`20261013T200641Z_e3286e`）；增加两个真实CLI子进程验收后最终70 passed，11.65 s（`20261013T200742Z_8da7c3`），覆盖新增八项选择器边界、既有launch冻结门与make / schedule回归。原生首次启动和 --remake无效输入均exit2、stderr说明mode错误、stdout无make输出，Study文件逐字节不变；合法选择仍正常进入假执行并不改jobs清单。

原始证据在 `.tmp/b040-launch-modes/` 及上述test-results，原生stdout/stderr在独立basetemp，均不随clone提供。427个正式Study文件hash保持；没有改旧Run、配置、checkpoint、数据或来源。仍未commit / push / PR；上一项98项完整CPU流程属于本次校验顺序调整之前，不冒称本次完整重跑。未验证其他CLI参数错误、副作用并发、真实GPU、异机、长实验或远端CI。

### 17. B-039 底层调度器在旧结果失效前核对 freeze

继续检查Bash静态清单与直接调度API的拒绝边界。`launch_jobs` 原先在spawn前失效旧成功train与sibling eval，直到子进程Runner才核对来源；即使没有开始新训练，旧结果已被改写。三种假进程反例覆盖修改Study源码、修改config字节、缺provenance；真实CPU Toy/Stub单seed、train/eval、2-step完成后，只给Study声明加注释再执行保存Bash清单，result与派生产物变化。原成功结果另存临时备份，首次4 failed / 2 passed（`20261013T195423Z_46b123`，UTC）。此为Issue #36的上游相邻路径核验，不把它写成下游已经报告或验证的范围。

先补[共同调度合同](../code/structure.md#10-make)，再在Run ID / 重复校验后、任何结果失效与spawn前，复用artifact的check_frozen和现有Study声明读取；不导入Flow或另写来源比较。拒绝抛出原异常，Study原文件全部不变。两个兼容对照仍允许未冻结 / 来源未变的显式清单执行；没有把底层调度器改成CLI的成功Run自动过滤，也未修改分组、train/eval屏障或重试。

定向97 passed，33.69 s（`20261013T195757Z_e320fa`），覆盖调度、make、Bash / PowerShell及两层既有冻结保全。当前标准CPU integration/e2e门98 passed / 426 deselected，106.38 s（`20261013T200001Z_4410bb`）；全部六项新增回归通过，无skip。真实Bash拒绝返回非零且stderr保留ProvenanceChangedError，整个临时Study的配置、来源、result、checkpoint、tracker、日志与派生产物逐字节保全。真实训练预算显式progress_unit=step、num_steps=2，无GPU或数据下载，不作科学数值判断。

证据在 `.tmp/b039-scheduler-freeze/` 与上述test-results，不随clone提供；原生拒绝输出和原成功JSON在独立basetemp保留。427个正式Study文件hash继续不变，未操作原Run、数据、checkpoint或迁移备份。仍未commit / push / PR；此前core对应更早源码，不作为当前完整重跑。没有验证源文件在调度中变化、并发写者、GPU、跨宿主、性能、长实验或远端CI，前置核对不等于原子事务保证。

### 16. B-037 / B-038 Bash 路径与目录切换失败边界

继续同一工作分支核对Bash侧。B-037：仓库与解释器路径仅被双引号包裹，美元变量、反引号仍被shell展开，含中文 / 空格 / 单引号的组合路径会失效。B-038：cd失败后仍在调用者目录继续执行Python；假库启动器最终exit0，覆盖了原目录错误。真实Git Bash和标准库生成的隔离Python环境覆盖普通路径、仓库特殊路径、解释器特殊路径及两者特殊路径，初次4 failed / 1 passed（`20261013T194647Z_2da024`，UTC）。测试只用独立假库，无真实算法；变量固定为无敏感内容的测试值，反引号中的命令仅为true。

先明确[脚本合同](../code/structure.md#10-make)，再复用标准库shlex.quote，对两个路径做Bash字面量转义，cd失败立即exit1；Python heredoc、wait分组、train→eval屏障、重试、结果判定和最终process均不改。新增[真实Bash脚本回归](../../tests/rpipe/structure/make/test_bash.py)，Windows明确选择Git Bash而不误用WSL入口，未安装时skip；临时环境不安装pip或RPipe。

定向make / schedule / PowerShell / Bash门66 passed，17.11 s（`20261013T194803Z_5cf22d`）。另用当前源码和现有源码venv，在全新含特殊字符路径的临时Study执行make→生成Bash启动→process / status / logs / report；version `bash-boundary-20261014-v1`，两seed × train/eval四个Run成功、两个Experiment完整，train仅2 step。随后直接CLI launch正确跳过成功Run，结果字节及单份Flow start保持；没有重复执行Bash静态清单，不宣称Bash与CLI的重复启动语义相同。

证据在 `.tmp/b037-bash/`（原始反例、定向日志、native各命令stdout/stderr与JSON）及上述test-results，不随clone提供。427个正式Study文件hash保持，未改历史脚本、Run、checkpoint、数据或来源。前一节core / 87项流程门对应此次Bash修改之前的源码，不把它们记为本次完整重跑；当前定向和新的原生流程证据覆盖实际修改。仍未commit / push / PR，未验证真正Linux / macOS宿主、GPU、长训练、吞吐或远端CI。

### 15. 当前本机开发改动的完整 CPU 核心门与交付核对

当前工作分支仍为 `codex/powershell-launch-boundaries`，基线 dev `a424be5`；本地 main `7a15c46` 与 dev 的代码树相同。B-031 至 B-036 的生产修改仅涉及 `schedule.py`、`flow/cli.py`、`flow/runner.py`，三份新增回归覆盖参数 / 路径 / 退出码、自动重建的冻结门、执行前拒绝的旧成功保全；设计与开发记录同步。没有依赖、算法、数值精度、持久化格式或公共参数变更。

为核对共同 Runner / CLI 改动对其他模块的影响，运行当前完整 core：420 passed / 2 skipped / 91 deselected，11.36 s，8 warnings（`20261013T194122Z_b26b54`，UTC）。两项skip为Windows符号链接权限；warning来自原有train / HF Trainer测试的checkpoint部分恢复提示，保留不抑制。此次源码与上一项最终CPU流程批次一致，核对生产及新增测试hash后复用该批次的87项integration通过证据；其中另有153项unit通过及2项skip，与core重叠，不能将批次数字直接相加。按test_id去重为507 passed / 2 skipped，仍不是全支持范围验收。

按成本、层级和资源标记核对三份新回归，均为integration / content / p1 / c2，当前CPU流程门会选择它们；缺PowerShell的环境明确skip其相关测试，未宣称所有平台均已运行。GitHub复查仍只有开放Issue #36，正文未变；本机已核验并修复的事实不等于已回复、合并、关闭或由下游验证。

汇总证据与源文件hash位于 `.tmp/dev-boundary-final/`，core及上一项流程原始事件留在各自 `.tmp/test-results/`，不随clone提供。427个正式Study文件继续hash保全，既有数据、checkpoint和迁移备份没有清理。当前为可复核的未提交本机改动，遵守不commit / push / 发布授权；后续集成仍需工作PR进入dev及当前提交的必需CI，不直接修改main/dev。未执行新包构建 / 安装、异机、GPU、并发写者、科学数值、吞吐或长实验验收；持续开发目标没有据此标为全部完成。

### 14. Issue #36 核验：B-035 保全改进与 B-036 冻结门遗漏

继续读取实际下游反馈，发现 [Issue #36](https://github.com/DreamSoul-AI/RPipe/issues/36)：v0.3.1 的原生 run-one 在计算前因 freeze 拒绝时，将旧 succeeded result 覆盖成 failed。原行为符合当时统一失败写回文档，此项按 B-035 证据保全改进处理，不声称旧数值算法错误。以真实 CPU Toy/Stub、linear、单seed、2-step 复现默认 / 启用 Study hook 两条路径，源码改动仅为给 study.yaml 追加注释，身份与计算配置不变；旧成功 result 的字节被替换，日志与资产仍在。

扩大到相邻失败边界时另确认 B-036：Runner 只在 Study hook 启用时做前置 freeze，关闭 hook 且直接选择 execute 子集会漏掉检查；这与任何阶段开始前检查的既有合同不一致。首轮14项中8 failed / 6 passed（`20261013T193614Z_9d5d10`，UTC），包括未进入阶段的断言失败和旧成功结果不变的断言失败。没有用真实GPU、领域算法或下游私有输入，也没有把下游 torch-first 导入改动带入上游修复。

先更新 [Runner 失败合同](../code/flow.md#3-runner-与失败)，再在共同 Runner 中对 freeze=true 始终检查前置门；仅 `current is None` 且异常为 `ProvenanceChangedError` 时复用现有 succeeded 保全路径。旧成功 result 的原字节保留，拒绝仍有失败日志、原异常和 CLI 非零退出码；没有旧成功仍写 failed。execute / summarize / write 真正失败继续覆盖旧成功为 failed，process 的原保全规则不变。未引入新 Result 字段、事务记录或启动包装器。

定向53 passed（`20261013T193722Z_401442`）。随后补充原成功 JSON 落盘备份及 checkpoint / tracker 等非日志资产的字节保全断言，当前最终 CPU 流程门240 passed / 2 skipped / 271 deselected，90.71 s（`20261013T193820Z_5dd6e1`），覆盖全部14项新反例、此前 launch 冻结门和 PowerShell 回归。两项skip仍为Windows符号链接权限，不计通过。真实 run-one 冻结拒绝exit1、stderr保留ProvenanceChangedError；结果和非日志资产不变，日志允许追加拒绝诊断。

本机原始证据在 `.tmp/b035-issue36/`（red / green / cpu日志及源码hash）和上述 `.tmp/test-results/`；真实拒绝 stdout/stderr 与原成功 JSON 在该批次 basetemp，均不随clone提供。427个正式Study文件与原保护清单hash一致；没有改正式Run、数据、checkpoint或历史来源。尚未commit / push / PR，未向Issue发评论或改状态，尚无当前远端CI或下游固定版本复验；Issue仍作为待集成反馈保留。未验证并发写者、GPU、长实验或跨设备科学数值。

### 13. B-034 自动计划重建前检查冻结来源

继续当前工作分支检查失败边界。既有 `_execute_launch` 在 jobs 缺失 / 损坏或资源参数不匹配时先执行自动 make，再检查 freeze；make 已重签来源，导致被冻结来源的变化被自动接受。五种反例包括缺 jobs、损坏 jobs、资源变更、缺 provenance、缺 provenance/index 但仍有 Run config。首轮5 failed / 4 passed（`20261013T193030Z_dab78d`，UTC），四个通过项为首次启动、来源未变、未冻结和显式 --remake 的兼容对照；假执行器证明被错误允许进入执行，不用于数值判定。

先补 [freeze 合同](../code/flow.md#142-来源清单) 和 README 命令表，再在自动 make 前复用现有来源检查：有 provenance、index 或 Run config 的默认重建必须先通过冻结门。拒绝时 exit 2，任何原配置、清单、脚本和来源均不改写；首次 launch 可以 make，显式 `make` / `launch --remake` 仍表示主动接受当前来源。没有变更持久化格式、公共参数、训练算法或 Run 重试语义，也未将显式重建改成隐式拒绝。

针对性首次复验50 passed / 2 skipped（`20261013T193113Z_4071d0`），其后增加两个真实 CLI 子进程反例，分别覆盖缺 jobs 和资源参数不匹配：exit2、原生 stderr 说明被冻结来源变化，所有 Study 文件逐字节不变。最终受影响的 make / artifact unit 与本地 CPU integration/e2e 门226 passed / 2 skipped / 271 deselected，88.08 s（`20261013T193153Z_52f775`）；覆盖全部11项新冻结边界、此前10项 PowerShell 回归及既有流程。两项skip是Windows无符号链接创建权限，不计质量通过。

证据在 `.tmp/b034-freeze-launch/`（red / green / cpu日志、源码hash）与上述 `.tmp/test-results/`；真实子进程 stdout / stderr 保存到本批次 basetemp。这些本机证据不随 clone 提供。427个正式 Study 文件hash仍与迁移后清单一致；本次不修改正式Run、checkpoint、数据或历史provenance。保持同一本地工作分支且未提交 / 推送 / PR；未执行真实GPU、正式实验、长训练、性能矩阵或远端CI。当前已完成本机冻结门修复，持续开发目标仍包含后续下游问题收纳与边界探索。

### 12. B-033 PowerShell 脚本保留生成时的资源参数

继续同一本机工作分支，核对脚本与直接 CLI 的计划复用边界。已有生成入口没有向 CLI 传递 `init_gpu` / `num_gpus` / `round`；以 `--init-gpu 3 --num-gpus 0 --round 2` make 的 CPU Toy/Stub 计划会因默认参数不匹配而进入自动 make，改写 `jobs.json` 和 `provenance.json`。两种 PowerShell 的真实计划保全反例及八项假 CLI 参数反例均先失败，共10 failed（`20261013T192619Z_491e76`，UTC）。没有启动 CUDA，只对非默认设备参数传递做 CPU 反例；不能据此声称 GPU 调度实测通过。

先补充[脚本合同](../code/structure.md#10-make)，再只在生成命令加三个已有 CLI 参数；显式 round 使用清单的有效并发值，自动装箱原有 batches 继续由清单复用。没有修改 CLI 对显式资源变更的重建规则、自动装箱或训练算法。实际修改的共同入口仍为 `write_launch_scripts`；根 CLI 和 make exports 不变，Bash 已直接携带分组，不在本次改动范围。

定向回归61 passed，14.67 s（`20261013T192706Z_949e30`）。新增两个真实子进程回归分别调用 Windows PowerShell / PowerShell 7：freeze=true、单 seed、Toy/Stub CPU 1-step，Run succeeded，index、provenance、jobs 和 config 四份文件的字节与 make 后完全一致。原八项路径 / 退出码反例同时验证参数传递，既有 make / schedule 回归保持通过。前一项的138项 CPU 流程门是参数补齐之前的证据，不冒称当前改动的完整重跑。

原始日志与独立临时目录在 `.tmp/b033-plan-reuse/` 及上述 `.tmp/test-results/`，不随 clone 提供。427个正式 Study 文件与本机迁移后的保护清单 SHA256 一致；未改历史脚本、Run、checkpoint、数据或来源。仍未 commit、push 或创建PR；旧脚本需按既有 make 流程更新，正式研究须保留旧身份与来源并另用新version。未验证真实GPU、多卡、远端CI、吞吐或长实验。

### 11. B-031 / B-032 PowerShell 调度入口边界

继续开发的本机工作分支为 `codex/powershell-launch-boundaries`，从 dev `a424be5dedac46d8f459ff8748f18387ebb05169` 建立；main 的发布合并提交为 `7a15c46`，两者代码树相同。GitHub connector 检索当时开放 Issues 和 PowerShell 相关 Issues 均为空；没有对外报单或自动监控。依据既有调度合同做小范围反例，不实施 brainstorm 中尚未确认的吞吐 / 滚动补位改造。

B-031：原 `launch.ps1` 设置 `$ErrorActionPreference=Stop` 但没有传递原生 Python 退出码。Windows PowerShell 5.1 / PowerShell 7 的真实 `-File` 子进程中，临时 CLI 返回 7，外层都返回 0。B-032：路径使用 `json.dumps`，反斜杠加倍、中文成为字面 `\uXXXX`，美元符号 / 反引号被 PowerShell 展开；普通路径已改变参数字节，特殊路径直接导致 Set-Location 失败。八项最小反例先全部失败（`20261013T192100Z_253e12`，UTC）。

先明确 [make 脚本合同](../code/structure.md#10-make)，再修改同一生成入口：PowerShell 单引号字面量及内部单引号转义、UTF-8 BOM 编码和 `exit $LASTEXITCODE`。没有新增配置、依赖或另一套启动器，也没有修改算法、Run 状态、失败重试、资源分组、Bash 生成或科学精度。新增 [PowerShell 子进程回归](../../tests/rpipe/structure/make/test_powershell.py)，使用临时假 CLI 检查真实参数与退出码；缺相应 shell 的环境明确 skip，不计通过。

定向 make / schedule / PowerShell 回归 59 passed（`20261013T192148Z_764c2e`）；受影响的 module_make unit 与本地 CPU integration/e2e 门 138 passed / 348 deselected，69.22 s（`20261013T192232Z_313c42`）。统一入口使用独立 basetemp，本机环境通过 `.tmp/runtime/activate.ps1` 启动，SEQUENTIAL / Agg / 空 GPU 可见性；未使用真实 GPU 或下载数据。

真实 CLI 的新增 `.tmp` Toy/Stub Study 使用中文 / 美元符号 / 反引号 / 单引号路径，version `powershell-boundary-20261014-v2`，两 seed × train/eval 四个 Run succeeded、两个 Experiment complete，训练仅 2 step。Windows PowerShell 启动完成后再由 PowerShell 7 重复启动，原 Run 日志只有一份 Flow start。缺 Study 声明时 native CLI、两种脚本均返回 1。临时验证器首轮默认 GBK 读取中文 JSON 失败；第二轮误将缺声明的原生退出码预期为 2，保留原日志后显式 UTF-8 并对照实际 native 退出码，独立失败出口核验通过。这两次是验证脚本错误，没有修改库 CLI 错误处理掩盖它们。

本机原始证据在 `.tmp/b031-powershell/` 和上述 `.tmp/test-results/`，不随 Git clone 提供。正式 studies 的源码、声明、调度及结果不改写；原本机迁移与 MKL 证据仍保留。旧生成 `.ps1` 不自动修改，下次按现有 make 流程应用修复，不能为更新脚本覆盖历史实测身份或来源。此次尚未 commit / push / PR，未通过远端跨平台门，也未运行正式 Study、GPU、长训练或性能矩阵。

### 10. 研究工作、下游反馈与 v0.3.1 发布准备

维护者授权全面文档检查、当前工作分支经PR进入dev，再由dev发布PR进入main，出v0.3.1；不跑新大规模实验。按最小改动保留已有README修正、历史实测和失败，不把历史日期的状态批量改写为当前结论。

完整读取指定飞书目录的研究计划规范 / 模板（均revision3），核对knowledge提交09078015的research-plan、literature-search、paper-analysis、pytorch-project正文和research-plan中文指南，适配为[研究工作要求](research.md)。宏观方案与具体Study分工不变，不增加运行依赖或安装skills；[下游反馈规范](feedback.md)明确归属、证据、公开脱敏、去重、授权提交与上游核验。README最后的prompt包含可删除的限范围Issue反馈授权，本轮没有实际对外报单。

版本元数据与README同步0.3.1，接入示例改为标签核对并保存确切commit；完整版本区间、迁移和未覆盖范围见[发布报告](../releases/v0.3.1.md)。活动文档修正已移除的B-022锚点、Flow表头、库默认write / Study扩展权限和safety默认值的边界；补查后核对实际Config与SystemFactory，明确标准字段不等于YAML全部必填、生态表不代表已实现、没有SystemRegistry。导航接通AGENTS、layout、Studies与PLAN模板，不改研究计算源码或正式数字。

本机core420 passed / 2 skipped（20261013T173200Z_03aa68，UTC），CPU integration52 passed（20261013T173211Z_8da839），使用独立basetemp。默认Conda / NumPy首次收集前因缺MKL runtime失败，采用既有MKL_THREADING_LAYER=SEQUENTIAL后复验；历史异机venv路径本机不存在，不能作为本机环境证据。两项skip为Windows符号链接权限，不计通过；未改代码掩盖环境缺失。

wheel / sdist在.tmp源码副本构建并分别安装，均0.3.1、各9条CLI及4个CPU Toy/Stub Run通过，复用本机依赖不代替远端干净环境。sdist首次默认pip缓存写入被拒绝，改.tmp及禁用缓存后通过，失败与原输出保留。77份Markdown的编码、围栏及1186个相对链接 / 锚点扫描无活动失效目标；427个Run / 临时引用是本机产物，55个历史快照引用仍按原目录语境解释，不为嵌套路径改变快照字节。7份v030清单102个文件hash / 大小通过。

本机证据位于.tmp/release-v0.3.1及上述test-results，不随Git提供；当前源码与旧804b3823的完整实验分别记录。提交、PR与main集成CI按现行门禁接续，实际发布URL、commit和检查链接补入GitHub Release，不把准备状态写成已发布。

### 9. B-030 报告来源校验与分支清理

本项于2026-10-13开始，跨日完成修复、回归与文档收口；下方实验来源仍是原日期与804b3823，不将旧实测重记为当天新增。

复现来自804b3823：新version / fresh make产生新index，旧process保留；原report只把index当日志链接来源且吞掉读取异常，仍exit0导出64个旧ID。正式复现保留在[Study证据](../../studies/main_exp/docs/v030_stale_report_repro.json)。先明确报告合同，再补反例；首轮18 failed / 2 passed，覆盖旧聚合、索引缺失 / 损坏 / 非法形状、重复Run与写入失败。修复仅在readout/report核对Study名及全部计划Run ID、seed、因素归属，复用已有atomic writer；不引入新process格式或索引哈希字段，不修改make、聚合、算法或数值门。

缺失或损坏索引、重复 / 缺少Run、身份不符均在导出前拒绝，旧process、NUMBERS和研究结论不变；旧格式清单、合法部分聚合和排列调整兼容。先前测试fixture漏写实际seed / 因素，已补齐；首次绿色复验19 passed / 1 failed因新增测试插在旧用例末段断言前，导致saved_figures越域，恢复原用例完整断言后31项通过，失败原样留在.tmp。清单匹配不证明同一Run再次执行后的指标新鲜度、运行期间索引未并发修改或研究质量；本次不扩展成并发快照 / 事务系统。

维护者确认后，brainstorm按资源与调度、环境准备、历史记录与边界分组，旧待办记录转入按日期备查，保留原始事实与链接，不实施吞吐或滚动补位改造。README的重构prompt移到最后，参考地址固定为https://github.com/DreamSoul-AI/RPipe/。已验证PR #32合并且内容保全后删除远端 / 本地codex/v030-local-studies，以及PR #28对应的旧本地codex/study-flow-layout和已合入的codex/downstream-usage；仅保留main/dev与本次codex/fix-stale-report。后续验证与PR状态以实际结果补记。

最终定向38 passed（20261013T155929Z_4aa95e）、core420 passed / 2 skipped（20261013T155957Z_df3f46）、CPU integration52 passed（20261013T160008Z_6384e0）；run_id使用UTC，integration实际本地日期为2026-10-14，两个批次使用独立basetemp。两项skip为Windows软链接创建受限用例，不计通过。复制正式新index与旧process的CLI复验exit2、原文件及旧表不变；匹配的当前聚合exit0，64个Run ID全部正确。7份v030清单的102个快照文件hash匹配，正式Studies相对dev无diff，证据为.tmp/b030-20261013/evidence-result.json；未训练、未改旧provenance或实测门限。

## 2026-10-13

### 8. 本地模板文档与云端实验分支同步

维护者指定 DreamSoul-AI/RPipe 的 codex/v030-local-studies 分支，将本地文档修正合入并经 PR 进入 dev。origin 已改为组织仓库；远端实验提交406d7a0基于发布804b3823，未修改库与测试源码。本地修改先提交为ef13595保全，再合入指定分支；唯一文本冲突为本文件同日新增记录，分别保留并区分章节，其他重叠文档自动合并。README 的源码模板、同名两柱、项目入口与可调整预算约定保留为接入规则，远端 safety=0.5 / 两并发仅为该次实验的实测条件，不转成通用要求。

远端 Study 报告、JSON、图和快照保持原字节，不重写实际计算来源；B-030保留open。首页补充本轮报告入口，同时保留旧示例图与日期；README、CI/CD及项目元数据的仓库链接更新为组织地址。本次同步不重跑研究实验，验证与实际PR状态另记后续结果，不将远端训练记录称为本地新增实测。

同步工作树的CPU core为395 passed / 2 skipped（20261013T153444Z_203e96），CPU integration为51 passed（20261013T153455Z_3ed556），两个进程使用不同basetemp；跳过不计通过。远端新增或修改的121个JSON可解析，7份v030快照清单的102个文件大小与SHA256匹配，两个正式Study相对于406d7a0无diff；根README等活动入口的93个相对链接 / 锚点、项目元数据组织地址和git diff --check通过。源码、测试和workflow均未变，临时检查及原始测试输出在.tmp，不随clone提供；本机未重新取得异机Run / checkpoint或执行GPU矩阵。

### 7. 源码模板接入与 Study 指南迁移

按维护者确认的接入方式，根 README 改为将固定 RPipe 源码复制到下游 `src/rpipe/` 并长期保留，研究包在 `src/<项目名>/` 下沿用相同 structure/flow 及层内职责，Study 导入项目实现。删除外部 pip / Git URL 依赖接入的推荐；通用 Study 指南从 studies/README.md 移至根 README，原目录入口保留导航和历史事实，设计、Agent 与模板的活动引用同步。

README 与 CI/CD 明确下游 main/dev/工作分支流转，以及 Actions 可选、本地验证和实际分支规则核对的边界；RPipe 自身的 Actions、Ruleset 和 workflow 未修改。

在 `.tmp/downstream-readme-20261013/` 隔离项目中从正式 main 提交804b3823导出并复制101个Python文件，逐文件字节核验；README 的复制命令拒绝覆盖已有src/rpipe，实际导入指向下游副本。四个CPU Toy/Stub Run成功，另一个Run通过recipe注册src项目算法并通过Study collect导入src项目阶段函数。验证仅证明源码接入、阶段和产物合同，不是新研究数值验收；原始检查与输出只在.tmp保留，不随clone提供。未测试网络clone、GPU或下游分发包，未提交推送或扩大实验。

后续按维护者要求保留原有实例曲线，移除 README 的安装章节和 recipe 目录项/示例，artifact 只在 structure 内说明。项目与 Study 的结构图均展示完整同名目录，不展开阶段文件；Studies 入口补充从 docs/research 宏观方案提取具体问题、计划和验收的步骤。通用命令以项目实际入口为准，rpipe 仅作为本仓库实例入口；显存预算和系数按项目校准调整，不规定统一比例。

隔离复验同时覆盖模板入口的四个 CPU Run 和研究项目自己的 my_project 入口；项目入口接入算法与 Study collect 的 Run 成功。容量 API 传入不同 safety 值时预算相应变化，仅证明参数可调整，不是 GPU 容量校准；原 CLI 未新增 safety 参数，库默认常量未改。原曲线图、生产源码和正式实验配置保持不变，证据为.tmp中的result-template-final.json。

README 的项目接入章节补充可复制的 Agent 重构指令，明确先读项目与模板文档，再按同名两柱映射、源码复用、宏观问题拆分 Study、来源冻结、真实入口及小规模验收实施。指令保留并行校准、分支/可选 CI 和当次授权边界，不自动授权下游推送、清理证据或大规模实验；本次只新增文档指令，未执行下游重构。

### 6. 本机接续编排与观察器修复

原调度进程终止后，未取回原退出码或观察器trace；没有把过期running字段当运行证据。临时观察器首次接续时未处理psutil.disk_io_counters返回None，在任何正式Run / checkpoint启动前退出；失败日志保全后以null表示不可用指标，保留进程I/O，并使用授权隐藏后台启动。接续期间持续取到新鲜资源 / 阶段观测，启动失败与修复范围见[观察器记录](../../studies/main_exp/docs/v030_observer_repair.json)，不是已证实的RPipe训练算法缺陷。

接续调用现有make接口，仅替换两个中断train及其配套eval的身份version；计算源码、声明、精度与科学预算一致，原已完成结果 / provenance未回写。原生launch和最终正式审计完成，来源、报告与中断证据见[Study报告](../../studies/main_exp/docs/STUDY_REPORT.md)。B-030仍open；本轮用额外来源核对避免错误导出，不把辅助核对当软件缺陷已经修复。没有实施brainstorm里的调度改造或发布新版本。

### 5. 发布后本机环境与目录核对

按维护者要求重新fetch并核对最新main：origin/main、本地main、工作分支HEAD及v0.3.0解引用提交均为 `804b3823`；本地main旧引用经祖先及worktree检查后快进，未切换正在实验的工作区。实际两Study计算源码无diff，来源冻结仍通过，见[来源核对](../../studies/main_exp/docs/v030_git_audit.json)。

main_exp根prepare / process及main_probe旧根阶段目录仅残留被Git忽略的`.pyc`；核对无源码、无重解析点且目标路径在工作区后，全部移入`.tmp/v030-20261013/old-phase-cache/`并保留hash清单。正式flow / structure源码与运行资产保持原路径；AGENTS新增切换或迁移后核对实际目录及忽略缓存的通用约定。

发现B-030的旧聚合报告来源缺陷，公开CLI独立元数据复现已保存，状态仍open，见[bugs.md](bugs.md)。细分预算、耗时均衡和归档依赖就绪检查记为[后续建议](brainstorm.md)，未自动实施。环境、数值门、校准及进行中的完整矩阵分别记录在[main_exp](../../studies/main_exp/docs/v030_execution.md)和[main_probe](../../studies/main_probe/docs/v030_execution.md)报告，开发记录不替代科学终验。

### 4. v0.3.0 发布准备

维护者在PR #30全绿后明确授权新版本与main同步。发布基线按正式v0.2.0标签75ebad0核对，包含此前首页PR #17、清理与分支规则PR #19、Flow / Study接续PR #21–#28及修复PR #29，不只记录最后一轮。新版本为0.3.0，元数据 / 包版本 / 首页同步；完整变更、迁移、历史与当前验收边界见[发布报告](../releases/v0.3.0.md)。

本轮不调整算法、门限或依赖，不重跑GPU长实验；版本号改变导致来源哈希变化，旧实测不回填新来源。版本工作仍通过工作PR→dev→main发布PR，当前提交与实际main集成分别验证，之后才创建新标签 / GitHub Release；不覆盖旧标签，不发布PyPI或二进制包。

0.3.0 元数据一致性、core 395 passed / 2 skipped、CPU integration/e2e 51 passed及新版本wheel / sdist安装验证通过，各安装环境9条CLI / 4个Toy/Stub Run完成。报告ID分别为20261012T183622Z_4c3128、20261012T183532Z_5a397b（UTC）；首轮20261012T183523Z_2ed7be出现238项setup错误，原因是新basetemp父目录不存在，保留失败后建立父目录并换新ID复验，没有修改库代码。安装证据在.tmp/release-v0.3.0/的wheel-smoke与sdist-smoke，不随clone提供；本机依赖复用不替代当前提交的远端干净安装。

### 1. 缺陷修复与整仓检查

基线为dev ce8b3e6与此前未提交的原生恢复修复。按文档检查公共入口、配置与持久化读写、调度、训练 / metric、compare及测试覆盖；不把未知功能或正常数值边界列成新缺陷，不引入跨会话锁 / 自动学习容量框架。

| **编号** | **反例及根因** | **修复与验收范围** |
| --- | --- | --- |
| B-002（用户B2） | CPU旧计划仍有GPU绑定，直接CPU环境覆盖父CUDA_VISIBLE_DEVICES；两条反例先失败 | 缓存计划拒绝残留绑定，直接CPU调度不应用GPU标签，日志显示CPU；48项make / schedule专项通过 |
| B-025 | 严格比较把2^53和2^53+1整数转double误判一致，复数虚部丢失；三条真实artifact反例先失败 | 整数标量保留精度，零容差整数tensor精确比较，复数保留虚部；普通浮点容差与CLI仍通过 |
| B-026 | jobs / batches重复Run并发争写；重复seed的Study展开也覆写同ID config，顺序聚合重复计数；三条反例先失败 | 展开在写config / index前拒绝重复ID，launch在启动 / 旧状态失效前拒绝实际分组重复；正常重试及train/eval屏障保留 |
| B-027 | Run名接受父目录、绝对路径或Windows别名；13条反例失败，外部路径尝试被沙箱阻止 | 任何创建前校验安全单目录组件；index的id与run_dir都校验，调度结果读入口复用；后续反例使用禁写替身，不再尝试外部写入 |
| B-028 | float / int32标签向量按batch维topk，得到200%准确率；预测样本数不匹配被广播；三条反例先失败 | 按目标形状解释标签 / 类别分数，样本数不一致报错；int64、one-hot与top-k保持 |
| B-029 | 正step / epoch预算的空train迭代器直接退出并定稿成功；两条反例先失败 | 空迭代器失败；已完成预算、恢复中的合法epoch尾段不误拒绝 |
| B-021补充边界 | 非CPU/CUDA后端没有对应设备RNG采集，不应标为完整恢复 | 仅CPU/CUDA声明完整状态；其他设备按部分恢复。测试只在CPU模拟声明，不冒充真实MPS实测 |

最初MPS声明测试漏替换eval阶段CPU传输，在无MPS的Windows失败；补齐测试替身后通过。重复展开修复首次回归99 passed / 1 failed，原因是expand_study按既有合同先建空目录；改验收为没有config / index写入，不改变正常布局。原红测和两次fixture失败保留于.tmp/test-results，不随clone提供。

### 2. B-022 / B-023 实测收口

恢复完整缓存后固定raw hash通过；B-022完成代表性串行 / 并发 / 混合的真实原生Flow容量、严格数值及总体吞吐验证，并用现有显式预算配置改进main_exp的后续自动装箱，默认safety / 系数不变。B-023完成同固定best的新进程串行 / 并发FP32验收及TF32负对照，取得实际CUDA kernel和逐样本预测差异。具体数字、条件与失败只留在[main_exp速度与精度报告](../../studies/main_exp/docs/speed_calibration.md#九2026-10-13-cifar10-容量与总体吞吐)。

首次容量测量在两份库文件行尾字节变化后被freeze拒绝，工具随后未记录缺失产物并退出；固定工具后换新ID与独立字节快照重测，没有关闭freeze。源逻辑核对不等于旧来源字节未变化，首轮仍为中止，不改写为成功。旧B-023权重未找回，不能还原其literal kernel；已修复的benchmark导入覆盖 / 未固定精度合同与遗留设备考古分别记录，不把新checkpoint当旧反例。

### 3. 回归与安装

- 核心395 passed / 2 skipped / 55 deselected；两项skip是本机无法创建目录符号链接。证据为.tmp/test-results/20261012T173135Z_04d87d（UTC，对应本地2026-10-13）。
- CPU integration/e2e 51 passed / 401 deselected，证据20261012T173226Z_9f21bc。legacy / HF部分恢复警告是兼容边界，不当作完整恢复通过。
- wheel / sdist均构建成功，在两个新的.tmp虚拟环境安装，各9条公开CLI、4个Toy/Stub Run通过。环境复用本机科学依赖，不代替远端干净安装。构建前端仅装.tmp；首次构建工作目录 / 输出位置错误被沙箱阻止，生成的本仓库egg-info移入.tmp；sdist首次pip缓存写出权限失败，改用.tmp缓存和新工作区重测，失败保留。
- 安装证据位于.tmp/main-ready/wheel-smoke/SMOKE_RESULT.json及sdist-smoke2/SMOKE_RESULT.json，不随clone提供。当前源码的八格600-step前置门、CUDA恢复和跨平台CI接续验证，不以旧长矩阵代替。

CUDA跨进程恢复复验1 passed / 10 deselected，证据20261012T174629Z_407c95；末轮边界专项89 passed。main_probe新版八个真实60-step成对Run全部成功，Study完整终验通过；原version05子集清单逐字节保存，具体研究数字见[probe报告](../../studies/main_probe/docs/STUDY_REPORT.md)。main_exp第一次八格600-step门通过；随后候选version标签及手动工具参数校验变化，使用新验证ID再跑最终源码门，旧PREFLIGHT未覆盖。显式容量fresh planning的64任务均唯一，ResNet18训练自动wait分组为2/2/2/2，不启动长训。

本轮不重跑32条80000-step矩阵，不改旧正式结果；main发布仍须工作PR进入dev、必需检查通过、dev发布PR检查通过，再到用户要求的main确认边界。

最终候选源码的main_exp600-step八格门全部通过，源哈希不变；科学原始值见Study报告。最后core复验仍395 passed / 2 skipped（20261012T175336Z_111c00）。132份Python语法、68份JSON（新增快照前统计）及5份历史snapshot清单校验通过；正式文档缺失入库目标为0，84个历史本机产物链接和2个模板占位明确不随clone。旧probe图从未修改前Git blob补存快照，主图更新为本轮8组；main_exp旧完整证据原样归档，未重建缺失Run。提交仅源码、测试、文档与Study科学证据，.tmp / runs / shared / 安装包全部保持忽略。

## 2026-10-12

### 8. B-022 校准预算接入与代表性短测

- 沿用 dev `ce8b3e6` 加未提交修复；先同步容量合同，再通过现有 SystemConfig 扩展增加 `vram_bytes` 显式单进程预算。不增加校准数据库或自动学习机制，不调整默认经验系数、安全系数、正式 Study 声明或并发数。预算复用既有 jobs.json 与装箱器，严格校验整数、固定开销下限与嵌套优先级。
- 复用 main_exp 的 `recipe calibrate`，增加数据选择、实际 raw hash / runtime、源码不变检查和 reserved 累计峰值。仅执行现有数据上的代表性短测；数值、阶段时间、显存与三轮失败全部留在 [速度报告第八节](../../studies/main_exp/docs/speed_calibration.md#八2026-10-12-本机-mnist-短校准)，有效复验使用新测量 ID，未覆盖旧失败或 PREFLIGHT。修正校准工具误把浮点汇总差异当预测分歧的判断，按原计划比较正确样本整数，Loss / 参数门保持不变。
- make 针对性首轮71 passed；最终 core 363 passed / 2 skipped / 55 deselected，CPU integration/e2e 51 passed / 369 deselected，Study 计数门 doctest 与非有限 / 非整数拒绝自检通过。符号链接权限仍跳过，旧 checkpoint 的部分恢复警告仍保留。CPU CLI 合同同时核对显式预算进入清单且实际 System 仍采用嵌套 CPU 设置。
- 测试证据在 `.tmp/test-results/20261012T153815Z_5d7cb1/`（首轮make）、`20261012T154640Z_10f59d/`（core）、`20261012T154652Z_6d6c8a/`（integration）；临时 checkpoint / 日志留在 `.tmp/speed-20261012-*/`，均不随 clone 提供。历史正式来源、结果及科学数字未改写，修复前 B-021 等未提交改动保留。
- B-022 的预算入口已接通，但原 CIFAR10 任务的完整进程峰值、受控并发与总体吞吐未验证，不关闭。B-023 原 checkpoint 不在当前工作区、CIFAR10 raw 不齐，无法做原 kernel 反例重放；本轮没有改 GPU 数值策略来绕过问题。未执行长矩阵、跨平台 CI 或全新安装，尚未 commit / push。

### 7. 容量配置修正与 B-024 停止条件修复

- 基线仍为 dev `ce8b3e6` 加本机未提交的 B-021 修复。本轮不修改正式 Study 配方、旧结果或来源快照，不运行长矩阵；范围为 make 配置解释 / 估算、原生训练停止合同及对应回归。
- make 与 runtime 使用相同的 config 优先于顶层 extras 语义，修正嵌套 algorithm 预算 / 评测设置和 system.device 的误读。训练行数不再借用 test 基数；估时计入梯度累积的实际 forward batch；显存估算计入实际 test batch，并保留固定开销。旧计划的 mode / device 与已有配置不一致时拒绝缓存、重新 make。安全系数 `0.5` 和模型经验系数不变，这些正确性修复不证明增加并发安全或更快。
- 核对 epoch 估时发现独立 B-024：每轮5 batch、step_period=2、num_epochs=2 原本多跑至3轮；每轮2 batch、step_period=3 原本无法更新并持续循环。三条有界反例先失败。现按声明 epoch 停止并正确判断恢复已完成预算，整轮无更新明确失败；恢复旧轮的尾 batch 不误判为整轮失败，梯度尾部丢弃行为不变。新增随机 transform / Dropout 的跨进程 epoch-tail 对照，连续、保存中断和恢复的样本轨迹、最终参数、optimizer / scheduler / RNG / tracker 精确一致。B-024 修复验证后移出开放清单，编号不复用。
- 最终 core 355 passed / 2 skipped / 54 deselected；最终 CPU integration/e2e 51 passed / 361 deselected；CUDA 新进程恢复 1 passed / 10 deselected。两项跳过仍为 Windows 目录符号链接权限。CPU 集成包括嵌套 CPU 配置的真实 CLI make / run-one，禁止 GPU 探测并核对实际 System；GPU 仅运行微小张量恢复回归，不是正式模型实验或吞吐校准。
- 修复前失败证据：`.tmp/test-results/20261012T151120Z_351188/`（8条估算 / 设备反例）、`20261012T151335Z_8b0e9f/`（3条停止反例）、`20261012T152126Z_08266c/`（2条旧计划反例）。最终证据：`20261012T152128Z_3aafb6/`（core）、`20261012T152658Z_5866a0/`（CPU integration）、`20261012T152733Z_145191/`（CUDA）。临时证据和执行输出在 `.tmp/capacity-fix/` 及上述测试目录，不随 Git clone 提供。
- B-022 只完成上述配置与估算修正，显存峰值、混合并发、总体吞吐和调度开销仍待受控校准；本机 CIFAR10 raw 下载未完成，不作为正式数据就绪。B-023 的默认精度 kernel 根因未追加诊断，两项继续 investigating。本轮尚未提交 / 推送，跨平台 CI、全新安装和当前源码的正式 Study 数值复验未执行。

### 6. B-021：原生 checkpoint 保存 RNG 与数据位置

- 基线为 dev `ce8b3e6` 加本机未提交修复。在 `.tmp/resume-fix/baseline/` 使用该提交的旧训练、Data、resume 与 tracker 实现做新进程反例：6-step 相同预算在 step2 保存后中断，续训的数据后缀和最终权重与连续运行不同；反例保留，不把旧 Study 中断归因于本缺陷。
- 先明确 [恢复合同](../code/structure.md#6113-resumetrain-与-eval-的特殊接口)，再在 native checkpoint 增加 `training_state.version=1`：保存 Python / NumPy / Torch CPU / 已初始化 CUDA RNG、数据 batch cursor / generator、训练配置及预算、early-stop 状态和 percent 快照进度。对象 / 优化器和完整预算 loader 重建后恢复位置与 RNG，只重放索引，不再次执行已消费样本的 transform；epoch 内恢复不重复计数，最后一个 batch 的收尾和已停止状态有针对性回归。
- tracker 新存 weighted_sum，避免半段恢复时 mean × n 的浮点舍入。旧 tracker / checkpoint 继续读取；旧 checkpoint、multi-worker、无独立 generator、自定义数据 / metric、整段 metric 和 HF 不宣称精确恢复，日志和 RuntimeWarning 明确提示部分恢复。有完整状态但预算 / 算法 / 数据 / 实际 runtime 不匹配时拒绝，不能换短预算充当前缀。
- 新进程恢复专项 10 passed，包括八条 CPU 连续 / 中断 / 恢复对照、一条自定义 metric 不标记完整恢复、一条真实 CUDA 对照；默认分类路径对照实际样本 / 三种随机 transform、Dropout、参数、optimizer、scheduler、RNG、tracker 和 early-stop 收口，均精确一致。最终 core 335 passed / 2 skipped / 53 deselected；CPU integration/e2e 49 passed / 341 deselected。两项跳过仍是 Windows 目录符号链接权限，部分恢复警告是明确的兼容边界，不隐藏为已完整支持。
- B-021 的原生恢复缺陷按上述支持合同本机修复并移出开放清单；multi-worker / 私有 transform RNG / HF / 整段 metric 扩展未实现，具体限制留在设计与 README，不外推短回归。B-022 / B-023 仍开放，本轮没有修改容量保护或 GPU kernel、没有重跑科学长实验，也没有修改 main_exp 的历史续训拒绝门。
- 本机证据：`.tmp/test-results/20261012T061240Z_73ec58/`（恢复专项）、`20261012T061334Z_5e4536/`（最终 core）、`20261012T061452Z_50549c/`（最终 integration）。首次旧用例把预算从2改为4而失败，已改为明确拒绝预算变化的回归；临时输出与旧反例在 `.tmp/resume-fix/`，均不随 clone 提供。此轮尚未提交推送，跨平台 CI 和正式 Study 复验未执行。

### 5. 提交前修复与回归

- 提交前重新检查发现 collect 包入口恢复为旧实现；原因未确认，不归因于杀毒软件或其他编辑器。重复定义与子模块同名，导入 `rpipe.flow.collect.run` 后包级 run 变为模块，两个导出回归失败。本轮重新删除旧实现，只导出既有 run.py；使用现有导出身份与行为测试验证，不添加新框架。
- 首次审查回归因 basetemp 父目录未创建发生 setup 错误；补齐临时父目录后复测为 319 passed / 2 failed / 2 skipped，两项失败都指向 collect。修正入口后最终核心回归 321 passed / 2 skipped / 43 deselected，CPU integration/e2e 40 passed / 326 deselected；两项跳过仍为目录符号链接权限。科学实验未重跑，旧失败、result 和 freeze hash 保留。
- 15 个库及 Study Flow 入口无业务实现，collect 函数 AST 未变；仅 collect 包入口 hash 与已完成短实验不同，历史快照 hash 全部通过。交接及 brainstorm 同步为实际基线 `7eb16438` 和已完成的有限短实测，原完整矩阵结论不扩称当前源码验收。
- 在 `.tmp/precommit-review/` 的源码副本使用现有 setuptools 构建 wheel 与 sdist，分别在独立虚拟环境安装并运行现有 package_smoke：两次均完成九条 CLI 命令、四个 Toy/Stub CPU Run、两组聚合及重复 launch 跳过检查。安装模块确认为虚拟环境中的分发包，不是仓库 src；依赖复用本机 site-packages，因此不替代远端干净环境检查。初次 sdist 安装因默认 pip 缓存目录写入权限失败，缓存改到 `.tmp/` 后通过；构建准备中的错误许可证文件名 / 工作目录已修正，未改变正式源码或全局 Python 环境。
- 本轮授权为整理这批重构、配套文档 / 测试 / 正式短实验证据，回归通过后创建工作分支、commit / push，再按 PR 必需检查合入 dev。临时验证与原始运行产物在忽略目录，不提交；不运行大规模实验，不直接 push dev，也不发布 main。实际提交与合并状态以 Git / GitHub 为准。

### 4. Study 短复验与入口遗漏清理

- 对齐两个 Study 的计划、README、累计报告与复跑记录，明确历史完整验收与当前短实验的范围。真实基线是 `7eb16438` 加未提交改动；实验过程和数字留在 [main_probe 报告](../../studies/main_probe/docs/STUDY_REPORT.md) 与 [main_exp 报告](../../studies/main_exp/docs/STUDY_REPORT.md)，没有扩大矩阵、提交或推送。
- 保留原探针失败与可取得的历史快照。缺少 evaluate 导致 prepare 失败后，只在 `.tmp/runtime` 补齐归档导入依赖，并换新 version 复验；下载仅完成 MNIST，CIFAR 未完成，不视为全数据就绪。
- 实验后复核发现上一轮 collect 的实现已复制到 `run.py`，但 `__init__.py` 仍有重复定义，之前“入口全清理”核查遗漏了这一处。本轮删除该重复实现，只导出 run；旧函数与既有 run.py 的 AST 一致，15 个库及 Study Flow 包入口重新核查均无业务实现。
- 最终针对性回归 57 passed / 2 skipped，覆盖导出身份、collect 行为、Study loader / 阶段边界及 provenance。跳过项为本机目录符号链接权限限制；原始科学结果与 freeze hash 未改写，最终入口清理后的完整数值复验仍须新 version / make。

### 1. 接通 Study 两柱与来源发现

- Study 专用建构 / 成对 Algorithm 已收进 `structure/`，阶段检查收进 `flow/`；清理冗余根入口、旧缓存与重复回归结果，保留正式结果、失败证据和历史快照。具体目录与清理事实见两个 Study 的 README，不改旧实测数字。
- 库加载改为 `flow/<phase>/__init__.py`，在独立 Study 命名空间中按正常包语义导入，支持跨两柱及根 recipe 的相对引用；空目录和没有包入口的手动工具不执行。旧根目录阶段包在显式启用时提示迁移，不静默漏跑。阶段顺序、prepare.before 时机、write 成功定稿及 process 失败边界不变。
- provenance 自动记录两柱全部 Python 源码，不要求显式 include 或阶段启用；保留额外输入的 include、路径边界、生成产物隔离和 freeze。两个 Study 删除重复的源码 include，声明、库和设计文档同步。旧来源清单必须重新 make；后续正式复跑使用新 version，不能沿用 version03 的历史通过记录。
- 针对性回归 44 passed / 2 skipped；CPU core 308 passed / 2 skipped / 40 deselected；本地 integration 门 37 passed / 313 deselected。两项跳过均为本机无目录符号链接创建权限，未宣称该边界已实际验收。首轮针对性运行因新 basetemp 的父目录未创建而发生 setup 错误（7 passed / 39 errors），补齐目录后复验通过，原失败事件保留。
- 两个真实 Study 的隔离副本分别展开 64 / 8 个配置，自动来源覆盖和 freeze 变更检测通过；小型自检通过真实 loader 的来源 / runtime 门、缺失 raw 拒绝、成对计算后置门与 Run process scope。自检首轮新增 runtime 调用参数位置错误，仅属于临时验证脚本，按现有接口修正后通过。没有数据下载、实际训练、数值探针或正式矩阵，未覆盖原 Run / 前置报告；本轮未提交或推送。
- 本机证据：`.tmp/test-results/20261011T192433Z_116ee5/`（首次 setup 失败）、`20261011T192442Z_cce0a1/`（针对性复验）、`20261011T192545Z_24e5b1/`（core）、`20261011T192629Z_521dd3/`（integration），每次保留 manifest / events / report；专用自检在 `.tmp/study-refactor/check.py`，不会随 Git clone 提供。测试 ID 采用 UTC 日期，开发记录按本机日期分组。

### 2. 对齐设计与接续文档

- 局部更新 concept / layout / code / structure / flow，说明 Study 两柱分工、hook 加载路径与手动入口；补清 provenance 的执行计划含义、摘要、覆盖行为、环境仅记录及非源码备份的边界。
- 更新 [handover.md](handover.md) 和 [brainstorm.md](brainstorm.md)，把当前 CPU 合同验证与历史 version03 实测分开；接续建议为新 version 单组合短数值门，再按计划与授权扩大。修正 bugs 和本节摘要中搬迁后的源码链接，开放缺陷状态不变。
- 本轮仅改文档，检查相对链接和 diff 空白；未新增代码验证、科学实测、提交或推送。上一轮 CPU 回归仍只覆盖当时的代码合同，不代表真实数值或异机验收。

### 3. 将 Flow 阶段实现移出包入口

- 文档原先只画阶段目录，且 Study 指南要求在 `__init__.py` 定义 `run`，不能清楚区分入口与实现。本轮先修订 layout / flow / code 与 Study 指南：库及 Study 的 Flow 包入口只做说明、导入和导出，普通阶段实现放 `run.py`，复杂阶段按职责拆模块；没有扩展不代表跳过库默认链。
- 搬迁库 prepare / execute / collect / summarize / write 和 main_exp 两个、main_probe 六个阶段，共 13 个实现模块；库 process 原已拆分，保留 `individual.py` / `study.py` / aggregate / curves。包级调用接口、prepare.before、write 定稿时序、异常与原数值门保持不变。未改结构柱实现或历史结果。
- 更新测试 fixture 为 `run.py` 实现与包入口导出，验证两层文件均进入 provenance，新增包入口结构及公开 callable 的回归，并覆盖 `run.py` 新增 / 修改 / 删除时的 freeze 拒绝。Flow 针对性 103 passed / 2 skipped，CPU core 321 passed / 2 skipped，CPU integration 40 passed；两个跳过项仍为本机目录符号链接权限限制。
- 本机 AST 自检确认 13 个搬迁模块与搬迁前完全一致，15 个库 / Study Flow 包入口仅有说明、导入和公开导出；真实 Study 隔离副本的注册、来源发现、freeze、runtime / raw / 成对后置门和 process scope 自检再次通过，没有实际计算或训练。
- 首轮 Flow 测试退出 0，但已有 pytest-cache 出现 WinError5 写缓存警告；随后 core 与 integration 使用本轮独立的 cache / basetemp，均通过。未修改共享缓存权限。证据在 `.tmp/flow-entries/` 和 `.tmp/test-results/20261011T194652Z_d22612/`（Flow）、`20261011T194723Z_1695df/`（core）、`20261011T194751Z_81b16b/`（integration）；临时证据不随 Git clone 提供。本轮未 make 正式 Study、未改 version、未提交或推送，后续实测仍须新 version / 来源冻结。

## 2026-10-11

### 1. 明确 System TF32 控制与 Study 新基线

- System runtime 增加可选 cudnn_allow_tf32 / matmul_allow_tf32 布尔配置，在 Data / Model 构造前应用并将实际值写入 System 结果；未声明时保留 Torch 当前设置，非法类型明确拒绝。设计与兼容边界见 [structure §7](../code/structure.md)。
- runtime 针对性回归 13 passed；核心回归 295 passed / 40 deselected。CPU Flow 集成 / CLI 回归 37 passed / 298 deselected。main_probe 新版八格真实验收通过，main_exp 归档导入修复后的 09e495b 前置门八格通过，完整矩阵于 19:48 通过原生 Flow 终验；实验数字与质量边界见 [正式报告](../../studies/main_exp/docs/STUDY_REPORT.md)。同步最终报告和数字表，保留旧失败；历史速度校准按 [测量方案](../../studies/main_exp/docs/speed_calibration.md) 接续，B-021 / B-022 与默认精度 kernel 调查尚未完成。
- 两个 Study 显式统一两侧 FP32 精度，检查声明 / 前置门条件，使用新 version / Run ID。旧版探针通过、长实验执行成功但终验失败的事实与原始证据均保留；实验数字与停止失败的恢复条件见 [main_exp 记录](../../studies/main_exp/docs/RERUN.md)、[main_probe 记录](../../studies/main_probe/docs/RERUN.md)，开发记录不重复实验结论。
- B-023 的串行反例排除了仅调度并发的解释；精度修复的真实完整复验已通过，具体默认精度 kernel 根因未完全确定，B-021 / B-022 保持开放。未发布或合入 main。

### 2. 保留归档导入前的设备开关

- 原归档 train_model 导入覆盖 cudnn.benchmark，导致实际条件与声明不一致。Archive 在 finally 恢复导入前值，归档源码字节保持不变；prepare 保存实际开关并拒绝不匹配，recipe 检查前置源码清单。实际归档 false / true 保留与异常拒绝回归通过，完整长实验需按新源码复验。设计和正式证据见 [恢复方案](../../studies/main_exp/docs/eval_recovery.md)、[复跑记录](../../studies/main_exp/docs/RERUN.md)。

### 3. 历史速度校准与归档准备优化

- 六组真实串行速度校准通过数值门，原始对照和阶段边界见 [速度报告](../../studies/main_exp/docs/speed_calibration.md)。接入 recipe calibrate，独立测量 ID 保留旧结果，不覆盖正式前置门。
- 归档 blob 从逐文件 Git 子进程改为 cat-file --batch；固定来源字节和 manifest 不变，拒绝不同内容与损坏响应，真实归档导入及运行条件回归通过。容量安全系数未修改，准备优化不等于并发问题已经解决。
- 当前核心回归 295 passed / 40 deselected。首次并行运行核心与集成测试共用默认 pytest 临时目录，集成产生 WinError5 / 随后 FileNotFoundError 的清理冲突（36 passed / 1 setup error），原日志 .tmp/batch-integration-tests.log 保留。改用独立 basetemp 后集成 / e2e 37 passed / 298 deselected；后续并行验证各自使用独立临时目录。日志仅本机提供。
- 准备优化提交 `8c17d5d` 的新八格前缀回归已通过，旧完整矩阵与前置报告未覆盖；证据见 [独立回归](../../studies/main_exp/docs/BATCH_EXPORT_PREFLIGHT.json)。PR #26 首轮与补入证据后的最终提交31acb5b，其 Unit tests、Build package、Branch flow 均实际通过。PR于20:29合入 dev 51a4ce0，本机已快进同步，原实验工作分支已删除；main 未变。合并后提交的 CI 单独核对，不沿用旧 HEAD 的绿灯。

## （一）2026-10-10

### 1. Flow / Study 阶段接续

- 从远端 `refactor/flow` 的 `0e3fec5` 读取交接。PR #21 的 Unit Tests、Package Check、Branch flow 全部通过后合入 dev（`2917e3e`），在 `refactor/study-phases` 接续；本轮不重跑 main_exp / main_probe，不发布新版本。
- 在设计文档先明确显式启用、库先/Study 后、相对导入隔离、Run/Study scope、recipe 建构前注册与 prepare 建构后检查的分工。实现可选阶段加载、顺序校验，以及每条 Run 前和 Study 聚合前的 freeze 检查；阶段包全部 Python 源码自动记入 provenance。
- main_exp 的完整曲线比较实现移入 `process/curves.py`；根 compare.py 保留手动完整/partial 入口。新增 prepare 来源检查、Study process 终验；不完整聚合只产生 partial，完整门失败报错，既有结果与通用聚合保留。
- main_probe 保留同进程受控计算，用 write 模块把已观测的 step30/60、latest/best 写为两侧 artifact 投影，调用库 compare 并保存 RUN_COMPARISON.json。仅规范化已知模型包装前缀，投影来源写入 result；不改变输入/RNG/样本计数/逐段参数/自身 best 门，也不伪称投影为新训练 Run。
- 按维护者明确要求删除本机 `studies/main_historical/` 残留目录（数据、Run、缓存与派生文件）；这些文件已不在当前分支 Git 跟踪中。保留现行 main_exp / main_probe；先前正式报告的 Git 提交链接仍可读取。
- 新增 16 项回归通过。首次完整测试因当前 Python 缺 kornia，core 为 280 passed / 2 failed / 35 deselected，integration/e2e 为 31 passed / 1 failed / 285 deselected；失败记录保留在本机 `.tmp/test-results/`。设置 `MKL_THREADING_LAYER=SEQUENTIAL`、`MPLBACKEND=Agg` 并复用 `.tmp/runtime` 的 kornia 0.8.3 后，core **282 passed / 35 deselected**，CPU integration/e2e **32 passed / 285 deselected**。
- CPU 小张量投影检查通过：相同快照及 latest/best 对比通过；修改 best 权重能检出；已有投影拒绝覆盖。该检查没有数据集、模型训练或 GPU 执行。本机临时检查及原始测试报告不随 Git clone 提供。

以下 §2–§5 保留 PR #21 前的开发事实；其中待办只描述当时状态。

仓库清理在分支 `refactor/cleanup`，提交 `f534358`，[PR #19](https://github.com/diaoenmao/RPipe/pull/19) 合向 `dev`。分支从 `main` 的 `71143ab` 拉出；当时 `dev` 落后 `main` 三个发布提交，合并会一并带入。本轮只整理目录、文档和测试输入，不改 flow，也不把 Study 接到新的库接口。

### 2. 文档

- 设计文档移到 `docs/code/`：`concept.md`、`layout.md`、`code.md`、`structure.md`、`flow.md`
- 开发记录移到 `docs/development/`：`brainstorm.md`、`bugs.md`、`testing.md`；`SUMMARY.md` 改名为 `record.md`
- 文档文件名改为小写。目录入口用 `README.md`。Study 内的 `PLAN.md`、`STUDY_REPORT.md`、`NUMBERS.md` 保持原名
- 原 Study 指南与 `studies/` 导航合并为 [studies/README.md](../../studies/README.md)
- 删除整份过时文档：`HANDOVER.md`、`MAIN_MIGRATION.md`、`REPOSITORY_CLEANUP.md`、`REPOSITORY_CLEANUP_RESULT.json`、`STUDY_GUIDE.md`、`docs/releases/v0.1.0.md`、`docs/releases/v0.2.0.md`。这些文件仍在 [`71143ab`](https://github.com/diaoenmao/RPipe/tree/71143ab/docs)
- `AGENTS.md` 与根 `README.md` 的入口和链接按新路径更新。Study 的计划、报告、图、正式数字和复跑代码只放在对应的 `studies/<name>/`；本文件只记开发事实，实验结论链接到 Study 报告

### 3. Study 与测试输入

- `studies/` 只保留 `main_historical`、`main_reproduction`、`_template`。删除 `mnist_train_size`、`mnist_native_vs_hf`、`cifar_grid`、`main_base`、`checkpoint_recovery`、`mnist_cnn_lr`、`mnist_cnn_budget`、`mnist_cnn_budget_repeat`、`local_model_matrix`、`support_data_smoke`、`support_model_smoke`，包括本机数据。计划与报告见 [`71143ab` 的 studies](https://github.com/diaoenmao/RPipe/tree/71143ab/studies)
- 保留的 Study 带上各自代码。`main_reproduction/code/` 补入 24 个复跑脚本，按原内容入库，因为部分脚本会校验自己的 SHA-256。这些脚本不在 `code/` 里直接运行，复跑前要先复制到 `.tmp/main-reproduction-<date>/`，并放回它们依赖的原始数据缓存
- 测试仍要用的 `mnist_train_size`、`mnist_native_vs_hf`、`cifar_grid` 声明移到 `tests/_data/studies/`，用例复制到临时目录再跑
- `.gitignore` 按 GitHub Python 模板加上 RPipe 规则重建

### 4. 验证与未做事项

本机 Anaconda 的 `libiomp5md.dll` 曾被改名为 `libiomp5md.dll.bak-before-torch-cu130`，裸 `import numpy` 会以 `0xc06d007e` 退出。两轮测试都设置了 `MKL_THREADING_LAYER=SEQUENTIAL`，没有改 Anaconda 安装。

- `tests/run.py --core`：268 passed / 23 deselected
- CPU 的 integration 与 e2e，c1/c2，排除 external / gpu / slow：20 passed / 271 deselected
- 相对链接检查后，仍打不开的是 Study 报告里指向本地 Run 日志和 `.tmp/` 证据的链接，以及模板占位图。这些文件本来不随 Git 提供

远端结果以 PR #19 的 Checks 为准。Flow 重构、Study 改接到新库接口，以及只含 `.gitkeep` 的测试占位目录，都留到后续分支；改完后需要重新跑验证。

### 5. CI/CD 接入

按《CI CD 执行规范》和《多人协作代码开发规范》写下 [cicd.md](cicd.md)。仓库只有一名维护者，合入 `dev` 和 `dev → main` 都不要求另一人 Approve，必需检查通过后自行合并。

分支路径由 `.github/workflows/branch-flow.yml` 检查：工作分支的 PR 目标必须是 `dev`，`main` 只接受来自 `dev` 的 PR。`dev` Ruleset 与 [main Ruleset](https://github.com/diaoenmao/RPipe/rules/24809015) 都要求 `Unit tests`、`Build package`、`Branch flow`，并禁止直接 push、force push 和删除。`main` 的旧版 branch protection 已删除，Ruleset 是唯一闸门。仓库打开了 Auto-merge。不自动删除头分支，避免发布 PR 删掉 `dev`。对外发布和安装包归档仍未接入。

受控失败：[PR #20](https://github.com/diaoenmao/RPipe/pull/20) 用 `tmp/branch-flow-probe` 直接向 `main` 开 PR。[Branch flow](https://github.com/diaoenmao/RPipe/actions/runs/37976079219) 失败，`mergeStateStatus` 为 BLOCKED。PR 已关闭，远端分支已删除。同一次的 Unit Tests 与 Package Check 在确认阻断后取消。

### 6. Flow：Study 扩展点进库

现有 Study 各自写了 launch、run-one、哈希清单和比较脚本，原因是库缺三样东西。设计见 [flow.md](../code/flow.md) §14，已在 `refactor/flow` 实现：

1. recipe：`study.yaml` 写 `recipe: recipe.py`，prepare 在每条 Run 建构前调用其 `register(ctx)`，`run-one`、`launch` 子进程和顺序 `run` 行为一致；不进 Run id。
2. 来源清单：make 写 `provenance.json`（库源码、声明、recipe、`provenance.include`、计划的哈希，环境与 git 提交）。`freeze: true` 时有变化则 launch 退出 2、prepare 失败；result 记录 `environment` 与清单摘要。
3. `python -m rpipe compare <run_a> <run_b>`：比较 result 指标、tracker history 与 checkpoint 的 model / optimizer / scheduler，按 `--atol` / `--rtol` 判定。

新增10个测试。本地 `tests/run.py --core` 为 272 passed / 29 deselected；CPU 的 integration 与 e2e（c1/c2，排除 external / gpu / slow）为 26 passed / 275 deselected，均设置 `MKL_THREADING_LAYER=SEQUENTIAL`。

`main_historical` 已改接：`study.yaml` 加 `recipe: recipe.py`、`freeze: true` 和 `provenance.include`；`recipe.register(ctx)` 承担原 `run.py one` 的线程、CUBLAS、拒绝续跑中断训练和 preflight 检查，preflight 入口移到 `recipe.py preflight`；206 行的 `run.py` 删除，调度改用通用 `rpipe make / launch / status / report`。在 `.tmp/` 副本上展开，64 个 Run ID 与 `docs/COMPARISON.json` 记录的一致。`compare.py`、`verify_group.py`、`verify_preflight.py`、`publish_figures.py`、`prepare_data.py` 是本 Study 的终验与审计，保留；`verify_preflight.py` 的源码哈希改用库的 `source_files`。该状态保存在提交 `18cd76c`。

随后按用户决定两个 Study 都将重跑，不再保留旧证据：`main_historical` 改名 `main_exp`，`main_reproduction` 改名 `main_probe`，Run 的 `experiment` 与 `version` 随之更新，Run ID 会变。`main_exp` 只留声明、`recipe.py`、`prepare_data.py`、终验 `compare.py`、`runtime-requirements.txt`、`TARGET.md`、`REFERENCE_CURVES.json`、原图归档，以及从 `main_reproduction` 移入的数据清单 `EXPECTED_DATA.json`；删除 `verify_group.py`、`verify_preflight.py`、`publish_figures.py` 和全部旧报告与证据。`main_probe` 只留声明和 `probe.py`（原 `current_device.py`，去掉对旧 dev 提交 `8bccbac` 的 HEAD 硬性检查），删除 `code/` 下 24 个脚本和全部旧证据；`study.yaml` 改为与探针一致的 deterministic=true、benchmark=false。两个 Study 重写了 README 与 PLAN。开发记录、brainstorm 与 `TARGET.md` 中指向旧文件的链接改为 `18cd76c` 永久链接。两个 Study 均未运行；`probe.py` 改为输出 Run 目录并用 `rpipe compare` 比较，计划在重跑前完成。

### 7. 交接与后续授权

- 会话交接文档：[handover.md](handover.md)（放在 `docs/development/`）
- **2026-10-10 维护者要求：本轮不再重跑 `main_exp` / `main_probe` 实验**；旧实测以 [`18cd76c`](https://github.com/diaoenmao/RPipe/tree/18cd76c/studies) 为准。接续工作限于合并 PR #21、实现 Study 与 Flow 同构阶段目录、代码整理，不含未授权的 launch / 长训 / 探针矩阵。

### 8. 成对探针与文档审校

- PR #23 将 main_probe 成对计算接入 Flow，移除独立 probe.py 入口；三个必需检查通过，合入 dev `22442f9`
- 审校 DreamSoul 的文档规范后，按仓库 Markdown 规则统一表头、标题空行、中英文混排和现行文档名称。更新 handover 当前状态，保留历史交接快照链接和开发失败证据
- 修正指南中“Study 只剩 recipe”、recipe 不可准备缓存及 Study process 必为独立进程等过时说明，区分 Run 数值门与 write 后派生失败
- main_exp 使用已有的 `flow.prepare_shared: false`，避免 recipe 注册前构造专用 source。临时 CPU 副本的 make 正常生成 64 Run、jobs 与 provenance，未构造归档 Data 或执行训练
- 24 份 Markdown 渲染解析通过，正式相对链接与锚点核对通过，表头加粗和示例代码保全检查通过。8 处历史临时证据路径标注当前缺失且不随 Git 提供；LICENSE 与实验配方参数保持原样

### 9. Flow 阶段职责重构与复跑准备

本次将构造前准备和写入前验收纳入库阶段合同，并拆开 main_probe 的实际阶段职责，变更对比见第一节（一）§5。main_exp 的构造后来源检查和 Study 终验继续使用原有入口。AGENTS.md 增加阶段职责验收、独立 Run 复跑和失败证据记录要求；Study 指南清除已删除 probe.py 的陈旧说明。

本地 CPU 门通过，探针训练顺序、预算与固定门未调整。GPU 可用，但 main_exp/shared/data 当前不存在，需要恢复并按 EXPECTED_DATA.json 核对。后续在验证后的源码基线上使用新 version，先准备数据、执行计划前置门，再运行完整矩阵；当前暂无本版本正式数值结论。

### 10. 正式复跑环境与数据

PR #25 合入后新建 feature/study-rerun，使用 main-probe-flow-20261010-01 / main-exp-flow-20261010-01 的独立 Run ID。GPU 为 RTX 5090 D v2，Torch 2.11.0+cu130；原始数据 16/16 文件 SHA-256 匹配固定清单。CIFAR10 慢速下载进程主动停止并确认退出，随后复用本机完整 raw 缓存；未完成 part 不作为有效数据。训练配方、完整矩阵和固定门限未调整。当前尚未取得正式数值结果，实际执行进展记入各 Study 的 RERUN.md。

### 11. 首轮真实探针失败与修复复验

version01 的 MNIST/linear（f6db09ee2f38fec9，提交5f1ff5c）在 prepare 失败：拆分遗漏 os 导入（B-019），来源扫描误含生成的32个原版归档 Python 文件（B-020）。数据与模型 CPU 数值检查通过，但整体准备门未通过；未进入正式训练。失败 result、日志和准备证据保留。

提交 e682f25 补充导入，并复用库来源清单，原归档逐字节检查及固定数值门不变。名称依赖检查通过；新增生成归档/真实源码变更回归，CPU 集成37项通过。version02 八组合真实 GPU 全链及 Study 终验通过，16段参数最大差0、Loss 最大差6.66e-16、独立 eval Loss 差0，库 compare 全通过，B-019 / B-020 已验证关闭。正式证据见 [探针报告](../../studies/main_probe/docs/STUDY_REPORT.md)，main_exp 600-step 八格前置门已启动，长曲线尚未完成。

## （二）2026-10-04

最新远端 dev 基线为 `8bccbac`。本机完成 [历史 main 曲线复现](https://github.com/diaoenmao/RPipe/blob/18cd76c/studies/main_historical/docs/STUDY_REPORT.md)：32 条四 seed 连续80000-step 训练与32条自身 best 独立评测全部成功，每条400个完整 test 点。八组的终点、末50点均值、七锚点、末段波动及两数据集模型排序通过预先冻结的原图估读合同，最大终点差0.207502个百分点。

[现代 main 本机对照](https://github.com/diaoenmao/RPipe/blob/18cd76c/studies/main_reproduction/docs/CURRENT_DEVICE_RESULT.md) 在原 Stats 全精度 profile 与统一确定性配置下60-step 八格通过，step30/60共16段参数最大差0；本机历史600-step 八格前缀桥及两类独立 CPU 存档核验也通过。原默认条件的失败、旧设备结果和原图缺失运行记录的边界保留；不把前缀对齐扩称为80k 权重逐位相同。

本轮另整理全仓库导航、正式 Study 入口、冗余占位和事务标记，补充安装后 CLI 与 CPU 验收。完整变更与实际验证见 [REPOSITORY_CLEANUP.md](https://github.com/diaoenmao/RPipe/blob/71143ab/docs/REPOSITORY_CLEANUP.md)。该文件已在 2026-10-10 从当前树删除，链接指向删除前的提交。

以下2026-10-03记录保留当时“仅探针、未执行长矩阵”的真实范围，不再代表最新完成状态。

### 1. 当前设备的完整验收

| **验收对象** | **实际完成** | **证据边界** |
|---|---|---|
| 历史候选同机计算桥 | 8组合、seed0、step200/400/600，24段参数/buffer 逐位一致，SGD/scheduler/RNG/实际输入一致 | 600-step 前缀，不能代替全80k 原码/current 权重证明 |
| 历史四 seed 长曲线 | 32train × 80000step，每条400次完整 test；32own-best eval；完整固定门通过 | PNG 估读，未知原 seed 与环境，未验收原图阴影 std 及全400点逐位差 |
| 现代 main 受控计算对照 | 8格 seed0、60step/eval30、16段参数 maxdiff0，独立完整 test 一致，CPU 重载通过 | 原 Stats B250全精度 profile、deterministic 条件；未覆盖原默认 benchmark=true/无 profile |
| 本轮整理与迁移准备 | 首页与14个 Study 导航、专用 Registry 入口、旧接口迁移说明、CPU 和安装交付门 | 具体执行结果及远端待执行项见 [整理报告](https://github.com/diaoenmao/RPipe/blob/71143ab/docs/REPOSITORY_CLEANUP.md)；未实际合并 main |

历史矩阵主控制器于2026-10-04 09:12:29 +08:00结束；完整曲线及独立审计于当日晚间收口。[最终结果](https://github.com/diaoenmao/RPipe/blob/18cd76c/studies/main_historical/docs/FINAL_RESULT.json)与[执行审计](https://github.com/diaoenmao/RPipe/blob/18cd76c/studies/main_historical/docs/FINAL_EXECUTION_AUDIT.json)保留真实来源。提前排班去重成功、无训练 resume；CIFAR linear 评测四 worker 成功，但记录控制器因 WinError5退出1，仅恢复记录，没有重跑或改写原退出码。

旧进度和前三组 JSON 快照保留原样。[Study 使用指南](../../studies/README.md)区分正式可交付成果与本机 runs/shared/.tmp 证据，清理没有删除数据、checkpoint 或失败记录。旧 main 无扩展 checkpoint 与新的.pt/bundle、模型 key 和 Stats 对象存在实际格式差异，普通新 CLI 不直接 resume 旧任务。

## （三）2026-10-03

### 阶段摘要

当天工作由 Cursor 的实现与 Codex 的接续复核组成，重点是修复基础问题、核对 main 的计算结果，并完成限定范围的历史超参探针。

#### 1. 推进过程

1. 修复 B-014～B-018 的配置入口、依赖、训练均值、topk 计数和零评测预算问题，调整曲线的真实进度与回滚分支处理
2. 补验 FashionMNIST、CIFAR100、SVHN，以及 ResNet10 和两种 WideResNet 的真实数据训练与评测
3. 对照当前 main 的 60-step 配方，核对数据、初始化、采样、参数、scheduler、best checkpoint 和独立 eval
4. 追溯 README 历史图的候选旧配方，保留旧 BN、CPU 增强、归一化和裁剪，仅执行 seed 0 的 200-step/eval200 前缀探针

#### 2. 验证结果

- 本地回归：**286 passed / 3 deselected**，接续复核确认 91 个生产源文件与最新实验快照一致
- 支持范围短验收：**12/12 执行成功**，严格 Accuracy 复算 5/6 一致。ResNet10 相差一个正确样本，保留未通过判定
- 当前 main 对照：默认条件严格数值门 **4/8**，原代码重复也有分歧。统一 CUDA 确定性条件后 **8/8** 对齐
- 历史超参探针：计算对照及存档复核均 **8/8**，参数差值为 0，内部耗时约 **149 秒**。[探针报告](https://github.com/diaoenmao/RPipe/blob/18cd76c/studies/main_reproduction/docs/HISTORICAL_PREFIX_PROBE.md)保留完整原值与边界

#### 3. 当前状态

- 没有开放缺陷，本轮探针未发现新生产 bug。维护关闭与根因修复的区别保留在详细记录中
- 本轮到探针结束，**完整历史曲线尚未复现**。没有执行 80000-step/4-seed 长实验，也未重建旧依赖环境
- Git 交付从 `feat/main-base` 集成至 `dev`，远程电脑使用 `dev` 的实际提交作为测试基线。交付范围和发布前回归见详细记录 §12；临时脚本、缓存和原始验证输出仍放在 `.tmp/`，不随 Git 同步

### 1. 本轮收尾与交接

本轮接续复核、历史超参前缀探针和 bug 列表整理已收口。用户限定的200-step/eval200探针8/8通过；固定当前 main 的确定性对照8/8，原默认严格门仍为4/8。README 历史图未复现，不把本轮探针完成写成完整结果复现。

| **交付** | **内容与证据** |
|---|---|
| B-014 / B-015 | Data/Model Factory 严格匹配 name/source，拒绝隐式 stub；补齐 SVHN 的 scipy 依赖及验证 |
| B-016 / B-017 / B-018 | 按评测段保存 train 均值；topk 按样本计数；零评测预算在读取数据前报错 |
| 用户选定3.1 | 曲线按 optimizer step / epoch 对齐，排除回滚分支，按共同坐标聚合并记录逐点 n；3.2/3.3不做 |
| 最新本地回归 | unit + integration、c1/c2：286 passed / 3 deselected，排除 external/slow/gpu；Codex 接续测试报告（临时证据，当前缺失，不随 Git 提供），原路径 `.tmp/test-results/20261003T110214Z_95a976/report.md` |
| 当前源码完整对照 | 8 train + 8 eval 全部成功；确定性门8/8，step30/60参数与 test 指标差值0，独立 eval/best 一致；[逐格证据](https://github.com/diaoenmao/RPipe/blob/18cd76c/studies/main_reproduction/docs/DETERMINISTIC_COMPARISON_AFTER_B018.json) |
| 历史超参前缀探针 | seed0、200-step/eval200、scheduler T_max80000；原代码/当前链8/8，存档独立复核8/8，参数差值0，内部148.740s；[探针报告](https://github.com/diaoenmao/RPipe/blob/18cd76c/studies/main_reproduction/docs/HISTORICAL_PREFIX_PROBE.md) |
| 当前 bug 清单 | 没有开放缺陷；移除“开放”标题下的已关闭说明，本探针未发现新生产 bug，不把维护关闭说成根因修复 |
| 研究报告 | [main对照](https://github.com/diaoenmao/RPipe/blob/18cd76c/studies/main_reproduction/docs/STUDY_REPORT.md)、[数据支持验收](https://github.com/diaoenmao/RPipe/blob/71143ab/studies/support_data_smoke/docs/STUDY_REPORT.md)、[模型支持验收](https://github.com/diaoenmao/RPipe/blob/71143ab/studies/support_model_smoke/docs/STUDY_REPORT.md)；各报告保留未通过门限和证据边界 |

探针收口时的交接位置：分支`feat/main-base`，HEAD `d938874`，当时后续改动尚未提交或推送，之后的 Git 交付见 §12。正式记录在本文件、[BRAINSTORM](brainstorm.md)与各 Study 的 docs；原始日志、探针和固定源码副本在`.tmp/main-reproduction-20261003/`等临时目录，保留以供核对，不作为已提交成果。当前91个生产源文件与 B-018快照一致；接续阶段新增的训练仅为用户明确要求的200-step 前缀探针，没有长实验或重复相同回归。

当前决定（2026-10-03）：用户选择 README 历史图对应的 main 超参路线，随后明确本轮只做到探针，并要求处理实际开放 bug。200-step/eval200、seed0的8格前缀对照与计时已完成并收口；不跑80000-step/4-seed 长实验。原默认4/8、确定性8/8及历史图未复现结论保留；原历史运行记录仍缺失。详细范围见[本轮计划](https://github.com/diaoenmao/RPipe/blob/18cd76c/studies/main_reproduction/docs/PLAN.md#用户确认的历史超参前缀探针2026-10-03执行前)。

### 2. 目标与对照

最终目标是复现 main 的研究结果。当前本地 main / origin/main 均为 `98648f3a5c7db7dccf3ca806410d5b6fdee9484c`，与已有 main_base PLAN 固定的对照一致。新架构的 Run 成功、更多训练步数或更高精度都不能替代对照验证。

main 的 base 矩阵是 MNIST / CIFAR10 × linear / mlp / cnn / resnet18，seed 0；batch 250、test batch 1000、60 optimizer steps、完整 test 每 30 steps；SGD lr=0.1 / momentum=0.9 / Nesterov / weight decay=0.0005、cosine T_max=60；按 test Loss 选择 best，独立 test 加载 best。

main Git 中只有 README 的曲线图，没有对应的可读原始结果、权重和依赖锁定。先在本机运行固定提交建立可核对的新对照，逐项注明与历史图结果的证据边界；不从图片猜出精确数值。

### 3. 已完成阶段

| **阶段** | **已验证的内容** | **尚不能证明的内容** |
|---|---|---|
| 结构迁移与执行链 | Study / Experiment / Run、make / launch / process、日志、结果及 checkpoint 路径 | 原 main 数值结果已复现 |
| checkpoint 修复 | B-009–B-013 的恢复、原子写入与 sibling eval 保护；真实 Windows 保存问题补测 | RNG / sampler 逐位恢复 |
| 真实训练验证 | main_base 16/16，600-step 多模型矩阵 24/24；报告保留各自配方 | main_base 每 5 步 eval 与 main 每 30 步 eval 等价 |
| 配置入口修复 | B-014 / B-015：拒绝未知来源、显式 stub、SVHN scipy 依赖与真实数据短验收 | 所有数据 / 模型组合与全新环境安装 |
| 曲线进度修复 | 新记录按 optimizer_step / epoch 对齐，回滚分支保留原始日志、有效轨迹单独聚合，逐点 n；本地门 279 passed / 3 deselected | 历史没有进度字段的曲线可以自动恢复真实坐标 |
| 训练均值修复 B-016 | native step 模式按评测段保存训练均值，空段不覆盖最后有效值；合并回归 282 passed / 3 deselected | 完整 60-step 真实数据结果与 main 等价 |
| 完整 main 源码对照 | 真实数据60-step / eval30，原默认 CUDA 配方两套各16/16执行成功、严格数值门4/8；确定性控制两套各16/16成功、数值门8/8，step30/60参数与 test 指标差值均为0 | 原默认 benchmark 配方逐位一致；历史 README 图已复现 |
| 指标修复 B-017 | topk 参数按样本计数；本地门283 passed / 3 deselected；修复后 CPU 探针8/8、当前16次真实数据 CUDA 执行与原 main 确定性存档数值对照8/8 | 新增 topk 配置入口；原默认非确定性/历史图结果的额外复现 |
| 评测预算修复 B-018 | eval_num_steps=0在读取数据前报错；本地门286 passed / 3 deselected，main 原代码 CPU 探针8/8；当前源码完整16次 CUDA 执行成功，确定性数值门8/8 | 原默认严格门4/8的额外验收；历史图复现 |

详细记录与原始验证入口见 [BRAINSTORM](brainstorm.md)、各 Study 报告；未解决缺陷见 [BUGS](bugs.md)。本文件记录开发事实，不取代设计文档。

### 4. main 配方与行为审计

- 工作区 HEAD `d938874`，保留已有未提交修复与两份支持范围验收 Study。
- 已核对 main 固定提交、base 矩阵、数据增强、模型结构 / 初始化、采样器及优化器；原代码只读导出在 `.tmp/main-reproduction-20261003/reference/`，不切换或覆盖工作区。
- B-016 已确认并修复：固定权重、前两步 Loss≈0 / 后两步 Loss≈10、eval_period=2，原实现最后 train_loss≈5，main 的最后段应≈10。修复前 step 反例失败、epoch 对照通过（1 failed / 1 passed），证据 `.tmp/main-red-38ebbacd7ca6403f87b14f187885a238/`；修复后两种模式通过。native step 每次周期 test 前保存并重置 train 段，收尾跳过空段；epoch 模式与参数更新顺序保留。
- 固定 main 原代码与新实现的 CPU 探针 8/8 通过：MNIST / CIFAR10 × linear / mlp / cnn / resnet18，合成固定输入、4 steps / eval2。初始化和采样顺序一致，step2 / 4 的参数、train / test Loss、Accuracy 差值均为 0，scheduler 状态一致。脚本与结果见 `.tmp/main-reproduction-20261003/cpu_parity.py`、`cpu-parity.json`、`cpu-parity.log`。这不是完整真实数据训练验收。
- 原代码导出未修改；缺失依赖 evaluate / datasets / multiprocess / xxhash 仅安装到 `.tmp/main-reproduction-20261003/deps/`。Windows 启动时先导入 numpy 再导入 torch，以适配本机运行环境；未重建 main 的历史依赖版本。
- 定向 algorithm / process 回归 110 passed；最终 unit + integration c1 / c2 本地门 282 passed / 3 deselected（排除 external / slow / gpu），验证报告（临时证据，当前缺失，不随 Git 提供），原路径 `.tmp/test-results/20261002T213434Z_f4a444/report.md`，临时工作目录 `.tmp/main-final-715e6cdaac9c49faa00315c0761604d0/`。
- 完整 main 对照已进入 [main_reproduction 计划](https://github.com/diaoenmao/RPipe/blob/18cd76c/studies/main_reproduction/docs/PLAN.md)：复制已缓存的真实原始数据，32 个文件副本 SHA-256 一致；旧 main_base CIFAR10 目录读取被拒绝，改用已有 support_model_smoke 的可读缓存副本，不改目录权限。原 main 的完整 train Stats 以 batch250、dim1 计算，完整精度写入新 Study；两套完整 train / test 像素和标签逐项一致。
- 真实数据初始化预检8/8：当前 Factory 重建的初始参数与原 main 实际训练前存档相同，初始化后 Torch CPU RNG 相同；每个数据集的全部15000个训练采样索引一致。两套原默认60-step 配方各16/16执行成功，严格数值门4/8（linear / mlp 通过，CNN / ResNet18未通过）。原 main 子进程计时合计205.609s，Rpipe launch 外部计时125.277s，计时边界不同，不作为算法加速结论。
- 隔离重复原代码3条训练，发现同一源码、数据、seed 在默认 benchmark 配方下也出现参数与指标分歧。额外完整确定性控制：两套同改 deterministic=true / cudnn.deterministic=true / benchmark=false / CUBLAS_WORKSPACE_CONFIG=:4096:8，其余条件和门限不变；各16/16成功、数值门8/8，step30/60参数、test Loss、test Accuracy 差值为0，scheduler 一致、best 均 step60、独立 eval 与 best 一致。训练 Loss 最大差6.66e-16、训练 Accuracy 最大差1.42e-14个百分点，为浮点均值累加顺序差异。原默认4/8判定保留，未把控制条件改成原配方结论。所有数字、来源与图见 [main_reproduction报告](https://github.com/diaoenmao/RPipe/blob/18cd76c/studies/main_reproduction/docs/STUDY_REPORT.md)。
- README 历史图核对：两个 PNG 与最后修改它们的`4ccb28d`（2024-01-08）逐字节相同。该提交候选配方为80000 steps / eval200 / lr0.01 / batch250 / test250 / 4 seeds、梯度裁剪1、CNN 含 BN、torchvision CPU 增强与旧常量统计；与当前 main 明显不同。旧图 Epoch 轴为评测段序号，不代表 MNIST 完整遍历400轮。没有历史原始运行配置或结果，旧提交配方仍为候选依据。已向用户说明最终对照选择，未启动长实验或擅自更改模型；当前 main 源码计算对齐与历史图复现分别记录。
- 核对2402个可读历史 Study 文件 size/mtime 与当前源码/声明94文件哈希，均未改变；main_base 的 CIFAR10解压目录无法枚举，未访问或修改。生产代码本轮未新增改动，沿用已通过的282项本地回归。
- 已取消的 1800-step 扩展、自动环境留档功能及独立 eval 数值诊断项目仍不作为本轮任务。此次手工记录复现环境只服务实际对照。

### 5. 后续历史审计与指标修复

- 历史 CPU 模型探针核对8个组合：linear / mlp / resnet18初始化参数与固定输入 logits 一致；旧 CNN 有4层 BN、当前无 BN，两个数据集 logits 最大差0.64609 / 0.60425。只扩大预算不能复现历史结构。
- 直接执行旧提交的 Metric.compare，Accuracy 序列95→90→92最终保存92%的第三个 checkpoint，覆盖95%的全程最佳；原因是旧方法即使没有改善也更新比较基准。当前 main 与 Rpipe 均保留95%的第一个 checkpoint。历史独立 eval 的权重来源必须考虑这个旧缺陷，归档代码不修改；见 [历史审计](https://github.com/diaoenmao/RPipe/blob/18cd76c/studies/main_reproduction/docs/HISTORICAL_AUDIT.md)。
- 当前工作树 B-017已修复：`accuracy_value(topk=2)`原本将3个样本展开为6个索引并抛 RuntimeError，现在保留候选轴，命中任一候选即计为该样本正确。没有增加配置入口；默认 top1保留。最小回归修复前1 failed，`.tmp/metric-red-ab3d83106e99449eacd5d33ced97a9e1/`。
- 修复后本地门283 passed / 3 deselected，测试报告（临时证据，当前缺失，不随 Git 提供），原路径 `.tmp/test-results/20261002T220008Z_f2b7db/report.md`；默认 top1 CPU 原代码探针8/8。当前源码以新临时 Study 重跑完整8 train + 8 eval，16/16成功、对照原 main 既有确定性存档数值门8/8，step30/60参数与 test 指标差值0，独立 eval 与 best 一致，launch86.401s。源码和逐格证据见 [报告后续章节](https://github.com/diaoenmao/RPipe/blob/18cd76c/studies/main_reproduction/docs/STUDY_REPORT.md)。此前 SOURCE_MANIFEST 和所有原始 Run 保留，未覆盖旧证据。

### 6. 历史候选短接入对照

- 现有 DataRegistry / ModelRegistry 的临时 builder 复用归档 dataset/model，样本 dict 转 tuple，`f(x)`返回 logits；旧 BN、CPU 增强/常量归一化和 clip1均保留，无生产源码或新配置改动。
- MNIST / CIFAR10 × linear / mlp / cnn / resnet18，seed0、batch250/test250、SGD lr0.01、cosine T_max80000，两边仅运行4个 optimizer steps，在 step2/4各评测完整10000张 test。数值门8/8通过；初始化/RNG、1000个实际训练索引、增强后输入哈希、scheduler 相同，参数与 buffer 差值0。train 指标差值0；test Loss 最大差8.88e-16、Accuracy 最大差3.55e-15个百分点，正确样本数一致。
- 16个原始数据文件复制 SHA256一致，旧 dataset 自行生成 pickle 缓存；原归档与既有证据不改写。进程内数据复制+对照36.841s，启动导入不计。91个当前源文件与2402个可读旧 Study 文件保护复核通过；归档历史36文件、当前 main40文件逐字节相同。正式证据见 [HISTORICAL_BRIDGE](https://github.com/diaoenmao/RPipe/blob/18cd76c/studies/main_reproduction/docs/HISTORICAL_BRIDGE.json) 与 [历史审计](https://github.com/diaoenmao/RPipe/blob/18cd76c/studies/main_reproduction/docs/HISTORICAL_AUDIT.md)。
- 这支持继续复用 Registry 承载候选旧配方，不先给生产 CNN 增加 BN 开关。eval2改变了随机数消费，这不是历史完整运行的前4步；未执行 eval200/80000-step/4-seed，未验证历史独立 eval 或复现旧 best 缺陷。该短对照未发现新的生产 bug。最终对照选择及长实验仍待明确，已拒绝的两项功能仍不做。

### 7. 评测预算修复 B-018

发现`eval_num_steps=0`仍先评测第一批，然后才判断上限，产生与预算不符的 Loss/Accuracy。先在 structure / STUDY_GUIDE 明确：正整数限 batch，缺省/负数表示完整 test，零预算无法产生有效指标，应报错。共享`eval_test_split`在迭代前拒绝非正的显式上限，训练中 test 与独立 eval 共同覆盖；配置负数经已有`eval_batch_limit`转为 None，完整 test 配方仍有效。

最小回归修复前1 failed，失败报告（临时证据，当前缺失，不随 Git 提供），原路径 `.tmp/test-results/20261002T221547Z_b97619/report.md`；首次测试另有全局 pytest 缓存写入警告，最终验证改用独立临时 cache。修复后 unit + integration、c1/c2、排除 external/slow/gpu：**286 passed / 3 deselected**，无警告，通过报告（临时证据，当前缺失，不随 Git 提供），原路径 `.tmp/test-results/20261002T221619Z_6a712b/report.md`。回归直接验证零预算报错且不访问数据，并验证完整 split 及1/2批上限。

main 原代码 CPU 探针以新输出目录重跑8/8，step2/4参数、train/test Loss 与 Accuracy 差值0、scheduler 一致。证据见 [CPU_PARITY_AFTER_B018](https://github.com/diaoenmao/RPipe/blob/18cd76c/studies/main_reproduction/docs/CPU_PARITY_AFTER_B018.json)；脚本`.tmp/main-reproduction-20261003/cpu_parity_after_B018.py`，目录`.tmp/main-reproduction-20261003/cpu-after-B018/`。这是4-step 固定合成输入对照，该修复阶段先完成 CPU 验证，后续完整 GPU 复核见下节。91个源文件中仅 eval_hook.py 相对 B-017快照改变，[新快照](https://github.com/diaoenmao/RPipe/blob/18cd76c/studies/main_reproduction/docs/SOURCE_AFTER_B018_MANIFEST.json)保留；2402个可读旧 Study 文件及两份归档逐字节/size/mtime 保护核对通过。B-018已从开放缺陷移除，下个编号 B-019；已拒绝的自动留档和独立 eval 诊断项目未实施。

### 8. 当前源码完整收口复核

新的临时 Study/version `main-98648f3-deterministic-after-B018`运行8 train + 8 eval，60-step/eval30、完整10000张 test、原固定 Stats 及 CUDA 确定性控制。当前16/16执行成功，与已有原 main 确定性存档比较数值门8/8：step30/60全部参数和 test Loss/Accuracy 差值0、scheduler 与 best 步数一致、独立 eval 与本实现 best 一致；train Loss/Accuracy 仅约6.66e-16/1.42e-14的浮点均值累加差异。launch 外部墙钟86.176s，make0.410s，无 train 恢复或 retry，原代码存档不重复执行。

逐格证据见 [DETERMINISTIC_COMPARISON_AFTER_B018](https://github.com/diaoenmao/RPipe/blob/18cd76c/studies/main_reproduction/docs/DETERMINISTIC_COMPARISON_AFTER_B018.json)，[当前学习曲线](https://github.com/diaoenmao/RPipe/blob/18cd76c/studies/main_reproduction/docs/figures/learning_curves_after_B018.png)已目检，test 坐标30/60、n=1。91个源文件与 B-018快照相同，2402个可读旧 Study 文件及两份归档保护核对通过；生产源码未再改动，沿用286项本地回归。历史候选长实验没有执行，原默认严格门4/8不改写。

### 9. 历史依据与实验工作量核对

本地非 shallow 仓库的13个可达 refs 包含150个提交；main 祖先91个，图最后更新提交的祖先39个。旧历史曾跟踪61个路径，按原始结果/权重后缀及 output/data 路径检查，只找到`src/config.yml`，没有找回运行指标或权重；未 fetch 远端或搜索外部存储，不能宣称外部记录不存在。两张 PNG 的 Software 字段均为 Matplotlib3.7.1，而旧 requirements 固定3.7.0，依赖文件不能证明实际绘图环境；这不提供训练 Torch 版本的证据。

在隔离目录执行归档 make.py 的生成阶段，4seed、单 GPU/round1，32条 train + 32条 test 命令及 wait 数通过核对，生成命令未执行。一套80000-step 候选需256万次 optimizer 更新、6.4亿 train 样本处理、1.28亿曲线 test 样本处理，双实现对照翻倍；不按短探针混合墙钟猜时长。正式证据见 [历史搜索与工作量](https://github.com/diaoenmao/RPipe/blob/18cd76c/studies/main_reproduction/docs/HISTORICAL_EVIDENCE_SEARCH.json)、[历史审计](https://github.com/diaoenmao/RPipe/blob/18cd76c/studies/main_reproduction/docs/HISTORICAL_AUDIT.md)。

已再次以明确选项询问最终按当前 main 源码的60-step 确定性对照验收，还是继续 README 历史图路线；答案未到。后者的条件下一步是先规划200-step/eval200前缀对照及分项计时，再制定完整预算；当前不执行依赖目标选择的实验。此次只有证据搜索、调度草稿与文档改动，没有新生产代码或 bug 修复。

### 10. Codex 接续复核

回到 Codex 后，按当前工作树重新核对交接：HEAD 仍为 `d938874`，后续代码、文档和三份 Study 尚未提交。当前 91 个生产源文件的 SHA-256 全部与 `SOURCE_AFTER_B018_MANIFEST.json` 相同，最近的完整 GPU 对照证据仍对应当前源码。

重新执行 unit + integration、c1 / c2、排除 external / slow / gpu 的本地门：**286 passed / 3 deselected**，无警告，本次报告（临时证据，当前缺失，不随 Git 提供），原路径 `.tmp/test-results/20261003T110214Z_95a976/report.md`。首次调用未创建临时父目录，导致 pytest setup 错误；补建父目录后原测试集合通过，未修改生产代码或测试。

复用原 `compare_after_B018.py` 的比较逻辑，直接读取两套已保存的 step30 / 60、best checkpoint、scheduler、独立 eval 结果及日志，将输出写入新的临时目录。重算 **8/8 通过**，逐格数值与正式 `DETERMINISTIC_COMPARISON_AFTER_B018.json` 完全一致；本次是存档证据复核，没有重新训练或覆盖历史结果。脚本及新输出在 `.tmp/codex-handoff-2d8d26a628d5420290defad76e254941/`。

接续时曾需明确最终验收对象：当前 main 源码的计算对照，或 README 历史图复现。连续三个目标轮次未收到选择，2026-10-03 曾将 goal 标记为 blocked（等待验收范围决定），不是完成；当时只核对源码哈希与既有报告，没有重跑测试或训练。用户随后选择历史超参路线并限制本轮只做探针，该阻塞已解除；原默认严格门4/8、确定性控制8/8及历史图未复现的结论保留。

### 11. 完成判定

收口审计按实际目标区分交付与未完成范围：

| **要求** | **当前证据与判定** |
|---|---|
| 查找并修复影响执行/验收的 bug | B-014–B-018修复、最小反例与回归已完成；最新本地门286 passed / 3 deselected |
| 阶段性 brainstorm 与开发 summary | 两份文档已同步到当前源码复核、历史证据搜索及本轮200-step 前缀探针 |
| 用户选定3.1曲线进度 | 已落地，真实当前矩阵按 optimizer step 记录；3.2/3.3保持不做 |
| 固定当前 main 计算对照 | 当前源码16次执行成功，确定性门8/8；原默认严格门4/8仍未通过 |
| README 历史图 | 候选配方与短接入验证已核对；原运行记录未找回，长实验未执行，未复现 |
| 本轮验收对象 | 用户已选历史图对应超参，并限制只做200-step/eval200前缀探针；不执行80000-step/4-seed 长实验 |
| 历史超参前缀探针 | 8格执行结束并通过，独立读取 checkpoint/optimizer/tracker 复核8/8，分项计时和正式报告已保存；没有新生产 bug |

本轮用户要求的接续理解、历史超参前缀探针、实际开放 bug 核对及报告已完成。长实验不属于用户最后确认的本轮范围；未来继续历史曲线或长期收敛需另行决定，不自动执行。

此前当前 main 对照逐格核查数据、结构、初始化、采样/增强、更新次数、scheduler、step30/60、best 和独立 eval；原默认严格门未通过，确定性控制8/8，历史 README 图未复现。上次等待范围决定的阻塞已解除。本轮按用户明确的探针范围完成8格200-step/eval200前缀对照、分项计时、报告和实际当前 bug 处理，不要求长实验或完整历史曲线通过。

### 12. Git 交付与异机测试基线

2026-10-03 用户确认将现有本地成果推送至工作分支，再合并至开发分支。远端开发分支实际名称为 `dev`，集成路径为 `feat/main-base → dev`，具体交付提交及合并状态以 Git / PR 为准。

- 交付内容：B-014～B-018 修复、曲线进度与恢复处理、对应回归测试、三份 Study 的正式声明与报告，以及更新后的 `AGENTS.md` 和项目文档
- 发布前本地回归：unit + integration、c1/c2，排除 external / slow / gpu，**286 passed / 3 deselected**。首次执行仅有旧 pytest cache 写入权限警告，改用 `.tmp/` 内独立缓存后复测无警告，不涉及生产代码修改
- 不随 Git 交付：`.tmp/`、数据缓存、checkpoint、生成调度脚本及 Study 运行产物。正式报告中指向这些本地证据的链接不会随 clone 提供
- 历史配方边界：200-step 探针的临时 Registry 接入与归档脚本仍在 `.tmp/`，尚未整理为正式长实验入口。拉取本次代码不能直接替代完整历史配方交付
- 远程验收：目标环境的安装、GPU、正式 Study 执行与中断恢复尚未验证，本次 Git 集成不等于异机测试通过，也没有启动长实验

# 三、后续合并 main

本阶段版本为 v0.2.0，将正式复跑入口、声明、报告、图和仓库整理改动一并交付。长期分支仅保留 `main` 与 `dev`，同步本阶段内容；实际标签、Release 与分支状态以远端为准。

发布前核对的旧 main 是 dev 的祖先，没有 main 独有提交。先通过受保护 dev 的 PR 与跨平台 CPU / 隔离安装门，再将通过验收的成果进入 main 并发布阶段版本；临时发布和已合并功能分支在确认提交已保留后清理。

第二节、科学报告和 [仓库整理验收](https://github.com/diaoenmao/RPipe/blob/71143ab/docs/REPOSITORY_CLEANUP.md) 中的 refs 与“未合并”描述保留其记录时点；它们不是发布后 main 的动态引用。原 source SHA、数值与失败判定不随发布改写。

本次是公共接口和产物格式的迁移：旧脚本调用需要更新，旧 checkpoint 不能直接作为新 Run resume；旧提交和原产物保留用于读取与回退。具体命令、配置、Stats 精度、std 及恢复边界见 [MAIN_MIGRATION.md](https://github.com/diaoenmao/RPipe/blob/71143ab/docs/MAIN_MIGRATION.md)。该文件已在 2026-10-10 从当前树删除，链接指向删除前的提交。
