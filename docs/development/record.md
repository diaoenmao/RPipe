# 一、摘要

RPipe 已从旧 main 的独立训练脚本演进为可安装的研究执行库。本节按内容归纳当前 dev 与本机新增成果的累计变化；具体实施日期、失败和阶段边界保留在第二节。

## （一）改前与改后对比

### 1. 结构与公共接口

本表以旧 main `98648f3` 对比当前 dev。旧版已经有 YAML、命令参数、调度、聚合和 checkpoint；变化在于组织方式、公共接口与产物合同。

| **范围** | **改前** | **改后** | **依据** |
| --- | --- | --- | --- |
| 调用方式 | 从 `src/` 运行 `train_model.py`、`test_model.py`、`make.py` 等独立脚本，共享全局 `cfg` | 安装 `rpipe` 包，以统一 CLI 执行 make / launch / process，源码集中到 `src/rpipe/` | [代码结构](../code/code.md) |
| 配置与编排 | `config.yml` 配合 `control_name` 解析；按 seed 与配置拼 tag，make 生成 Bash / wait 调度 | Study YAML 声明 axes / seeds，展开 Experiment × seed 的 Run；配置内容生成 Run ID，launch 执行固定阶段链 | [概念](../code/concept.md)、[Study 指南](../../studies/README.md) |
| 运行产物 | 配置、脚本、实验、结果和图分别落到 `output/config`、`script`、`exp/<tag>`、`result`、`vis`；checkpoint / best 为旧分件格式 | 按 Study / Run 统一落盘 config、result、tracker、log 和新 checkpoint；index / process 支持 status / logs / report | [目录约定](../code/layout.md) |

### 2. 计算与恢复修复

本表的“改前”是 dev 中对应问题修复前的行为；这些问题不能统称为旧 main 的缺陷。

| **范围** | **改前** | **改后** | **依据** |
| --- | --- | --- | --- |
| 来源与依赖（B-014 / B-015） | 未知 name / source 可能落入默认或占位路径；缺正式输入仍可能产出结果；SVHN 所需 scipy 未列入基础依赖 | Factory 严格匹配，未知/空 source 报错，省略时保留正常默认；占位必须显式 stub，缺输入失败落盘；声明并核验 scipy | 第二节 §1、[数据验收](https://github.com/diaoenmao/RPipe/blob/71143ab/studies/support_data_smoke/docs/STUDY_REPORT.md) |
| train 均值（B-016） | step 模式跨评测段累计，反例最后一段应约10却报告约5 | 每个评测周期收口并重置 train 段，同一反例报告约10；空段不覆盖最后有效均值 | 第二节 §4 |
| topk 计数（B-017） | topk=2 将3个样本的候选展平为6个索引，触发形状/计数错误 | 保留候选轴，每个样本命中任一候选只计一次；修复既有函数，未新增 CLI topk 配置 | 第二节 §5 |
| 零评测预算（B-018） | `eval_num_steps=0` 仍读取并评测第一批 | 读取数据前报错；正数限批、缺省/负数完整 test 的语义保留 | 第二节 §7 |
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
| 导航与清理 | Study 入口分散，设计文档与开发记录同在 `docs/` 根目录；正式 Study 与测试用声明混放 | 设计文档在 `docs/code/`，开发记录在 `docs/development/`；Study 指南是 `studies/README.md`；`studies/` 只留 `main_exp`、`main_probe`（原 `main_historical`、`main_reproduction`，待按新流程重跑）、`_template`，测试声明在 `tests/_data/studies/` | 第二节（一）、[Study 使用指南](../../studies/README.md) |
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

## 2026-10-13

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
