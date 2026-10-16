> 当前（2026-10-17）：B124–B127/#170–#173同分支字段/事件身份里程碑本机完成，尚未推送/开PR；接受dev仍bef7de9。
> 最新：合法消息/提示的行边界、原生事件字段定位、标准traceback过滤四项统一收敛，详§209–213，独立红证据保留。
> 验证：相邻420 passed；core2009 passed/1既有skip、CPU Flow354 passed，新增78（21 unit+57 integration）全执行/0新skip。
> 下一步：完整主题一个PR，最终head/实际dev全部实际CI验收后关170–173、清已合主题；无决策阻塞。
> 保护：清理基线161计数2、remote44；旧日志不重写、legacy不猜，main/tag/GPU/数据/结果/文件与产物布局不动。

> 当前：PR169双侧全部实际CI验收，devbef7de9；#167/#168已关闭，已合主题本地/remote均清理。
> 最新：末条Runner失败留证与半行日志独立error读回统一完成；正常日志/原字节/追加权限保持，见§206–208。
> 验证：相邻298、core1988 passed/1既有skip、CPU Flow297 passed；32新全执行，最终head与实际dev所有矩阵/聚合成功。
> 下一步：开放Issue快照为空，codex/correctness-review-next继续三类复查，本地主题积累、完整里程碑再PR，无决策阻塞。
> 保护：已删codex/correctness-next-milestone本地+remote，累计remote44，基线161计数2；main/tag/GPU/数据/结果/布局不动。

> 当前：B122/#167、B123/#168同主题日志失败/中断读回里程碑，本机完整检查完成，准备统一一个PR。
> 最新：末条IO拒绝绕过留证与旧半行吞新ERROR身份均复现；先独立红回归再761700f/97a1443最小修，详§204–207。
> 验证：相邻298 passed；core1988 passed/1既有skip、CPU Flow297 passed，32新全执行（14 unit+18 integration）/0新skip。
> 下一步：同分支统一PR，验最终head与实际dev所有CI后关167/168并清已合主题；无决策阻塞。
> 保护：接受dev仍70a8750/PR165，清理基线161计数1、remote43；不合main/tag/GPU/测速，不删数据/结果、不改布局。

> 当前：B122/#167已本机修复并完整验证，尚未推送/开PR；接受dev仍PR165/70a8750。
> 最新：末条Runner日志绕过失败留证；先红12项8失败/4控制，再761700f只缩进原语句，见§204–205。
> 验证：Runner49 passed；core1974 passed/1既有skip、CPU Flow291 passed，新12全执行/0skip；源码仅两行diff。
> 下一步：同一codex/correctness-next-milestone继续三类及相邻失败边界复查，相关主题完整里程碑再PR；167仍OPEN，无决策阻塞。
> 保护：清理基线161计数1、累计remote43；未合分支/数据/结果保留，无GPU/测速/Study/main/tag或布局变更。

> 当前：PR165已双侧全实际CI验收，dev70a8750；#163/#164/#166已关闭，已合主题本地/remote均清理。
> 最新：B119/B120/B121统一加载隔离、dataclass身份、动态导入与checkpoint兼容，详§202–203；原候选反例保留。
> 验证：core1974 passed/1既有skip、CPU Flow279 passed，新61全执行；最终head与实际dev所有矩阵/聚合均成功。
> 下一步：开放Issue快照为空，codex/correctness-next-milestone继续三类复查；本地主题积累、完整里程碑再PR，无决策阻塞。
> 保护：已删codex/study-loader-isolation本地+remote，累计remote43，清理基线161计数1；main/tag/GPU/数据/布局未动。

> 当前：B119/#163、B120/#164、B121/#166统一PR165，动态导入候选回归已补修，本机完整检查完成。
> 最新：原dev动态helper返回37、原候选ModuleNotFoundError；先红10项8失败/2控制，再9166cbd修复，见§202。
> 验证：当前core1974 passed/1既有skip，CPU Flow279 passed，新增61全执行、0新skip；目标67含6既有节点。
> 下一步：推同一PR新head验全部实际CI，再正常合dev并验实际dev；三Issue仍OPEN，无决策阻塞。
> 保护：清理基线161计数0，活动未合分支保留；不合main/打tag/跑GPU或测速，不删数据/结果、不改布局。

> 当前：#163/#164同分支本机修复与完整回归完成，尚未推送PR；已验收dev仍PR161/8480d79。
> 最新：Study私有导入生命周期、dataclass模块身份及旧/迁移checkpoint兼容统一修；原候选晚导入/运行时误接管反例均保留，见§198–201。
> 验证：当前core1964 passed/1既有skip，CPU Flow279 passed，新增51（45 unit+6 integration）全执行、0新skip；4源码范围已核对。
> 下一步：按20:56在原分支完整里程碑合批、少量PR；最终head/实际devCI未执行，163/164仍OPEN，无外部决策阻塞。
> 保护：清理基线161/计数0，活动未合分支保留；无GPU/测速/下载/Study/main/tag/数据删除或文件架构/产物布局变更。

> 当前：PR161已双侧验收/关160与162/清已合主题/第5批盘点；本地继续B119/#163、B120/#164加载隔离。
> 最新：31节点候选通过后，4a53165补晚导入2红均失败，确认候选回归；当前33节点未验收，详§197。
> 验证：此前core1950 passed/1既有skip、Flow273 passed不含新增2红，不能替代当前验收；所有反例保留。
> 下一步：修正注册函数晚导入的私有模块生命周期，再跑当前完整集合；仅本地合批，不立即推新PR，无外部决策阻塞。
> 保护：活动未合study-loader-isolation保留，清理基线161；无GPU/测速/下载/Study/main/tag/数据删除或布局变更。

> 当前：PR161已双侧全实际CI验收，160/162关闭、已合主题本地/remote清理，第5批盘点完成，新基线161。
> 最新：B119/#163顺序helper串用，18红14失败/4控制；B120/#164合法dataclass加载失败，8红4失败/4控制，见§192–195。
> 验证：PR161合批core1919 passed/1既有skip、Flow273 passed；新加载回归仅红执行，生产修改尚未开始。
> 下一步：按§195实现并核对加载隔离，相关B项在同一本地分支积累，遵循20:56少量PR规则，不立即开PR。
> 保护：活动未合study-loader-isolation保留；无GPU/测速/下载/Study/main/tag/数据或布局变化，无外部决策阻塞。

> 当前：PR161统一收尾B117/#160与B118/#162临时进程状态并发修复，唯一未合codex/correctness-followup。
> 最新：两次独立正式红8项4失败/4控制、9项4失败/5控制；fffd872/0aaae0b后两处相邻27项全绿。
> 验证：合批core1919 passed/1既有skip、CPU Flow273 passed，新17全执行；详§188–191，原证据含拒绝均保留。
> 下一步：最终head与实际dev全部实际CI验收后关两Issue/清主题/第5批盘点；后续按20:56规则攒相关B项里程碑，少量PR。
> 边界：无下载/GPU/测速/正式Study/main/tag/数据删除或布局变化；当前无代码阻塞。

> 当前：B117/#160本机修复完成，唯一未合codex/correctness-followup；PR159双侧验收/清理已完成。
> 最新：并发端点串用/残留；8正式红4失败/4控制→fffd872后13新旧端点测试全绿，见§188–189。
> 验证：core1910 passed/1既有skip，CPU Flow273 passed，新8全执行；真实Factory双根原版失败/当前正确。
> 下一步：完整并发里程碑PR→dev，分别验最终head/实际devCI；无代码阻塞，验收后清已合主题。
> 保护：无网络下载/GPU/测速/正式Study/main/tag/数据删除或布局变化；原失败与收集拒绝保留。

> 当前：PR159已双侧全部实际CI验收，Issue158已关闭，已合readout-time-order本地/remote均已删除。
> 最新：28新回归全执行；core1902 passed/1既有skip，CPU Flow273 passed，详见§184–187。
> 验收：实际dev ab5c825，Package38901125267/Unit38901125320全部实际成功；周期基线149累计4。
> 下一步：继续三类正确性复查，验收文档本地积累随下一完整里程碑合入，不单开小文档PR；无代码阻塞。
> 边界：GPU准入新规见§183，仅静态复核；无GPU探测/任务干预、测速、main/tag或布局变更。

> 当前：PR157双侧全部实际CI验收，Issue156已关闭，已合主题本地/remote已清；周期149累计3。
> 最新：B116/#158跨偏移/DST/UTC日志顺序反例；28正式红8失败/20控制→ba665de后全绿。
> 验证：core1902 passed/1既有平台skip，CPU Flow273 passed；28新节点全执行，见§184–186。
> 下一步：完整只读时序里程碑按频率规则流转PR→dev，最终head与实际devCI分别验收；无代码阻塞。
> 保护：旧时间fallback/原日志字节与格式、布局/数值/科学门保持；无GPU/测速/Study/main/tag。

> 当前：PR135实际dev已全CI验收、Issue134关闭，工作分支本地/remote普通删。
> 最新：B104/Issue136同Study路径别名翻转多根锁顺序，真实并发循环已复现。
> 修复：四回归2红/2控制→设计→共享物理身份排序；shared76及双并发空目录迁移通过。
> 下一步：core/CPU Flow、最终head与实际devCI；无代码阻塞，见record§159–160。
> 保护：无GPU/测速/正式Study、main/tag、数据删除或产物布局变更；以下旧状态保留。

> 当前：PR133实际dev已全CI验收、Issue132关闭，已删train-failure-finalizer本地/remote各1条。
> 最新：B103/Issue134路径别名锁自争用，四红42c8b5b→设计afd8ea0→最小修复41ce552。
> 验证：共享72/core1697 passed，1既有平台skip；证据见record§157–158与.tmp/audit103。
> 下一步：CPU Flow、最终head及实际dev全部实际CI；当前无代码阻塞。
> 保护：无main/tag/GPU/测速、无数据删除或产物布局修改；以下旧状态保留为历史。

> 当前：B102/#132已红1e8c5d1→设计75e9885→最小生产384bf2c完成本机；待PR最终/实际dev CI。
> 最新：22新节点全执行/通过；core1693+1既有POSIX skip、CPU Flow207通过；成功路径错误仍传播。
> 补强：a758de4只加3条实际tracker staging清理断言，冻结旧函数20 failed/2控制passed；当前22绿。
> 下一步：PR→dev，最终head及实际dev全部实际CI后关闭132、普通清本地/remote；无阻塞。
> 保护：原训练/数值/周期flush、checkpoint/RNG、schema/layout保持；无main/tag/GPU/测速/正式Study。

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
| Git 来源失败（B-061，已合 dev） | status 非零退出仍被写成 dirty 布尔值 | 失败返回空 Git 元数据；11 项新增边界、本机相关 87 项通过，已由 PR #46 集成 | 第二节 2026-10-14 §48 |
| 共享引用并发（B-062，已合 dev） | 同字节引用发布临时名可被同 tick 写者共用 | 每个 staging 使用 UUID；6 项新增边界、本机相关 93 项通过，已由 PR #46 集成 | 第二节 2026-10-15 §49 |
| 可选 Git 元数据（B-063，已合 dev） | 文件读取/格式错误可令 pytest 配置失败或误记 SHA | 错误归为可选 null，保留快速路径和 CI 优先级；本机相关 108 项通过，已由 PR #46 集成 | 第二节 2026-10-15 §50 |
| 长文件名 staging（B-064，已合 dev） | 文本/迁移临时名追加完整目标名，合法245字符名失败 | 两处短 UUID-only 名；4 项真实反例修复，相关187通过、5未选中，已由 PR #46 集成 | 第二节 2026-10-15 §51 |
| 单次来源去重（B-065，暂缓） | 历史草案新增读取次数去重 | 已按最终只做正确性的要求撤回优化，历史证据保留 | 第二节 2026-10-15 §57 |
| 可选活动读取（B-068，已合 dev） | 非法UTF-8或存在性权限错误使status异常，遮蔽正式结果 | 忽略不可用提示、保留index退出合同；新13项与status联合34通过，已由 PR #46 集成 | 第二节 2026-10-15 §56 |
| 冻结采集错误（B-067，已合 dev） | 文件缺失/拒读或坏JSON异常可覆盖旧成功，非法mapping还会抛AttributeError | 读取时校验mapping，冻结入口转现有拒绝异常并保留cause；34项定向及214相邻检查通过，已由 PR #46 集成 | 第二节 2026-10-15 §54 |
| 来源键遮蔽（B-066，已合 dev） | Study同键可覆盖库hash，真实库修改仍被freeze放行 | 保留库键域并验证物理身份，14边界和202相邻core通过，已由 PR #46 集成 | 第二节 2026-10-15 §53 |
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
| 准备与执行 | 临时工作区、脚本自建当前侧对象 | Run assets 内隔离准备；成对执行复用 Flow 的 Data / Model / System / Tracker，恢复初始化 RNG，原版先、当前版后 | [计划](<../research/研究计划 RPipe.md#study-main-probe>) |
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

## 2026-10-16


### 139. PR119实际dev验收、设计状态交付与第二十一条远程清理

PR119实际dev `4d12719d7eba4f94fa279d75614e3a108a38fe10` 的 Unit38817527562、Package38817527778全部实际矩阵/聚合成功；DEV_CHECKED SHA256 `fb60756f81845f584dc043e43f33594887fc35c6da5d14f528222870647c3287`。已交付设计状态更新：原三类PR52/54/55和B086 PR98、B088 PR102均标已实施并列实际merge记录，保留红历史/标题锚点/相对链接与第七节不实施限制，不重新等待批准。B096修复同批全绿，Issue118留至下版。

核对祖先/无开放PR/实际CI后删除结束remote codex/scheduling-query-parse，本地tip与当前活动codex/optional-state-stat保留，本目标累计21条结束remote已清；本机证明.tmp/audit096不随clone。不合main/强推/tag或删数据，main仍v0.4.1。

### 138. PR119最终全实际CI后合dev，设计状态更新已集成

最终head `fdf4f8fb30a7b42fb15809206a22c2798d4a21a8` 的 Unit38816839360、Package38816839047、Branch38816839074全部实际矩阵/聚合成功；PR_CHECKED SHA256 `69da6b1a97a063212dbce7c93e6e9ff5b3dd88e01f5edde1426dbc76156bdd27`。PR119普通merge为dev `4d12719d7eba4f94fa279d75614e3a108a38fe10`，包含已审定的设计状态订正文档与B096修复；实际devUnit/Package观察中，全绿后才清结束remote。当前新B097基于该merge，仅一条未合分支，不再等设计批准，不把待验CI误记为完成。

### 137. 可选state完整I/O边界与合同矩阵收敛（B-097 / #120）

2026-10-16：沿可选读取合同核对metadata/read/decode是否同一边界。_state_history的is_file在原try外；对实际load_curve_series/load_tracker_history、两种primary状态，在该state的Path.stat点注入PermissionError，四场景均整体抛错且原文件/peer字节保持。明确这是目标方法点的文件系统故障注入，未改系统ACL/权限，不宣称原生OS拒绝复现。原REPRO/栈/原输入在.tmp/audit097，不随clone。

先Issue120，764a613先红。复用既有test_optional_state_keeps_io_and_programming_error_boundary，不新增平行测试函数：原2项扩成stat/read×Permission/EIO/missing/RuntimeError×curve/history×primary，共32合同实例，净增30。矩阵8 failed/24控制（只有stat的Permission/EIO失败）；全文件69项8 failed/61 passed，无error/skip。read侧原回退、FileNotFound空回退与RuntimeError同对象传播均有正控制；输入核对在故障上下文退出后执行，不因注入遮蔽证据检查。旧2项函数原文另保留，未删除失败样本或挑选通过。

设计先更新structure：仅把原_state_history存在性检查放入既有try，沿原OSError/ValueError/RecursionError/UnicodeError/TypeError边界回退；不新增异常吞咽、helper、文件或架构依赖。普通RuntimeError仍传播，既有JSONL读取/恢复指针、正常state fallback、科学聚合与所有产物位置/字节合同保持。先红后最小移动，再针对性及必要CPU门，不运行真实Study/GPU/性能测试。

06dfe86只把原is_file放入已有try，源码总行数与异常列表不变，SOURCE_BYTES_CHECKED证明其他字节不动；32矩阵全部转绿，process相关104 passed、core1482 passed/1既有POSIX executable mode bits skip、CPU Flow189 passed，无failure/error。净增30个节点，未把原2个控制重复算新增；RED/GREEN/CORE均执行全矩阵，普通RuntimeError同对象传播、缺失/OSError回退、有效primary/旧fallback和输入/peer字节保持。

LOCAL_CHECKED SHA256 `bb378811152d5e2c72641297bea30eaed2f2aed6c1d197071edb5b37da6a43fc`；REPRO/before.zip/旧2项函数/原日志/JUnit与红测试实际CRLF字节留.tmp/audit097，不随clone。测试blob LF与执行CRLF经精确重建核对，修后断言未改。明确只有目标stat/read故障注入，不改变系统ACL，也不冒充原生权限或科学验收；最终head/实际dev CI待验。

### 136. PR117实际dev验收与第二十条结束远程清理

PR117实际dev `20027afb231bc5637324e37310dd69e7ec09e1a7` 的 Unit38815660146、Package38815660139全部实际矩阵/聚合成功；DEV_CHECKED SHA256 `5663016e4300ce92377ebbee459dd17a0c7a6904d7282f4ae454a6bf8a7ba6bc`。核对祖先/无开放PR/实际CI后删除结束remote codex/optional-json-decode，原本地tip与当前活动codex/scheduling-query-parse保留，本目标累计20条结束remote已清。Issue116保留open至下版；证据.tmp/audit095不随clone，不重复合并/派发CI。main仍v0.4.1，未合main/tag/发布或删数据。

### 135. PR117最终head全实际CI后合dev，实际dev验收接续

最终head `000b39a233819fd84ca232dd964c9c7da792da3d` 的 Unit38814885801、Package38814885885、Branch38814885873全部实际矩阵/聚合成功；PR_CHECKED SHA256 `5caef99ad91857cc6098801d3689cc82388e9e6ed590ed78e696a53d443b9f89`。PR117普通merge为dev `20027afb231bc5637324e37310dd69e7ec09e1a7`，实际devUnit/Package在观察，待实际全绿后清结束remote。新B096基于该merge，只有一条未合codex/scheduling-query-parse；不把前批待验文字当当前待审，不重复CI/合并。

### 134. 调度查询的可恢复解析边界与设计状态订正（B-096 / #118）

2026-10-16：复查剩余解析入口，区分严格输入与原可恢复查询。八个实际文件场景中，深层jobs缓存抛RecursionError；result的list/null/bool根抛CorruptArtifactError、深层抛RecursionError，共5异常；原坏JSON语法保守False、有效succeeded True与有效空plan复用为3正常控制。原config/index/result/jobs及peer字节保全，未执行launch/进程/GPU或改解析限制。原样本/hash/栈/REPRO在.tmp/audit096，不随clone。

先Issue118，e3c3c12先20新回归13 failed/7控制，完整80项13 failed/67 passed、0error/skip。覆盖include_done开关、五种非object根/深层/语法/合法结果，严格load_result保持CorruptArtifactError/RecursionError/JSONDecodeError，run_succeeded保守False与合法成功True，缓存过滤保留坏记录/跳过成功peer，原文件不改；普通RuntimeError与非法调用参数的原控制保持。

设计先更新structure：复用_run_result的原空mapping与load_launch_plan的原None行为，具体捕获解析CorruptArtifactError/RecursionError，不改strict decoder或创建统一parser模块；调用路径校验/当前数字参数错误/合法成功与sibling新鲜度/重试策略/旧成功保留范围保持。

解析入口合同核对（代码分析，不代表所有故障组合已实测）：

| **入口组** | **分类与当前处理** |
| --- | --- |
| status/logs | 既有三态，PR115隔离解析递归；配置优先、单格/peer合同保留 |
| activity/process旧state/JSONL | 可选提示/回退/记录，PR117统一具体JSON值/递归解析异常 |
| schedule成功查询/保存计划 | 既有保守False/None，B096本批真实反例锁定再最小修复 |
| index/config/result/provenance、readout report/compare | 权威输入严格读取或身份校验，不把拒绝坏输入当需吞错的缺陷 |
| shared路径声明/不可变cache manifest | 配置/身份严格拒绝并保全诊断，不默默重建覆盖缓存 |
| checkpoint分片meta/tracker/logger | 恢复权威可变状态，不能忽略坏片继续假恢复 |
| make展开/data profile与来源marker | 声明/来源验证保持，不扩大本批到Study功能或科学输入迁移 |

监督助手本轮明确审定设计已实施：e44b94d单独文档提交修订artifact_consistency_design的过时状态，读取核对PR52/54/55/98/102均MERGED及实际merge commit，原三类实际CI见record§75。B086/B088红数据标明历史修复前，链接对应已合PR；六旧Issue与53/62已发布关闭的状态也核对，main只读确认v0.4.1。全部原标题锚点与12相对链接保持；第七节仍不实施，不因文档状态更新修改代码架构/布局或重新等设计批准。原文/差异/hash/链接核对在.tmp/audit096/DESIGN_STATUS_*，不随clone。

bbd33b4复用artifact已有CorruptArtifactError导出，仅修改schedule既有导入与两个异常元组；不加函数/模块，源码总行数不变，SOURCE_BYTES_CHECKED证明只有三处精确字节替换。make相关157 passed，core1452 passed/1既有POSIX executable mode bits skip，CPU Flow189 passed，无failure/error；20新节点在RED/GREEN/CORE完整执行。严格decoder原错、普通RuntimeError与调用参数错误、include_done、合法成功与peer过滤、原输入字节保持，不启动研究或改变旧成功保留/重试规则。

LOCAL_CHECKED SHA256 `d398e7c2029b52956ee3b1b031e71b3baea7576bfa4230bb30f269b10c2b8741`；原REPRO/before.zip/hash/RED/GREEN/CORE/FLOW、红测试实际CRLF字节与设计状态检查留.tmp/audit096，不随clone。Git测试blob的LF与执行CRLF经精确重建核对，断言修后未改。设计状态原锚点及12相对链接保持；最终head/实际devCI待验，旧CI不替代本批检查。

### 133. PR115实际dev验收与第十九条结束远程清理

PR115实际dev `b28f6a219ce967472a8d1f6c300995eb2c84e6cf` 的 Unit38813819579、Package38813819609全部实际矩阵/聚合成功；DEV_CHECKED SHA256 `b240c99b9e6a996cc1df1070c1a9853ed88a65ee02d843fbc0fb7ef4bc9ce243`。核对祖先、无开放PR、实际CI后删除结束remote codex/status-parser-boundary，原本地tip保留，当前唯一未合codex/optional-json-decode未动。本目标累计19条结束remote已清，不与历史§82九条重复计数。Issue114保留至下版，证据.tmp/audit094不随clone。main与v0.4.1不变，无tag/发布/数据删除。

### 132. 三处可选JSON解析失败的统一边界（B-095 / #116）

2026-10-16：在PR115观察期间按类复查可选JSON读取。5000位整数、约5049字节的原生文件超过本机默认4300位安全限制，json.loads抛ValueError而非JSONDecodeError：activity直接/CLI缺index、有效JSONL+坏state三个场景失败；坏JSONL同输入已有ValueError保护正常，4场景3失败/1控制。未改变系统限制或跑Study。进一步不足21KB、10000层的JSON在activity、state、JSONL三处真实RecursionError；1200层JSON本机正常的旧控制保持，不能混称任意深度均失败。原字节/hash/栈/REPRO在.tmp/audit095（不随clone）。

先开Issue116；从最新merge devb28f6a2建立唯一未合codex/optional-json-decode。abe24a3先12红6失败/6控制；新增同类递归复现后f1b0d5f扩为26新合同14 failed/12控制，全部78项14 failed/64 passed，无error/skip。首红日志/字节与扩展RED2分别保留，不覆盖失败；f64917b另在原源码上补1通过的JSONL普通RuntimeError传播合同，三处均要求不掩盖编程错误。原生整数限制若平台未启用或预算外则明确skip，不调高/修改全局限制，不冒充通过。

设计先写structure：复用原有三个JSON try，activity与旧state用ValueError代替过窄的JSONDecodeError，再同JSONL一起具体捕获RecursionError。坏提示忽略但不绕过missing-index错误；坏state不拖垮有效primary，坏JSONL行跳过且保留后续真实轨迹，正常旧回退保持。不建新模块/helper、不改strict loader、恢复指针、科学统计、schema或布局，不catch普通RuntimeError，不新增配置或提高安全上限。

843daa4只替换activity与aggregate中的三处异常列表，SOURCE_BYTES_CHECKED证明其他字节不变。针对性含原生process97 passed；core1432 passed/1既有平台skip；CPU Flow189 passed，0failure/error，27新合同均在core实际执行且无新增skip。保留正常提示/缺index错误、正常旧回退/有效primary、坏JSONL后的合法记录、输入与peer字节；普通RuntimeError三处均不吞，严格加载器/科学聚合/恢复指针不改。

LOCAL_CHECKED SHA256 `66a718582facb280fc78eaed9252419fe56d86c4c55ce2786f64e0064dfa2d39`；首RED64项6失败/58通过、扩展RED2的78项14失败/64通过、单独编程错误控制1通过以及GREEN/CORE/FLOW全部保留.tmp/audit095，不随clone。首RED实际CRLF字节已冻结；RED2 CRLF按f1b0d5f的Git LF blob重建并核对最终执行字节前缀，明确是重建，不伪装原始快照；最终仅追加f64917b在修前已通过的编程错误控制。最终工作head/实际dev CI待验，不以本机数字代替远端或科学验收。

### 131. PR113实际dev验收/第十八条远程清理与PR115接续

PR113实际dev `bf24574c3b3008b82cf65f702b92f055caad32a5` 的 Unit38812391082、Package38812390945全部实际矩阵/聚合成功；DEV_CHECKED SHA256 `01e6e0d66d747783fcde9afdc534439a9f0bf5f94b1772df4d2d2102649d64df`。核对祖先/无开放PR/实际CI后删除结束remote codex/test-evidence-unicode，本地tip保留，当前活动分支未动。本目标累计18条清理；Issue111留至下版，证据.tmp/audit093不随clone。

PR115最终head343012569af65ec2c35cc02c7c0b79448fc8590d的 Unit38813118717、Package38813118918、Branch38813119229全部实际矩阵/聚合成功，PR_CHECKED SHA256 `b22d5c60209619ac0c8f7655d3154ea1254c694a4585e23253dbed8745f32687`；普通merge devb28f6a219ce967472a8d1f6c300995eb2c84e6cf。实际dev Unit/Package仍在观察，全绿后才清结束remote，不能写已完成。新B095基于该merge，只有一条未合工作分支；无main/tag。

### 130. 可选配置解析递归错误的三态边界（B-094 / #114）

2026-10-16：PR113等待实际dev CI期间，只读扫描现有安全读取层，发现具体未捕获解析异常。在现有三Run fixture的单一config.yaml写入约2435字节、1200层未使用序列字段，yaml.safe_load真实抛RecursionError，list_runs整体中断，peer无法展示。三个同深度JSON（result/trajectory/state）在本机Python3.13正常，明确不算缺陷。原始输入/栈/hash/脚本在.tmp/audit094/repro-v2，不随clone。初次探针漏建tracker子目录而setup失败，其部分样本和HARNESS_SETUP_FAILURE保留，不当库证据。

先开Issue114；在PR113已merge后的最新devbf24574开唯一未合codex/status-parser-boundary。f5d0e01先提交九项真实配置回归（ok/bad/wait三个状态×deep/shallow/syntax），新3 failed/6控制、0error；严格load_config深层RecursionError与语法错误仍明示，普通RuntimeError既有传播合同保留。RED.xml与RED_CHECKED保留。

设计先更新structure：仅现有_read_optional异常元组具体加入RecursionError，把不可解析配置按既有unreadable→mode unknown/error降级，正常peer/status/counts、失败Run原error和所有磁盘字节保留。不catch普通RuntimeError、不提高递归上限、不改变严格加载器或执行者，不新建模块/helper/持久化格式；不从本机JSON可解析案例倒推其他平台边界。

74aaa49仅向_read_optional现有异常元组加入具体RecursionError；原文件before.zip和SOURCE_BYTES_CHECKED证明除这一处字节替换外无变化。九新节点全部转绿，readout500 passed、core1405 passed/1既有平台skip、CPU Flow189 passed，0failure/error。三个状态的计划行/计数、成功指标、失败原error、CLI行列及所有输入字节保全；strict load_config仍抛原解析错误，既有decoder普通RuntimeError传播测试通过。

LOCAL_CHECKED SHA256 `6a10bc4f96bd2fce6209776cac4bcde704e7bdbfa5a57088f65a185fc986034e`；RED90项3失败/87通过（新九项3失败/6控制），GREEN500/Core/Flow/JUnit及红测试执行CRLF字节均在.tmp/audit094，不随clone。先红→设计→fix→本机验收，最终head/实际dev CI尚待验，不把本机通过或对照计数当科学验收。

### 129. PR113最终head全实际CI后合dev，实际dev验收接续

最终head335e203cbbc0903468607b5f19a9f80a6709514c的 Unit38811546168、Package38811546209、Branch38811546234全部实际矩阵/聚合成功；PR_CHECKED SHA256 `1f24bc8917729d9c16d37d089bbcba0e23797ecc5514bec3709f986f01af23a6`。PR113普通merge为devbf24574c3b3008b82cf65f702b92f055caad32a5，实际dev的Unit/Package仍在验证，不能写已验收或清分支完成。

已合旧分支codex/test-evidence-unicode只等实际dev CI后清remote；新B094从该实际merge基线开始，仅一条未合工作分支。Issue111保留至下版，生产src在PR113未变。

### 128. 测试证据 Unicode 传输合同（B-093 / #111）

2026-10-16：Issue111 已先于修复登记；不重复登记计数格问题。独立复制原生入口证据 `.tmp/audit093/NATIVE_REPRO.json` 显示旧 writer 遇到 U+D800 时退出3、events0字节且 peer 未执行。旧源码 SHA256 `b732100cd80b2d9872789c316964147318cce1b1e4e32da38647f59ad3595456`。这些本机证据不随 clone，portable 回归在 tests 根的既有设施测试文件。

2b4c0d1 先提交红回归：20新节点15 failed/5正常Unicode控制 passed；连同5既有节点为25项15 failed/10 passed、0error/skip，RED.xml/RED_CHECKED 保留。覆盖8事件、4manifest、8真实复制入口；call/setup、高/低/混合代理与有效中文/emoji全部执行。子进程原退出3的真实故障由外层断言失败记录，未排除坏输入或挑选通过。

方案先写入 testing：不增加helper/模块/目录，两个现有JSON文本写入点加 errors=backslashreplace，保留 ensure_ascii=False 与有效UTF8，只把非法代理转成可无损解码的JSON转义。既有4000字符上限/状态/phase起点/来源优先级不变。修复后删除B092计数测试的临时ASCII诊断shortcut，恢复相同直接断言；src生产代码不变。终端编码设置仅保障采集，不掩盖显式JSON writer故障。

58f9aaf只在manifest write_text和events append open各加一个errors参数；原ensure_ascii=False保持。计数测试同步删2行绕行/注释、恢复原直接相等断言，src生产树与基线49748695完全相同。针对性175 passed；core1396 passed/1既有POSIX mode skip；CPU Flow189 passed，无failure/error。12新unit在core、8新native在Flow、60计数在core均实执行。8独立native子进程各退出1，事件failed1+passed1且peer执行，报告同样失败1/通过1；这是预期失败测试的正确落盘，不是子任务质量通过。有效中文/emoji原UTF8与代理解码字符串均核验。

LOCAL_CHECKED SHA256 `cea6fe304c936747d6deaff3dc0389c11c9aaf8f9510fb39f86234c1e76b41cc`；原始before.zip/hash/RED/GREEN/CORE/FLOW及红测试实际CRLF字节保留.tmp/audit093，不随clone。首次核对器把pytest的current目录别名重复计入为9条而拒绝；原脚本及LOCAL_CHECK_REFUSED保留，按resolve去重后确认8独立结果，未重跑测试或删结果。此核对拒绝不当库失败。最终工作head与实际dev CI待验，不把本机通过替代远端。

### 127. PR112实际dev验收与第十七条结束远程清理

PR112普通merge到 `4974869527bffe706d59fd4da92a6cfcbafa711e`。最终head `07a32762ec777b29d4c4134621d2f62fa617da5f` 的 Unit38808338109、Package38808338125、Branch38808337874，以及实际dev Unit38809234579、Package38809234571的全部实际矩阵/聚合成功。DEV_CHECKED SHA256 `32bfa3c0833f9bd946a5e2157c4369844056ae6e79b8c1430482d5bfad2076e3`。证据 `.tmp/audit092` 不随clone。

B09260新节点完整红45失败/15控制，修复后readout491、core1384+1既有平台skip、CPU Flow181通过。核对祖先/无开放PR/实际CI后已删remote codex/report-count-cells，本地tip保留；本目标累计17条结束remote已清。Issue110保留open至下版；B093/#111独立接续，不重跑已完成CI，不合main/tag。

### 126. NUMBERS计数格的统一类型与行列合同（B-092 / #110；另发现B-093 / #111）

记录日期2026-10-16，第二类只读展示系统复查。report直接str拼stat.n、空metrics的Experiment.n与stat缺n时的Experiment回退，前期B086覆盖factor/metric文本而未覆盖计数类型；不重开旧合法文本反例。实际15文件场景：三个分支各4坏输入（pipe/newline/bool/negative）及1合法整数，12违反合同/3控制；pipe使7列表格多边界符，newline拆行，bool/负数假计数。原REPRO.json保留，REPRO_CHECKED另记几何+单元格联合判据，不能仅对换行后的碎片误认count；每个rendered.md可核对原文。index/process原字节保全，未跑计算。

先[Issue110](https://github.com/DreamSoul-AI/RPipe/issues/110)，首60回归在第52节点因测试证据writer编码崩溃中止：RED.xml记36 failed/1error/15控制，不当作完整60执行。原测试FIRST_RED_TEST_BYTES/RED/log/hash保留；tests/conftest.py ensure_ascii=False序列化失败longrepr含U+D800，UTF8写失败造成pytest internalerror。另开[Issue111](https://github.com/DreamSoul-AI/RPipe/issues/111)登记B093，紧接当前批次处理，不把测试设施失败当计数产品失败或吞掉。当前断言将同一单元格相等判断先存bool，仅诊断用ASCII（shortcut明确关联#111），所有原数据及案例含U+D800保留，不排除节点、不放宽判断。c1add39正式红提交，完整RED2为60项45 failed/15 passed、0error/skip，源码仍冻结原字节。

设计先于生产：report.py现有模块内增私有_count，三分支共享非负int十进制/其他“-”规则；bool、负数、float/非有限、text/分隔符/代理、array/object/null均为坏格。4整数控制含0/大整数及缺失回退控制保留，缺stat.n沿exp.n，缺exp.n默认0仍原语义。不按n_planned重算或改写科学计数，不改_fmt浮点精度、计划一致性、Run归属、peer或JSON/schema/布局；不新增模块。关联研究计划RQ2/H2的类型视图，不把显示可用当数据真实性或科学质量通过。

原部分RED XML细核：suite汇总tests=52、36 failures/1 internalerror；实际命名完成节点51（36失败/15控制），另有1无名占位和1 pytest.internal合成条目，所以testcase标签共有53。初步通用核对器误把无名占位算成第16个pass，LOCAL_BEFORE_RED_ROW_CORRECTION保留，正式LOCAL_CHECKED已拒绝该计法并分列，不把占位/内部错误算实际完成节点；原RED2及GREEN/CORE/FLOW均无此问题。

生产ce72f79只改原report.py +6/-2（净+4）：私有_count一处判断、两处渲染调用覆盖三个来源，输入JSON/聚合计算/_fmt精度和缺失回退不变。readout491 passed/0skip、core1384 passed/1既有POSIX script mode skip、CPU Flow181 passed/0skip，0fail/error；60新节点在RED2、GREEN、CORE完整执行，包含未配对代理字符，旧index/process/人工报告及peer字节保全，NUMBERS是正常替换的派生视图。没有排除“难过”节点或用旧通过覆盖新失败，B093原内部错误仍单独未修；首部分RED与后完整RED2不混算。执行CRLF/Git LF分别hash并保留原执行字节，原源码ZIP/XML/log/LOCAL_CHECKED在.tmp/audit092，不随clone。最终工作head/实际dev CI待验，Issue110留至下版；本批后优先修已确认的Issue111。

### 125. PR109实际dev验收与第十六条结束远程清理

[PR109](https://github.com/DreamSoul-AI/RPipe/pull/109)最终head `9e2f0efbe546a64c4fbbbe4caf1fed326cd396bd` 的Unit38804944730/Package38804944700/Branch38804944725全部实际矩阵/聚合成功，PR_CHECKED SHA256 `baf857baac64d4c81c11b7d6f93f2dc72211a4b939e694b2243f3a37fbce1b24`；普通merge实际dev `fccccc4c28f9f6797fe5873b0700eead3ba39404`，[Unit38805781593](https://github.com/DreamSoul-AI/RPipe/actions/runs/38805781593)/[Package38805781634](https://github.com/DreamSoul-AI/RPipe/actions/runs/38805781634)也全部实际成功，DEV_CHECKED SHA256 `a8fa26c46478379644d4342248f6f730701e811c396295fa650e2c2a6a8fc89e`。24新增本机Windows全部执行，其他平台4原生只读项skip另记，见§124，不宣称任意权限必清或更改共享硬链接mode。

祖先/无开放PR/实际CI核对后普通删除结束remote codex/shared-cleanup-errors（tip9e2f0ef），原本地保留；目标累计十六条结束remote清理，前十五见§123，§82历史不重复计数。PR/Issue108原正文已补实际验收，Issue留至下版，main仍00e70b1/v0.4.1。数据/Run/provenance/原只读文件与所有失败保留；临时proof/删除前后/五行COMPLETION在.tmp/audit091，不随clone。当前分支处理新计数合同，未重派旧CI或重开旧题。

### 124. 共享引用/迁移的原发布结果保全（B-091 / #108，红合同与设计）

记录日期2026-10-16，第三类共享链接清理复查，不将独立文本的属性恢复套到硬链接。实际Windows两条路径：link_input模拟无symlink权限走原生hardlink，migrate_shared原生hardlink；只读旧目标先使replace WinError5，finally unlink同样WinError5，后者遮蔽原发布错误，最终filename2为None，原目标仅在context。源/目标字节与权限均未变，原始文件/硬链接与NATIVE_REPRO/源码ZIP/hash在.tmp/audit091，不随clone。没有迁移正式输入或修改系统ACL/设置。保留诊断链接比修改与原始输入共享的权限安全；此处修的是错误归属及发布结果，不宣称任意IO必清。

先[Issue108](https://github.com/DreamSoul-AI/RPipe/issues/108)再f1c6426红合同：24项15 failed/9 passed、0error/skip；四路为引用默认/硬链接回退、data/model迁移，覆盖发布失败的权限/普通IO清理拒绝、完成后清理IO拒绝、RuntimeError程序错误以及真实原生只读/正常控制。回归断言原异常identity或原生filename2、成功后的samefile、源/目标/peer字节权限，禁止清理非owned路径。本机Windows全部24执行，其他平台四个原生只读项明确skip不计通过，其余照常；原红保留。

设计先于生产：只在shared.py两处既有owned finally内捕获unlink的OSError、best-effort保留诊断；原发布异常及已提交成功返回值不被覆盖。RuntimeError不吞，未owned路径不清理，不chmod硬链接，不动原atomic_replace六次有界重试、锁次序、配置/返回结构、schema或布局，无新helper/模块。全仓实际调用核对为DataFactory的reference_tree/link_input及CLI shared迁移；prepared_resource独立构建入口仅核对、不改其诊断目录，checkpoint恢复合同也不改。关联研究计划RQ1/H1与RQ3/H3的首错/发布不变量，不新增科学实验。

生产a8fb381仅原shared.py两处owned finally +8/-2（净+6），无chmod或新helper；OSError清理拒绝不覆盖原发布异常/成功，RuntimeError控制仍传播。atomic/shared120 passed/1既有POSIX mode skip、core1324 passed/1同skip、CPU Flow181 passed，0fail/error，24新增节点在本机Windows的联合与core全部实际执行转绿。原生四个只读节点在其他平台明确skip不计通过，可移植合同仍执行。所有源/目标/peer字节权限保留；原只读样本/硬链接未清或改属性，不把残留诊断称为已删除。执行CRLF与Git LF的红字节分别hash核对、保留准确原执行字节；原XML/log/ZIP/LOCAL_CHECKED在.tmp/audit091，不随clone。最终head/实际devCI待验，Issue108等下版。

### 123. PR107实际dev验收与第十五条结束远程清理

[PR107](https://github.com/DreamSoul-AI/RPipe/pull/107)最终head `168456adf0fba9223bad5113a5e8d05eec955ad4` 的Unit38802303742/Package38802303741/Branch38802303901全部实际矩阵/聚合成功，PR_CHECKED SHA256 `a0b6f8552f3590d0792891f73b6be24728f6a0035d3de0d0caff1ce80eaa2f28`；普通merge实际dev `3a72bd8cef7efc178d9f51b49d4dc05c53d83b36`，[Unit38803008248](https://github.com/DreamSoul-AI/RPipe/actions/runs/38803008248)/[Package38803008361](https://github.com/DreamSoul-AI/RPipe/actions/runs/38803008361)也全部实际成功，DEV_CHECKED SHA256 `34f0f676b0c264e313a075f58703e221a2c5b0af126d8d70910013873d90289a`。12新增节点本机Windows均执行；其他平台3原生只读节点skip另记，不冒充通过，详§122。

祖先/无开放PR/实际CI核对后普通删除结束remote codex/atomic-readonly-cleanup（tip168456a），原本地保留；目标累计十五条结束remote清理，前十四见§121，§82历史不重复计数。PR与Issue106原正文已追加实际验收，Issue等下版。main仍00e70b1/v0.4.1；所有数据/Run/provenance和旧失败保留。临时proof/删除前后/五行COMPLETION在.tmp/audit090，不随clone；当前新分支仅处理共享链接结果保全，不重复旧CI或重开旧缺陷。

### 122. owned只读文本staging的清理边界（B-090 / #106，红回归与设计）

记录日期2026-10-16，第三类原子写入口复查。真实Windows：对只读旧target调用atomic_write_text，stage复制只读mode后原生replace最终WinError5，旧target仍原字节，但finally的unlink同样拒绝，完整候选内容残留。REPRO.json与原生目标/残留、源码ZIP/hash留.tmp/audit090，不随clone。全仓调用检索确认config/result/index/provenance/activity/Run与Study process/report/compare都复用既有atomic_write_text；文本新建副本与共享硬链接不能混用chmod清理，因为后者改属性会影响源。此次不对共享引用或checkpoint诊断恢复做无反例扩展，也不改atomic_replace既有Windows重试。

先[Issue106](https://github.com/DreamSoul-AI/RPipe/issues/106)，再红提交50aac22，12合同5 failed/7 passed、0error/skip；真实Windows三公共writer（text/config/result）及正常控制，可移植删除权限拒绝/恢复/再次拒绝/非权限IO，未取得owned的只读碰撞控制。输入字节与权限保全，异常identity不变。native readonly三项其他平台按明确Windows语义skip不计通过，非Windows仍执行可移植故障合同；原生复现未改系统ACL或设置。

设计：仅在现有owned finally范围，先照旧unlink；仅PermissionError触发临时独立文本chmod0o600后再unlink一次。仍不可删除时best-effort保留诊断并传播原发布异常，不更改target/peer或未owned路径。保留默认成功路径、编码/部分写入/长名/碰撞/模式保全合同；不新建helper/模块/格式，修新宏观计划RQ3/H3的发布/清理不变量。文档先于生产，相关源码与全部调用边界已核对。

生产f2c0ab9仅既有_atomic.py净+6，无新helper/import；先unlink的成功路径不变，只有PermissionError进入有界属性恢复与第二次删除，仍拒绝时保留诊断并保持原发布异常。联合atomic/shared 96 passed/1既有POSIX script mode skip，core 1300 passed/1同skip、CPU Flow181 passed，0fail/error，12新增节点全部在本机Windows的联合与core实际执行转绿。无真实权限/ACL系统调整，首次原生只读目标和残留内容保留不动；不可恢复chmod/第二次删除/非权限IO与碰撞控制均通过。Git LF与执行CRLF的红字节单独hash核验并保留准确原字节，不混称Git blob字节相同。LOCAL_CHECKED/XML/log/ZIP/hash在.tmp/audit090，不随clone；最终工作head/实际dev CI待验，Issue106留至下版。

### 121. PR105实际dev验收与第十四条结束远程清理

[PR105](https://github.com/DreamSoul-AI/RPipe/pull/105)最终head `39f132bfa24244a9e6d622760c79a06a78e679f7` 的Unit38799213179/Package38799213252/Branch38799213142全部实际矩阵/聚合成功，PR_CHECKED SHA256 `106af455567197daad5d3cbdeb5a265050031423ef0b68478bd9858842695f62`；普通merge实际dev `83615d727122de805a280aa3d0c1d7f217fa62aa`，[Unit38800132891](https://github.com/DreamSoul-AI/RPipe/actions/runs/38800132891)/[Package38800132794](https://github.com/DreamSoul-AI/RPipe/actions/runs/38800132794)也全部实际成功，DEV_CHECKED SHA256 `2e684b7c19ddf822f882cdbde07b07d463d3b2ce73108e4cab6b27730fcabe5e`。108新增节点的前置96和后补12执行范围见§120，未把后补当原红；pairedNaN旧语义与有限容差保留，不是科学质量验收。

祖先/无开放PR/实际CI核对后已普通删除结束remote codex/compare-nonfinite-contract（tip39f132b），原本地保留；目标累计十四条结束remote已清，前十三见§119，§82历史清理不重复计数。原PR/Issue104正文已追加验收，Issue保持open到下版。main仍00e70b1/v0.4.1；所有数据/Run/provenance/旧失败保留。临时proof/清理前后/五行COMPLETION在.tmp/audit089，不随clone，当前新分支只处理B090，不重派旧CI。

### 120. 只读比较的非有限值合同（B-089 / #104，红回归与设计）

记录日期2026-10-16，按三类合同继续系统复查。当前dev233fb2a的_close/_tree在指标、history、checkpoint tensor共用比较树；有限值与±inf、相反inf在rtol1e-5时因diff与bound都inf而错误通过，相同inf因相减NaN错误失败。实际18场景12判定不符，pairedNaN与有限相等控制正常；原字节不变，所有pt为自建微小CPU载荷。冻结原码ZIP/hash/实际导入路径、脚本/REPRO在.tmp/audit089，不随clone。已检索所有调用，三个读取来源汇入同一比较树，不能只修CLI或屏蔽显示。

先[Issue104](https://github.com/DreamSoul-AI/RPipe/issues/104)，96个API/CLI合同红提交19313c7；首轮96失败中含测试错误：--out后CLI最后一行是输出路径，不是passed/failed。原RED保留；1821c3c更正尾两行断言，生产字节与冻结基线相同，RED2为40 failed/56 passed、0error/skip。该错误测试不当作产品缺陷，18实际复现独立成立。参数覆盖四类来源（metrics/history/real tensor/complex tensor）、exact/relative、双向±inf/同值/NaN/有限控制和所有输入原字节。

设计只改原compare.py共享判断：标量先判NaN/inf身份；tensor将已相同非有限元素差归零，不匹配非有限差归inf且相对界归零，防止inf与NaN界绕过。保持既有pairedNaN、有限值容差、整数精度、字段选择、缺失/错误传播与JSON差值null处理；不新建helper/模块/字段、不改变训练计算或checkpoint恢复。结构合同先更新再生产。复数已有NaN行为沿用，不借此新设逐分量复数协议；任意有限极值算术或精度扩展不伪装为本批已覆盖。

生产2444097仅原compare.py +6/-2（净+4），96前置合同转绿，完整compare152、core1276 passed/1既有POSIX mode skip、CPU Flow181 passed，0fail/error。随后dc5988c补12混合tensor控制：相同±inf/NaN旁仍检有限值inside/outside漂移，冻结原函数实际12场景有4不符，候选全符，输入字节保全；全compare164再通过，源字节未变，未重复整套core/Flow、不把后补12项冒充旧完整门执行。原96项均在core及GREEN_MIXED实际执行。XML/log/LOCAL_CHECKED记录分层范围，平台skip不计通过。

首次直接比较RED2工作区hash与Git blob拒绝：执行为CRLF、Git为LF。RED_NEWLINE_CHECK分记两hash，按Git文本换行策略恢复的红工作区字节与原记录hash逐字节一致；原RED/RED2不覆盖，不把换行差异当库失败。测试差异与错误假设均公开保留。对应新宏观计划RQ2/H2的“错误通过数”合同；这里只修只读判定，不改变训练载荷或旧科学结果。当前最终PR/实际dev验收待办，Issue104留至下版。

### 119. 研究计划PR103实际dev验收与第十三条结束远程清理

[PR103](https://github.com/DreamSoul-AI/RPipe/pull/103)最终head `b631d3a5d81375dba099a112e8d8d263f204c1a7` 的Unit38795995407/Package38795995108/Branch38795995082全部实际矩阵/聚合成功，PR_CHECKED SHA256 `d9b70699720041322e333904a0c71266eec22822eff96e97fe6cf9986767b60c`；普通合实际dev `233fb2a9512520431332cd75873200605b512850`，[Unit38796958288](https://github.com/DreamSoul-AI/RPipe/actions/runs/38796958288)/[Package38796958133](https://github.com/DreamSoul-AI/RPipe/actions/runs/38796958133)也全部实际成功，DEV_CHECKED SHA256 `6328e7c70b2d094a51d5e41fb7326bb5257e5acf71612182971975057e05e512`。中文/英文各十章、36布局合同、103本地链接、三正本字节证明见§118；未跑新科学实验。

祖先/无开放PR/实际CI核对后普通删除结束remote codex/rpipe-research-plan（tipb631d3a），原本地保留；目标累计十三条清理，前十二见§117/114，不重复算§82历史。PR正文与本机五行报告已写实际验收，无其他待清结束remote。原staging未提交/删除，main仍00e70b1/v0.4.1；所有Run/data/provenance/失败证据保留。临时proof/COMPLETION在.tmp/research-plan-20261016，不随clone；文档任务完成后当前唯一未合为非有限值比较合同分支。

### 118. 按新规范与模板重做 RPipe 研究计划（文档批次）

记录日期2026-10-16。Enmao当晚优先任务改为项目研究计划，暂停新增审计批次，正确性目标仍active。原仓库仅有通用研究要求、规范/模板和历史Study计划，没有统一宏观课题计划；按layout既有职责新建[中文计划](<../research/研究计划 RPipe.md>)和[完整英文](<../research/Research Plan RPipe.md>)，逐章覆盖新模板十章。RQ1–RQ3为失败证据保全、可选/严格读取与引用归属、原子发布；RQ4为条件性历史原码科学对照。明确不扩旧成功保留范围、不更改结构或落盘，历史main_probe/main_exp计划/结果不事后重写；旧版科学验收与当前dev未重跑分开。

[文献证据](../research/literature.md)登记2026-10-16四组实际查询、四条来源的版本/阅读范围/六方面/用途/缺口。L1仅官方摘要；L2固定arXiv v2定向章节；L3仅官方检索摘录且直读失败，不比较系统能力；L4固定PyTorch2.9整页，不据此升级Study依赖。ACM候选原页访问失败不作依据，不称系统综述、独立复现或外部优越性。没有下载PDF、安装工具或新建论文库/科学数据。完整读取本地Knowledge 0d7c68d的检索skill、研究笔记模板与wiki，仅作方法参考，不采用引用量排名。

可读DEM文献规范完整623行/34809字节原样加入[standards](../research/standards/README.md)，SHA256 `2605cf562aee59b2023ecb0954c3639658958c1874ddce79a623dd3df63489a8`；与两份既有研究计划中文正本一同用-text保全字节，未用本机较旧同名计划覆盖新交付正本。原件内部缺失DEM链接保留，导航单独说明，不复制未读规范。同步research.md、根README、Studies入口，正式资料不依赖_staging_from_DEM；暂存保留不提交。文档批次仅校验文档/来源/布局与必要CI，不为填模板启动训练、测速、GPU或外部基线实验。临时证据.tmp/research-plan-20261016不随clone。

本地验收：中文/英文各十章，RQ/H、D1–D4、E0–E4关键对应齐备；103条新增/修改本地链接及锚点通过，UTF-8/代码围栏/改动范围通过；三个中文正本工作区、Git index blob和强制autocrlf=true隔离checkout逐字节一致。既有layout合同36项实际执行通过，0失败/错误/跳过；src/tests、版本、layout合同与两个历史Study均无改动。仅文档更改未重复全套本地core/Flow，最终head/实际dev仍按CI验收。首次文档辅助脚本因旧目标文本不匹配而assert停止，未写目标文件；更正后检查通过，记为文档脚本边界而非库缺陷。开放Issue逐项核对均为待下版已修项或探索89/90，无依据批量关闭；无其他开放PR。

### 117. PR100/102实际dev验收及第十一、十二条结束远程清理

PR100普通merge的dev `187bb2e562979d1911fa2911702ea35bdae707da`，[Unit38792231762](https://github.com/DreamSoul-AI/RPipe/actions/runs/38792231762)/[Package38792231924](https://github.com/DreamSoul-AI/RPipe/actions/runs/38792231924)全部实际矩阵/聚合成功；DEV_CHECKED SHA256 `2c3822b240f2b9eebb14d4de04db84ac3b0f297e413376b5f18c3461e60092af`。PR102最终head00500c325676fbe2f38ed522d69236c2d9d5fa04的Unit38793099410/Package38793099261/Branch38793099281全部实际矩阵/聚合通过后普通merge dev `ef6084681160a68588e0b2a359ab7cc07c398879`，实际dev[Unit38793891029](https://github.com/DreamSoul-AI/RPipe/actions/runs/38793891029)/[Package38793891048](https://github.com/DreamSoul-AI/RPipe/actions/runs/38793891048)也全部实际成功，DEV_CHECKED SHA256 `634cde73338bb11e8e76ec570f4c78ee1d6e33b1e4ac647d3b75709d5f14dcd0`。§115/116的待验收仅为历史过程，不重复派发。

祖先、无开放PR、实际CI核对后，普通删除远程codex/optional-tracker-decode（tip8e8ee94）与codex/compare-checkpoint-boundary（tip00500c3），原本地分支保留，当前rpipe-research-plan保留。两PR/Issue99/101原正文已补实际验收，Issue保持open至下版，不作为“过期”关闭；探索89/90保持独立。当前目标累计十二条结束remote清理，前十见§114，不重复计§82历史九条。main/dev未因删除改变，main仍00e70b1/v0.4.1；数据、Run、provenance和旧失败均保留。临时proof/清理前后/命令在.tmp/audit087、audit088，不随clone。

### 116. compare checkpoint归属与动态键显示合同（B-088 / #101）

三类系统复查继续到compare参数边界。此前§111将compare key当代码固定的判断不完整：checkpoint.<name>确实由参数参与构造，本节据真实反例更正，不追改旧记录掩盖遗漏。4组原生CPU自建小文件证明父路径/Run外绝对stem/文件symlink读到同一个外部pt而checkpoint项pass，正常自身不同权重control fail；追加5组补齐相同metrics/history后三种场景整个report错误true，Unicode LS合法名称使4行断成5行，正常control false。外部与各Run输入原字节保全，未加载正式模型、不可信payload、GPU或执行测速。原始REPRO/REPRO2/原码ZIP在`.tmp/audit088`，不随clone。

先[Issue101](https://github.com/DreamSoul-AI/RPipe/issues/101)，冻结原源码实际导入与hash核对，17节点10 failed/7 passed、0error/skip。同字节红回归a0682e3先于设计6aea0fc与生产5280815；portable合同在[test_compare.py](../../tests/rpipe/structure/artifact/readout/test_compare.py)。API/CLI父路径、绝对、外部文件alias及checkpoint根目录alias拒绝；合法nested、自身abs、内部文件/目录alias、missing、普通name和真实Windows extended-own为绿色兼容控制。Windows专用extended节点在本机执行，其他平台skip不计通过。

仅原compare.py +11/-2：原regular-file/missing/非文件/IO预检保留，存在普通文件加载前检查解析checkpoint根属于Run、文件属于该根；归属比较复用B085既有Windows namespace key，文件IO仍使用原生路径。format_compare复用B086的_text_cell，只改单行视图；JSON原key、容差、数值与旧--out位置不变。未改变训练checkpoint保存/恢复、结构或产物布局，没有新helper/模块/依赖。不存在的可选文件仍原missing，不新增清理/迁移/锁或安全沙箱协议。

本地完整compare56、相关983 passed/1既有POSIX脚本mode skip、core1180 passed/1同skip、CPU Flow181 passed/0skip，17前置节点全执行转绿，0fail/error。所有完整门结束后，8925e22对既有8个拒绝节点补torch.load调用为零断言，再单独GREEN_LOAD全文件56通过；生产字节不变，未重复整套核心/Flow门，也不把后补断言伪装成初始冻结测试字节。`.tmp/audit088/LOCAL_CHECKED.json`区分原红/完整门/后补断言及源码hash，本机材料不随clone。当前唯一未合codex/compare-checkpoint-boundary，准备最终head/实际dev两轮实际CI后清remote，Issue101等下版。

### 115. 共享可选解码PR100按最终CI普通合dev

[PR100](https://github.com/DreamSoul-AI/RPipe/pull/100)最终head `8e8ee9443d8fb87c3e2db1beea2b3f1bacd0b451`的Unit38791314476 / Package38791314478 / Branch38791314505全部实际矩阵与聚合成功，PR_CHECKED SHA256 `b2d7e2394ade099d2f42e02a28741b959fbf9d6bf436b83d10157fd72e02771a`。普通merge为实际dev `187bb2e562979d1911fa2911702ea35bdae707da`，自动触发Unit38792231762/Package38792231924。本节记录时实际dev仍执行中，不用工作head通过代替集成结果；观察原句柄、不重复派发，实际全绿后立即清optional-tracker-decode远程，当前compare分支保留。24节点与单行修复见§113，Issue99留至下版。

### 114. PR98实际dev验收与第十条结束远程清理

PR98实际dev `8dc6a914a6f0e0f42a2353f43e73505880672087`的[Unit38790569174](https://github.com/DreamSoul-AI/RPipe/actions/runs/38790569174) / [Package38790569148](https://github.com/DreamSoul-AI/RPipe/actions/runs/38790569148)所有实际矩阵及聚合全成功，DEV_CHECKED SHA256 `3de37da86aaa3a95336e7a90c1044b9f6bd2f5088cdfd83a77bb2094e84318c0`；最终head验收见§112。原pending仅历史过程，不重派。祖先/无开放PR核对后已普通删除结束远程codex/readout-text-cells，原本地分支及当前活动分支保留；PR98/Issue97原正文已补实际验收，Issue仍open等下版。

本目标累计十条结束remote已清：artifact-contracts-correctness、yaml-mapping-roots、declared-batch-contract、saved-plan-read-boundary、result-status-contract、activity-hint-fields、research-plan-masters、provenance-declarations、log-reference-contract、readout-text-cells（均codex/前缀）。与§82历史九条清理不重复计数；删除前后main/dev引用不变，main仍00e70b1/v0.4.1。删除证据`.tmp/audit086/BRANCH_DELETED_CHECKED.json`不随clone；全部数据/Run/provenance和旧失败保留。DEM中文原件本轮再与staging逐字节比对仍一致，英译和两项独立调研已PR92正式完成，不部署logger/shared或重复迁移。


### 113. 可选 tracker_state 解码失败的共享读取修复（B-087 / #99）

在第二类合同扫描中，真实坏UTF8状态文件经load_curve_series触发UnicodeDecodeError；同一_state_history被JSONL主轨迹、legacy history、Run process、Study process与曲线复用，缺失/坏JSON/读取IO原有空history规则未包含解码失败。5原生隔离探针中，3个循环symlink候选均已正常降级、普通result控制正常，只有坏UTF8证实异常，不把正常行为登记为缺陷。原始REPRO/ZIP在`.tmp/audit087`，不随clone。先开[Issue99](https://github.com/DreamSoul-AI/RPipe/issues/99)，再以冻结源码实际导入路径/hash核对后执行24合同：18 failed/6 passed、0error/skip；测试正式同字节提交6d11281，随后设计41e7130、生产f5f85b7。可移植单元测试位于[test_aggregate.py](../../tests/rpipe/flow/process/test_aggregate.py)，Run/Study调用集成在[test_process.py](../../tests/rpipe/flow/process/test_process.py)。

三种坏编码 × 主轨迹有/无 × 两个公共读取函数，以及Run/Study真实入口，均复用同一边界；绿色控制覆盖坏JSON、非对象、缺字段、有效旧history、读取IO和RuntimeError原异常identity。最终只在既有_state_history的except列表加UnicodeError，原单个生产文件1行替换，无新helper/模块/依赖，不扩大stat前置异常规则。有效JSONL主轨迹、peer、数字、旧input字节保全；读取损坏状态不清理/回写它。严格训练checkpoint/RNG恢复、percent、统计、曲线坐标/截断、Run/Study文件布局和失败结果保全规则均不变。

本地隐藏GPU，RELATED966 passed/1既有POSIX脚本mode skip，CORE1163 passed/1同skip，FLOW181 passed/0skip；0fail/error。全部24前置节点在相关门执行，其中18unit在core、6集成在Flow执行，不把未选集成计成core通过。读取者只返回空可选旧history或有效主轨迹，Run/Study仍写各自既有process，peer与所有输入精确字节不变。`.tmp/audit087/LOCAL_CHECKED.json`保存实际JUnit/命令和源码hash，不随clone。当前唯一未合codex/optional-tracker-decode；待最终head/实际dev CI，Issue99保持open至下版。不测速/GPU、不迁logger/shared、不合main/tag。

### 112. 文本单元格PR98按最终head CI合dev，实际集成验收接续

[PR98](https://github.com/DreamSoul-AI/RPipe/pull/98)最终head `58d8393f88db802407958357e134c8bf4333264d`的Unit38789761211/Package38789760971/Branch38789760983全部实际矩阵与聚合成功，PR_CHECKED SHA256 `302079bb740b32d40492db9af884d39a150e6875b3f19129b68c8e5c2efb6eeb`。普通merge为实际dev `8dc6a914a6f0e0f42a2353f43e73505880672087`，自动触发Unit38790569174/Package38790569148；本节记录时仍在执行，不把pending计作通过，不重复派发。实际dev全绿后按祖先/活动保护立即清codex/readout-text-cells远程，当前optional-tracker-decode保留；Issue97等后续版本。完整本地56合同与初轮白名单假设修正见§111，原始证据不覆盖。


### 111. 文本单元格合同统一：先红回归再最小修复（B-086 / #97）

只读合同批次从dev bf57c88继续，先开[Issue97](https://github.com/DreamSoul-AI/RPipe/issues/97)。7组真实隔离probe确认合法多行factor、tab、pipe或metric名称打破既有TSV/Markdown几何：单Run factor LF令4→5行，tab令9→10列，pipe令report7/5列各增1；metric LF令两表断段。普通/单反斜线pipe为几何控制，index/process/peer原字节均不变。原始REPRO及ZIP保留在`.tmp/audit086`，不随clone；可移植合同进入既有[test_report.py](../../tests/rpipe/structure/artifact/readout/test_report.py)及[test_status.py](../../tests/rpipe/structure/artifact/readout/test_status.py)。

红提交1801e5b首轮66节点49失败/17通过，其中10项误假设status支持任意metric；核对既有accuracy/best_accuracy/train_loss白名单后用7b451a7仅缩窄测试，保留首轮原字节/日志及TEST_ASSUMPTION_CORRECTION。生产不变时权威RED2为56项39 failed/17 passed、0error/skip；不把测试假设错误当缺陷。设计8bb68a4明确只改显示视图、不限制合法配置或改磁盘产物。d59cc70在原report.py放两个私有纯文本formatter，status复用已有report依赖；统一splitlines/tab显示空格、Markdown先反斜线再pipe，替代error/note不完整的局部LF/tab规则。因TSV与Markdown分隔符不同保留两步，未新增模块/依赖或改变数字/状态/指标选择。只有原两生产文件+19/-11、净增8行；日志链接仍复用B085，不移动任何文件或布局。

本地GPU隐藏，GREEN只读306全部通过；RELATED 942 passed/1既有POSIX脚本mode skip，CORE1145 passed/1同skip，FLOW175 passed/0skip。56前置节点在GREEN/RELATED/CORE全执行，0fail/error；比对peer/数字/链接/计数及精确输入字节。`.tmp/audit086/LOCAL_CHECKED.json`与各命令/日志/JUnit保留，不随clone；不是GPU、性能或Study科学验收。当前唯一未合codex/readout-text-cells，准备PR最终head/实际dev两轮实际CI；Issue97留至下版，不提前关闭。compare的item展示键由代码固定且不打印动态problem键，logs已逐行读取，不把两者无依据加入本次TSV/Markdown修复。

### 110. PR96最终head与实际dev验收，结束远程分支清理

[PR96](https://github.com/DreamSoul-AI/RPipe/pull/96)普通merge为dev `bf57c88fdf0694c69c7df6003241a8b19d33db4f`。最终head ac1d404的Unit38787098038 / Package38787098355 / Branch38787098335全部实际矩阵/聚合成功，PR_CHECKED SHA256 `f52bf7fdd56312618ca836be597b073e959b49510834e74521b9fcff538a8d1a`；实际dev [Unit38788640662](https://github.com/DreamSoul-AI/RPipe/actions/runs/38788640662)及[Package38788640570](https://github.com/DreamSoul-AI/RPipe/actions/runs/38788640570)全部实际矩阵/聚合成功，DEV_CHECKED SHA256 `529de1182147730b5d9912b06c94360c5f419f0d09ad2d16c10eee4f297910da`。最终90节点、本地各门与原失败证据见§109，不使用旧a5 CI代替最终代码。

按长期明确授权，核对最终tip已为origin/dev祖先且无开放PR后立即普通删除远程`codex/log-reference-contract`，本地保留，当前readout-text-cells保留。本目标累计九条结束remote已删（前八见§107），与历史§82清单不重复计数；删除前后main/dev SHA一致，main仍00e70b1/v0.4.1。PR96及Issue95原正文已追加验收事实，Issue95保持open等后续版本。`.tmp/audit085/BRANCH_DELETED_CHECKED.json`等本机证据不随clone，原失败/拒绝/旧成功均保留。bugs移出已解决B085，不以GitHub仍open误判未修。DEM源文档/英译及89/90独立调研已在PR92实际验收，不再迁移或重复开PR；staging保持未跟踪供对照。

### 109. 日志可用性与原生路径别名最终组合本地验收

记录日期2026-10-16，继续同一B085/#95/PR96/工作分支，未合旧a5 head。d6bbbae循环红3项2失败/1通过；790de87目录/真实Windows alias红6项5失败/1通过。首次stat预检实现修正循环/目录，但仍有3个extended别名失败：AVAILABILITY_GREEN为6通过/3失败；PowerShell分行调用未在首子进程失败后阻断扩大GREEN2，该GREEN2也保留同3失败（无CORE2/FLOW2执行），不能算完整通过。已把后续依赖执行改为Python subprocess.run(check=True)，先针对性通过才启动完整门，不以shell最后命令成功掩盖前错。

根因是Path.resolve仍可保留显式Windows namespace锚点，不能把返回字符串和普通锚点直接relative_to后据此判越界，也不能为了比较剥掉用于原生IO的前缀。先补a8eebf0的3个Windows UNC关系合同（同share大小写别名可归属、不同share/server拒绝），初版缺比较函数的3个ImportError为前置合同失败，不额外声称3个真实数据故障；纯Path关系，不挂载/访问网络share。随后最小bce96cb：既有log_path中stat区分普通/缺失/不可用，缺失仍允许未来日志；私有_log_path_key仅供归属比较与href，把常见Windows盘符/UNC扩展拼法对齐，文件IO继续用原生Path。普通本Run拼法/内部父段/别名保留，只有不同锚点但已证明归属时用规范拼法；不猜不同volume/share、不笼统吞RuntimeError，不搬模块或造I/O框架。[Microsoft命名空间说明](https://learn.microsoft.com/en-us/windows/win32/fileio/naming-a-file)作语法核对，真实短路径samefile/原生错误/旧reader对照见§108，不外推任意网络文件系统或长路径能力。

最终只改现有status.py/report.py，相对dev2b7c769共+57/-18，现有函数共用同一引用guard与比较键；没有新依赖/模块/字段/目录/日志追加或科学聚合变更。早前3a5/c89候选、a5全部CI、所有失败实现/测试收集错误及输入/link/peer全保留。原84前置节点（最初72+本次12）跨互不重叠红批为54失败/30通过，修复后均执行转绿；另6既有兼容追加控制执行，共90新节点在GREEN3/CORE3执行。本机Windows最终针对性12通过；相关886 passed/1 POSIX mode平台skip、core1089 passed/1同skip、Flow175 passed/0skip，0fail/error。Windows特有6节点在POSIX明确skip，不当作通过；本机链接创建实际执行，不改权限。

LOCAL_CHECKED_FINAL记录完整范围/最终两源码hash，位于.tmp/audit085，不随clone；旧LOCAL_CHECKED/AVAILABILITY_GREEN/GREEN2/红/命令/XML等未覆盖，portable90节点已提交。新的比较函数有3条纯Windows share归属合同，真实namespace文件有3个公共reader回归；没有把纯路径测试冒充UNC网络实测。正式Study/layout/版本及release正文无变化，GPU/性能测试0，不覆盖下游。接着最终新head全部实际CI及实际dev单独验收后才合dev/清remote；不拿旧a5绿证明本修订，Issue95保留到下版。

### 108. 合并前补齐循环/非文件可用性与Windows路径别名（仍B-085 / #95）

记录日期2026-10-16。PR96初候选head `a5b83970e65e948e6eb5c83c8a9c57ca161ad998` 的[Unit38782730979](https://github.com/DreamSoul-AI/RPipe/actions/runs/38782730979)、[Package38782730953](https://github.com/DreamSoul-AI/RPipe/actions/runs/38782730953)、[Branch38782731079](https://github.com/DreamSoul-AI/RPipe/actions/runs/38782731079)全部实际矩阵/聚合成功，proof SHA256 `147b739a52c4e39a887003a8c015e8ce734b502c509cb4b621ba3f302fcbc5c1`备份为.tmp/audit085/ci/PR_CHECKED_a5b8397.json。但合并前复查发现以下真实边界，**初候选不计最终验收、不合此旧head**；不是历史pending重派或取消CI挑结果。

本机Python3.13.9真实循环文件链接：原生stat为OSError errno22/WinError1921，非严格Path.resolve仍返回路径，候选status保留无效log格、report仍输出循环链接；logs已跳过并保留peer。Python官方[3.10 resolve](https://docs.python.org/3.10/library/pathlib.html#pathlib.Path.resolve)说明循环会RuntimeError，[3.13变更](https://docs.python.org/3.13/library/pathlib.html#pathlib.Path.resolve)说明非严格不再因此抛异常。后者为本机实测，前者为文档支持的版本风险，不声称已在本机跑3.10；py launcher查询不可用不等于已枚举全机解释器。LOOP_REPRO保存原生错误与链接身份，无GPU/计时/系统权限变更。

另一个本机实证：Windows命名空间短路径别名与普通本Run日志samefile=True，原v0.4.1后基线reader能读OWN-EXTENDED，但a5候选因词法锚点不同返回'-'。旧reader实际通过冻结snapshot导入并核对来源，EXTENDED_REPRO保存路径/原始事件/两个结果；字节不改。只证明本机短路径extended别名，未改LongPathsEnabled或宣称通用长路径/任意文件系统能力。

继续同一Issue95/PR96、同分支，先补d6bbbae循环链接3项红：2失败/1通过、0error/skip，测试快照用readlink保全链接本体而不跟随坏链接；再790de87的目录/Windows别名6项红：5失败/1通过、0error/skip。首RED_FILES因测试路径字面量转义导致SyntaxError，pytest退出2、collection error1、实际节点执行0；原文件/日志/XML保留FILES_SETUP_FAILURE，不算库失败或通过。修正后先AST检查，再以新basetemp/RED_FILES2执行真实6节点，不覆盖失败。

设计仍在既有log_path统一：原生stat已有对象必须普通文件；FileNotFound允许既有未来日志链接；目录、循环或其他OSError为不可用，随后验证规范物理目标留在Run/Study。逻辑锚点正常时保留原路径拼法，有效命名空间/别名锚点不同则使用已验证的规范路径，不把同文件别名误判成越界；其他Run/Study外实际目标仍拒绝。保持RuntimeError等程序异常原样，不按错误字符串猜测或笼统吞异常；不新建模块/依赖/目录、不更改磁盘合同。该原生预检避免旧版本先走resolve的循环RuntimeError路径，同时缺失文件不误伤为不可用。

原72前置合同与6后补控制保留，新9项在本可用性修订前先红；接着用新输出/独立basetemp跑合并合同、core与Flow，最终head重新验收CI后才合dev。旧代码通过/新反例、首测试收集错误、所有链接和peer证据均保留.tmp/audit085，不随clone。当前仍无B086已登记缺口。

### 107. 来源证据实际dev验收与日志引用统一修复本地门

记录日期2026-10-16。#91/#93由[PR94](https://github.com/DreamSoul-AI/RPipe/pull/94)普通merge为dev `2b7c76986950fd6242afbf5bfc0bf81892dfeeaa`，最终head三门及实际dev [Unit38780743725](https://github.com/DreamSoul-AI/RPipe/actions/runs/38780743725) / [Package38780743746](https://github.com/DreamSoul-AI/RPipe/actions/runs/38780743746)全部实际矩阵及聚合成功。DEV_CHECKED SHA256 `dd63d4eb87bc6b9ee4613e3b9a8f6f91e33700394ec7e4fc02276c7c5582655c`。祖先/无开放PR/目标非活动分支核对后，普通删除结束remote `codex/provenance-declarations`（tip e59a226）；PR/两个Issue验收正文已追加，Issue保留open至下版。

本目标累计八条已删结束remote清单：codex/artifact-contracts-correctness、codex/yaml-mapping-roots、codex/declared-batch-contract、codex/saved-plan-read-boundary、codex/result-status-contract、codex/activity-hint-fields、codex/research-plan-masters、codex/provenance-declarations。各批清理前后/命令/祖先与CI证据留对应.tmp，不随clone；旧本地分支保留，main/dev未因删除改变，历史§82九条不重复计。当前唯一未合codex/log-reference-contract；推送前远程只剩main/dev，后续本分支保持至实际dev验收。

B085/#95先红1a2fb23（60项34失败/26通过）、9abee79（6项错误边界合同失败），设计fee867e后再补Unicode LS/NEL行分隔6项：4失败/2通过，红6ffed36仍先于生产。三批节点不重叠，共72项44失败/28通过；权限/程序错误是前置守卫合同，不是额外声称实际数据损坏。最小生产3a5c98f仅修改现有status.py与report.py，共+33/-13：既有log_path统一可选引用类型、行字符、Run/Study词法及物理范围，status构建行时复用一次并不展示坏引用；logs沿已有调用自动复用；report函数内复用同一guard避免_fmt模块循环，用标准库quote输出合法文件URI，坏href仅本格'-'。不新建模块、读取层或配置，不改writer追加/字段/schema/布局、状态/计数及科学聚合。OSError/非法路径可选降级，RuntimeError原样。

合法默认/null/空串、相对/自身绝对、Unicode空格、内部别名和未来日志路径已覆盖；再补c89a6d1的6项本Run内合法父段路径/Study目录别名兼容控制（生产后追加，不冒充原红）。相关874 passed/1 POSIX mode平台skip、core1077 passed/1同skip、Flow175 passed/0skip，0fail/error；72前置节点与6后补共78节点均在GREEN及CORE执行。日志外链、文件/目录别名在本机真实执行，未改系统权限；其他平台可选链接skip不算通过。输入及peer字节均保全，report测试只在合成Study渲染，不覆盖正式报告。

完整红/绿/命令/XML/原码与真实输入在.tmp/audit085（本机材料，不随clone），LOCAL_CHECKED记录两份生产hash；78个portable节点已提交。没有GPU/性能测量/服务启动/迁shared或logger，不干预下游vendor/实验。接着本批最终head与实际dev分别全部实际CI验收、正常合dev后删remote，Issue95保持open至下版；当前没有B086已确认问题。

### 106. 来源证据PR合dev与日志引用统一合同红测试（B-085 / #95）

记录日期2026-10-16。[PR94](https://github.com/DreamSoul-AI/RPipe/pull/94)最终head `e59a22694b3fc12053490b3398d47c0842e9c22d` 的[Unit38779395986](https://github.com/DreamSoul-AI/RPipe/actions/runs/38779395986)、[Package38779395962](https://github.com/DreamSoul-AI/RPipe/actions/runs/38779395962)、[Branch38779395955](https://github.com/DreamSoul-AI/RPipe/actions/runs/38779395955)全部实际矩阵及聚合成功，PR_CHECKED SHA256 `061a0e0a5d7ad4aea675af8fc7864c9a2c472e4f0050c0445013ae6f19e232b1`。普通merge至dev `2b7c769`，实际dev独立观察中；验收后立即删结束remote，Issue91/93等后版，不以候选CI代替实际dev。

B085/#95依据现有structure中“index.log指向该Run的run.log”合同复查，共7真实隔离场景：5坏引用为相对Study外、绝对Study外、peer Run、带换行、Run内file symlink指Study外；2合法自身/缺省控制正常。status/logs读出外部合成事件，peer错误可被算入两个Run并重复显示，NUMBERS输出外部/peer错误链接，含换行的单字段使status应有4行变5行、Markdown行也截断。全部index/process/日志/peer原字节不变，无正式日志泄漏证据。source ZIP/REPRO留.tmp/audit085，不随clone。

先开Issue95，在当前PR94最终CI观察期间仅对冻结快照加测试，实际导入快照src，读取文件hash与当前同字节；60节点34 failed/26 passed、0error/skip。数据路径兼容矩阵包括合法默认/null/空串、自身相对/绝对、Unicode空格、内部别名、尚不存在日志，另证实旧report对合法自身绝对/Unicode空格路径产生错误URI。PR94合后从新dev开唯一未合codex/log-reference-contract，同字节portable红由1a2fb23先提交，再补6个解析权限错误/非预期RuntimeError前置合同，6失败、0error/skip，红9abee79先于生产。后6项是设计守卫边界而非6个新实际数据故障；66节点跨两批合计40失败/26通过，不重复跑来挑结果。

设计先明确在既有log_path约束字符串、无行控制符、词法和物理路径留在本Run及Study内，并由status/logs/report共同复用。非法或权限不可用引用仅本格'-'/跳过事件/不输出坏href，peer、状态/计数/其他字段不变；缺省/null/空串仍默认本Run日志，合法别名/Unicode空格/自身绝对路径及未来日志链接兼容。report使用标准库文件URI编码；以函数内导入复用path guard，避免status既有_fmt导入形成模块循环，不搬模块或新造安全读取层。OSError/路径错误可选降级，程序RuntimeError原样。引用检查不扫描硬链接/inode所有权、不承诺任意并发文件树事务、不迁移日志/数据或改共享布局。

本次只读也核对数值_fmt已有OverflowError保护、activity清理已有OSError保护，不将已合规项当新缺陷。正确性目标active；研究89/90仍独立探索，不触发架构改动。接着最小修复、66节点/相邻/core/Flow验收，一个PR合dev，不合main/tag、不测速/GPU。

### 105. 来源证据两项根边界统一修复与组合本地验收

记录日期2026-10-16，唯一未合codex/provenance-declarations统一处理#91/#93，不为同类另开零碎PR。源码只改provenance.py（相对devf303d72共+18/-6），不新增模块/helper/字段或目录。B083红d1baff1→设计2b29e89→source_files最小fix4f23ae4；B084红3804bce→设计4e1dbc3→plan_files最小fix2cd634c，保留每项红测试先于其生产修复的顺序。

source_files以显式None回退保留缺省/null，统一拒绝非法provenance mapping/include list类型；plan_files复用现有load_index/validate_run_directory/_inside，约束权威index及物理输入留在Study内、保留内部链接的逻辑Run键，数据TypeError转ValueError并保cause给既有freeze拒绝，RuntimeError不吞。缺index/合法空/partial/缺config等旧语义保留，write_provenance原发布函数不变；未扩大旧成功保全范围、make多文件事务、hash缓存或研究能力。

组合本机CPU：相关784 passed/1 POSIX mode平台skip；core987 passed/1同平台skip；Flow175 passed、0skip，均0fail/error。B083的38原节点与B084的67原节点共105个全部在GREEN与CORE执行；原来分别30失败/8控制、56失败/11控制，现全部绿。另在生产修复后补a721ec0的12项缺失/null Run id及配套run_dir越界控制，独立EXTRA12通过/0skip，不把它们冒充原红节点或已执行旧core节点，不把core数字写成999。当前117个新节点均有执行证据；后续最终head CI将收集全套。没有为挑通过重复跑同字节测试；扩展根边界后使用全新a84g/c/f，原B083门保留，不替代本批。

LOCAL_CHECKED在.tmp/audit084，生产SHA256 `182ede3fb83286e8cf8f41b210b73cc50f6fadf59f6c4182f2a6d9da23c84c36`，全部XML/命令/日志/红原字节/真实外链输入保留；这些临时证据不随clone，117 portable节点随Git。相关受影响调用方make、freeze/prepare、Study process与已有Run/peer保全合同共同回归；不是make全局事务证明、任意网络文件系统并发证明或Study科学验收。layout、版本/发布材料及正式Study无变化；无GPU/性能测试或下游覆盖。下一步同一PR最终head与实际dev全部实际矩阵验收，成功后删结束remote、Issue91/93留至下版。

本轮只读检查activity清理路径也确认其已有OSError保护，不再将其当新缺陷。当前无B085已确认反例，继续按三类合同核查而不制造任务或实施探索89/90的架构改动。

### 104. 研究文档实际dev验收与同类来源计划边界红回归（B-084 / #93）

记录日期2026-10-16。[PR92](https://github.com/DreamSoul-AI/RPipe/pull/92)已普通merge至dev `f303d72328d2dafb5f10129c31a776fdcb3b7553`，实际[Unit38776926700](https://github.com/DreamSoul-AI/RPipe/actions/runs/38776926700)/[Package38776927062](https://github.com/DreamSoul-AI/RPipe/actions/runs/38776927062)与最终head三门全部实际矩阵/聚合成功。DEV_CHECKED SHA256 `adb17938a1d2662e02a5a9fdab21c92950b7c89cad003128cdf69da6ed181ac2`。核对tip eadb392在dev、无开放PR且当前为另一个工作分支后，普通删除远程codex/research-plan-masters，本地保留；main/dev未因删除改变。清理前后/命令/proof保留.tmp/research-docs；本目标累计七条结束remote清理（前六条见§101/103），当前活动provenance分支未推时远程只剩main/dev。原PR92和探索89/90均已追加真实文档/CI入口，保持探索open，不自动实施。不合main/打tag。

B-083/#91红d1baff1之后设计2b29e89、最小生产4f23ae4在source_files仅+6/-2统一容器约束；38节点全部执行并转绿，相关717 passed/1 POSIX mode平台skip、core920 passed/1同平台skip、Flow175 passed，0fail/error。完整本机旧批证据.tmp/audit083保留；以下新同类反例使该批不再能替代最终组合验收。

在同一来源证据发布类发现B-084/#93：plan_files自己解析原index，绕过已修load_index。12个隔离小探针中10坏输入：8被接受、2泄漏AttributeError/TypeError；同样10坏输入主load_index均拒绝，另2合法控制通过。`../../external`真实纳入Study外config摘要并允许write_provenance覆盖旧合成来源；外部输入字节不变。2个真实file symlink（index/config各1）也接受外部target。只读合成目录，不触碰正式Run/数据；不是B077主reader重修。

先开Issue93，再提交红3804bce：67节点56 failed/11 passed、0error/skip；48坏index在plan、write-new、write-old、freeze四入口拒绝，8外部链接拒绝（含哈希调用守卫），11合法空/partial/缺config、内部链接逻辑键、缺index与RuntimeError控制。期望复用既有load_index/validate_run_directory/_inside，坏类型仅在plan数据边界转换成ValueError保cause，使既有freeze正确拒绝；必需Run id与物理输入限制在Study内，内部链接仍以逻辑Run键记录。不改index主reader/schema/layout、hash算法、缓存或旧成功决策，不扩make全局事务，RuntimeError不吞。设计先写flow，再最小生产；原ZIP/REPRO/LINK_REPRO/红XML/命令/hash在.tmp/audit084，不随clone。

按Enmao“按类统一修”把#91与#93合在同一未合分支和一个来源证据PR，先前修复提交不改写，新增红先于其生产修复。接着组合回归及最终head/实际dev独立验收，不为同一来源类开零碎PR、不得拿旧717/core920/Flow175冒充新代码通过。当前没有B085已确认缺口。

### 103. 研究规范合dev、activity实际验收与来源声明红合同（B-083 / #91）

记录日期2026-10-16。B-082/#86的PR88实际dev `8ee1913de69871692c99d7228384730f81c18bb9` 的[Unit38775349489](https://github.com/DreamSoul-AI/RPipe/actions/runs/38775349489) / [Package38775349361](https://github.com/DreamSoul-AI/RPipe/actions/runs/38775349361)全部实际矩阵及聚合成功，DEV_CHECKED SHA256 `7096b74293fb906fa298d35200dc015371d6f80e2da0979500f3be8148a0f458`。核对tip63e2c84为dev祖先、无开放PR后普通删除远程`codex/activity-hint-fields`，本地保留，main/dev与活动文档分支不因删除改变。PR/Issue验收已追加，Issue86保持open；本目标至此共六条结束远程删除，前五条见§101。

[PR92](https://github.com/DreamSoul-AI/RPipe/pull/92)最终head `eadb39219d4c3c1425f0fc087a1127e843b38f58` 的[Unit38776167214](https://github.com/DreamSoul-AI/RPipe/actions/runs/38776167214)、[Package38776167175](https://github.com/DreamSoul-AI/RPipe/actions/runs/38776167175)、[Branch38776167404](https://github.com/DreamSoul-AI/RPipe/actions/runs/38776167404)全部实际矩阵与聚合成功，PR_CHECKED SHA256 `dd10585ffaa472c9fe804f377400c72c5f926e2021ddd76800eb78a777a50be4`，普通merge为dev `f303d72`，实际dev独立观察中；完成后清文档remote，不冒充已验收。DEM中文正本/英译/入口及两个独立调研已入dev，source字节与换行checkout证据见§102。监控调研固定dayan a7a1b40已用GitHub commit API确认远程存在，不依赖未推本机commit。#89/#90是探索议题，不是待立刻实施的bug，不与正确性修复或模板绑定。

新B-083/#91真实10种非法声明被接受（4种provenance容器、6种include值），缺省/null/空mapping/list与有效列表共6控制，True block按已有规则ValueError拒绝为控制。单文件合成包与临时Study避免扫描正式输入，environment/Git采集mock；直接write_provenance把旧合成provenance覆盖，旧SHA256 `83c56e89413dfa79b426c211b5777e9f364a1ff99947869efdb09a52ece7e6d1`、新 `7ddc267434d52ab14259c45a45141ba5fafe50c9ac8bb7334651e3f25096dbeb`，不声称正式数据已损坏。

先开Issue91，等待文档CI时仅在.tmp冻结快照加portable测试，实际导入该快照src、生产provenance字节与活动基线一致。38节点30 failed/8 passed、0error/skip：10项source拒绝、20项有/无旧target发布拒绝及peer保全、8项合法/历史声明控制（含glob/Path）。源码/新增测试SHA及真实导入路径在RED_SNAPSHOT_V2_CHECKED，XML/日志/命令/REPRO/BEFORE.zip全留.tmp/audit083，不随clone。首快照git archive缺--format=zip，tar送入ZIP reader产生BadZipFile，pytest尚未启动，错误及原tar字节保留；修正导出后才执行红回归，不计为库失败/通过。文档PR合后从新dev建唯一未合codex/provenance-declarations，同字节测试由d1baff1先提交，尚未修生产；未重复跑红测试来挑结果。

设计先明确在既有source_files区分None/非法类型、复用ValueError及下游既有拒绝边界；write_provenance发布前失败保全原来源/peer。合法声明/模式/键域/hash和错误/冻结/旧成功决策不变，不将make既有config/index步骤改成多文件事务、不加新模块/布局/缓存。这是同类声明根边界统一，未散加CLI补丁，不测速/GPU或干预下游。下一步最小修复与相邻/core/Flow正确性验收。

### 102. DEM 研究计划中文正本、英译与两个独立调研议题

记录日期2026-10-16，按Enmao新增授权接续文档任务，正确性目标仍active。B-082/#86 的[PR88](https://github.com/DreamSoul-AI/RPipe/pull/88)最终head `63e2c84097a4db34fc4b5970c8d6be40215be1bb` 的[Unit38774244145](https://github.com/DreamSoul-AI/RPipe/actions/runs/38774244145)、[Package38774244266](https://github.com/DreamSoul-AI/RPipe/actions/runs/38774244266)、[Branch38774244169](https://github.com/DreamSoul-AI/RPipe/actions/runs/38774244169)全部实际矩阵与聚合成功；PR_CHECKED SHA256 `6b36a135c1779f38e8911fcb03cf91fbba16b382f9071c75d23b09eeaeec6ef8`。普通merge为dev8ee1913，实际dev独立验收后立即清远程，不拿候选CI代替；Issue86等下版。

研究规范从用户指定的本机_staging_from_DEM实际读取，未直读Office DEM原路径。两中文文件按原字节复制并保留中文文件名至[正式入口](../research/standards/README.md)，ASCII别名与中文名一致；规范184行/25618字节 SHA256 `61fc9fbe1130810b41ef25d3655eef2f83289b60306458ad5e4c17b9d4658bd3`，模板122行/9309字节 SHA256 `b53a7250fd57f754a57a2ac3ddb918eefe7a0ac7a4c28096f39c858486370f44`。保留中文正本编码/换行/内容/链接；首次add出现autocrlf将转CRLF提示，未把提示当成已转换，随后仅两份原件设置.gitattributes -text并核对Git blob/工作区/原件逐字节一致，避免未来checkout改变正本。新增规范8章与模板10章完整英译；逐章/表核对72与29表格行数量，数量检查不替代语义阅读。research.md明确指向四文件，中文优先，宏观计划与Study PLAN原分工不变，未重写已有Study。新增docs子入口在layout的文档范围说明，不改运行目录/产物。

原规范的四处内部链接涉及三类缺失DEM依赖：文献规范（2处）、通用调研规范、校准调研大纲；为原样保留不改正本，在README逐项说明不可随clone访问，译本用明确依赖说明。其他本次文档相对文件引用全部存在，UTF-8无替换字符、fence配对，中文逐字节一致；36项既有layout正确性测试通过，未新增镜像实现测试，不跑速度/GPU。完整DOCS_CHECKED/LAYOUT XML/命令/来源hash在.tmp/research-docs（不随clone），核心生产/测试/Study/版本/发布材料无变更。

两议题分别开[探索Issue89](https://github.com/DreamSoul-AI/RPipe/issues/89)与[探索Issue90](https://github.com/DreamSoul-AI/RPipe/issues/90)，不是已确认bug、不占B编号，不互相或与模板绑定实施。[跨项目shared调研](../issue/cross_project_shared.md)比较现有shared.root与表层shared symlink，保留跨Study不可变输入/Run状态独立，尚未部署/迁数据；[监控logger调研](../issue/experiment_monitoring.md)只读RPipe与legacy-dayan固定a7a1b40源码以及Python/Microsoft/PyTorch/TensorBoard/MLflow/W&B官方文档，区分文本logger、指标tracker和展示；建议先明确缺失需求，再决定是否最小展示适配，不迁回旧聚合/恢复语义、不启服务或加依赖。下游文件与Git blob经换行归一一致，未干预运行。原文二级论文/skill未重新核验或安装，不把原作者核验日期当本次行为。

暂存保留对照，未纳入Git、未删除；仅docs主题分支，按本次“模板任务可并行开PR”的明确授权独立提交，不等待前批已合PR的实际dev观察；只有一个未合工作分支，结束旧远程仍在实际dev验收成功后立即删。额外用core.autocrlf=true的checkout-index仅导出两份正本再按原字节核对通过，未触碰其他文件。已确认的新provenance声明反例已开Issue91，38合同节点仅在.tmp预备/语法检查尚未实际执行，后续红回归先于修复，不与文档PR混改。

### 101. activity 字段候选本地验收与第五条结束远程清理

记录日期2026-10-16。B-081/#85 经[PR87](https://github.com/DreamSoul-AI/RPipe/pull/87)普通 merge 为 dev `37d0262b58fd1ac8d833b04fbb0fdcdcfce61235`，最终 head `49fbf56f9d595728a32b0cf2a1f169999636cecb` 的 Unit38770856180 / Package38770856256 / Branch38770856234、实际 dev [Unit38771938958](https://github.com/DreamSoul-AI/RPipe/actions/runs/38771938958) / [Package38771938853](https://github.com/DreamSoul-AI/RPipe/actions/runs/38771938853) 全部实际矩阵及聚合成功。DEV_CHECKED SHA256 `726461b0a65d27999523f96c7495b773a5f034836dbaafd33de374578afb3a69`；已更新原 PR/Issue 验收正文，Issue85仍open至下版。

已证明旧tip纳入dev、无开放PR、目标不是当前工作分支后，普通删除远程 `codex/result-status-contract`（tip49fbf56）。本目标累计结束远程五条：`codex/artifact-contracts-correctness`、`codex/yaml-mapping-roots`、`codex/declared-batch-contract`、`codex/saved-plan-read-boundary`、`codex/result-status-contract`。清理前后与命令保存各批.tmp；本地旧分支保留，main/dev未因清理改变，不重复计历史§82九条。

B-082/#86：红313f826为25项21失败/4控制，文档c16a7e2后最小生产59785be仅在read_activity增加非空白str phase及可选str/null detail检查（+4/-1）。坏提示沿既有不可用分支，合法自定义/Unicode阶段、缺省/null/空detail不变；未改writer/clear/CLI、阶段枚举、TTL或所有权。25项全部执行并转绿；相关436 passed、core882 passed/1 POSIX mode平台skip、Flow175 passed，0fail/error；CPU子进程隐藏CUDA，不是GPU/科学验收。核心XML共883节点，skip不计通过。生产仅activity.py变化，layout/版本/发布正文/Study均无变更。完整红绿/XML/命令与LOCAL_CHECKED在`.tmp/audit082`，不随clone；待本批最终head及实际dev CI验收，不拿旧CI替代。

本机出现与本次变更无关的未跟踪`_staging_from_DEM/`，保持原样，未打开/移动/删除/忽略。临时CI checker只允许这一条精确未跟踪状态或全干净，仍拒绝任何tracked/staged改动及其他未跟踪项；不更改仓库配置或必需CI。目标active，下一批仅针对已复现的provenance声明类型缺口，原始REPRO/BEFORE.zip在.tmp/audit083；不把“新反例”计为已修。

### 100. 可选activity字段类型与权威index的边界（B-082 / #86）

记录日期2026-10-16。上一批[PR87](https://github.com/DreamSoul-AI/RPipe/pull/87)最终head49fbf56f9d595728a32b0cf2a1f169999636cecb的Unit38770856180/Package38770856256/Branch38770856234全部实际矩阵/聚合成功后普通merge dev37d0262b58fd1ac8d833b04fbb0fdcdcfce61235，实际dev独立验收中；状态修复remote待该验收后删。当前唯一未合codex/activity-hint-fields，继续已确认#86。

1. 先Issue86及9个真实小探针：6个坏字段（phase布尔/数字/list/object/纯空白或detail object）均被当有效提示，缺index时status退出0；合法缺detail/null detail为2控制，空dict不可用/退出2为1控制。提示原字节全部保留，无模型/真实进程/GPU/ACL变更。旧B-068 stat/编码错误已修，本轮是独立最低结构缺口，不重复登记。
2. 红回归313f826先于生产：25节点21 failed/4 passed、0error/skip。10非法形状各覆盖reader与缺index CLI（20红），另1合法index/peer不应有坏前缀的红；4个Unicode/custom/缺detail/null/空detail兼容控制，返回原mapping、不归一化。准备稿先在隔离stagecheck编译，未把编译计通过；实际RED.xml/log/命令/计数与原源码ZIP/REPRO在.tmp/audit082，不随clone。
3. 设计先写flow.md现有activity合同：仅在read_activity验证根mapping后phase为非空白str、detail缺省/null/str，坏字段按原可选不可用None处理。合法index继续权威显示，缺index保留原错误2；合法提示仍可让尚未生成index的make进度可见。保留自定义非空phase和原字节，不新建阶段enum、owner/PID/TTL或读时修复。
4. 已核对全部调用方：reader仅CLI status；announce在CLI make和data.prepare调用，现有参数都是文字/f-string，producer、clear_activity、阶段/路径/磁盘格式均不改。不反向依赖readout或新建helper/module，原RuntimeError仍上抛，stat/read/解码处理保持。生产修复待做，#77/#79/#81/#82已修dev、#85已合dev验收中、#86当前红合同；全部等后版关Issue。
5. 目标active，按当前唯一未合分支→PR→dev与最终head/实际dev全部实际CI接续，再删已结束remote；本地旧分支与数据/Run/provenance保留。无测速/GPU/480s调查、layout/文件结构变化、main合并或tag，不因准备稿或历史pending重复派发已有验证。

### 99. 可选缓存最终验收、第四条远程清理与状态候选

记录日期2026-10-16，目标active。先清已结束分支，再推进已确认正确性队列，不合main/打tag或关闭待下版Issue。

1. B-080/#82已通过[PR84](https://github.com/DreamSoul-AI/RPipe/pull/84)普通merge dev6863f7ffe5f2e8b3e1883e39dfd5106cdb234493。最终head1516a8bacc0188c210f92ecff4aec0354fdb1c5d的[Unit38768342129](https://github.com/DreamSoul-AI/RPipe/actions/runs/38768342129)/[Package38768342472](https://github.com/DreamSoul-AI/RPipe/actions/runs/38768342472)/[Branch38768341996](https://github.com/DreamSoul-AI/RPipe/actions/runs/38768341996)及实际dev的[Unit38769261653](https://github.com/DreamSoul-AI/RPipe/actions/runs/38769261653)/[Package38769261884](https://github.com/DreamSoul-AI/RPipe/actions/runs/38769261884)全部实际矩阵/聚合成功；DEV_CHECKED SHA256 2157b2029574020fde2afbe73d092919eb80e15e243b9dbad8b6d5e8abf471d4。观察3302/28794均正常结束，Issue82/PR84正文补齐验收，不重派流程。
2. 实际新增删除远程codex/saved-plan-read-boundary，旧tip1516a8bacc0188c210f92ecff4aec0354fdb1c5d已是dev祖先、无活动PR引用；普通push --delete。删除后当时仅main00e70b1/dev6863f7f，随后仅推当前codex/result-status-contract；本地旧分支和数据保留。本轮新增累计4条（前三条详§95/97），早前9条另记§82。删除前/日志/复核在.tmp/audit080，不随clone。
3. B-081/#85本地候选完成：红fc704fa为29节点16 failed/13 passed，设计50e18f5先于fixcd06c5d（两原文件+9/-2，复用VALID_STATUS，不改通用load_result或process schema）。402相邻通过、core857通过/1 POSIX权限平台skip、CPU Flow175通过，0fail/error，29新增节点均执行。原writer/peer/输入字节保留；status合法/legacy/partial7控制不变。LOCAL_CHECKED与原RED/RED2/源码ZIP在.tmp/audit081，不随clone；最后head/实际dev以本批PR正文实际CI验收为准。
4. 接续真实B-082/[Issue86](https://github.com/DreamSoul-AI/RPipe/issues/86)：9个小探针中6个坏activity字段被当有效，缺index时status错误退出0；2合法提示、1空dict不可用为控制。所有原字节不改，未启动进程/GPU。仅需既有read_activity最低字段类型检查，不建阶段enum或ownership/TTL；writer现有调用均为字符串，保留自定义非空phase与缺省/null detail，不重构。已先Issue，便携红测试/生产修复尚未做；原证据在.tmp/audit082。该形状缺口不同于B-068已修的IO错误。
5. main仍已发布v0.4.1/00e70b1；五个开放Issue77/79/81/82/85中前四已修dev、85当前候选，86新确认队列，均留到后版关闭。未测速/GPU/480s调查、重训Study、改layout/磁盘格式、覆盖vendor或删数据/Run/provenance；继续目标，不卡等待反馈。

### 98. Run状态字段校验与只读有类型视图（B-081 / #85）

记录日期2026-10-16。上一批[PR84](https://github.com/DreamSoul-AI/RPipe/pull/84)最终head1516a8bacc0188c210f92ecff4aec0354fdb1c5d的Unit38768342129/Package38768342472/Branch38768341996全部实际矩阵/聚合成功后普通merge dev6863f7ffe5f2e8b3e1883e39dfd5106cdb234493，实际dev独立验收中；codex/saved-plan-read-boundary待通过后清理。当前唯一未合codex/result-status-contract，继续已真实确认的#85，不恢复测速或布局重构。

1. 先8个隔离输入探针（.tmp/audit081/REPRO.json）：三个容器status使validate_result和write_result抛unhashable TypeError；七个非法字段在status没有损坏诊断，其中list-label/float/未知queued造成状态计数总和小于planned。null是legacy pending控制。所有旧target/输入/peer字节不改，实际损害是校验/展示合同，不夸大为已发生数据丢失。
2. 已先[Issue85](https://github.com/DreamSoul-AI/RPipe/issues/85)，再红测试fc704fa：29节点16 failed/13 passed、0error/skip。validator的3容器为红、6其他非法值原errors为控制；writer有/无旧target各3容器为红并核对peer/无staging；读out7非法字段为红、7缺失/null/空/pending/成功/失败/partial为兼容控制。首RED诊断assert用了None的in导致表达式不清晰，改成显式短路并保留RED2，同样红绿计数，生产未动；不把这个断言表达式错误混记库TypeError来源。
3. 设计先写structure.md：validator先验str再使用既有VALID_STATUS，状态枚举/写入异常接口不变；status在现有list_runs针对所读Run状态字段复用同一枚举，非法字段按已有文件unreadable处理，保留计划mode/seed、peer与输入。显示保留现有缺失/null/空/pending、合法部分映射兼容，不扩大未知状态标签；不把Run写validator用于通用process映射或修改全局load_result。
4. 已搜索所有validate_result直接调用方（write_result）以及读出list_runs/CLI；不改load_result、process科学聚合、schema/持久格式或目录布局。原字节ZIP/红日志在.tmp/audit081不随clone，portable合同已入库。旧任务的head/实际dev分别验收、remote已结束再删；所有Issue等后续版本才关。目标active，不弹窗、不测速/GPU、不合main/打tag、不删数据/Run/provenance。

### 97. 批次证据保护最终验收与可选缓存候选完成

记录日期2026-10-16。目标模式active；先处理已确认队列，所有四个Issue77/79/81/82保持open，已合dev不等同下版正式发布。

1. B-079/#81通过[PR83](https://github.com/DreamSoul-AI/RPipe/pull/83)普通merge dev4536ec497cce48c56db77ead6f8ce3009a186142。最终headf617d4366ebf9570080996dc36fdba3d1b2bff75的[Unit38765832139](https://github.com/DreamSoul-AI/RPipe/actions/runs/38765832139)/[Package38765831882](https://github.com/DreamSoul-AI/RPipe/actions/runs/38765831882)/[Branch38765831857](https://github.com/DreamSoul-AI/RPipe/actions/runs/38765831857)及实际dev的[Unit38766766425](https://github.com/DreamSoul-AI/RPipe/actions/runs/38766766425)/[Package38766766311](https://github.com/DreamSoul-AI/RPipe/actions/runs/38766766311)均全部实际矩阵/聚合成功。DEV_CHECKED SHA256 b99811e3e81e77e47248049f072fd88319ff7eaedeee5f76a39582924ecdc840，Issue81/PR83正文已补全验收。
2. 一次查询故障：观察29334因`gh pr view83`的GraphQL POST EOF退出1，最后已知Unit38766766425为pending，未重派CI、未当测试失败/通过、未提前删分支。DEV_OBSERVER_NETWORK_EOF.json保留；退避30秒后重查同组既有流程，全实际通过才验收。不是自动审批拒绝，不需要弹窗或调整计费。
3. 实际新增删除远程codex/declared-batch-contract，删除前tipf617d4366ebf9570080996dc36fdba3d1b2bff75已是dev祖先，无活动PR引用；普通push --delete后main00e70b1/dev4536ec4未变，本地分支与数据保留。本轮累计新删3条：artifact-contracts-correctness、yaml-mapping-roots、declared-batch-contract（均codex/前缀，逐项SHA/验证见§92/95/本节）；早前9条另见§82。当前唯一未合codex/saved-plan-read-boundary保留，正常推送。原始删除前/日志/复核在.tmp/audit079，不随clone。
4. B-080/#82候选完成：红9add4cc为22节点10 failed/12 passed；设计ef11eda先于fixaf56370（load_launch_plan原函数+10/-8，缓存stat/读取/解码/存储数值转换归入现有错误边界）。371相邻通过、core834通过/1 POSIX权限平台skip、CPU Flow169通过，0fail/error，22新增节点全执行；输入/peer字节不改，非法当前参数与原RuntimeError仍抛出。既有batch一致性、合法缺省/整数文本/round空值与模式/依赖/retry不变。LOCAL_CHECKED/原红/源码ZIP在.tmp/audit080，本机证据不随clone。最终head与实际dev以本批PR正文实际CI验收为准。
5. 只读核查dayan、量化、output-space三会话最新反馈，内容为其各自研究/集成；未从中取得新的已复现RPipe反例，未发任何消息、覆盖vendor或干预实验。本批没有测速、GPU、480s调查、布局/格式迁移、main/tag或数据/Run/provenance删除；保留原失败、网络错误与全部旧证据。

### 96. 可选缓存读取的完整文件错误边界（B-080 / #82）

记录日期2026-10-16。上一批[PR83](https://github.com/DreamSoul-AI/RPipe/pull/83)最终headf617d4366ebf9570080996dc36fdba3d1b2bff75的Unit38765832139/Package38765831882/Branch38765831857全部实际矩阵/聚合成功后普通merge dev4536ec497cce48c56db77ead6f8ce3009a186142，实际dev独立验收中。当前codex/saved-plan-read-boundary是唯一未合分支，先处理已确认#82，不重开已修Issue或转去无反例新任务。

1. 先Issue82及.tmp/audit080的8小探针：坏UTF8、坏init_gpu文本、null num_gpus、坏round文本、Infinity round、注入stat拒绝共6异常逃逸；坏JSON→None及合法空计划为2控制。只有7个物理缓存，均保留原字节；stat为显式注入，不改真实ACL，不启动子进程/GPU。
2. 便携红回归9add4cc先于生产改动：22节点，10 failed/12 passed、0error/skip。8个存储数值错误、1坏UTF8和1stat拒绝为红；已有坏JSON/坏根、read拒绝、4合法缺省/整数文本、3非法当前调用参数、意外RuntimeError同对象传播及缺缓存为控制。真实Run/数据不参与。
3. 设计先写structure.md：仅扩大现有load_launch_plan读取边界，stat/read/jsondecode/已存数值转换归到同一明确OSError/TypeError/ValueError/OverflowError处理，返回既有None让CLI依原逻辑make；只读loader不重写缓存、不宣称完成恢复。当前调用参数转换仍在错误边界外，合法整数文本、round缺省/null/0兼容不改，程序RuntimeError不吞。
4. 调用方为CLI复用计划入口、生成PowerShell经CLI；producer是现有write_jobs_json/write_launch_scripts，B-079显式jobs/batches一致性与过滤/依赖/retry规则不改。不新增helper/模块/API/格式，不做其他缓存schema或资源规划改动；复用已有函数和错误分类。
5. 原源码/测试字节ZIP、REPRO/RED.xml/log/命令/计数在.tmp/audit080，不随clone；portable合同已入库。#77/#79/#81修复后仍open等后版，当前#82生产修复待做。目标active，不测速、GPU、main/tag、数据/Run/provenance删除；旧结束远程在实际dev通过后即清理并保留清单。

### 95. 两批dev验收、远端清理及持续正确性队列

本节记录本地日期2026-10-16的收尾与接续；§92—94中的2026-10-15开发/复现事实保留，不改写为当天新成果。目标模式active，目标为持续正确性；不以已发布v0.4.1或某批CI完成当作等反馈空档。

1. B-078/#79最终验收：[PR80](https://github.com/DreamSoul-AI/RPipe/pull/80)最终head dd8c7f8e72ff28601b9a3d9a8e523b74bbdec28b的[Unit38763144958](https://github.com/DreamSoul-AI/RPipe/actions/runs/38763144958)/[Package38763144991](https://github.com/DreamSoul-AI/RPipe/actions/runs/38763144991)/[Branch38763144864](https://github.com/DreamSoul-AI/RPipe/actions/runs/38763144864)全部实际矩阵/聚合成功。普通merge devf4acb936ce66ac1db4907bccb4813a073cf80f71，实际dev的[Unit38764359287](https://github.com/DreamSoul-AI/RPipe/actions/runs/38764359287)/[Package38764359383](https://github.com/DreamSoul-AI/RPipe/actions/runs/38764359383)亦全部实际成功；DEV_CHECKED SHA256 52449a5e141b98ca03eb5bdb4c5607ee0ac58a739ed2999c7ce871a975821e12。Issue79与PR80正文补齐，Issue保留open到下版。观察4196/52389已正常结束，不重复派发。
2. 实际新增远程删除清单：codex/artifact-contracts-correctness（959f2a5675b1f9b309a53d4ee592c7faa659b73d，PR78/实际dev1667f51完成后删，详§92）；codex/yaml-mapping-roots（dd8c7f8e72ff28601b9a3d9a8e523b74bbdec28b，PR80/实际devf4acb93完成后删）。均先核对祖先、没有活动PR引用、main/dev未动；普通git push --delete，无force。删除后当时远端仅main/dev，随后只推当前未合codex/declared-batch-contract。本地旧分支及数据/Run/provenance不删。此前9条清单继续见§82，不把它们重新计成本次新增删除。
3. B-079/#81本地完成：红5597727为23 failed/3 passed；设计aa6b56b先于生产fix0f57ddb（schedule.py+12/-1，既有函数内共享一次分组规则及两处调用）。26新增节点全部执行，349相邻通过、core812通过/1 POSIX权限平台skip、CPU Flow169通过，0fail/error。保护原native peer/目标/旧scripts字节；旧重复与路径、合法重排、模式/已成功过滤、retry屏障不变。LOCAL_CHECKED与原红/旧字节在.tmp/audit079，本机证据不随clone；最终head/实际dev仍按下一PR实际CI验收。
4. 接续真实B-080/[Issue82](https://github.com/DreamSoul-AI/RPipe/issues/82)：可选jobs缓存8个隔离探针中6个stat/编码/缓存数值转换异常逃逸，坏JSON返回None与合法空计划可用为2控制；输入不改、真实ACL/进程/GPU均未动。已先Issue，尚未写便携红测试/修生产；根因是现有可选缓存错误边界缺口，不借此重构schema或资源规划。原字节ZIP/REPRO在.tmp/audit080。按当前任务单一未合分支接续，不与#81候选混做已完成。
5. 仅CPU正确性，main保持00e70b1f83ac23d113b9582a4270a41cfadb1444及v0.4.1；不测速、GPU、480s调查、科学重跑、布局/格式迁移、main合并或tag。#77/#79已修dev仍open，#81当前候选、#82新确认队列；均留待之后版本关闭。无审批弹窗、花钱或数据删除。

## 2026-10-15

### 94. 声明jobs与显式batches一致性，先拒绝再写入（B-079 / #81）

记录日期2026-10-15。目标模式按Enmao明确指令创建active，不待命、不弹窗。PR80最终head dd8c7f8e72ff28601b9a3d9a8e523b74bbdec28b的Unit38763144958/Package38763144991/Branch38763144864全部实际矩阵/聚合成功后普通merge dev f4acb936ce66ac1db4907bccb4813a073cf80f71，实际dev独立验收中。首次合并工具误传短SHA被GitHub422拒绝，未发生合并；使用验收的完整40位SHA重试成功。MERGE_INVALID_ARGUMENT保留，不当自动审批拒绝、CI失败或强推。

1. 新真实反例：schedule.wait_groups不核对显式batches与jobs，native启动batches投影；jobs仅planned却mock启动peer，并在启动前把未列入任务的旧成功peer失效。只有隔离文件和Popen替身，真实子进程0，GPU/训练/测速0；旧字节和前后hash在.tmp/audit079/REPRO.json/PEER_BEFORE.bin，原始源码/测试在BEFORE.zip，不随clone。
2. 生成脚本也绕过原jobs：最终只嵌入foreign peer groups，再从groups重建jobs，planned消失。只生成文本、没有执行脚本。首补充调用漏round_size/split_round的TypeError保留并纠正；“同时包含planned/peer”的探针假设为False，已按实际脚本修正Issue正文与SCRIPT_CHECKED，不据该假设作结论。
3. [Issue81](https://github.com/DreamSoul-AI/RPipe/issues/81)先创建；随后从dev f4acb93建立唯一未合分支codex/declared-batch-contract，红测试5597727共26节点、23 failed/3 passed、0error/skip。三控制为合法同批、合法重排、原生已有重复拒绝；红例分根校验、原生失效/启动前保护、脚本旧文件保护、渲染、缓存include_done两分支，全部保存，不挑通过。
4. 设计先写structure.md：只在既有wait_groups归纳显式批次的完整多重集与字段一致性；write_jobs_json发布前和load_launch_plan成功过滤前复用。合法重排/成功过滤/缺省batch/retry/依赖合同不变，缓存不一致沿既有None重建路径，直接native/生成明确ValueError。全部调用方wait_groups→native/render、write_jobs_json→write_launch_scripts、load_launch_plan→CLI已核对；不搬模块、新增格式或承诺多文件事务。
5. 此批只修有红回归支撑的批次一致性，不实施猜测中的其他缓存字段/schema扩展；现有重复Run与路径检查保留。上一批YAML远程分支在实际dev验收通过后清理，本地旧分支和当前未合分支保留。#77/#79/#81修复Issue留到下版，不合main/打tag。

### 93. 严格YAML mapping读取的空值判定统一（B-078 / #79）

记录日期2026-10-15。PR78已通过最终head959f2a5所有实际Unit38761019602/Package38761019537/Branch38761019535后普通合dev1667f517d8585627ba04639ba14f7612ead8b885；实际dev CI另行验收中。不开等待反馈空档，从该dev建立唯一未合工作分支codex/yaml-mapping-roots，上一条已合分支待实际dev验收后按新授权清理，当前分支保留。

1. 归类：继续第二类共享解析边界，确认decode_config和load_study_yaml两处同一`safe_load(...) or {}`反例。各7个隔离输入中false/0/[]/空字符串误接受，空文档/null/{}是兼容控制。四个坏Run配置令status误显示train且无error；原输入均未改写，不代表真实正式数据已损坏。已先创建[Issue79](https://github.com/DreamSoul-AI/RPipe/issues/79)，不重开旧Issue。
2. 设计先写structure.md：复用现有类型验证，只把truthiness空值判定改成None判定，保留两入口异常类型与合法mapping。load_config所有调用方和load_study_yaml调用方均搜索核对；stats可选loader已对同类输入返回None，明确不改。文件架构/layout/产物格式、科学聚合、异常分类和默认train合法配置语义不变。
3. 红测试先提交9f4d796：两严格parser各6非法根值（4falsy反例+true/1已有拒绝控制）与4空文档/null/{}兼容控制，另4个真实status peer/字节保全，共24节点、12 failed/12 passed、0error/skip。源码原字节ZIP、探针与RED.xml/log/命令/计数在.tmp/audit078，本机证据不随clone；portable回归已入库。
4. 两处最小修复5d15070已完成（+4/-2），323相邻全部通过、完整core786通过/1 POSIX权限平台skip、CPU Flow169全部通过，无fail/error；24新增节点全部执行，所有旧字节/peer保全。LOCAL_CHECKED记录精确范围，非GPU/科学数值验收。随后按类一PR进dev；不测速/GPU、不合main/打tag、不删除数据/Run/provenance。#77/#79均保留至后续版本，最终head/实际dev验证分别记录，不把历史pending当新失败。

### 92. v0.4.1后共享index容器合同缺口（B-077 / #77）

记录日期2026-10-15。开始时工作区干净、fetch成功，HEAD与origin/dev均为2c8b6fbeab617dfabfff3db03789b21f6a29c23c；GitHub开放Issue为空。按最新授权继续三类正确性复查，不重开已发布Issue。本批锁定读取类共同边界，不修改发布main00e70b1或v0.4.1标签。

1. 读取入口：已逐一搜索load_index的status/logs/report、Study process、resume、schedule和main_probe调用；根对象与Run路径已有检查，中间experiments/runs/factors的容器检查缺失。status/logs的既有CLI只捕获文件/类型/值错误，标量条目的AttributeError逃逸；空object则经`or []`变成假空计划。
2. 真实小型复现：两个非法标量条目分别调用status/logs，4次AttributeError；空experiments object与列表对factors分别调用两命令，4次误成功，均未改输入。只用隔离JSON，不运行阶段、GPU或性能测量。原始源码/测试/设计字节在.tmp/audit077/BEFORE.zip，探针在REPRO.json；临时材料不随clone提供。
3. [Issue #77](https://github.com/DreamSoul-AI/RPipe/issues/77)先于测试/修复创建。红测试提交f85f6bb先于生产改动：10个共享loader拒绝、8个实际CLI拒绝、6个缺省/null/合法空容器兼容控制，总计24节点，18 failed/6 passed，0error/skip。RED.xml/log/COMMAND和RED_CHECKED保留，不将红例当最终CI。
4. 方案先写入structure.md：仅在现有load_index验证list/object，保留返回原mapping、不补字段、不迁移、不修改生产者或布局。所有调用方统一收到现有TypeError；不增加展示层逐处catch，不吞意外AttributeError。现有report额外科学身份检查、process必须有experiments列表的规则不放宽。
5. 相邻两类已检查入口：整体替换文本继续走_atomic；Logger/Tracker追加及checkpoint/lock的专用保全合同不改。未复现写入类新缺陷，不把诊断staging或显式单写者限制登记为bug。此批不宣称穷尽任意第三方扩展。
6. 分支清理复核（本轮新指令）：fetch后以ls-remote列出实际远端，仅main、dev、codex/artifact-contracts-correctness；当前工作分支明确保留，长期分支受保护，没有其他可删分支，本次新增删除0条。此前已删9条为artifact-consistency-design、cache-lock-budget、failure-preservation-policy、maintenance-closeout、open-issue-correctness、safe-readout-contracts、status-metric-shape、status-readout-errors、v040-layout-release（均codex/前缀），完整tip/祖先/删除证据见§82；本轮BRANCH_CHECKED保留实际三条远端SHA，不重复删除或恢复旧分支。
7. 生产修复823c9b0仅index.py增加15行/删除2行，在第一次dereference前验证容器；299项相邻CPU回归全部通过，0fail/error/skip，24项新增全部执行。第一次绿色命令误指不存在test_resume.py、收集前退出4，GREEN.log/命令保留，不当库失败或已执行回归；纠正为真实test_resume_sibling.py后GREEN2全部通过。CPU Flow现有句柄40809已正常结束，169 passed，0fail/error/skip；FLOW/GREEN等本机材料在.tmp/audit077，不随clone。最终工作提交与实际dev CI仍分别验收，不拿本机通过代替远端矩阵。


8. 最终验收：PR78最终head959f2a5675b1f9b309a53d4ee592c7faa659b73d的[Unit38761019602](https://github.com/DreamSoul-AI/RPipe/actions/runs/38761019602)/[Package38761019537](https://github.com/DreamSoul-AI/RPipe/actions/runs/38761019537)/[Branch38761019535](https://github.com/DreamSoul-AI/RPipe/actions/runs/38761019535)全部实际矩阵/聚合成功后普通merge至dev1667f517d8585627ba04639ba14f7612ead8b885；实际dev的[Unit38762035791](https://github.com/DreamSoul-AI/RPipe/actions/runs/38762035791)/[Package38762035520](https://github.com/DreamSoul-AI/RPipe/actions/runs/38762035520)也全部实际矩阵/聚合成功。DEV_CHECKED SHA256 ded731c77c1478b88880524e62148bfdb96a21d2e5ee55cc660bdebbf4271866；无存活验证句柄31287/40304，不重复派发。
9. 按最新“合dev且CI通过即删已结束远程”指令，实际删除codex/artifact-contracts-correctness（旧远端tip959f2a5已验证为dev祖先，无活动PR引用）；BRANCH_DELETE_BEFORE/log/BRANCH_DELETED_CHECKED保留。本地旧分支、main/dev及全部数据保留，正在用的新批codex/yaml-mapping-roots保留并正常推送。删除后当时远端仅main00e70b1/dev1667f51，随后新批活动分支推送另计；本次实际新增删除1条，早先“可删0”是该分支尚活动时的事实。Issue77与PR78正文已补全真实验收，Issue仍open，不关、不合main/打tag。

### 90. v0.4.1 正式发布完成与五项 Issue 关闭

按Enmao本次明确授权完成[发布PR75](https://github.com/DreamSoul-AI/RPipe/pull/75)：dev7025f62经独立最终/实际dev全实际CI后，发布PR的Unit38754510141 / Package38754509384 / Branch38754509522全部实际矩阵和聚合成功，普通merge main `00e70b1f83ac23d113b9582a4270a41cfadb1444`，保留两个父提交。实际main Unit38755466192 / Package38755466195全部实际矩阵/聚合成功后才发布，MAIN_CHECKED SHA256 0536b6fe48314724f0c948428b151bab69789f894b66dc0a7c62d139bafc29c0。

annotated [tag v0.4.1](https://github.com/DreamSoul-AI/RPipe/tree/v0.4.1) 的tag对象`66c2057b0ad3fd976a3a1e60de1e72330ecb80c0`，remote peeled commit精确指向上述main；[GitHub Release](https://github.com/DreamSoul-AI/RPipe/releases/tag/v0.4.1)编号413944175，published_at=2026-10-15T15:00:36Z（Asia/Shanghai为2026-10-15 23:00:36）。正文取main该提交的docs/releases/v0.4.1.md，只将14个相对链接转为精确提交链接并附发布PR/实际main验收；发布正文SHA256=92e3bc98a338d04164012ef3de842265efcac1ca92cd95a8892a0a2205eb40c8。无二进制/数据附件，只保留GitHub源码归档；未发布PyPI。

首次git push origin refs/tags/v0.4.1因Empty reply from server返回128；在核对远端尚无该ref、本地annotated tag目标不变后，等待并以同一标签对象重试成功。PUBLISH_ATTEMPT1_FAILED与TAG_REMOTE_AFTER_NETWORK_FAILURE证据保留；该失败是网络错误，不是审批拒绝、库失败或通过。没有强推、覆盖旧tag或删除分支/数据。

Release核对后依次留言引用同一Release并以completed关闭五Issue，ISSUES_CLOSED逐项读取state/stateReason核实；原“保持open到下版”已由本次授权发布完成满足，不自动重开：

| **Issue** | **最终状态** | **发布留言** |
| --- | --- | --- |
| [#53](https://github.com/DreamSoul-AI/RPipe/issues/53) | completed | [版本留言](https://github.com/DreamSoul-AI/RPipe/issues/53#issuecomment-6195181701) |
| [#62](https://github.com/DreamSoul-AI/RPipe/issues/62) | completed | [版本留言](https://github.com/DreamSoul-AI/RPipe/issues/62#issuecomment-6195183571) |
| [#66](https://github.com/DreamSoul-AI/RPipe/issues/66) | completed | [版本留言](https://github.com/DreamSoul-AI/RPipe/issues/66#issuecomment-6195185485) |
| [#67](https://github.com/DreamSoul-AI/RPipe/issues/67) | completed | [版本留言](https://github.com/DreamSoul-AI/RPipe/issues/67#issuecomment-6195187710) |
| [#68](https://github.com/DreamSoul-AI/RPipe/issues/68) | completed | [版本留言](https://github.com/DreamSoul-AI/RPipe/issues/68#issuecomment-6195190276) |

草稿PR73最终及实际dev全绿（实际dev Unit38751749854 / Package38751749836）；版本PR74最终head0227e78全实际Unit38753402509 / Package38753402914 / Branch38753402917与实际dev7025f62 Unit38754391374 / Package38754391691也全成功。旧head3633255全绿、main祖先补齐与旧观察者自然完成记录保留，不能替代最终head证明。36项本机版本/布局合同通过。所有原始JSON/XML/log、渲染脚本/正文hash、tag/Release/Issue核对在.tmp/v041-draft-20261015，不随clone；本节和入库发布报告提供便携事实。

### 91. 发布后近期三类修复的正确性复查

以正式main/tag的原字节为基线，复查近期PR61/63/64/69/70/71的交互边界，不重新做无差别全仓扫描。准备四项绿色合同控制：owned staging在chmod拒绝时仅清自己；Logger与Tracker已有文件不重开；Tracker初始化后begin_run保留原始失败后缀并用原offset追加marker。另加强已有compare --out拒绝回归，要求失败发布前不输出成功/失败摘要；重跑既有staging/引用/迁移/长名、status/logs/report/compare、计划模式/读取失败、冻结保全相邻集合。

这些是既有政策的回归控制，不将已通过的控制登记为新缺陷；下一编号B-077暂不占用。未调整生产代码、保留旧成功范围、原格式/目录、恢复协议或数值门。生产及Study跟踪文件逐字节hash保护，4个测试原字节ZIP保留；本地复查子进程CUDA_VISIBLE_DEVICES空串，仅CPU，不改系统或其他会话环境。实际执行与结论在本节补记，不以准备的测试替代执行。

实际复查：测试提交`4dd0c872701ad1feb0600cf49e06fb9090172411`，337节点中336 passed /1 POSIX executable mode bits平台skip，0失败/错误；四个新控制及加强的compare拒绝案例均实际执行。没有发现近期修复引入的新缺陷，不登记B-077、不改生产逻辑。原报告403函数组绑定先前扫描基线，本轮只增加验证控制，不把旧全仓表当作含新增测试的实时统计。

REVIEW_LOCAL_CHECKED核对全部464个跟踪生产/Study文件的原字节SHA不变；layout、版本、README和已发布v0.4.1报告均不变。旧四测试文件ZIP、命令/源码hash和XML/log保留于.tmp/v041-draft-20261015（不随clone）。首个追加脚本在compare已有“错误消息包含”断言锚点拒绝，当时未执行测试，三文件已追加内容保留；仅修正实际断言后首轮检查336/1skip通过，REVIEW_STAGE_ANCHOR_REJECTED保留，不当库失败或新缺陷。

本批仅四测试与四开发文档：发布报告事实/五Issue关闭、当前版本入口、近期复查结论。后续最终工作head和实际dev分别验收全部实际CI，结果以本批PR验收正文为准，不用发布前CI冒充新增控制通过；不重发Release、不改main/tag、不删分支或数据、不测速/GPU。

### 89. v0.4.1 正式发布授权与版本准备

Enmao明确覆盖前一阶段“仅草稿/不合main/不打tag/不发release”的限制：草稿PR73正常合dev23d6907aec49a5950ea4bd323a19280256323991，最终head7461522三项必需CI全部实际矩阵/聚合成功；实际dev独立验收。发布按工作PR→dev→main发布PR普通merge→实际main全实际CI→annotated tag v0.4.1→GitHub Release→逐项引用release关闭#53/#62/#66/#67/#68执行，不强推、不删分支/数据。

pyproject、rpipe.__version__与README同步0.4.1；release notes由草稿转为本次授权发布正文，逐项PR/迁移说明不变，不增加目录/schema/运行时改动。按“立即5行以内现状”要求已在本报告顶部记录当前步骤、数字与出处、下一步和无阻塞状态。已有四项近期交互控制草稿仅在.tmp准备，按最新顺序留到正式发布完成后执行复查，不混入本次发布候选。

版本/布局36项全部通过、无skip；VERSION_LOCAL_CHECKED确认生产仅__version__字符串改变且其余原字节相同，测试与layout无改动，14个发布正文相对文件链接有效。首个范围checker仅规范化working换行而未处理原blob同为CRLF，拒绝证据VERSION_SCOPE_CHECK_REJECTED保留；精确按原字节替换版本后核对相等，不当代码失败。原始XML/log位于.tmp/v041-draft-20261015，不随clone；候选与实际dev仍需各自CI。

发布前祖先检查发现dev缺main两个历史发布merge（54f8d31/v0.4.0与7a15c46/v0.3.1，main...dev为2/40）。merge-tree预演与原工作树tree完全相同、无冲突；经当前工作分支普通merge补齐main历史，不直接推dev/main。只改Git父关系，36项本机验证仍绑定相同代码/测试/版本字节。旧head3633255的CI/状态保留在.tmp/v041-draft-20261015/version-ci-original和BRANCH_ANCESTRY_BEFORE；旧观察者自然完成，保守终止检查未执行任何终止；最终head需重新验收，不能用旧head替代。

### 88. v0.4.1 Release Notes 草稿与近期修复复查立项

前轮收尾PR72实际dev e39f7f82c3d07fa600b62727fee17587d65338c6已全实际验收：Unit38749067743 / Package38749067736全部矩阵/聚合成功，DEV_CHECKED SHA256为37c43d95eea8ad874acccf7af9f7a6b1abc84cb7717e372d57385f30ce215eb8，工作区干净，保留工作分支已普通快进同步。

按最新授权在同一分支编写[下一版草稿](../releases/v0.4.1.md)：五个open Issue逐项列PR/回归/迁移影响，区分#53已在v0.4.0发布的原始index修复和PR61后续Study发布边界。三个统一合同包含PR64失败保全控制及PR69/70/71根入口修复；无文件/schema/shared位置迁移，明确status占位与compare退出语义变化。仅写docs，不升包版本、不合main/打tag/发release/关Issue；正式发布由父会话决定。

草稿经本地链接/事实核对与正常PR/CI合dev后，继续只读审查最近合入三类修复是否引入同类回归，再按真实反例决定是否需要红测试和最小修复。不重做全仓无差别扫描，不制造新缺陷，不改性能/文件架构/产物布局，不跑GPU。临时证据在.tmp/v041-draft-20261015，不随clone。

### 87. 三类系统扫描最终集成验收与下一版边界

7895169分支清理记录已由PR65合dev并完成最终head/实际dev全实际CI。此后按第三类→第二类→第一类各自先红回归、最小生产修复、独立PR普通merge：三批代码及相应便携回归均已在dev集成。下表工作提交与实际集成提交分别核对所有实际矩阵/聚合，不用rollup或旧提交替代。

| **批次** | **最终工作提交 / CI** | **实际 dev / CI** |
| --- | --- | --- |
| 第三类 / [PR69](https://github.com/DreamSoul-AI/RPipe/pull/69) | `0f393f1a48acf7f14824dd49af108f72548d0fe8`；[Branch flow 38742367302](https://github.com/DreamSoul-AI/RPipe/actions/runs/38742367302) / [Package Check 38742367353](https://github.com/DreamSoul-AI/RPipe/actions/runs/38742367353) / [Unit Tests 38742367339](https://github.com/DreamSoul-AI/RPipe/actions/runs/38742367339) | `b858c2ba689f1f8e77fe7065b53ee551fe42325d`；[Package Check 38743310055](https://github.com/DreamSoul-AI/RPipe/actions/runs/38743310055) / [Unit Tests 38743309828](https://github.com/DreamSoul-AI/RPipe/actions/runs/38743309828) |
| 第二类 / [PR70](https://github.com/DreamSoul-AI/RPipe/pull/70) | `02f6f05a5f9dd35fb58c9565a24305b37ae0923d`；[Branch flow 38744687921](https://github.com/DreamSoul-AI/RPipe/actions/runs/38744687921) / [Package Check 38744688048](https://github.com/DreamSoul-AI/RPipe/actions/runs/38744688048) / [Unit Tests 38744687993](https://github.com/DreamSoul-AI/RPipe/actions/runs/38744687993) | `869300bd446edb24cfd95bdd3bc9a3a65bf29702`；[Package Check 38745527623](https://github.com/DreamSoul-AI/RPipe/actions/runs/38745527623) / [Unit Tests 38745527574](https://github.com/DreamSoul-AI/RPipe/actions/runs/38745527574) |
| 第一类 / [PR71](https://github.com/DreamSoul-AI/RPipe/pull/71) | `6a25b3b9cf88ab47badfd9b5e06526d5c3465754`；[Branch flow 38746397054](https://github.com/DreamSoul-AI/RPipe/actions/runs/38746397054) / [Package Check 38746397615](https://github.com/DreamSoul-AI/RPipe/actions/runs/38746397615) / [Unit Tests 38746397175](https://github.com/DreamSoul-AI/RPipe/actions/runs/38746397175) | `fb022a41230ba5c11916941e3a25617b33f04d57`；[Package Check 38747297474](https://github.com/DreamSoul-AI/RPipe/actions/runs/38747297474) / [Unit Tests 38747297462](https://github.com/DreamSoul-AI/RPipe/actions/runs/38747297462) |

本轮最终代码dev为`fb022a41230ba5c11916941e3a25617b33f04d57`。本机最终core734 passed/1平台skip/173 deselected，CPU Flow169 passed/0skip；三批新增合同分别7、32、4节点全部执行，合计43节点。原失败（第三类6 failed、第二类26 failed/6 passed、第一类2 failed/2 passed）、首候选局部失败、未执行路径错误及全部核验拒绝保留，不挑通过覆盖历史。CPU门中的平台/可选依赖skip不是通过，不代表GPU或Study科学验收。

全仓215个受跟踪Python文件、1313候选调用的当前403函数组清单见§83（生产/Study146，验证257）；13个根边界按三类统一修复，无本轮已确认而未修的缺口。原始基线388组及逐字节源hash保留于.tmp/systematic-contracts-20261015；FINAL_INVENTORY_CHECKED与三批*_LOCAL_CHECKED/CI_CHECKED可执行复核材料不随clone，便携回归/清单/设计已入库。所有当前Python字节仍与最终扫描快照一致。专用追加、checkpoint恢复、锁、二进制/图片、外部builder与单写者边界已逐项注明；不把静态清单称为任意第三方扩展的完整证明。

#53/#62与本轮#66/#67/#68均已修并有dev回归，但故意保持open至下一版发版时关闭，不自动关闭。main仍v0.4.0/54f8d31；本轮没有合main、打tag、发release、测速、GPU、下载或正式Study重跑，layout与产物schema/含义不变，原数据/Run/provenance不覆盖。唯一工作分支codex/artifact-contracts-correctness按明确授权保留，已删九条旧远程分支不恢复，本地旧分支不删。后续只处理新证据或新立项正确性工作，不为维持活跃重复扫描/生成同质任务，性能与结构扩张仍不实施。

### 86. 第一类：追加诊断初始化保全与三类逐点收敛（B-075 / #67）

本轮复用同一工作分支。第二类PR70最终head02f6f05的Unit38744687993/Package38744688048/Branch38744687921全部实际矩阵与聚合成功，普通合dev869300bd446edb24cfd95bdd3bc9a3a65bf29702；实际dev独立验收。第一类portable红提交f3605f1：到达交错2 failed，目录拒绝2 passed，原构造器确实把旧证据变成空字节；FIRST_RED/XML/log保留于.tmp/systematic-contracts-20261015，不随clone。

最小方案只在Logger.__init__与AlgorithmTracker.__init__的既有is_file分支内用append创建文件；不把已存在只读文件提前重开，不改emit/flush、Tracker.begin_run的offset恢复、状态原子发布或Runner/schedule保全predicate。该分支内创建遇到目录仍抛原OSError，不吞程序错误、不增加锁或新目录。系统扫描发现的多个调用方继续委托这两个根入口，不复制补丁。

第一类生产提交e072305，仅两个原构造器4行增/2行删；相关162 passed /1平台skip、core734 passed /1平台skip /173 deselected（8条既有partial-resume warning）、CPU Flow169 passed /0失败/0skip。FIRST_LOCAL_CHECKED核对四个新节点均执行、输入到达字节与目录错误合同、生产仅两文件。首次相邻检查误写tests/rpipe/flow/test_schedule.py导致未执行/退出4，原FIRST_GREEN.log保留；实际路径structure/make/test_schedule.py的GREEN2及完整CPU门均通过，不将未执行误计失败或通过。

最终逐点复核215个受跟踪Python文件、1313个候选调用，共403函数组（生产/Study146、验证257）；13个已修根边界全部标明对应合同，无本轮已知待修项。§83表格按当前源码刷新，原388组清单、红例与源字节ZIP不覆盖。FINAL_INVENTORY_CHECKED绑定e072305源hash，403条相对文件链接均存在；layout的Git规范化blob与基线相同，未改目录/格式/含义。直接比较checkout CRLF与Git blob LF曾拒绝，FINAL_LAYOUT_BYTE_CHECK_REJECTED与原字节hash保留，随后分别记录checkout SHA和相同Git blob，不把换行形态差异算生产变更。

三类代码与便携回归均已完成，本次只执行工程CPU正确性门；head/实际dev的最终CI与合并事实分别写入对应PR验收正文，不能用前一类CI替代下一类。#53/#62及#66/#67/#68均按指令保持open至下版；main保持v0.4.0/54f8d31，不打tag、不删活动分支或数据，不测速/跑GPU、不修改文件架构、正式研究或旧provenance。所有临时证据在.tmp/systematic-contracts-20261015，不随clone；第三方扩展与专用二进制协议的限制仍以§83说明为准。

### 85. 第二类：严格 compare 读取与输出边界（B-076 / #68）

第三类PR69最终head0f393f1的Unit38742367339/Package38742367353/Branch38742367302全部实际任务与聚合成功，正常合dev b858c2ba689f1f8e77fe7065b53ee551fe42325d，实际dev独立CI接续。第二类复用同一工作分支，portable红提交c954b49：32项26 failed/6 passed，程序RuntimeError原样上抛、历史null和已拒绝的负无穷作为控制通过。

在最初21项预检之后，实际读取点继续覆盖：目录冒充result/tracker/checkpoint、空/坏pickle/截断zip的原生解析错误，以及巨大整数有限绝对/相对容差。原21项16失败/5通过及后续SECOND_PREFLIGHT/XML全保留；该补充使§83初始标合规的_checkpoint也归入B-076，不能用最初清单覆盖新反例。

统一方案在既有compare.py与CLI函数内：_load_json同时校验根对象与所用metrics/splits对象（null兼容），history逐级校验；严格文件错误拒绝，不套用status占位。torch.load解析错误仅在实际加载调用处转ValueError并保留cause，意外程序RuntimeError不吞。整数阈值使用标准库Fraction对应已配置浮点容差的精确值，保留差值整数，格式化不用g隐式float；普通浮点/NaN分支保持。CLI把--out发布也放入原错误边界，成功后才展示结果；不改Run文件、schema、计算或物理布局。

第二类生产修复提交dbf0837；相关161 passed、core730 passed /1平台skip /173 deselected（8条既有partial-resume warning），CPU Flow169 passed /0失败/0skip。SECOND_LOCAL_CHECKED确认32个新增节点均执行、旧32项红例26失败/6通过，生产仅compare.py与CLI，输入/旧输出字节保护及程序异常传播控制均通过。原始XML/log/check清单在.tmp/systematic-contracts-20261015（不随clone）；未运行GPU、研究或性能入口。

第三类实际dev b858c2b已验收：Unit38743309828 / Package38743310055所有实际矩阵/聚合成功；DEV_CHECKED SHA256为8c4374c8e361113b0a1078f44044bf361e4cf79be7f54804efa1c232649b4850。PR69正文与Issue66一次性追加验收，Issue仍open。第二类后续PR/head与实际dev独立验收，不用本机或上一类CI替代。

### 84. 第三类：排他所有权覆盖文本与共享引用（B-074 / #66）

先Issue66与portable红提交a080bc6（6 failed），再设计/逐点清单5d2e174，补活动stage真实线程交错a99fef0，生产8f9cc2c只改现有_atomic/shared两文件。atomic_write_text以排他touch获得owned后才写/清，shared晋升去除递归并统一软链→硬链创建，迁移在link成功后才清本写者；短名、Windows有界replace、外层锁、返回值和路径保持。

相邻105 passed /1 POSIX skip，七个新节点均执行；core698 passed /1 skip /173 deselected，8条既有partial-resume warning；CPU Flow169 passed /0 fail/0 skip，THIRD_LOCAL_CHECKED逐项核对完整XML与七个新节点。原6失败与所有源字节ZIP不改写。首个编辑只完成_atomic，shared为LF而脚本错用CRLF锚点导致中止，THIRD_GREEN原3失败/101通过/1skip与PARTIAL_EDIT_NOTE保留；按文件原换行补完后GREEN2为104/1skip，增加线程合同GREEN3为105/1skip。未将工具错误当新库问题或隐藏首候选失败。

本批不扩大任意同Run多写者或跨文件事务，不迁移正式数据、不调用研究/性能入口，#53/#62及新Issue均留到下版。第二类B-076和第一类B-075仍按清单分别接续，不把本批全绿当它们已修。

### 83. 全仓写入与只读展示逐项清单（系统扫描基线99e9e09）

按 Enmao 本轮新要求，7895169已通过[PR65](https://github.com/DreamSoul-AI/RPipe/pull/65)普通合dev99e9e0961ccf2dce9a16cf17e2e00270fd605571；最终head Unit38738711058/Package38738711163/Branch38738711093与实际dev Unit38740119413/Package38740119450所有实际矩阵/聚合均成功。§82“不额外开PR”是新授权前事实，本轮明确覆盖该边界。仍复用保留工作分支，不重建已删九分支。

本次不是只跑旧文本架构门：以git ls-files枚举全部215个受跟踪Python文件，AST记录文本/二进制写、stream open、replace/link/unlink/mkdir、委托writer、日志与终端展示；补列全部readout函数、activity和纯展示入口，并人工追踪TrainAlgorithm.on_checkpoint的getattr(system,'save_checkpoint')→writer(payload,name)到System发布者。§83逐函数列出全部候选行/调用，字符串replace单独排除，验证fixture单独列出，避免误禁造坏输入。

最初扫描基线99e9e09共388个函数组：生产/Study146组、验证242组，当时12个根函数/边界待修。后补真实反例使_checkpoint归入同一B-076。下表已按三类修复后源码重新生成：共403组（生产/Study146、验证257），13个根边界均有红绿合同覆盖，无本轮待修项。行号对应本轮最终源码；原始基线清单、hash与失败仍在本机证据中，不把新增测试组误记成新生产缺陷。

| **类** | **统一待修项** | **真实未修结果** | **最小方案** |
| --- | --- | --- | --- |
| 第三类 | B-074/[#66](https://github.com/DreamSoul-AI/RPipe/issues/66)，临时名碰撞时写入/清理未取得所有权的peer | 原portable红提交a080bc6，6 failed；包含target与stage同名、stat拒绝、同字节递归和迁移到达peer | 排他创建成功才拥有staging；只清自身；共享引用移除递归晋升并复用原发布；不改外层锁/软硬链fallback/短名 |
| 第二类 | B-076/[#68](https://github.com/DreamSoul-AI/RPipe/issues/68)，compare坏容器/整数/容差/--out错误越界 | 合成21项16 failed/5 passed，合法历史null与负无穷拒绝控制保留 | 严格文件形状用ValueError/CLI2，不转status占位；整数精确、一般浮点口径不动；输出纳同一错误边界 |
| 第一类 | B-075/[#67](https://github.com/DreamSoul-AI/RPipe/issues/67)，Logger/Tracker初始化w模式截断刚到达证据 | 合成4项2 failed/2目录拒绝控制passed | 只在现有构造器用append创建，不扩大旧成功保留规则或任意同Run多写者承诺 |

全部合成预检31项24 failed/7 passed，原字节ZIP/hash与PREFLIGHT2/XML/log在.tmp/systematic-contracts-20261015，本机材料不随clone，原失败不改写。临时扫描器首轮对Name形式replace调用取.value而失败，已补AST类型判断并重跑；SCAN_FIRST_FAILURE保留，不当生产缺陷，也不把不完整扫描计完成。

判定依据是[layout](../code/layout.md)与[既有三类设计](artifact_consistency_design.md)，不是“全部文件都必须文本原子”：追加日志/恢复offset、checkpoint恢复bundle、锁inode、校验后发布二进制partial、固定Git blob与派生图片均按原协议精确标注。单写者与单文件保证不升级成多文件事务或任意多写者；下载/第三方builder及任意注册扩展具有其外部协议，表格登记其委托入口，不声称静态AST可证明未提供扩展的任意Python副作用。CI的build/pip只写临时runner环境与包，使用已跟踪tests/package_smoke.py；workflow YAML未包含新增业务写入点。README/文档中的代码块与未跟踪历史缓存不执行、不纳为当前代码。

以下“合规”指入口符合既有合同；13个已修根边界由函数行与上表单列，调用方委托同一入口，不复制补丁。验证代码允许部分写/坏字节以复现，不能因扫描误报删除这些用例。所有行均有明确类别，未知调用会让清单生成拒绝，不用默认“合规”掩盖漏审。

#### 生产与Study写入/展示点（146个函数组）

| **位置/函数** | **行:调用（当前复核）** | **状态** | **合同与依据** |
| --- | --- | --- | --- |
| [src/rpipe/flow/cli.py](../../src/rpipe/flow/cli.py) `_execute_compare` | `345:write_compare; 347:print; 349:print; 349:format_compare; 351:print ` | 已修（本轮） | B-076/#68：严格对象/文件/数值/输出边界；compare新增32项合同；PR70 |
| [src/rpipe/flow/cli.py](../../src/rpipe/flow/cli.py) `_execute_data` | `404:print ` | 合规 | 终端进度/错误或只读入口；保留退出语义，非正式结果替代；CLI/流程回归 |
| [src/rpipe/flow/cli.py](../../src/rpipe/flow/cli.py) `_execute_launch` | `276:print; 277:print; 285:print; 290:print; 292:print; 294:print; 311:print; 313:print; 315:print ` | 合规 | 终端进度/错误或只读入口；保留退出语义，非正式结果替代；CLI/流程回归 |
| [src/rpipe/flow/cli.py](../../src/rpipe/flow/cli.py) `_execute_logs` | `381:format_logs; 383:print; 386:print; 388:print ` | 合规 | 终端进度/错误或只读入口；保留退出语义，非正式结果替代；CLI/流程回归 |
| [src/rpipe/flow/cli.py](../../src/rpipe/flow/cli.py) `_execute_process` | `394:print; 395:print ` | 合规 | 终端进度/错误或只读入口；保留退出语义，非正式结果替代；CLI/流程回归 |
| [src/rpipe/flow/cli.py](../../src/rpipe/flow/cli.py) `_execute_report` | `367:write_numbers; 369:print; 372:print; 374:print ` | 合规调用 | 委托既有统一文本入口；B-074已在根入口修复；调用方不重复补丁；原子/流程回归 |
| [src/rpipe/flow/cli.py](../../src/rpipe/flow/cli.py) `_execute_run` | `236:print; 237:print; 239:print ` | 合规 | 终端进度/错误或只读入口；保留退出语义，非正式结果替代；CLI/流程回归 |
| [src/rpipe/flow/cli.py](../../src/rpipe/flow/cli.py) `_execute_run_one` | `435:print ` | 合规 | 终端进度/错误或只读入口；保留退出语义，非正式结果替代；CLI/流程回归 |
| [src/rpipe/flow/cli.py](../../src/rpipe/flow/cli.py) `_execute_status` | `413:print; 417:print; 423:print; 426:print; 428:print; 428:format_status ` | 合规 | 终端进度/错误或只读入口；保留退出语义，非正式结果替代；CLI/流程回归 |
| [src/rpipe/flow/cli.py](../../src/rpipe/flow/cli.py) `_prepare_shared` | `58:print ` | 合规 | 终端进度/错误或只读入口；保留退出语义，非正式结果替代；CLI/流程回归 |
| [src/rpipe/flow/cli.py](../../src/rpipe/flow/cli.py) `_print_launch_counts` | `360:format_status; 361:print ` | 合规 | 终端进度/错误或只读入口；保留退出语义，非正式结果替代；CLI/流程回归 |
| [src/rpipe/flow/cli.py](../../src/rpipe/flow/cli.py) `_print_make_paths` | `244:print; 245:print; 246:print; 248:print; 249:print; 252:print; 252:summarize_capacity ` | 合规 | 终端进度/错误或只读入口；保留退出语义，非正式结果替代；CLI/流程回归 |
| [src/rpipe/flow/cli.py](../../src/rpipe/flow/cli.py) `_provenance_gate` | `327:print; 332:print ` | 合规 | 终端进度/错误或只读入口；保留退出语义，非正式结果替代；CLI/流程回归 |
| [src/rpipe/flow/cli.py](../../src/rpipe/flow/cli.py) `main` | `548:migrate_shared; 550:print; 553:print ` | 合规 | 终端进度/错误或只读入口；保留退出语义，非正式结果替代；CLI/流程回归 |
| [src/rpipe/flow/execute/run.py](../../src/rpipe/flow/execute/run.py) `run` | `28:logger.info; 32:logger.exception; 35:logger.error; 43:tracker.flush_state ` | 合规调用 | 委托Logger追加/Tracker原子state，B-074/B-075在根入口统一覆盖；不吞原阶段异常 |
| [src/rpipe/flow/process/curves.py](../../src/rpipe/flow/process/curves.py) `write_learning_curves` | `90:dest.parent.mkdir; 124:fig.savefig ` | 合规（专用协议） | 派生二进制图，不属整体文本硬门；计划核对先行，未承诺多文件事务 |
| [src/rpipe/flow/process/individual.py](../../src/rpipe/flow/process/individual.py) `run` | `46:atomic_write_text ` | 合规调用 | 委托既有统一文本入口；B-074已在根入口修复；调用方不重复补丁；原子/流程回归 |
| [src/rpipe/flow/process/study.py](../../src/rpipe/flow/process/study.py) `run_study` | `49:str(figure.relative_to(study_dir)).replace; 51:atomic_write_text ` | 合规调用 | 委托既有统一文本入口；B-074已在根入口修复；调用方不重复补丁；原子/流程回归 |
| [src/rpipe/flow/runner.py](../../src/rpipe/flow/runner.py) `FlowRunner._log_failure` | `76:logger.exception; 79:logger.error ` | 合规 | 失败保全predicate/既有rerun失效；委托Result原子入口；42+12/CLI/schedule合同 |
| [src/rpipe/flow/runner.py](../../src/rpipe/flow/runner.py) `FlowRunner._write_failed_result` | `116:write_result ` | 合规 | 失败保全predicate/既有rerun失效；委托Result原子入口；42+12/CLI/schedule合同 |
| [src/rpipe/flow/runner.py](../../src/rpipe/flow/runner.py) `FlowRunner.run` | `39:logger.info; 57:logger.info ` | 合规 | 失败保全predicate/既有rerun失效；委托Result原子入口；42+12/CLI/schedule合同 |
| [src/rpipe/flow/write/run.py](../../src/rpipe/flow/write/run.py) `run` | `21:write_result ` | 合规 | 失败保全predicate/既有rerun失效；委托Result原子入口；42+12/CLI/schedule合同 |
| [src/rpipe/structure/algorithm/base.py](../../src/rpipe/structure/algorithm/base.py) `Algorithm.resume` | `76:logger.info ` | 合规调用 | 委托Logger追加/Tracker原子state，B-074/B-075在根入口统一覆盖；不吞原阶段异常 |
| [src/rpipe/structure/algorithm/eval/__init__.py](../../src/rpipe/structure/algorithm/eval/__init__.py) `EvalAlgorithm.run` | `23:logger.report; 56:logger.info ` | 合规调用 | 委托Logger追加/Tracker原子state，B-074/B-075在根入口统一覆盖；不吞原阶段异常 |
| [src/rpipe/structure/algorithm/eval_hook.py](../../src/rpipe/structure/algorithm/eval_hook.py) `eval_test_split` | `56:logger.report; 60:tracker.flush_state ` | 合规调用 | 委托Logger追加/Tracker原子state，B-074/B-075在根入口统一覆盖；不吞原阶段异常 |
| [src/rpipe/structure/algorithm/hf_map.py](../../src/rpipe/structure/algorithm/hf_map.py) `hf_optim_name` | `30:str(raw or 'SGD').lower().replace ` | 非IO（排除） | 字符串/配置名转换，不是文件替换 |
| [src/rpipe/structure/algorithm/hf_map.py](../../src/rpipe/structure/algorithm/hf_map.py) `hf_scheduler_name` | `39:str(raw).lower().replace ` | 非IO（排除） | 字符串/配置名转换，不是文件替换 |
| [src/rpipe/structure/algorithm/kwargs.py](../../src/rpipe/structure/algorithm/kwargs.py) `torch_attr` | `49:raw.lower().replace; 52:raw.lower().replace; 54:attr.lower().replace ` | 非IO（排除） | 字符串/配置名转换，不是文件替换 |
| [src/rpipe/structure/algorithm/optim.py](../../src/rpipe/structure/algorithm/optim.py) `make_scheduler` | `66:str(name).lower().replace; 68:str(name).lower().replace ` | 非IO（排除） | 字符串/配置名转换，不是文件替换 |
| [src/rpipe/structure/algorithm/progress.py](../../src/rpipe/structure/algorithm/progress.py) `parse_checkpoint_mode` | `158:str(raw).lower().replace ` | 非IO（排除） | 字符串/配置名转换，不是文件替换 |
| [src/rpipe/structure/algorithm/resume.py](../../src/rpipe/structure/algorithm/resume.py) `partial_resume_warning` | `43:logger.warning ` | 合规调用 | 委托Logger追加/Tracker原子state，B-074/B-075在根入口统一覆盖；不吞原阶段异常 |
| [src/rpipe/structure/algorithm/tracker.py](../../src/rpipe/structure/algorithm/tracker.py) `AlgorithmTracker.__init__` | `61:self.root.mkdir; 64:self.jsonl_path.parent.mkdir; 66:self.jsonl_path.open ` | 已修（本轮） | B-075/#67：现有日志/曲线构造器append创建；到达证据及目录拒绝四项合同 |
| [src/rpipe/structure/algorithm/tracker.py](../../src/rpipe/structure/algorithm/tracker.py) `AlgorithmTracker.append_jsonl` | `167:self.jsonl_path.parent.mkdir; 169:self.jsonl_path.open; 178:handle.write ` | 合规（专用协议） | JSONL追加/恢复offset与state原子发布；tracker/恢复回归 |
| [src/rpipe/structure/algorithm/tracker.py](../../src/rpipe/structure/algorithm/tracker.py) `AlgorithmTracker.begin_run` | `159:self.jsonl_path.open; 163:handle.write; 164:handle.write ` | 合规（专用协议） | JSONL追加/恢复offset与state原子发布；tracker/恢复回归 |
| [src/rpipe/structure/algorithm/tracker.py](../../src/rpipe/structure/algorithm/tracker.py) `AlgorithmTracker.flush` | `188:self.append_jsonl; 189:self.flush_state ` | 合规（专用协议） | JSONL追加/恢复offset与state原子发布；tracker/恢复回归 |
| [src/rpipe/structure/algorithm/tracker.py](../../src/rpipe/structure/algorithm/tracker.py) `AlgorithmTracker.flush_state` | `183:atomic_write_text ` | 合规（专用协议） | JSONL追加/恢复offset与state原子发布；tracker/恢复回归 |
| [src/rpipe/structure/algorithm/train/__init__.py](../../src/rpipe/structure/algorithm/train/__init__.py) `TrainAlgorithm.on_checkpoint` | `122:logger.info ` | 合规调用 | 委托Logger追加/Tracker原子state，B-074/B-075在根入口统一覆盖；不吞原阶段异常 |
| [src/rpipe/structure/algorithm/train/__init__.py](../../src/rpipe/structure/algorithm/train/__init__.py) `TrainAlgorithm.on_eval_period` | `90:logger.info ` | 合规调用 | 委托Logger追加/Tracker原子state，B-074/B-075在根入口统一覆盖；不吞原阶段异常 |
| [src/rpipe/structure/algorithm/train/__init__.py](../../src/rpipe/structure/algorithm/train/__init__.py) `_run_supervised` | `395:logger.report; 450:tracker.flush_state ` | 合规调用 | 委托Logger追加/Tracker原子state，B-074/B-075在根入口统一覆盖；不吞原阶段异常 |
| [src/rpipe/structure/algorithm/train/__init__.py](../../src/rpipe/structure/algorithm/train/__init__.py) `_run_supervised.finish_train_segment` | `313:logger.report; 317:tracker.flush_state ` | 合规调用 | 委托Logger追加/Tracker原子state，B-074/B-075在根入口统一覆盖；不吞原阶段异常 |
| [src/rpipe/structure/algorithm/train/__init__.py](../../src/rpipe/structure/algorithm/train/__init__.py) `_run_supervised.fire_ckpt` | `351:algo.on_checkpoint ` | 合规调用 | 算法回调委托既有System.save_checkpoint；包含getattr writer(payload,name)动态入口；checkpoint/Hook合同 |
| [src/rpipe/structure/algorithm/transformers_trainer.py](../../src/rpipe/structure/algorithm/transformers_trainer.py) `_run_hf_trainer._EpochHook.on_epoch_end` | `302:logger.report; 308:tracker.flush_state ` | 合规调用 | 委托Logger追加/Tracker原子state，B-074/B-075在根入口统一覆盖；不吞原阶段异常 |
| [src/rpipe/structure/algorithm/transformers_trainer.py](../../src/rpipe/structure/algorithm/transformers_trainer.py) `_run_hf_trainer._EpochHook.on_train_end` | `331:tracker.flush_state ` | 合规调用 | 委托Logger追加/Tracker原子state，B-074/B-075在根入口统一覆盖；不吞原阶段异常 |
| [src/rpipe/structure/algorithm/transformers_trainer.py](../../src/rpipe/structure/algorithm/transformers_trainer.py) `_run_hf_trainer.fire_ckpt` | `249:algo.on_checkpoint ` | 合规调用 | 算法回调委托既有System.save_checkpoint；包含getattr writer(payload,name)动态入口；checkpoint/Hook合同 |
| [src/rpipe/structure/artifact/_atomic.py](../../src/rpipe/structure/artifact/_atomic.py) `atomic_replace` | `15:source.replace ` | 非IO（排除） | 字符串/配置名转换，不是文件替换 |
| [src/rpipe/structure/artifact/_atomic.py](../../src/rpipe/structure/artifact/_atomic.py) `atomic_write_text` | `25:target.parent.mkdir; 33:tmp.touch; 35:tmp.write_text; 37:tmp.chmod; 38:atomic_replace; 42:tmp.unlink ` | 已修（本轮） | B-074/#66：排他取得stage才写/清；atomic/shared七项所有权回归；PR69已合dev |
| [src/rpipe/structure/artifact/activity.py](../../src/rpipe/structure/artifact/activity.py) `announce` | `42:atomic_write_text; 46:print ` | 合规 | 可选activity读取/独立原子提示/显式clear；activity回归 |
| [src/rpipe/structure/artifact/activity.py](../../src/rpipe/structure/artifact/activity.py) `clear_activity` | `34:path.unlink ` | 合规 | 可选activity读取/独立原子提示/显式clear；activity回归 |
| [src/rpipe/structure/artifact/activity.py](../../src/rpipe/structure/artifact/activity.py) `read_activity` | `18:display-contract ` | 合规 | 可选activity读取/独立原子提示/显式clear；activity回归 |
| [src/rpipe/structure/artifact/asset/__init__.py](../../src/rpipe/structure/artifact/asset/__init__.py) `ensure_assets` | `13:layout.assets_dir.mkdir; 14:(layout.assets_dir / kinds.TRACKER).mkdir; 15:(layout.assets_dir / kinds.LOGS).mkdir; 16:(layout.assets_dir / kinds.CHECKPOINTS).mkdir ` | 合规 | 目录创建或字符串转换，不截断已有产物；layout/Factory回归 |
| [src/rpipe/structure/artifact/config/io.py](../../src/rpipe/structure/artifact/config/io.py) `write_config` | `21:atomic_write_text ` | 合规调用 | 委托既有统一文本入口；B-074已在根入口修复；调用方不重复补丁；原子/流程回归 |
| [src/rpipe/structure/artifact/index.py](../../src/rpipe/structure/artifact/index.py) `_run_log_ref` | `32:f'runs/{run_id}/assets/{RUN_LOG}'.replace ` | 非IO（排除） | 字符串/配置名转换，不是文件替换 |
| [src/rpipe/structure/artifact/index.py](../../src/rpipe/structure/artifact/index.py) `write_index` | `94:atomic_write_text ` | 合规调用 | 委托既有统一文本入口；B-074已在根入口修复；调用方不重复补丁；原子/流程回归 |
| [src/rpipe/structure/artifact/layout.py](../../src/rpipe/structure/artifact/layout.py) `ArtifactLayout.ensure` | `57:self.root.mkdir; 58:self.assets_dir.mkdir; 59:self.shared_data_dir.mkdir; 60:self.shared_model_dir.mkdir; 61:self.docs_dir.mkdir ` | 合规 | 目录创建或字符串转换，不截断已有产物；layout/Factory回归 |
| [src/rpipe/structure/artifact/layout.py](../../src/rpipe/structure/artifact/layout.py) `ensure_study_layout` | `68:(study / paths.DOCS_DIRNAME).mkdir; 69:shared['data'].mkdir; 70:shared['model'].mkdir; 71:(study / paths.RUNS_DIRNAME).mkdir ` | 合规 | 目录创建或字符串转换，不截断已有产物；layout/Factory回归 |
| [src/rpipe/structure/artifact/provenance.py](../../src/rpipe/structure/artifact/provenance.py) `write_provenance` | `150:atomic_write_text ` | 合规调用 | 委托既有统一文本入口；B-074已在根入口修复；调用方不重复补丁；原子/流程回归 |
| [src/rpipe/structure/artifact/readout/compare.py](../../src/rpipe/structure/artifact/readout/compare.py) `_checkpoint` | `137:display-contract ` | 已修（本轮） | B-076/#68：严格对象/文件/数值/输出边界；compare新增32项合同；PR70 |
| [src/rpipe/structure/artifact/readout/compare.py](../../src/rpipe/structure/artifact/readout/compare.py) `_close` | `23:display-contract ` | 已修（本轮） | B-076/#68：严格对象/文件/数值/输出边界；compare新增32项合同；PR70 |
| [src/rpipe/structure/artifact/readout/compare.py](../../src/rpipe/structure/artifact/readout/compare.py) `_compare_tree` | `96:display-contract ` | 已修（本轮） | B-076/#68：严格对象/文件/数值/输出边界；compare新增32项合同；PR70 |
| [src/rpipe/structure/artifact/readout/compare.py](../../src/rpipe/structure/artifact/readout/compare.py) `_history` | `121:display-contract ` | 已修（本轮） | B-076/#68：严格对象/文件/数值/输出边界；compare新增32项合同；PR70 |
| [src/rpipe/structure/artifact/readout/compare.py](../../src/rpipe/structure/artifact/readout/compare.py) `_item` | `34:display-contract ` | 合规 | 只读三态/严格计划或纯格式；共享有限数/原子数字表；readout回归 |
| [src/rpipe/structure/artifact/readout/compare.py](../../src/rpipe/structure/artifact/readout/compare.py) `_load_json` | `105:display-contract ` | 已修（本轮） | B-076/#68：严格对象/文件/数值/输出边界；compare新增32项合同；PR70 |
| [src/rpipe/structure/artifact/readout/compare.py](../../src/rpipe/structure/artifact/readout/compare.py) `_missing` | `38:display-contract ` | 合规 | 只读三态/严格计划或纯格式；共享有限数/原子数字表；readout回归 |
| [src/rpipe/structure/artifact/readout/compare.py](../../src/rpipe/structure/artifact/readout/compare.py) `_number` | `19:display-contract ` | 合规 | 只读三态/严格计划或纯格式；共享有限数/原子数字表；readout回归 |
| [src/rpipe/structure/artifact/readout/compare.py](../../src/rpipe/structure/artifact/readout/compare.py) `_tree` | `42:display-contract ` | 合规 | 只读三态/严格计划或纯格式；共享有限数/原子数字表；readout回归 |
| [src/rpipe/structure/artifact/readout/compare.py](../../src/rpipe/structure/artifact/readout/compare.py) `compare_runs` | `152:display-contract ` | 已修（本轮） | B-076/#68：严格对象/文件/数值/输出边界；compare新增32项合同；PR70 |
| [src/rpipe/structure/artifact/readout/compare.py](../../src/rpipe/structure/artifact/readout/compare.py) `format_compare` | `209:display-contract ` | 已修（本轮） | B-076/#68：严格对象/文件/数值/输出边界；compare新增32项合同；PR70 |
| [src/rpipe/structure/artifact/readout/compare.py](../../src/rpipe/structure/artifact/readout/compare.py) `write_compare` | `224:display-contract; 235:atomic_write_text ` | 合规 | 只读三态/严格计划或纯格式；共享有限数/原子数字表；readout回归 |
| [src/rpipe/structure/artifact/readout/logs.py](../../src/rpipe/structure/artifact/readout/logs.py) `event_lines` | `15:display-contract ` | 合规 | 只读三态/严格计划或纯格式；共享有限数/原子数字表；readout回归 |
| [src/rpipe/structure/artifact/readout/logs.py](../../src/rpipe/structure/artifact/readout/logs.py) `format_logs` | `41:display-contract ` | 合规 | 只读三态/严格计划或纯格式；共享有限数/原子数字表；readout回归 |
| [src/rpipe/structure/artifact/readout/report.py](../../src/rpipe/structure/artifact/readout/report.py) `_factors` | `30:display-contract ` | 合规 | 只读三态/严格计划或纯格式；共享有限数/原子数字表；readout回归 |
| [src/rpipe/structure/artifact/readout/report.py](../../src/rpipe/structure/artifact/readout/report.py) `_fmt` | `20:display-contract ` | 合规 | 只读三态/严格计划或纯格式；共享有限数/原子数字表；readout回归 |
| [src/rpipe/structure/artifact/readout/report.py](../../src/rpipe/structure/artifact/readout/report.py) `_log_links` | `55:display-contract; 63:str(raw).replace ` | 合规 | 只读三态/严格计划或纯格式；共享有限数/原子数字表；readout回归 |
| [src/rpipe/structure/artifact/readout/report.py](../../src/rpipe/structure/artifact/readout/report.py) `_planned_runs` | `35:display-contract ` | 合规 | 只读三态/严格计划或纯格式；共享有限数/原子数字表；readout回归 |
| [src/rpipe/structure/artifact/readout/report.py](../../src/rpipe/structure/artifact/readout/report.py) `numbers_path` | `16:display-contract ` | 合规 | 只读三态/严格计划或纯格式；共享有限数/原子数字表；readout回归 |
| [src/rpipe/structure/artifact/readout/report.py](../../src/rpipe/structure/artifact/readout/report.py) `render_numbers` | `67:display-contract ` | 合规 | 只读三态/严格计划或纯格式；共享有限数/原子数字表；readout回归 |
| [src/rpipe/structure/artifact/readout/report.py](../../src/rpipe/structure/artifact/readout/report.py) `write_numbers` | `146:display-contract; 147:atomic_write_text; 147:render_numbers ` | 合规 | 只读三态/严格计划或纯格式；共享有限数/原子数字表；readout回归 |
| [src/rpipe/structure/artifact/readout/status.py](../../src/rpipe/structure/artifact/readout/status.py) `_factor_label` | `72:display-contract ` | 合规 | 只读三态/严格计划或纯格式；共享有限数/原子数字表；readout回归 |
| [src/rpipe/structure/artifact/readout/status.py](../../src/rpipe/structure/artifact/readout/status.py) `_metric_cell` | `119:display-contract ` | 合规 | 只读三态/严格计划或纯格式；共享有限数/原子数字表；readout回归 |
| [src/rpipe/structure/artifact/readout/status.py](../../src/rpipe/structure/artifact/readout/status.py) `_mode` | `37:display-contract ` | 合规 | 只读三态/严格计划或纯格式；共享有限数/原子数字表；readout回归 |
| [src/rpipe/structure/artifact/readout/status.py](../../src/rpipe/structure/artifact/readout/status.py) `_note_from_log` | `90:display-contract; 116:note.replace('\t', ' ').replace; 116:note.replace ` | 合规 | 只读三态/严格计划或纯格式；共享有限数/原子数字表；readout回归 |
| [src/rpipe/structure/artifact/readout/status.py](../../src/rpipe/structure/artifact/readout/status.py) `_read_optional` | `23:display-contract ` | 合规 | 只读三态/严格计划或纯格式；共享有限数/原子数字表；readout回归 |
| [src/rpipe/structure/artifact/readout/status.py](../../src/rpipe/structure/artifact/readout/status.py) `format_status` | `200:display-contract; 214:str(error).replace('\t', ' ').replace; 214:str(error).replace; 215:str(row.get('note') or '-').replace('\t', ' ').replace; 215:str(row.get('note') or '-').replace ` | 合规 | 只读三态/严格计划或纯格式；共享有限数/原子数字表；readout回归 |
| [src/rpipe/structure/artifact/readout/status.py](../../src/rpipe/structure/artifact/readout/status.py) `list_runs` | `129:display-contract ` | 合规 | 只读三态/严格计划或纯格式；共享有限数/原子数字表；readout回归 |
| [src/rpipe/structure/artifact/readout/status.py](../../src/rpipe/structure/artifact/readout/status.py) `log_path` | `81:display-contract ` | 合规 | 只读三态/严格计划或纯格式；共享有限数/原子数字表；readout回归 |
| [src/rpipe/structure/artifact/result/io.py](../../src/rpipe/structure/artifact/result/io.py) `write_result` | `22:atomic_write_text ` | 合规调用 | 委托既有统一文本入口；B-074已在根入口修复；调用方不重复补丁；原子/流程回归 |
| [src/rpipe/structure/artifact/shared.py](../../src/rpipe/structure/artifact/shared.py) `cache_lock` | `50:root.mkdir; 56:(root / '.rpipe.lock').open; 58:handle.write ` | 合规 | 资源锁/新staging/摘要校验/引用委托；不清diagnostic；shared回归 |
| [src/rpipe/structure/artifact/shared.py](../../src/rpipe/structure/artifact/shared.py) `link_input` | `103:target.parent.mkdir; 115:temporary.symlink_to; 117:os.link; 121:atomic_replace; 125:temporary.unlink ` | 已修（本轮） | B-074/#66：排他取得stage才写/清；atomic/shared七项所有权回归；PR69已合dev |
| [src/rpipe/structure/artifact/shared.py](../../src/rpipe/structure/artifact/shared.py) `migrate_shared` | `210:target.parent.mkdir; 212:os.link; 219:os.link; 222:atomic_replace; 225:tmp.unlink ` | 已修（本轮） | B-074/#66：排他取得stage才写/清；atomic/shared七项所有权回归；PR69已合dev |
| [src/rpipe/structure/artifact/shared.py](../../src/rpipe/structure/artifact/shared.py) `prepared_resource` | `159:staging.mkdir; 165:atomic_write_text; 168:staging.rename ` | 合规 | 资源锁/新staging/摘要校验/引用委托；不清diagnostic；shared回归 |
| [src/rpipe/structure/artifact/shared.py](../../src/rpipe/structure/artifact/shared.py) `reference_tree` | `138:link_input ` | 合规 | 资源锁/新staging/摘要校验/引用委托；不清diagnostic；shared回归 |
| [src/rpipe/structure/data/factory.py](../../src/rpipe/structure/data/factory.py) `_build_torch_vision` | `370:root.mkdir; 374:reference_tree; 382:discard_bad_archive; 386:reference_tree ` | 合规（委托协议） | 缓存/第三方builder入口，运行路径与资源锁由原Factory合同约束；不执行外网/下载 |
| [src/rpipe/structure/data/prepare.py](../../src/rpipe/structure/data/prepare.py) `_prepare` | `65:atomic_write_text ` | 合规调用 | 委托既有统一文本入口；B-074已在根入口修复；调用方不重复补丁；原子/流程回归 |
| [src/rpipe/structure/data/profile.py](../../src/rpipe/structure/data/profile.py) `profile_study` | `127:atomic_write_text ` | 合规调用 | 委托既有统一文本入口；B-074已在根入口修复；调用方不重复补丁；原子/流程回归 |
| [src/rpipe/structure/make/__init__.py](../../src/rpipe/structure/make/__init__.py) `expand_study` | `74:write_run_configs; 75:write_study_index ` | 合规调用 | 委托既有统一文本入口；B-074已在根入口修复；调用方不重复补丁；原子/流程回归 |
| [src/rpipe/structure/make/capacity.py](../../src/rpipe/structure/make/capacity.py) `summarize_capacity` | `530:display-contract ` | 合规 | 只格式化容量元数据，无磁盘写入；不执行容量/性能测量 |
| [src/rpipe/structure/make/expand.py](../../src/rpipe/structure/make/expand.py) `format_description` | `46:str(dotted).replace ` | 非IO（排除） | 字符串/配置名转换，不是文件替换 |
| [src/rpipe/structure/make/schedule.py](../../src/rpipe/structure/make/schedule.py) `_fail_result` | `75:write_result ` | 合规 | 失败保全predicate/既有rerun失效；委托Result原子入口；42+12/CLI/schedule合同 |
| [src/rpipe/structure/make/schedule.py](../../src/rpipe/structure/make/schedule.py) `bash_path` | `43:text[2:].replace; 44:text.replace ` | 非IO（排除） | 字符串/配置名转换，不是文件替换 |
| [src/rpipe/structure/make/schedule.py](../../src/rpipe/structure/make/schedule.py) `launch_jobs` | `528:print ` | 合规 | 终端进度/错误或只读入口；保留退出语义，非正式结果替代；CLI/流程回归 |
| [src/rpipe/structure/make/schedule.py](../../src/rpipe/structure/make/schedule.py) `launch_jobs._run_groups` | `478:print; 486:print; 497:print; 513:print ` | 合规 | 终端进度/错误或只读入口；保留退出语义，非正式结果替代；CLI/流程回归 |
| [src/rpipe/structure/make/schedule.py](../../src/rpipe/structure/make/schedule.py) `scripts_dir` | `169:path.mkdir ` | 合规 | 目录创建或字符串转换，不截断已有产物；layout/Factory回归 |
| [src/rpipe/structure/make/schedule.py](../../src/rpipe/structure/make/schedule.py) `write_jobs_json` | `257:atomic_write_text ` | 合规调用 | 委托既有统一文本入口；B-074已在根入口修复；调用方不重复补丁；原子/流程回归 |
| [src/rpipe/structure/make/schedule.py](../../src/rpipe/structure/make/schedule.py) `write_launch_scripts` | `359:atomic_write_text; 364:atomic_write_text ` | 合规调用 | 委托既有统一文本入口；B-074已在根入口修复；调用方不重复补丁；原子/流程回归 |
| [src/rpipe/structure/make/write.py](../../src/rpipe/structure/make/write.py) `write_run_configs` | `33:write_config ` | 合规调用 | 委托既有统一文本入口；B-074已在根入口修复；调用方不重复补丁；原子/流程回归 |
| [src/rpipe/structure/make/write.py](../../src/rpipe/structure/make/write.py) `write_study_index` | `53:write_index ` | 合规调用 | 委托既有统一文本入口；B-074已在根入口修复；调用方不重复补丁；原子/流程回归 |
| [src/rpipe/structure/origin.py](../../src/rpipe/structure/origin.py) `discard_bad_archive` | `142:path.unlink ` | 合规（显式策略） | 仅校验失败的下载归档显式剔除；origin回归，本轮不调用正式缓存 |
| [src/rpipe/structure/system/factory.py](../../src/rpipe/structure/system/factory.py) `System.checkpoint_dir` | `37:directory.mkdir ` | 合规（专用协议） | checkpoint bundle/.writing/.incomplete恢复合同；不以文本替换/清残留改写；checkpoint回归 |
| [src/rpipe/structure/system/factory.py](../../src/rpipe/structure/system/factory.py) `System.load_checkpoint` | `110:raw.replace('/', '_').replace; 110:raw.replace ` | 非IO（排除） | 字符串/配置名转换，不是文件替换 |
| [src/rpipe/structure/system/factory.py](../../src/rpipe/structure/system/factory.py) `System.save_checkpoint` | `48:str(name).replace('/', '_').replace; 48:str(name).replace; 59:torch.save; 60:atomic_replace; 61:torch.save; 64:shutil.rmtree; 65:tmp.mkdir; 70:torch.save; 72:atomic_write_text; 74:atomic_write_text; 81:atomic_write_text; 83:torch.save; 85:incomplete.touch; 86:folder.mkdir; 90:atomic_replace; 92:target.unlink; 93:atomic_replace; 95:incomplete.unlink ` | 合规（专用协议） | checkpoint bundle/.writing/.incomplete恢复合同；不以文本替换/清残留改写；checkpoint回归 |
| [src/rpipe/structure/system/logger.py](../../src/rpipe/structure/system/logger.py) `Logger.__init__` | `32:self.path.parent.mkdir; 34:self.path.open ` | 已修（本轮） | B-075/#67：现有日志/曲线构造器append创建；到达证据及目录拒绝四项合同 |
| [src/rpipe/structure/system/logger.py](../../src/rpipe/structure/system/logger.py) `Logger._emit` | `41:print; 42:self.path.open; 43:handle.write ` | 合规（专用协议） | Logger追加+flush；运行期typed指标，不把编程错误吞为缺文件；logger回归 |
| [src/rpipe/structure/system/logger.py](../../src/rpipe/structure/system/logger.py) `Logger.report` | `63:display-contract ` | 合规（专用协议） | Logger追加+flush；运行期typed指标，不把编程错误吞为缺文件；logger回归 |
| [studies/main_exp/flow/execute/capacity.py](../../studies/main_exp/flow/execute/capacity.py) `main` | `73:root.mkdir; 88:folder.mkdir; 90:write_config; 94:write_config; 100:atomic_write_text; 104:path.mkdir; 105:atomic_write_text; 106:atomic_write_text; 153:atomic_write_text; 154:print ` | 合规调用 | 委托既有统一文本入口；B-074已在根入口修复；调用方不重复补丁；原子/流程回归 |
| [studies/main_exp/flow/execute/capacity.py](../../studies/main_exp/flow/execute/capacity.py) `worker_write` | `44:atomic_write_text ` | 合规调用 | 委托既有统一文本入口；B-074已在根入口修复；调用方不重复补丁；原子/流程回归 |
| [studies/main_exp/flow/execute/precision.py](../../studies/main_exp/flow/execute/precision.py) `_worker` | `146:torch.save; 147:atomic_write_text ` | 合规（专用协议） | Study隔离version/Run的二进制观测/权重；单写者，保留原配方，不执行研究 |
| [studies/main_exp/flow/execute/precision.py](../../studies/main_exp/flow/execute/precision.py) `main` | `34:root.mkdir; 65:atomic_write_text; 66:print; 77:atomic_write_text ` | 合规调用 | 委托既有统一文本入口；B-074已在根入口修复；调用方不重复补丁；原子/流程回归 |
| [studies/main_exp/flow/execute/precision.py](../../studies/main_exp/flow/execute/precision.py) `main.execute` | `49:output.mkdir; 56:(output / 'worker.log').open ` | 合规（专用协议） | fresh output.mkdir后subprocess实时worker.log，不整体替换；架构门精确函数豁免，本轮不执行 |
| [studies/main_exp/flow/execute/speed.py](../../studies/main_exp/flow/execute/speed.py) `main` | `53:name.replace; 85:output.mkdir; 236:target.parent.mkdir; 237:atomic_write_text; 238:print ` | 合规调用 | 委托既有统一文本入口；B-074已在根入口修复；调用方不重复补丁；原子/流程回归 |
| [studies/main_exp/flow/execute/speed.py](../../studies/main_exp/flow/execute/speed.py) `main.current.MeasuredTrain.on_checkpoint` | `197:super(MeasuredTrain, self).on_checkpoint ` | 合规调用 | 算法回调委托既有System.save_checkpoint；包含getattr writer(payload,name)动态入口；checkpoint/Hook合同 |
| [studies/main_exp/flow/execute/speed.py](../../studies/main_exp/flow/execute/speed.py) `main.old.save` | `154:(output / 'old/best').mkdir ` | 合规 | 目录创建或字符串转换，不截断已有产物；layout/Factory回归 |
| [studies/main_exp/flow/prepare/run.py](../../studies/main_exp/flow/prepare/run.py) `validate_runtime` | `20:atomic_write_text ` | 合规调用 | 委托既有统一文本入口；B-074已在根入口修复；调用方不重复补丁；原子/流程回归 |
| [studies/main_exp/flow/prepare_data.py](../../studies/main_exp/flow/prepare_data.py) `download` | `65:print; 69:destination.parent.mkdir; 71:print; 73:partial.open; 79:handle.write; 82:print; 88:partial.replace ` | 合规（专用协议） | 二进制partial+校验后发布；失败保留诊断；脚本单写者，不新增任意多写者承诺 |
| [studies/main_exp/flow/prepare_data.py](../../studies/main_exp/flow/prepare_data.py) `expected_files` | `52:relative.replace ` | 非IO（排除） | 字符串/配置名转换，不是文件替换 |
| [studies/main_exp/flow/prepare_data.py](../../studies/main_exp/flow/prepare_data.py) `main` | `246:args.manifest.parent.mkdir; 247:atomic_write_text; 248:print ` | 合规调用 | 委托既有统一文本入口；B-074已在根入口修复；调用方不重复补丁；原子/流程回归 |
| [studies/main_exp/flow/prepare_data.py](../../studies/main_exp/flow/prepare_data.py) `prepare_cifar` | `157:destination.parent.mkdir; 159:partial.open; 160:shutil.copyfileobj; 163:partial.replace ` | 合规（专用协议） | 二进制partial+校验后发布；失败保留诊断；脚本单写者，不新增任意多写者承诺 |
| [studies/main_exp/flow/prepare_data.py](../../studies/main_exp/flow/prepare_data.py) `prepare_mnist` | `116:folder.mkdir; 128:partial.open; 129:shutil.copyfileobj; 132:partial.replace ` | 合规（专用协议） | 二进制partial+校验后发布；失败保留诊断；脚本单写者，不新增任意多写者承诺 |
| [studies/main_exp/flow/prepare_data.py](../../studies/main_exp/flow/prepare_data.py) `reuse_caches` | `106:destination.parent.mkdir; 107:link_input; 109:print ` | 合规 | 终端进度/错误或只读入口；保留退出语义，非正式结果替代；CLI/流程回归 |
| [studies/main_exp/flow/process/curves.py](../../studies/main_exp/flow/process/curves.py) `main` | `371:output.mkdir; 372:atomic_write_text; 374:print ` | 合规调用 | 委托既有统一文本入口；B-074已在根入口修复；调用方不重复补丁；原子/流程回归 |
| [studies/main_exp/flow/process/curves.py](../../studies/main_exp/flow/process/curves.py) `plot_report` | `352:destination.parent.mkdir; 353:fig.savefig ` | 合规（专用协议） | 派生二进制图，不属整体文本硬门；计划核对先行，未承诺多文件事务 |
| [studies/main_exp/recipe.py](../../studies/main_exp/recipe.py) `main` | `331:atomic_write_text ` | 合规调用 | 委托既有统一文本入口；B-074已在根入口修复；调用方不重复补丁；原子/流程回归 |
| [studies/main_exp/recipe.py](../../studies/main_exp/recipe.py) `preflight` | `145:output.mkdir; 183:cell.mkdir; 209:torch.save; 214:torch.save; 293:atomic_write_text; 294:print ` | 合规（专用协议） | Study隔离version/Run的二进制观测/权重；单写者，保留原配方，不执行研究 |
| [studies/main_exp/recipe.py](../../studies/main_exp/recipe.py) `preflight.CapturedTrain.on_checkpoint` | `174:super().on_checkpoint ` | 合规调用 | 算法回调委托既有System.save_checkpoint；包含getattr writer(payload,name)动态入口；checkpoint/Hook合同 |
| [studies/main_exp/structure/api/__init__.py](../../studies/main_exp/structure/api/__init__.py) `export_source` | `80:target.parent.mkdir; 81:target.write_bytes; 87:atomic_write_text ` | 合规（专用协议） | 固定Git blob字节，既有归档逐字节拒绝不符；manifest原子写，不转码历史源码 |
| [studies/main_exp/structure/data/cache.py](../../studies/main_exp/structure/data/cache.py) `datasets` | `32:prepared_resource ` | 合规调用 | 共享不可变输入委托同一shared入口；B-074已在根入口修复；shared/Study合同 |
| [studies/main_exp/structure/data/cache.py](../../studies/main_exp/structure/data/cache.py) `datasets.build` | `27:reference_tree ` | 合规调用 | 共享不可变输入委托同一shared入口；B-074已在根入口修复；shared/Study合同 |
| [studies/main_probe/flow/prepare/data.py](../../studies/main_probe/flow/prepare/data.py) `prepare` | `22:workspace.mkdir; 23:(workspace / 'original').mkdir; 38:destination.parent.mkdir; 39:link_input; 48:reference_tree; 56:stats_path.parent.mkdir; 57:link_input; 59:link_input; 101:print ` | 合规 | 终端进度/错误或只读入口；保留退出语义，非正式结果替代；CLI/流程回归 |
| [studies/main_probe/flow/prepare/inputs.py](../../studies/main_probe/flow/prepare/inputs.py) `prepared_inputs` | `35:prepared_resource ` | 合规调用 | 共享不可变输入委托同一shared入口；B-074已在根入口修复；shared/Study合同 |
| [studies/main_probe/flow/prepare/inputs.py](../../studies/main_probe/flow/prepare/inputs.py) `prepared_inputs.build` | `17:reference_tree; 26:atomic_write_text ` | 合规调用 | 委托既有统一文本入口；B-074已在根入口修复；调用方不重复补丁；原子/流程回归 |
| [studies/main_probe/flow/process/run.py](../../studies/main_probe/flow/process/run.py) `run` | `43:atomic_write_text ` | 合规调用 | 委托既有统一文本入口；B-074已在根入口修复；调用方不重复补丁；原子/流程回归 |
| [studies/main_probe/flow/write/run.py](../../studies/main_probe/flow/write/run.py) `run` | `67:write_compare ` | 合规 | 失败保全predicate/既有rerun失效；委托Result原子入口；42+12/CLI/schedule合同 |
| [studies/main_probe/flow/write/run.py](../../studies/main_probe/flow/write/run.py) `write_observed_run` | `26:write_config; 32:atomic_write_text; 34:folder.mkdir; 41:torch.save; 42:write_result ` | 合规（专用协议） | Study隔离version/Run的二进制观测/权重；单写者，保留原配方，不执行研究 |
| [studies/main_probe/structure/algorithm/paired.py](../../studies/main_probe/structure/algorithm/paired.py) `MainArchive.__init__` | `90:target.parent.mkdir; 91:target.write_bytes ` | 合规（专用协议） | 固定Git blob字节，既有归档逐字节拒绝不符；manifest原子写，不转码历史源码 |
| [studies/main_probe/structure/algorithm/paired.py](../../studies/main_probe/structure/algorithm/paired.py) `run_pair` | `367:run_root.mkdir; 411:cell.mkdir; 443:torch.save; 451:torch.save; 516:torch.save ` | 合规（专用协议） | Study隔离version/Run的二进制观测/权重；单写者，保留原配方，不执行研究 |
| [studies/main_probe/structure/algorithm/paired.py](../../studies/main_probe/structure/algorithm/paired.py) `run_pair.CapturedTrain.on_checkpoint` | `395:super().on_checkpoint; 403:torch.save ` | 合规（专用协议） | Study隔离version/Run的二进制观测/权重；单写者，保留原配方，不执行研究 |
| [studies/main_probe/structure/algorithm/paired.py](../../studies/main_probe/structure/algorithm/paired.py) `save_json` | `44:atomic_write_text ` | 合规调用 | 委托既有统一文本入口；B-074已在根入口修复；调用方不重复补丁；原子/流程回归 |

#### 验证入口与fixture写入/展示点（257个函数组）

| **位置/函数** | **行:调用（当前复核）** | **状态** | **合同与依据** |
| --- | --- | --- | --- |
| [tests/_data/launch_retry_recipe.py](../../tests/_data/launch_retry_recipe.py) `Once.run` | `9:(system.assets_dir / 'attempts.txt').open; 10:stream.write; 11:(system.assets_dir / 'exclusive').mkdir ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/_data/recovery_worker.py](../../tests/_data/recovery_worker.py) `<module>` | `25:root.mkdir; 81:(root / f'{action}_trace.json').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/conftest.py](../../tests/conftest.py) `pytest_configure` | `100:out_dir.mkdir; 117:(out_dir / 'manifest.json').write_text; 121:(out_dir / 'events.jsonl').write_text; 122:(out_dir / 'artifacts').mkdir ` | 合规 | 验证元数据独立run_id；写.tmp/报告，不改旧Run；测试入口/元数据回归 |
| [tests/conftest.py](../../tests/conftest.py) `pytest_runtest_makereport` | `213:(out_dir / 'events.jsonl').open; 214:handle.write ` | 合规 | 验证元数据独立run_id；写.tmp/报告，不改旧Run；测试入口/元数据回归 |
| [tests/conftest.py](../../tests/conftest.py) `pytest_sessionfinish` | `257:(out_dir / 'report.md').write_text ` | 合规 | 验证元数据独立run_id；写.tmp/报告，不改旧Run；测试入口/元数据回归 |
| [tests/package_smoke.py](../../tests/package_smoke.py) `main` | `26:workspace.mkdir; 72:study.mkdir; 73:(study / 'experiment_config.yaml').write_text; 89:(study / 'study.yaml').write_text; 134:(workspace / 'SMOKE_RESULT.json').write_text; 135:print ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/package_smoke.py](../../tests/package_smoke.py) `main.command` | `64:(workspace / 'SMOKE_RESULT.json').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/process/test_curves.py](../../tests/rpipe/flow/process/test_curves.py) `test_curve_only_uses_explicit_valid_coordinates` | `165:tracker.mkdir; 167:(tracker / 'scalars.jsonl').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/process/test_curves.py](../../tests/rpipe/flow/process/test_curves.py) `test_history_falls_back_when_jsonl_has_no_matching_values` | `72:tracker.mkdir; 73:(tracker / 'tracker_state.json').write_text; 77:(tracker / 'scalars.jsonl').write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/process/test_curves.py](../../tests/rpipe/flow/process/test_curves.py) `test_history_prefers_all_report_observations_over_sparse_history` | `23:tracker.mkdir; 24:(tracker / 'tracker_state.json').write_text; 35:(tracker / 'scalars.jsonl').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/process/test_curves.py](../../tests/rpipe/flow/process/test_curves.py) `test_history_skips_invalid_jsonl_rows_and_keeps_valid_observations` | `46:tracker.mkdir; 60:(tracker / 'scalars.jsonl').write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/process/test_curves.py](../../tests/rpipe/flow/process/test_curves.py) `test_restart_marker_survives_a_partial_jsonl_tail` | `148:tracker.jsonl_path.open; 149:handle.write ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/process/test_process.py](../../tests/rpipe/flow/process/test_process.py) `_write_succeeded` | `40:write_config; 41:write_result ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/process/test_process.py](../../tests/rpipe/flow/process/test_process.py) `test_process_does_not_rewrite_result` | `204:write_result ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/process/test_process.py](../../tests/rpipe/flow/process/test_process.py) `test_report_rejects_old_process_after_fresh_make_and_accepts_reprocess` | `172:study.mkdir; 173:(study / 'study.yaml').write_text; 177:(study / 'experiment_config.yaml').write_text; 186:(study / 'experiment_config.yaml').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/process/test_process.py](../../tests/rpipe/flow/process/test_process.py) `test_study_process_accepts_empty_current_index` | `252:write_index ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/process/test_process.py](../../tests/rpipe/flow/process/test_process.py) `test_study_process_aggregates_history_min_max` | `112:path.parent.mkdir; 113:path.write_text; 131:write_index ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/process/test_process.py](../../tests/rpipe/flow/process/test_process.py) `test_study_process_aggregates_history_min_max.capture_figure` | `85:savefig ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/process/test_process.py](../../tests/rpipe/flow/process/test_process.py) `test_study_process_rejects_unusable_index_without_touching_artifacts` | `235:output.write_bytes; 237:figure.parent.mkdir; 238:figure.write_bytes; 241:(tmp_path / 'index.json').write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/process/test_process.py](../../tests/rpipe/flow/process/test_process.py) `test_study_progress_summary_and_plot_share_coordinates_and_counts` | `280:write_result; 288:write_index ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/process/test_process.py](../../tests/rpipe/flow/process/test_process.py) `test_study_progress_summary_and_plot_share_coordinates_and_counts.capture` | `268:savefig ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_activity_status.py](../../tests/rpipe/flow/test_activity_status.py) `test_status_ignores_unusable_hint_and_keeps_index_contract` | `21:study.mkdir; 23:activity.write_bytes; 25:write_index; 27:write_result ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_capture_refusal.py](../../tests/rpipe/flow/test_capture_refusal.py) `test_capture_refusal_preserves_results_and_executes_no_phase` | `24:package.mkdir; 25:(package / '__init__.py').write_bytes; 26:study.mkdir; 28:extra.write_bytes; 34:write_config; 38:write_result; 41:extra.unlink; 51:(study / 'provenance.json').write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_capture_refusal.py](../../tests/rpipe/flow/test_capture_refusal.py) `test_concurrent_freeze_refusals_preserve_each_run_independently` | `128:package.mkdir; 129:(package / '__init__.py').write_bytes; 130:study.mkdir; 132:extra.write_bytes; 138:write_config; 145:write_result; 149:extra.unlink ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_capture_refusal.py](../../tests/rpipe/flow/test_capture_refusal.py) `test_prepare_capture_refusal_with_study_phases_disabled` | `83:package.mkdir; 84:(package / '__init__.py').write_bytes; 85:study.mkdir; 87:extra.write_bytes; 93:write_config; 95:write_config; 101:write_result; 103:extra.unlink ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_cli_retry.py](../../tests/rpipe/flow/test_cli_retry.py) `test_native_launch_retry_policy_preserves_attempts_and_failure` | `26:shutil.copyfile; 27:write_config; 31:write_config; 48:(tmp_path / 'launch.stdout').write_bytes; 49:(tmp_path / 'launch.stderr').write_bytes; 66:(tmp_path / 'later-default.stdout').write_bytes; 67:(tmp_path / 'later-default.stderr').write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_freeze_preservation.py](../../tests/rpipe/flow/test_freeze_preservation.py) `_success` | `24:write_result ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_freeze_preservation.py](../../tests/rpipe/flow/test_freeze_preservation.py) `test_existing_failure_preservation_decision_table` | `110:write_config; 112:layout.assets_dir.mkdir; 114:peer.write_bytes; 119:write_result ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_freeze_preservation.py](../../tests/rpipe/flow/test_freeze_preservation.py) `test_freeze_refusal_preserves_success_or_creates_failure` | `33:write_config; 35:write_config; 38:(tmp_path / 'study.yaml').open; 39:stream.write ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_freeze_preservation.py](../../tests/rpipe/flow/test_freeze_preservation.py) `test_native_run_one_preserves_real_cpu_success_on_freeze_refusal` | `76:write_config; 78:write_config; 88:(tmp_path / 'prior-success.json').write_bytes; 91:(tmp_path / 'study.yaml').open; 92:stream.write; 97:(tmp_path / f'refusal-{attempt}.stdout').write_bytes; 98:(tmp_path / f'refusal-{attempt}.stderr').write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_freeze_preservation.py](../../tests/rpipe/flow/test_freeze_preservation.py) `test_phase_and_failure_publication_errors_preserve_original_evidence` | `153:peer.write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_mnist_native_vs_hf_study.py](../../tests/rpipe/flow/test_mnist_native_vs_hf_study.py) `test_mnist_native_vs_hf_cpu_make_launches_both_sources_without_gpu` | `32:shutil.copytree; 38:shutil.copytree; 41:text.replace; 45:text.replace; 46:text.replace; 47:yaml_path.write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_mnist_train_size_study.py](../../tests/rpipe/flow/test_mnist_train_size_study.py) `test_mnist_train_size_make_launch_processes_train_before_eval` | `31:shutil.copytree; 37:shutil.copytree; 40:text.replace; 44:text.replace; 45:text.replace; 46:yaml_path.write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_readout_io.py](../../tests/rpipe/flow/test_readout_io.py) `test_native_readout_denied_log_preserves_results_and_peer_output` | `20:write_index; 23:write_result; 26:log.parent.mkdir; 27:log.write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_readout_io.py](../../tests/rpipe/flow/test_readout_io.py) `test_native_status_metric_shape_keeps_filtered_peer_and_bytes` | `54:write_index; 57:write_result ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_run_study.py](../../tests/rpipe/flow/test_run_study.py) `test_nested_cpu_device_make_and_run_one_agree` | `26:study.mkdir; 27:(study / 'study.yaml').write_text; 28:(study / 'experiment_config.yaml').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_run_study.py](../../tests/rpipe/flow/test_run_study.py) `test_run_study_new_version_replaces_index_and_preserves_old_run` | `84:study.mkdir; 85:write_config; 90:write_config; 95:write_result; 102:write_config; 120:write_result ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_run_study.py](../../tests/rpipe/flow/test_run_study.py) `test_run_study_skip_launch_index_groups_train_eval_experiments` | `57:shutil.copytree ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_runner.py](../../tests/rpipe/flow/test_runner.py) `test_failed_flow_writes_failed_result` | `51:write_config ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_runner.py](../../tests/rpipe/flow/test_runner.py) `test_full_flow_writes_succeeded_status` | `22:write_config ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_runner.py](../../tests/rpipe/flow/test_runner.py) `test_invalid_configuration_writes_failed_result` | `119:write_config ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_shared_inputs.py](../../tests/rpipe/flow/test_shared_inputs.py) `test_profile_directory_move_keeps_native_sampler_restore_identity` | `145:path.parent.mkdir; 146:path.write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_shared_inputs.py](../../tests/rpipe/flow/test_shared_inputs.py) `test_shared_raw_does_not_share_study_normalization` | `101:foreign_stats.mkdir; 102:(foreign_stats / 'stats.yaml').write_text; 106:study.mkdir; 107:(study / 'study.yaml').write_text; 108:(study / 'experiment_config.yaml').write_text; 113:local.mkdir; 114:(local / 'stats.yaml').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_shared_inputs.py](../../tests/rpipe/flow/test_shared_inputs.py) `test_two_native_studies_share_downloads_not_training_state` | `51:study.mkdir; 53:(study / 'study.yaml').write_text; 54:(study / 'experiment_config.yaml').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_shared_inputs.py](../../tests/rpipe/flow/test_shared_inputs.py) `test_two_native_studies_share_downloads_not_training_state.data_builder` | `27:raw.write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_shared_inputs.py](../../tests/rpipe/flow/test_shared_inputs.py) `test_two_native_studies_share_downloads_not_training_state.model_builder` | `41:torch.save ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_source_key_gate.py](../../tests/rpipe/flow/test_source_key_gate.py) `test_collision_refusal_executes_no_phase_and_preserves_success` | `25:package.mkdir; 26:(package / '__init__.py').write_bytes; 27:(study / 'rpipe').mkdir; 28:(study / 'rpipe/__init__.py').write_bytes; 30:write_config; 36:(study / 'provenance.json').write_bytes; 39:write_result ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_status_planned_modes.py](../../tests/rpipe/flow/test_status_planned_modes.py) `test_native_status_finds_pending_fixed_mode_without_writes` | `25:study.mkdir; 29:write_config; 30:write_config ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_status_planned_modes.py](../../tests/rpipe/flow/test_status_planned_modes.py) `test_native_status_finds_pending_fixed_mode_without_writes.command` | `38:(tmp_path / (name + '.stdout')).write_bytes; 39:(tmp_path / (name + '.stderr')).write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_study.py](../../tests/rpipe/flow/test_study.py) `test_disabled_study_directories_do_not_execute` | `34:phase.mkdir; 35:(phase / '__init__.py').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_study.py](../../tests/rpipe/flow/test_study.py) `test_flow_parent_and_structure_relative_imports_are_isolated` | `68:phase.mkdir; 69:(root / 'structure').mkdir; 70:(root / 'structure' / 'data.py').write_text; 71:(root / 'recipe.py').write_text; 72:(root / 'flow' / 'helper.py').write_text; 74:(root / 'flow' / '__init__.py').write_text; 75:(phase / '__init__.py').write_text; 76:(phase / 'run.py').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_study.py](../../tests/rpipe/flow/test_study.py) `test_helper_only_phase_is_not_executed` | `89:phase.mkdir; 90:(phase / 'speed.py').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_study.py](../../tests/rpipe/flow/test_study.py) `test_import_failure_does_not_leak_study_namespace` | `104:phase.mkdir; 105:(phase / '__init__.py').write_text; 106:(phase / 'run.py').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_study.py](../../tests/rpipe/flow/test_study.py) `test_invalid_hook_does_not_leak_modules` | `49:directory.parent.mkdir; 50:directory.write_text; 53:directory.unlink; 54:directory.mkdir; 55:(directory / '__init__.py').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_study.py](../../tests/rpipe/flow/test_study.py) `test_legacy_root_phase_requires_explicit_migration` | `96:phase.mkdir; 97:(phase / '__init__.py').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_study.py](../../tests/rpipe/flow/test_study.py) `test_recipe_owned_shared_preparation_is_not_called` | `121:(tmp_path / 'study.yaml').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_study.py](../../tests/rpipe/flow/test_study.py) `test_relative_helpers_are_isolated_between_studies` | `22:phase.mkdir; 23:(root / 'helper.py').write_text; 24:(phase / '__init__.py').write_text; 25:(phase / 'run.py').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_study.py](../../tests/rpipe/flow/test_study.py) `test_shared_preparation_requires_boolean` | `131:(tmp_path / 'study.yaml').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_study_hooks.py](../../tests/rpipe/flow/test_study_hooks.py) `_study` | `44:study.mkdir; 48:(study / 'recipe.py').write_text; 49:(study / 'helper.py').write_text; 52:(study / 'study.yaml').write_text; 53:(study / 'notes').mkdir; 54:(study / 'notes' / 'a.txt').write_text; 55:(study / 'experiment_config.yaml').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_study_hooks.py](../../tests/rpipe/flow/test_study_hooks.py) `test_freeze_blocks_launch_and_run_one_after_source_change` | `116:(study / 'notes' / 'a.txt').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_study_hooks.py](../../tests/rpipe/flow/test_study_hooks.py) `test_make_rejects_missing_or_invalid_recipe` | `103:(study / 'recipe.py').write_text; 105:(study / 'recipe.py').unlink ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_study_hooks.py](../../tests/rpipe/flow/test_study_hooks.py) `test_without_freeze_change_only_warns` | `131:(study / 'notes' / 'a.txt').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_study_phases.py](../../tests/rpipe/flow/test_study_phases.py) `_hook` | `30:path.parent.mkdir; 32:(path.parent / '__init__.py').write_text; 33:path.write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_study_phases.py](../../tests/rpipe/flow/test_study_phases.py) `_study` | `19:study.mkdir; 20:(study / 'study.yaml').write_text; 22:(study / 'experiment_config.yaml').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_study_phases.py](../../tests/rpipe/flow/test_study_phases.py) `test_freeze_includes_stage_helpers_and_blocks_before_selected_execute` | `163:helper.unlink; 165:helper.write_text; 168:helper.unlink; 170:helper.write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_study_phases.py](../../tests/rpipe/flow/test_study_phases.py) `test_generated_run_archive_does_not_change_source_provenance` | `127:archive.parent.mkdir; 128:archive.write_text; 131:(study / 'flow' / 'prepare' / 'helper.py').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_study_phases.py](../../tests/rpipe/flow/test_study_phases.py) `test_prepare_before_runs_before_recipe_and_factory` | `75:(study / 'recipe.py').write_text; 78:(study / 'study.yaml').open; 79:stream.write ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_study_phases.py](../../tests/rpipe/flow/test_study_phases.py) `test_write_gate_failure_cannot_leave_success_result` | `97:write_result ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/flow/test_svhn.py](../../tests/rpipe/flow/test_svhn.py) `test_svhn_mat_train_checkpoint_eval` | `27:root.mkdir ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/algorithm/test_checkpoint_train.py](../../tests/rpipe/structure/algorithm/test_checkpoint_train.py) `test_legacy_training_resume_warns_but_loads_weights` | `320:system.save_checkpoint ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/algorithm/test_resume_sibling.py](../../tests/rpipe/structure/algorithm/test_resume_sibling.py) `test_eval_run_loads_sibling_train_best` | `33:train_assets.mkdir; 34:eval_assets.mkdir; 40:train_system.save_checkpoint; 44:write_index ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/algorithm/test_resume_sibling.py](../../tests/rpipe/structure/algorithm/test_resume_sibling.py) `test_sibling_lookup_ignores_previous_version_outside_index` | `89:old_best.parent.mkdir; 90:old_best.write_bytes; 93:current_best.parent.mkdir; 94:current_best.write_bytes; 96:eval_assets.mkdir; 97:write_index ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/algorithm/test_tracker.py](../../tests/rpipe/structure/algorithm/test_tracker.py) `test_tracker_constructor_preserves_evidence_arriving_before_open.arriving` | `94:stream.write ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/algorithm/test_tracker.py](../../tests/rpipe/structure/algorithm/test_tracker.py) `test_tracker_constructor_still_rejects_directory` | `107:target.mkdir ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/algorithm/test_tracker.py](../../tests/rpipe/structure/algorithm/test_tracker.py) `test_tracker_keeps_jsonl_on_init` | `20:leftover.parent.mkdir; 21:leftover.write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/algorithm/test_train.py](../../tests/rpipe/structure/algorithm/test_train.py) `test_checkpoint_hook_normalizes_best_metric_for_all_train_callers` | `317:algo.on_checkpoint ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/asset/test_tree.py](../../tests/rpipe/structure/artifact/asset/test_tree.py) `test_list_asset_files_is_recursive` | `22:path.parent.mkdir; 23:path.write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/readout/test_compare.py](../../tests/rpipe/structure/artifact/readout/test_compare.py) `_run` | `24:(run / 'assets' / 'tracker').mkdir; 25:(run / 'assets' / 'checkpoints').mkdir; 26:(run / 'result.json').write_text; 30:(run / 'assets' / 'tracker' / 'tracker_state.json').write_text; 34:torch.save ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/readout/test_compare.py](../../tests/rpipe/structure/artifact/readout/test_compare.py) `test_compare_bad_containers_are_file_errors_without_evidence_overwrite` | `114:path.write_text; 116:output.write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/readout/test_compare.py](../../tests/rpipe/structure/artifact/readout/test_compare.py) `test_compare_corrupt_checkpoint_is_file_error_without_rewrite` | `203:path.write_bytes; 205:output.write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/readout/test_compare.py](../../tests/rpipe/structure/artifact/readout/test_compare.py) `test_compare_directory_is_bad_file_not_missing` | `189:path.unlink; 190:path.mkdir; 192:output.write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/readout/test_compare.py](../../tests/rpipe/structure/artifact/readout/test_compare.py) `test_compare_keeps_legacy_null_optional_containers` | `171:p.write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/readout/test_compare.py](../../tests/rpipe/structure/artifact/readout/test_compare.py) `test_compare_nonfinite_tolerances_are_argument_errors` | `155:output.write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/readout/test_compare.py](../../tests/rpipe/structure/artifact/readout/test_compare.py) `test_compare_output_refusal_uses_cli_file_error_boundary` | `127:output.write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/readout/test_compare.py](../../tests/rpipe/structure/artifact/readout/test_compare.py) `test_exact_checkpoint_comparison_preserves_integer_and_complex_differences` | `95:torch.save ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/readout/test_logs.py](../../tests/rpipe/structure/artifact/readout/test_logs.py) `test_log_decoder_programming_typeerror_is_not_optional_io` | `93:write_index; 95:path.parent.mkdir; 96:path.write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/readout/test_logs.py](../../tests/rpipe/structure/artifact/readout/test_logs.py) `test_logs_prints_events_in_time_order` | `26:folder.mkdir; 27:(folder / 'run.log').write_text; 32:write_index ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/readout/test_logs.py](../../tests/rpipe/structure/artifact/readout/test_logs.py) `test_logs_skip_only_unreadable_run_and_keep_peer_events` | `58:write_index; 61:path.parent.mkdir; 62:path.write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/readout/test_logs.py](../../tests/rpipe/structure/artifact/readout/test_logs.py) `test_logs_unexpected_probe_error_is_not_hidden` | `77:write_index ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/readout/test_report.py](../../tests/rpipe/structure/artifact/readout/test_report.py) `report_study` | `77:(study / 'docs').mkdir; 88:(study / 'index.json').write_text; 89:(study / 'process.json').write_text; 90:numbers_path(study).write_text; 91:(study / 'docs/STUDY_REPORT.md').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/readout/test_report.py](../../tests/rpipe/structure/artifact/readout/test_report.py) `test_nonobject_metrics_do_not_hide_other_run_or_rewrite_evidence` | `208:path.write_text; 210:render_numbers ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/readout/test_report.py](../../tests/rpipe/structure/artifact/readout/test_report.py) `test_report_accepts_legacy_partial_aggregation_and_reordered_runs` | `147:(report_study / 'index.json').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/readout/test_report.py](../../tests/rpipe/structure/artifact/readout/test_report.py) `test_report_rejects_stale_plan_without_overwriting_evidence` | `120:(study / 'index.json').write_text; 121:(study / 'process.json').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/readout/test_report.py](../../tests/rpipe/structure/artifact/readout/test_report.py) `test_report_rejects_unusable_current_index` | `135:index.unlink; 137:index.write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/readout/test_report.py](../../tests/rpipe/structure/artifact/readout/test_report.py) `test_report_rejects_unusable_process_without_overwriting_numbers` | `160:(report_study / 'process.json').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/readout/test_report.py](../../tests/rpipe/structure/artifact/readout/test_report.py) `test_report_writes_experiment_and_run_tables` | `22:(study / 'docs').mkdir; 23:(study / 'docs' / 'STUDY_REPORT.md').write_text; 24:(study / 'process.json').write_text; 49:(study / 'index.json').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/readout/test_report.py](../../tests/rpipe/structure/artifact/readout/test_report.py) `test_unusable_number_is_a_placeholder_in_aggregate_and_run_cells` | `189:path.write_text; 191:render_numbers ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/readout/test_status.py](../../tests/rpipe/structure/artifact/readout/test_status.py) `_planned_run` | `184:write_index; 185:write_config ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/readout/test_status.py](../../tests/rpipe/structure/artifact/readout/test_status.py) `_write_study` | `24:(study / 'runs' / 'ok' / 'assets' / 'logs').mkdir; 25:(study / 'runs' / 'bad' / 'assets' / 'logs').mkdir; 26:(study / 'runs' / 'wait' / 'assets' / 'logs').mkdir; 27:write_index; 70:write_result; 79:write_result ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/readout/test_status.py](../../tests/rpipe/structure/artifact/readout/test_status.py) `test_boolean_recorded_seed_is_not_equal_to_planned_zero` | `411:write_result ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/readout/test_status.py](../../tests/rpipe/structure/artifact/readout/test_status.py) `test_config_identity_conflict_does_not_change_disk_result_or_hide_peer` | `374:write_config ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/readout/test_status.py](../../tests/rpipe/structure/artifact/readout/test_status.py) `test_known_mode_does_not_require_legacy_config` | `231:(tmp_path / 'runs/r0/config.yaml').unlink; 233:write_index; 236:write_result ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/readout/test_status.py](../../tests/rpipe/structure/artifact/readout/test_status.py) `test_metric_shape_fallback_preserves_run_state_and_peer` | `314:write_result ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/readout/test_status.py](../../tests/rpipe/structure/artifact/readout/test_status.py) `test_missing_or_malformed_mode_config_is_unknown_with_diagnostic` | `204:path.unlink; 206:path.write_text; 208:path.write_text; 210:write_config; 212:write_config; 217:format_status ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/readout/test_status.py](../../tests/rpipe/structure/artifact/readout/test_status.py) `test_mode_diagnostic_preserves_execution_state_and_real_failure` | `253:(tmp_path / 'runs/r0/config.yaml').unlink; 255:write_result ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/readout/test_status.py](../../tests/rpipe/structure/artifact/readout/test_status.py) `test_optional_directory_is_unreadable_not_missing_and_programming_error_propagates` | `397:path.write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/readout/test_status.py](../../tests/rpipe/structure/artifact/readout/test_status.py) `test_optional_read_states_keep_input_bytes_and_strict_loader` | `335:path.write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/readout/test_status.py](../../tests/rpipe/structure/artifact/readout/test_status.py) `test_pending_note_is_last_epoch_or_error_not_traceback` | `109:(study / 'runs' / 'bad' / 'assets' / 'logs' / 'run.log').write_text; 121:log.write_text; 140:format_status; 141:print; 147:log.write_text; 152:format_status; 153:print ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/readout/test_status.py](../../tests/rpipe/structure/artifact/readout/test_status.py) `test_planned_mode_conflict_is_local_and_not_silently_coerced` | `352:write_config; 358:write_index ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/readout/test_status.py](../../tests/rpipe/structure/artifact/readout/test_status.py) `test_status_cli_prints_activity_while_index_is_missing` | `174:study.mkdir; 175:(study / 'activity.json').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/readout/test_status.py](../../tests/rpipe/structure/artifact/readout/test_status.py) `test_status_read_denial_keeps_other_runs_and_original_bytes` | `270:log.write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/readout/test_status.py](../../tests/rpipe/structure/artifact/readout/test_status.py) `test_unreadable_config_cannot_be_hidden_by_legacy_mode` | `385:(study / 'runs/bad/config.yaml').write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/readout/test_status.py](../../tests/rpipe/structure/artifact/readout/test_status.py) `test_unreadable_result_uses_planned_config_mode` | `245:(tmp_path / 'runs/r0/result.json').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/readout/test_status.py](../../tests/rpipe/structure/artifact/readout/test_status.py) `test_unusable_numeric_metric_is_local_and_keeps_valid_fallback` | `424:write_result ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/result/test_result_io.py](../../tests/rpipe/structure/artifact/result/test_result_io.py) `test_result_encoding_failure_preserves_target_and_allows_next_write` | `23:write_result; 26:peer.write_text; 29:write_result; 38:write_result ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_activity.py](../../tests/rpipe/structure/artifact/test_activity.py) `test_unexpected_activity_reader_bug_is_not_suppressed` | `54:path.write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_activity.py](../../tests/rpipe/structure/artifact/test_activity.py) `test_unreadable_activity_is_ignored_including_existence_check` | `26:path.write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_activity.py](../../tests/rpipe/structure/artifact/test_activity.py) `test_unusable_activity_is_ignored_without_rewriting` | `18:path.write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_activity.py](../../tests/rpipe/structure/artifact/test_activity.py) `test_valid_unicode_activity_keeps_original_bytes` | `47:path.write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_atomic.py](../../tests/rpipe/structure/artifact/test_atomic.py) `test_atomic_replace_transient_windows_denial_eventually_commits` | `15:source.write_text; 16:target.write_text; 30:atomic_replace ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_atomic.py](../../tests/rpipe/structure/artifact/test_atomic.py) `test_atomic_replace_transient_windows_denial_eventually_commits.held` | `26:replace ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_atomic.py](../../tests/rpipe/structure/artifact/test_atomic.py) `test_atomic_text_newline_option_preserves_native_and_script_bytes` | `233:atomic_write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_atomic.py](../../tests/rpipe/structure/artifact/test_atomic.py) `test_atomic_text_preserves_existing_script_mode` | `241:target.write_text; 242:target.chmod; 243:atomic_write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_atomic.py](../../tests/rpipe/structure/artifact/test_atomic.py) `test_atomic_text_success_replaces_target_and_keeps_peer_stage` | `94:target.write_text; 96:peer.write_text; 97:atomic_write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_atomic.py](../../tests/rpipe/structure/artifact/test_atomic.py) `test_atomic_write_permanent_failure_preserves_target_and_cleans_stage` | `38:target.write_text; 49:atomic_write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_atomic.py](../../tests/rpipe/structure/artifact/test_atomic.py) `test_concurrent_nonce_collision_cannot_delete_an_active_writer_stage` | `404:target.write_text; 423:_atomic.atomic_write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_atomic.py](../../tests/rpipe/structure/artifact/test_atomic.py) `test_index_encoding_failure_does_not_truncate_existing_plan` | `222:target.write_bytes; 224:write_index ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_atomic.py](../../tests/rpipe/structure/artifact/test_atomic.py) `test_partial_text_write_failure_cleans_only_its_stage` | `60:target.write_text; 62:peer.write_text; 70:atomic_write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_atomic.py](../../tests/rpipe/structure/artifact/test_atomic.py) `test_public_text_publishers_preserve_target_when_commit_is_refused` | `193:target.parent.mkdir; 194:target.write_bytes; 196:peer.write_bytes; 209:write_index ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_atomic.py](../../tests/rpipe/structure/artifact/test_atomic.py) `test_study_json_concurrent_publishers_keep_distinct_stages_and_peer` | `338:target.write_bytes; 339:peer.write_bytes; 344:replace ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_atomic.py](../../tests/rpipe/structure/artifact/test_atomic.py) `test_study_json_concurrent_publishers_keep_distinct_stages_and_peer.native_replace` | `356:replace ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_atomic.py](../../tests/rpipe/structure/artifact/test_atomic.py) `test_study_json_publisher_failure_preserves_evidence_and_peer` | `291:target.write_bytes; 292:peer.write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_atomic.py](../../tests/rpipe/structure/artifact/test_atomic.py) `test_study_json_publisher_success_preserves_payload_format_and_peer` | `320:peer.write_bytes; 323:(json.dumps(body, ensure_ascii=False, indent=2) + '\n').replace ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_atomic.py](../../tests/rpipe/structure/artifact/test_atomic.py) `test_text_encoding_failure_preserves_target_and_peer` | `82:target.write_bytes; 83:peer.write_bytes; 85:atomic_write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_atomic.py](../../tests/rpipe/structure/artifact/test_atomic.py) `test_text_stage_collision_preserves_unowned_bytes` | `377:target.write_bytes; 379:peer.write_bytes; 391:_atomic.atomic_write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_atomic_boundaries.py](../../tests/rpipe/structure/artifact/test_atomic_boundaries.py) `test_atomic_writers_publish_complete_payloads_and_keep_peer_stage` | `24:target.write_text; 26:peer.write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_atomic_boundaries.py](../../tests/rpipe/structure/artifact/test_atomic_boundaries.py) `test_atomic_writers_publish_complete_payloads_and_keep_peer_stage.publish` | `32:replace ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_atomic_boundaries.py](../../tests/rpipe/structure/artifact/test_atomic_boundaries.py) `test_windows_held_unicode_target_preserves_old_bytes_then_recovers` | `50:directory.mkdir; 52:target.write_bytes; 64:atomic_write_text; 71:atomic_write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_freeze_capture.py](../../tests/rpipe/structure/artifact/test_freeze_capture.py) `fail_capture` | `33:extra.unlink; 45:(study / 'provenance.json').write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_freeze_capture.py](../../tests/rpipe/structure/artifact/test_freeze_capture.py) `frozen` | `18:library.mkdir; 19:(library / '__init__.py').write_bytes; 20:study.mkdir; 22:extra.write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_freeze_capture.py](../../tests/rpipe/structure/artifact/test_freeze_capture.py) `test_load_retains_optional_legacy_mappings` | `90:(study / 'provenance.json').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_io.py](../../tests/rpipe/structure/artifact/test_io.py) `test_index_groups_runs_by_experiment_factors` | `87:study_dir.mkdir; 97:write_config; 119:write_index ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_io.py](../../tests/rpipe/structure/artifact/test_io.py) `test_index_rejects_unsafe_id_even_with_safe_directory` | `128:write_index ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_io.py](../../tests/rpipe/structure/artifact/test_io.py) `test_index_rejects_unsafe_run_directory_before_readout` | `49:write_index ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_io.py](../../tests/rpipe/structure/artifact/test_io.py) `test_write_config_roundtrip_preserves_slug` | `56:write_config ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_io.py](../../tests/rpipe/structure/artifact/test_io.py) `test_write_result_missing_status_raises_value_error` | `82:write_result ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_io.py](../../tests/rpipe/structure/artifact/test_io.py) `test_write_result_valid_payload_creates_file` | `75:write_result ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_provenance.py](../../tests/rpipe/structure/artifact/test_provenance.py) `_source` | `16:path.parent.mkdir; 17:path.write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_provenance.py](../../tests/rpipe/structure/artifact/test_provenance.py) `test_freeze_detects_two_column_helper_changes` | `48:path.unlink; 50:_source(tmp_path, f'{folder}/helper.py').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_provenance.py](../../tests/rpipe/structure/artifact/test_provenance.py) `test_two_column_source_cannot_escape_study` | `58:outside.mkdir; 61:study.mkdir; 63:(study / folder).symlink_to ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_provenance_git.py](../../tests/rpipe/structure/artifact/test_provenance_git.py) `test_capture_git_reports_real_repository_state` | `19:repo.mkdir; 21:hooks.mkdir; 28:(repo / 'tracked.txt').write_text; 35:(repo / 'tracked.txt').write_text; 41:(repo / '.git/index').write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_shared.py](../../tests/rpipe/structure/artifact/test_shared.py) `test_bad_shared_config_creates_no_layout` | `45:study.mkdir; 46:(study / 'study.yaml').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_shared.py](../../tests/rpipe/structure/artifact/test_shared.py) `test_corrupt_payload_and_incomplete_manifest_are_not_rebuilt_or_accepted` | `205:prepared_resource; 208:(cache / 'data').write_bytes; 210:prepared_resource; 212:(cache / 'data').write_bytes; 215:manifest.write_text; 217:prepared_resource ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_shared.py](../../tests/rpipe/structure/artifact/test_shared.py) `test_corrupt_payload_and_incomplete_manifest_are_not_rebuilt_or_accepted.build` | `204:(staging / 'data').write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_shared.py](../../tests/rpipe/structure/artifact/test_shared.py) `test_explicit_root_is_resolved_relative_to_study` | `37:study.mkdir; 38:(study / 'study.yaml').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_shared.py](../../tests/rpipe/structure/artifact/test_shared.py) `test_failed_reference_creation_keeps_duplicate_payload` | `118:source.write_bytes; 119:target.write_bytes; 125:link_input ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_shared.py](../../tests/rpipe/structure/artifact/test_shared.py) `test_identical_raw_duplicates_become_one_file` | `106:source.mkdir; 107:target.mkdir; 108:(source / 'input.bin').write_bytes; 109:(target / 'input.bin').write_bytes; 110:(source / 'stats.yaml').write_text; 111:reference_tree ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_shared.py](../../tests/rpipe/structure/artifact/test_shared.py) `test_input_reference_never_copies_or_overwrites` | `92:source.write_bytes; 93:link_input; 95:link_input; 97:unrelated.write_bytes; 99:link_input ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_shared.py](../../tests/rpipe/structure/artifact/test_shared.py) `test_interrupted_resource_retry_preserves_failed_staging_and_publishes_fresh` | `269:prepared_resource; 275:prepared_resource; 280:prepared_resource ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_shared.py](../../tests/rpipe/structure/artifact/test_shared.py) `test_interrupted_resource_retry_preserves_failed_staging_and_publishes_fresh.complete` | `274:(staging / 'data').write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_shared.py](../../tests/rpipe/structure/artifact/test_shared.py) `test_interrupted_resource_retry_preserves_failed_staging_and_publishes_fresh.interrupted` | `266:(staging / 'data').write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_shared.py](../../tests/rpipe/structure/artifact/test_shared.py) `test_invalid_resource_contract_is_rejected_before_writes` | `224:prepared_resource ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_shared.py](../../tests/rpipe/structure/artifact/test_shared.py) `test_migration_checks_all_conflicts_before_linking` | `76:data.mkdir; 77:model.mkdir; 78:(data / 'raw.bin').write_bytes; 79:(model / 'weights.bin').write_bytes; 81:target.mkdir; 82:(target / 'weights.bin').write_bytes; 84:migrate_shared ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_shared.py](../../tests/rpipe/structure/artifact/test_shared.py) `test_migration_deduplicates_identical_bytes_and_keeps_profiles_private` | `58:folder.mkdir; 59:(folder / 'input.bin').write_bytes; 60:(study / 'shared' / 'data' / 'stats.yaml').write_text; 62:migrate_shared; 69:migrate_shared ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_shared.py](../../tests/rpipe/structure/artifact/test_shared.py) `test_partial_build_is_retained_without_publishing_ready` | `196:prepared_resource ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_shared.py](../../tests/rpipe/structure/artifact/test_shared.py) `test_partial_build_is_retained_without_publishing_ready.fail` | `193:(staging / 'data').write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_shared.py](../../tests/rpipe/structure/artifact/test_shared.py) `test_private_data_does_not_disable_shared_model` | `28:study.mkdir; 29:(study / 'study.yaml').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_shared.py](../../tests/rpipe/structure/artifact/test_shared.py) `test_same_definition_reuses_input_and_changed_revision_builds_separately` | `184:prepared_resource; 185:prepared_resource; 186:prepared_resource ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_shared.py](../../tests/rpipe/structure/artifact/test_shared.py) `test_same_definition_reuses_input_and_changed_revision_builds_separately.build` | `183:(staging / 'weights').write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_shared.py](../../tests/rpipe/structure/artifact/test_shared.py) `test_valid_resource_names_keep_original_manifest_keys_and_reuse` | `450:prepared_resource; 456:prepared_resource ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_shared.py](../../tests/rpipe/structure/artifact/test_shared.py) `test_valid_resource_names_keep_original_manifest_keys_and_reuse.build` | `448:path.parent.mkdir; 449:path.write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_shared.py](../../tests/rpipe/structure/artifact/test_shared.py) `test_windows_resource_aliases_rejected_before_builder_or_layout` | `438:prepared_resource ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_shared_references.py](../../tests/rpipe/structure/artifact/test_shared_references.py) `test_concurrent_identical_promotions_own_distinct_staging` | `22:source.write_bytes; 23:target.write_bytes; 25:peer.write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_shared_references.py](../../tests/rpipe/structure/artifact/test_shared_references.py) `test_concurrent_identical_promotions_own_distinct_staging.publish` | `43:replace ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_shared_references.py](../../tests/rpipe/structure/artifact/test_shared_references.py) `test_failed_publisher_cleans_only_own_stage_while_peer_is_live` | `92:source.write_bytes; 93:target.write_bytes; 95:peer.write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_shared_references.py](../../tests/rpipe/structure/artifact/test_shared_references.py) `test_failed_publisher_cleans_only_own_stage_while_peer_is_live.promote` | `128:shared.link_input ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_shared_references.py](../../tests/rpipe/structure/artifact/test_shared_references.py) `test_failed_publisher_cleans_only_own_stage_while_peer_is_live.publish` | `124:replace ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_shared_references.py](../../tests/rpipe/structure/artifact/test_shared_references.py) `test_failed_reference_publication_preserves_original_and_peer` | `61:source.write_bytes; 62:target.write_bytes; 64:peer.write_bytes; 69:shared.link_input ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_shared_references.py](../../tests/rpipe/structure/artifact/test_shared_references.py) `test_migration_collision_cleanup_preserves_arriving_peer` | `178:source.parent.mkdir; 179:target.parent.mkdir; 180:source.write_bytes; 181:target.write_bytes; 192:shared.migrate_shared ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_shared_references.py](../../tests/rpipe/structure/artifact/test_shared_references.py) `test_migration_collision_cleanup_preserves_arriving_peer.arriving` | `188:peer.write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_shared_references.py](../../tests/rpipe/structure/artifact/test_shared_references.py) `test_reference_collision_never_owns_or_deletes_peer` | `154:source.write_bytes; 155:target.write_bytes; 157:peer.write_bytes; 170:shared.link_input ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_shared_references.py](../../tests/rpipe/structure/artifact/test_shared_references.py) `test_reference_promotion_preserves_valid_long_filename` | `80:source.write_bytes; 81:target.write_bytes; 82:shared.link_input ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_source_keys.py](../../tests/rpipe/structure/artifact/test_source_keys.py) `inputs` | `20:package.mkdir; 21:(package / '__init__.py').write_bytes; 22:(study / 'rpipe').mkdir ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_source_keys.py](../../tests/rpipe/structure/artifact/test_source_keys.py) `test_distinct_sources_cannot_share_library_manifest_key` | `33:copy.write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_source_keys.py](../../tests/rpipe/structure/artifact/test_source_keys.py) `test_existing_shadowed_freeze_is_refused_after_library_edit` | `42:copy.write_bytes; 45:path.write_bytes; 46:(package / '__init__.py').write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_source_keys.py](../../tests/rpipe/structure/artifact/test_source_keys.py) `test_hardlink_does_not_mask_deleted_library_path` | `91:os.link; 93:source.unlink ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_source_keys.py](../../tests/rpipe/structure/artifact/test_source_keys.py) `test_make_refuses_collision_without_overwriting_prior_provenance` | `54:(study / 'rpipe/__init__.py').write_bytes; 56:path.write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_source_keys.py](../../tests/rpipe/structure/artifact/test_source_keys.py) `test_reserved_namespace_refuses_extra_even_before_library_has_that_name` | `83:(study / 'rpipe/extra.py').write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_source_keys.py](../../tests/rpipe/structure/artifact/test_source_keys.py) `test_same_physical_source_keeps_one_key_and_detects_edits` | `68:copy.write_bytes; 72:os.link; 76:source.write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_source_keys.py](../../tests/rpipe/structure/artifact/test_source_keys.py) `test_similar_nonreserved_namespace_is_accepted` | `101:target.parent.mkdir; 102:target.write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_source_keys.py](../../tests/rpipe/structure/artifact/test_source_keys.py) `test_unreadable_source_identity_is_freeze_refusal` | `110:os.link ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_source_reads.py](../../tests/rpipe/structure/artifact/test_source_reads.py) `test_overlap_does_not_hide_invalid_extra_input` | `51:(study / 'structure').mkdir; 52:(study / 'structure/helper.py').write_bytes; 53:(tmp_path / 'outside.py').write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_source_reads.py](../../tests/rpipe/structure/artifact/test_source_reads.py) `test_overlapping_discovery_preserves_read_failure` | `36:target.parent.mkdir; 37:target.write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_source_reads.py](../../tests/rpipe/structure/artifact/test_source_reads.py) `test_overlapping_sources_keep_keys_and_detect_later_edits` | `20:path.parent.mkdir; 21:path.write_bytes; 28:(tmp_path / 'structure/helper.py').write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_staging_names.py](../../tests/rpipe/structure/artifact/test_staging_names.py) `test_atomic_text_accepts_valid_long_basename` | `22:target.write_bytes; 23:_atomic.atomic_write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_staging_names.py](../../tests/rpipe/structure/artifact/test_staging_names.py) `test_duplicate_migration_accepts_valid_long_basename` | `34:folder.mkdir; 35:(folder / ('x' * 245)).write_bytes; 38:shared.migrate_shared; 43:shared.migrate_shared ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_staging_names.py](../../tests/rpipe/structure/artifact/test_staging_names.py) `test_failed_migration_publication_keeps_copies_and_peer` | `50:folder.mkdir; 51:(folder / 'input').write_bytes; 53:peer.write_bytes; 58:shared.migrate_shared ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_staging_names.py](../../tests/rpipe/structure/artifact/test_staging_names.py) `test_two_preparers_publish_one_complete_resource.build` | `70:(staging / 'data').write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/artifact/test_staging_names.py](../../tests/rpipe/structure/artifact/test_staging_names.py) `test_two_preparers_publish_one_complete_resource.prepare` | `73:shared.prepared_resource ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/data/test_origin.py](../../tests/rpipe/structure/data/test_origin.py) `test_discard_bad_archive_removes_partial_and_keeps_matching_md5` | `72:partial.write_bytes; 73:discard_bad_archive; 78:good.write_bytes; 86:discard_bad_archive ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/data/test_prepare.py](../../tests/rpipe/structure/data/test_prepare.py) `isolated_shared_root` | `17:(tmp_path / 'study.yaml').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/data/test_prepare.py](../../tests/rpipe/structure/data/test_prepare.py) `test_prepare_announces_download_before_build` | `96:write_config ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/data/test_prepare.py](../../tests/rpipe/structure/data/test_prepare.py) `test_prepare_once_per_name_source` | `69:write_config ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/data/test_prepare.py](../../tests/rpipe/structure/data/test_prepare.py) `test_prepare_retries_partial_archive` | `118:write_config; 131:partial.mkdir; 132:(partial / 'cifar-10-python.tar.gz').write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/data/test_prepare.py](../../tests/rpipe/structure/data/test_prepare.py) `test_prepare_skips_existing_cache` | `47:write_config; 52:cached.mkdir; 53:(cached / '.ready').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/data/test_prepare.py](../../tests/rpipe/structure/data/test_prepare.py) `test_prepare_skips_stub` | `33:write_config ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/data/test_profile.py](../../tests/rpipe/structure/data/test_profile.py) `test_recorded_stats_replace_the_builtin_table` | `36:path.write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/make/test_capacity.py](../../tests/rpipe/structure/make/test_capacity.py) `test_attach_estimates_reads_calibrated_budget_from_run_config` | `307:write_config ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/make/test_capacity.py](../../tests/rpipe/structure/make/test_capacity.py) `test_cpu_capacity_summary_does_not_mention_gpu` | `233:summarize_capacity ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/make/test_make.py](../../tests/rpipe/structure/make/test_make.py) `_touch_config` | `32:layout.config_path.write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/make/test_make.py](../../tests/rpipe/structure/make/test_make.py) `_write_succeeded` | `41:layout.result_path.write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/make/test_make.py](../../tests/rpipe/structure/make/test_make.py) `test_cpu_make_skips_gpu_probe` | `365:study.mkdir; 366:(study / 'study.yaml').write_text; 370:(study / 'experiment_config.yaml').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/make/test_make.py](../../tests/rpipe/structure/make/test_make.py) `test_duplicate_expansion_is_rejected_before_writing` | `66:(tmp_path / 'study.yaml').write_text; 67:(tmp_path / 'experiment_config.yaml').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/make/test_make.py](../../tests/rpipe/structure/make/test_make.py) `test_launch_jobs_prints_each_wait_group` | `278:study.mkdir ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/make/test_make.py](../../tests/rpipe/structure/make/test_make.py) `test_launch_mode_eval_does_not_rewrite_jobs_json` | `329:(tmp_path / 'pyproject.toml').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/make/test_make.py](../../tests/rpipe/structure/make/test_make.py) `test_launch_prints_status_counts_when_nothing_remains` | `299:(tmp_path / 'pyproject.toml').write_text; 304:write_index ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/make/test_make.py](../../tests/rpipe/structure/make/test_make.py) `test_load_launch_plan_include_done_keeps_succeeded` | `243:(tmp_path / 'pyproject.toml').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/make/test_make.py](../../tests/rpipe/structure/make/test_make.py) `test_load_launch_plan_reuses_jobs_and_drops_succeeded` | `205:(tmp_path / 'pyproject.toml').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/make/test_make.py](../../tests/rpipe/structure/make/test_make.py) `test_plan_jobs_trains_before_eval` | `77:eval_path.write_text; 79:train_path.write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/make/test_make.py](../../tests/rpipe/structure/make/test_make.py) `test_render_bash_wait_every_round` | `153:(tmp_path / 'pyproject.toml').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/make/test_make.py](../../tests/rpipe/structure/make/test_make.py) `test_render_bash_waits_after_train_wave` | `87:(tmp_path / 'pyproject.toml').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/make/test_make.py](../../tests/rpipe/structure/make/test_make.py) `test_write_launch_scripts_split_round` | `171:(tmp_path / 'pyproject.toml').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/make/test_schedule.py](../../tests/rpipe/structure/make/test_schedule.py) `_fake_processes.Proc.wait` | `124:write_result ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/make/test_schedule.py](../../tests/rpipe/structure/make/test_schedule.py) `_study` | `33:config.write_text; 37:write_index ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/make/test_schedule.py](../../tests/rpipe/structure/make/test_schedule.py) `_succeed` | `42:write_result ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/make/test_schedule.py](../../tests/rpipe/structure/make/test_schedule.py) `test_checkpoint_directory_requires_material_to_be_independent` | `238:folder.mkdir; 240:(folder / material).write_bytes; 242:folder.with_suffix('.pt').write_bytes; 244:(folder / 'meta.json').write_text; 246:(folder / 'notes.txt').write_text; 248:artifact_layout(tmp_path, 'e0').config_path.write_text; 253:partial.parent.mkdir; 254:partial.write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/make/test_schedule.py](../../tests/rpipe/structure/make/test_schedule.py) `test_independent_eval_does_not_require_sibling_success` | `221:checkpoint.parent.mkdir; 222:checkpoint.write_bytes; 225:artifact_layout(tmp_path, 'e0').config_path.write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/make/test_schedule.py](../../tests/rpipe/structure/make/test_schedule.py) `test_plan_and_capacity_use_nested_runtime_device` | `54:layout.config_path.write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/make/test_schedule.py](../../tests/rpipe/structure/make/test_schedule.py) `test_relaunch_and_make_include_previously_successful_stale_eval` | `186:write_result ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/make/test_schedule.py](../../tests/rpipe/structure/make/test_schedule.py) `test_reused_plan_rejects_gpu_binding_on_cpu_job` | `79:layout.config_path.write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/make/test_schedule.py](../../tests/rpipe/structure/make/test_schedule.py) `test_reused_plan_rejects_stale_device_then_accepts_replanned_jobs` | `66:layout.config_path.write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/system/test_checkpoint.py](../../tests/rpipe/structure/system/test_checkpoint.py) `test_checkpoint_mirror_removes_stale_meta_format` | `137:system.save_checkpoint; 141:system.save_checkpoint; 143:bundle.unlink ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/system/test_checkpoint.py](../../tests/rpipe/structure/system/test_checkpoint.py) `test_checkpoint_publication_failure_preserves_committed_bundle` | `47:folder.mkdir; 48:torch.save; 50:system.save_checkpoint; 52:bundle.unlink; 85:system.save_checkpoint; 103:system.save_checkpoint; 112:shutil.copytree; 114:bundle.unlink ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/system/test_checkpoint.py](../../tests/rpipe/structure/system/test_checkpoint.py) `test_checkpoint_publication_failure_preserves_committed_bundle.fail_rename` | `78:rename ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/system/test_checkpoint.py](../../tests/rpipe/structure/system/test_checkpoint.py) `test_checkpoint_publication_failure_preserves_committed_bundle.fail_replace` | `73:replace ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/system/test_checkpoint.py](../../tests/rpipe/structure/system/test_checkpoint.py) `test_checkpoint_publication_failure_preserves_committed_bundle.fail_save` | `63:Path(path).write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/system/test_checkpoint.py](../../tests/rpipe/structure/system/test_checkpoint.py) `test_checkpoint_rejects_unsafe_stem` | `151:system.save_checkpoint ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/system/test_checkpoint.py](../../tests/rpipe/structure/system/test_checkpoint.py) `test_checkpoint_windows_denial_commits_or_preserves_old_bundle` | `160:system.save_checkpoint; 178:system.save_checkpoint; 184:system.save_checkpoint ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/system/test_checkpoint.py](../../tests/rpipe/structure/system/test_checkpoint.py) `test_checkpoint_windows_denial_commits_or_preserves_old_bundle.denied` | `173:replace ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/system/test_checkpoint.py](../../tests/rpipe/structure/system/test_checkpoint.py) `test_committed_checkpoint_survives_marker_cleanup_failure` | `129:system.save_checkpoint ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/system/test_checkpoint.py](../../tests/rpipe/structure/system/test_checkpoint.py) `test_committed_checkpoint_survives_marker_cleanup_failure.fail_marker_cleanup` | `125:unlink ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/system/test_checkpoint.py](../../tests/rpipe/structure/system/test_checkpoint.py) `test_system_save_checkpoint_writes_pt` | `21:system.save_checkpoint ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/system/test_logger.py](../../tests/rpipe/structure/system/test_logger.py) `test_logger_constructor_preserves_evidence_arriving_before_open.arriving` | `90:stream.write ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/system/test_logger.py](../../tests/rpipe/structure/system/test_logger.py) `test_logger_constructor_still_rejects_directory` | `103:target.mkdir ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/system/test_logger.py](../../tests/rpipe/structure/system/test_logger.py) `test_logger_exception_writes_traceback_on_each_line` | `58:logger.exception ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/system/test_logger.py](../../tests/rpipe/structure/system/test_logger.py) `test_logger_report_uses_metric_tags` | `41:logger.report ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/structure/system/test_logger.py](../../tests/rpipe/structure/system/test_logger.py) `test_logger_writes_timestamp_level_and_run_id` | `29:logger.info ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/test_layout.py](../../tests/rpipe/test_layout.py) `test_private_shared_paths_match_layout_documentation` | `119:study.mkdir; 120:(study / 'study.yaml').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/rpipe/test_layout.py](../../tests/rpipe/test_layout.py) `test_profile_path_matches_documented_private_directory` | `136:study.mkdir; 137:(study / 'study.yaml').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/run.py](../../tests/run.py) `main` | `75:print ` | 合规 | 验证元数据独立run_id；写.tmp/报告，不改旧Run；测试入口/元数据回归 |
| [tests/test_git_metadata.py](../../tests/test_git_metadata.py) `test_corrupt_direct_metadata_is_optional` | `66:(metadata / 'refs/heads').mkdir; 67:(metadata / 'objects').mkdir; 69:(metadata / 'HEAD').write_text; 70:(metadata / location).write_bytes ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/test_git_metadata.py](../../tests/test_git_metadata.py) `test_detached_sha_formats_keep_file_fast_path` | `96:(tmp_path / '.git').mkdir; 97:(tmp_path / '.git/HEAD').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/test_git_metadata.py](../../tests/test_git_metadata.py) `test_direct_metadata_read_failure_is_optional` | `79:(metadata / 'refs/heads').mkdir; 81:leaf.write_text; 83:(metadata / 'HEAD').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/test_git_metadata.py](../../tests/test_git_metadata.py) `test_git_fallback_unavailable_is_optional_metadata` | `53:(tmp_path / '.git').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/test_git_metadata.py](../../tests/test_git_metadata.py) `test_manifest_git_sha_handles_real_git_layouts` | `19:hooks.mkdir ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/test_manifest_bootstrap.py](../../tests/test_manifest_bootstrap.py) `test_native_test_entry_keeps_optional_corrupt_metadata` | `20:(tmp_path / 'tests').mkdir; 22:shutil.copy2; 23:(tmp_path / '.git').mkdir; 24:(tmp_path / '.git/HEAD').write_bytes; 25:(tmp_path / 'tests/test_minimal.py').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |
| [tests/test_run_entry.py](../../tests/test_run_entry.py) `test_captured_launcher_keeps_child_environment_output_and_exit` | `25:module.mkdir; 26:(module / '__init__.py').write_text; 27:(module / '__main__.py').write_text ` | 合规 | 隔离fixture/故障注入/测试输出；允许构造坏字节；不纳入生产文本硬门 |

### 82. Enmao 授权清理九条已合并远程分支

2026-10-15 按 Enmao 本次明确授权，只删除本仓库已合入 dev 的远程分支，保留当前工作分支 `codex/artifact-contracts-correctness`。删除前 fetch origin，核对远端12条heads、恰好9条候选、无活动PR使用候选；逐条 `merge-base --is-ancestor` 确认以下tip均已包含在 dev `401af5e553f7a5024ab7b68c3000d718327c3399`。使用普通 `git push --atomic origin --delete` 一次删除，不使用 force push，不删除本地分支。

| **已删除的远程分支** | **删除前 tip（均由 dev 历史保全）** |
| --- | --- |
| `codex/artifact-consistency-design` | `eb52c65b9128cd60b8eb2ab84592dbd9f4da8161` |
| `codex/cache-lock-budget` | `90b7d75323d266c4d272f786c28f4c00959d0432` |
| `codex/failure-preservation-policy` | `bbb2f41fa8af7107eada339f412ecefb05653ddb` |
| `codex/maintenance-closeout` | `b8648d54296ffe3a0050e8e570d8aa9cdb524495` |
| `codex/open-issue-correctness` | `8a3f21342ced528a885001629d2643df0127e0b2` |
| `codex/safe-readout-contracts` | `41982bfb62cc562d4005093dc6c8a57a3f190d29` |
| `codex/status-metric-shape` | `cbc8c0ae13c998e74b178209de0abd14a74cf8e7` |
| `codex/status-readout-errors` | `ed96c8ccbe388dba4fe07f0598a5c93bcb712af5` |
| `codex/v040-layout-release` | `fb15d6efd20fa97904c62b4b151e693495a7328a` |

删除后以 `git ls-remote --heads origin` 复核：远程只剩 `main`、`dev`、`codex/artifact-contracts-correctness`。删除操作未改变main `54f8d31c0f5505742677dd2ce6dec323ee114c61`、dev或保留工作分支的tip；后续报告提交正常追加到保留分支。所有本地分支、数据、Run、provenance与Git对象均保留，#53/#62仍open；没有改自动删除设置或其他仓库。

本轮仅清理远程引用与补文档，未修改源码、布局、版本或产物，不重跑代码测试/CI/性能/科学实验。报告提交并推送至保留工作分支，本轮不额外发起文档集成PR；后续入dev仍遵循正常PR流程。原始删除前/后清单、祖先核对与DELETE.log在本机.tmp/branch-cleanup-20261015，不随clone；本节保留全部分支名/tip与dev基线，正式清单不依赖临时材料。

### 81. 三类系统合同归纳与第一类双重失败留证

第二类实际dev验收完成：DEV_CHECKED.json SHA256 c2385f49cabfda50d994192e74a3a51fe14b250ea685db9ebdf99c424eeede89；第三类对应f5d5ca315cb450feb700a5ed038ac461eb2f654e69fd9f6aec238ac51ce78443。精确CI与最终验收也入PR61/63正文。B-072/B-073从开放软件缺陷表移除，原失败条目快照仍可追溯；#53/#62按本轮要求保持open到下一版本，不冒充已随v0.4.0发布。main仍54f8d31，无新tag/release。本批第一类测试/文档经[PR64](https://github.com/DreamSoul-AI/RPipe/pull/64)接续，最终head与实际集成CI仍分别核对。


按最新授权，先处理#53，再在库与受跟踪Study源文件中按三类归纳，不以逐个邻接补丁代替共享规则，也不改变[layout](../code/layout.md)。唯一工作分支codex/artifact-contracts-correctness依次复用，全部旧分支、数据、Run与provenance保留。

| **类别** | **统一规则与实际处理** | **回归与边界** |
| --- | --- | --- |
| 原子写与并发（第三类） | B-072的八个Study发布文件复用既有atomic_write_text；删除probe固定staging；原#53 index合同重验 | 先红19a3fbb、生产a20efbb；PR61最终a3e4a14全实际CI成功，实际dev fa18404的Unit38731981263/Package38731981260全实际成功。三态读取/二进制/日志协议不混入文本架构门 |
| status/log安全读取（第二类） | B-073把status/report的数值显示收为既有report._fmt；只接收object指标与有限数，不隐藏peer | 红a201d2f为20失败/2通过；生产8605fc3；119相邻、core691通过/1平台skip、CPU Flow157通过。PR63最终716a95c的Unit38732603420/Package38732603458/Branch38732603454全部实际成功，正常合dev72e4307；实际dev的[Unit38733284739](https://github.com/DreamSoul-AI/RPipe/actions/runs/38733284739)与[Package38733284680](https://github.com/DreamSoul-AI/RPipe/actions/runs/38733284680)全部实际矩阵/聚合成功 |
| 失败写入留证（第一类） | 保留既有_preserve_success决策，不新增生产函数或修改保留范围 | 新e6817b1加12项阶段失败+失败结果编码/提交再失败合同，连同Runner/CLI/schedule相邻共103通过；同时核对42项原决策表。完整CPU流程169 passed /0 skip；生产和layout逐字节差异检查为空，未重复本地core（生产未改），本批最终head仍执行全实际CI |

第一类核对范围：Runner外层/prepare/execute/collect/summarize/write/process；result验证→编码→原子提交；launch的rerun与train→eval屏障；make的全配置预验证与逐文件提交；Study输出调用方。二次写入失败仍返回原始阶段异常，原错日志、旧result和peer字节不变，只清本写者staging。process的旧成功短路仍按既有合同，不假称发生二次提交。

第二类核对status/logs/activity的三态I/O与mode/计划一致性，以及report显示；compare是严格比较，process是科学聚合，保持拒绝/判定，不套用显示层降级。第三类架构门覆盖src/rpipe与Study recipe/flow/structure；precision.main.execute流式worker.log、tracker追加/恢复offset、下载二进制、原Git blob、checkpoint与锁按既有协议精确豁免。

本批未发现第一类新的生产反例，因此只补合同测试，不新造Issue或扩大异常吞咽。单文件原子不承诺多文件事务或任意同Run多写者；不加attempts/锁/模块/字段，不动第七节结构能力。新Study源码的科学使用仍须新version/make与原科学门，未重签旧来源。临时证据.tmp/contract072—074不随clone；便携回归已进入原镜像测试文件。没有测速、性能修改、GPU、下载或正式研究计算。

### 80. 全仓三类合同第二批：安全数字显示（B-073 / #62）

最小生产修复8605fc3只改status/report两个文件；119项相邻回归全部通过，22新节点全部执行。core691 passed /1 POSIX skip /161 deselected，8条既有partial-resume warning；CPU Flow157 passed /0 skip。本机LOCAL_CHECKED.json逐项验证红20失败/2通过与三个绿色集合；临时XML、原字节快照和清单不随clone。科学计算、严格load/compare和process聚合均未改；无GPU/性能/正式Study执行。


第三类 PR61 最终head a3e4a140a29c75512d47090e273efc28cf5edbf0 的 Unit38731020292、Package38731020242、Branch38731020233 所有实际矩阵/聚合成功，普通merge到dev fa184045dc09f54c6b296d1f4c914a021853d041；实际dev随后Unit38731981263/Package38731981260全部实际成功，不以工作head代替验收。旧Python3.10失败保留，未重跑挑通过。

第二类在同一工作分支接续，不另建同质分支。Issue62先登记，22项便携回归先于生产修复，20 failed /2 passed：8项status、4项数字格式、8项非法容器失败，2项空列表控制通过。输入为临时合成index/result/process；没有改正式Study或既有产物。复用已有report._fmt，消除status的重复数值转换；严格执行/比较/聚合的异常不转成显示占位。临时证据.tmp/contract073不随clone。

### 79. 全仓替换文本发布合同接续（#53 / B-072）

首个远端head `5aceeda` 的 Unit 38730213954 保留真实失败：Ubuntu/Windows Python3.10各仅新增并发测试失败（stages为空），3.13与包门成功。原因是3.10 pathlib捕获os.replace的预绑定接口，单拦os.replace不能证明同步点到达。测试现同时拦原子发布边界与原生入口，用thread-local防重入，并显式模拟预绑定后端；原生产atomic_replace重试继续执行。中间portable.xml的2项WinError5来自测试替身直接调用os.replace而绕开重试，该候选未推送、原失败保留；portable2.xml相关48 passed /1 POSIX skip。生产八文件与a20efbb完全相同，不将测试钩子修正记为另一个库缺陷。本机没有Python3.10，跨版本验收以新head实际CI为准。


按 Enmao 新授权，从实际 dev `2e8d786` 建唯一未合分支 `codex/artifact-contracts-correctness`。#53 原 index 截断已在 PR52/v0.4.0 修复；本轮先重验旧合同，Issue 继续开放至下一版，不重复发布或关闭。

| **范围** | **改前** | **本批改后** |
| --- | --- | --- |
| 库 index / jobs / result / profile 等 | 已统一 atomic writer，原回归存在 | 保留原生产实现，重验原编码失败/替换拒绝合同 |
| 受跟踪 Study recipe / flow / structure | 八文件中的替换报告/清单仍直接写；架构门只扫描库 | 同一 atomic_write_text 发布；架构门同时覆盖 Study 源码，不扫描旧 runs/shared 或修改其中数据 |
| probe.save_json | 固定 `.writing`，部分写/拒绝会覆盖 peer，成功会移走 peer，并发共用临时文件 | 删除重复 staging/replace 逻辑，复用库随机临时名、finally 与原子替换；JSON/编码/换行和最终路径不变 |
| 专用协议 | 流式 worker.log、原始 Git blob、下载二进制与 checkpoint 独立合同 | 保持；流式日志精确豁免到 precision.main.execute，不把整个模块豁免 |

先红提交 `19a3fbb`：35项5 failed /29 passed /1 POSIX skip。八文件原字节/hash在本机 .tmp/contract072/before.zip/before.json；初轮 red.xml 的 replace 钩子未覆盖 os.replace，修正后 red2.xml 真实拒绝导致 peer 被覆盖，旧材料保留。候选首轮 green.xml 的唯一失败是日志豁免错写外层 main，后限定真实嵌套函数 main.execute；不把测试清单修正冒充生产修复。

设计/缺陷 `f8e6de1` 先于生产修复 `a20efbb`。相邻原子/引用合同47 passed /1 POSIX skip；core668 passed /1 POSIX skip，8条既有 partial-resume warning。便携用例在 tests/rpipe/structure/artifact/test_atomic.py；Study函数测试只从源码抽取 JSON 发布函数，在合成临时目录执行，不调用配对训练。SOURCE_CHECKED.json 核对除发布表达式/导入及 save_json 本体外，八文件计算 AST 与原字节快照一致；初次通用编辑曾规范化文件既有混合换行，提交前已恢复所有未改行原字节。

临时XML/自动report/源码快照不随clone；本地CPU integration/e2e流程157 passed /0 skipped；head/实际dev CI另补确切结果。没有运行性能入口、下载、GPU、数值研究或重训；不声称新源码完成科学验收。两个Study旧Run/result/provenance全部不改写，新计算须新version/make并按原科学门复验。第二类读取与第一类失败保全继续以同一范围系统核对，不扩大旧成功保留条件或吞咽程序错误。


### 78. v0.4.0 正式发布完成与六项 Issue 关闭

按Enmao本次明确发版授权完成发布，不再沿用§76的“仅准备/监督助手接续”旧边界。发布状态文档经[PR #58](https://github.com/DreamSoul-AI/RPipe/pull/58)正常合dev300308a，最终head b49fed6及实际dev全部实际CI分别通过；[发布PR #59](https://github.com/DreamSoul-AI/RPipe/pull/59)由dev→main正常merge，两个父提交为旧main7a15c46和dev300308a，未squash/rebase/强推。材料与目录合同来源[PR #57](https://github.com/DreamSoul-AI/RPipe/pull/57)及实际dev79bcce0也已独立全绿。

- **发布提交：** `54f8d31c0f5505742677dd2ce6dec323ee114c61`；main树与已验收dev一致。
- **实际main验收：** [Unit Tests 38654268816](https://github.com/DreamSoul-AI/RPipe/actions/runs/38654268816)、[Package Check 38654268693](https://github.com/DreamSoul-AI/RPipe/actions/runs/38654268693)，全部实际矩阵和聚合成功后才建立tag。
- **Annotated tag：** [v0.4.0](https://github.com/DreamSoul-AI/RPipe/tree/v0.4.0)，Git对象 `4b3f70ba2fed5cb19288b6f99b0d50f838e112c7`，本地类型tag，远端peeled commit与发布提交完全一致；普通推送，未覆盖旧标签。
- **GitHub Release：** [v0.4.0](https://github.com/DreamSoul-AI/RPipe/releases/tag/v0.4.0)，ID413273894，published_at为2026-10-14T23:35:48Z（本地2026-10-15），非draft/prerelease，无二进制或本机数据附件。
- **正文来源：** 发布提交内docs/releases/v0.4.0.md，转换16个仓库相对引用为精确提交链接并附PR/commit/main CI；实际Release正文SHA256 `e90f95c11e50dd639111fc0df599b92c94e61d0e9a35e7942fe0c80a76bc4f20` 与生成正文逐字节一致。连接无Release创建接口，按记录使用已认证gh的GitHub API正式创建并回读验证，未修改计费或安装额外插件。

Release实际发布后，六个Issue各自留言注明“已在v0.4.0修复”、回归位置与精确发布来源，再经GitHub连接以completed关闭；评论与状态回读确认如下。#53不在明确关闭清单，仍保持open，不扩大处置范围。

| **Issue** | **发布验收评论** | **状态** |
| --- | --- | --- |
| [#36](https://github.com/DreamSoul-AI/RPipe/issues/36) | [注明v0.4.0修复](https://github.com/DreamSoul-AI/RPipe/issues/36#issuecomment-6180860574) | closed / completed |
| [#37](https://github.com/DreamSoul-AI/RPipe/issues/37) | [注明v0.4.0修复](https://github.com/DreamSoul-AI/RPipe/issues/37#issuecomment-6180861076) | closed / completed |
| [#38](https://github.com/DreamSoul-AI/RPipe/issues/38) | [注明v0.4.0修复](https://github.com/DreamSoul-AI/RPipe/issues/38#issuecomment-6180861608) | closed / completed |
| [#41](https://github.com/DreamSoul-AI/RPipe/issues/41) | [注明v0.4.0修复](https://github.com/DreamSoul-AI/RPipe/issues/41#issuecomment-6180862310) | closed / completed |
| [#48](https://github.com/DreamSoul-AI/RPipe/issues/48) | [注明v0.4.0修复](https://github.com/DreamSoul-AI/RPipe/issues/48#issuecomment-6180862843) | closed / completed |
| [#50](https://github.com/DreamSoul-AI/RPipe/issues/50) | [注明v0.4.0修复](https://github.com/DreamSoul-AI/RPipe/issues/50#issuecomment-6180863455) | closed / completed |

本机TAG_CHECKED、RELEASE_CHECKED、ISSUES_CLOSED及完整CI核对在.tmp/v040，不随clone；便携证据为上述公开PR/commit/CI/tag/Release/评论链接。实际main核对文件SHA256 `7ba9692e617d5b830d5f6f0c2580e8973eb44a04bfce50ac7be393b595710e20`。旧失败、Run、provenance、所有工作分支及dev保留，未删数据/分支；未测速、跑GPU、更新下游vendor、重训或重签科学验收。本节为发布后dev文档补记，不移动v0.4.0标签或改写已发布正文；后续只按明确任务或可复现新反馈推进。

### 77. v0.4.0 发布授权接续（准备阶段）

Enmao明确更正“合dev后停止”，授权由本任务继续：dev→main发布PR、普通merge（不squash/强推/删分支）、实际main验收后 annotated tag v0.4.0、GitHub Release正文使用入库报告，发布后分别留言并关闭#36/#37/#38/#41/#48/#50。此授权取代§76及旧handover的“交监督助手”执行边界；不扩大到#53、数据/分支删除、PyPI或二进制发布。

复用已合PR57的codex/v040-layout-release，先ff到实际dev79bcce0，再仅更新README、发布报告和当前版本入口；运行源码、目录、版本、测试均不变。PR57最终head全实际CI已通过，实际dev79bcce0的Unit38652463042/Package38652463047另行核对，不把旧pending或通过记录混淆。最终发版事实与链接在完成后追加，当前节不是已发布声明。

GitHub连接未暴露创建Release工具接口；发布采用已认证gh对GitHub API的正式调用。不会为了补接口安装插件、使用浏览器凭据或更改计费。本轮不测速、不跑GPU、不干预下游实验，保留旧失败和全部分支/数据。

### 76. v0.4.0 目录合同、迁移材料与版本准备

按 Enmao 本次指令，从已验收 dev 5cff920 建唯一工作分支 codex/v040-layout-release。项目根 shared/data、shared/model 沿用现有实现与位置，作为 studies 下各 Study 共用的不可变输入；Run 的模型对象、checkpoint、sampler、RNG 及可变状态不共享。未迁移正式数据、未重训 Study、未修改产物格式或运行路径。

先以 d55461b 在 [layout.md](../code/layout.md) 固定项目顶层和 Study 级目录的位置、内容与写入方，明确以后先更新文档再改路径；在现有 test_layout.py 新增默认路径、私有 data/model、profile 与版本一致性回归。31个默认路径参数调用真实 layout / shared / schedule / process / readout 入口，与文档表格直接比较。已有镜像测试加35个新节点共36项；两项内存负控分别改文档和改实现路径，均被断言拒绝，不修改正式实现。

后以 fccc45c 准备 [v0.4.0 发布材料](../releases/v0.4.0.md)，说明0.3.x的旧 Study 私有缓存到项目根 shared 的显式迁移、rpipe shared --migrate、同卷硬链接/冲突/中途I/O失败限制及 study.yaml 两类独立私有设置；列出三类收敛与 #36/#37/#38/#41/#48/#50（另含#53）修复。pyproject、__version__、README 同步0.4.0，README明确标签尚未建立，未来标签命令仅发布后执行；不把本次候选叫成已发布版本。生产源码相对5cff920仅__init__.py的一行版本号变化，路径和迁移实现不改。

本机实际验证：

| **入口** | **结果与边界** |
| --- | --- |
| pytest tests/rpipe/test_layout.py | 新独立basetemp下36 passed；当前core再次包含全部36节点 |
| tests/run.py --core | 662 passed /1 skipped /161 deselected；跳过为Windows不适用POSIX脚本执行位，8条既有partial-resume警告保留 |
| tests/run.py --all --cost-class c1 --cost-class c2，筛选CPU integration/e2e | 157 passed /667 deselected，无失败或跳过；未选中的GPU/外部/slow不算通过 |
| 文档/实现路径漂移负控 | 两项均触发预期AssertionError；原文档/代码保持不变 |
| 编码、版本、链接与变更范围 | UTF-8、CRLF、新相对文件链接与版本一致；无运行路径、结果schema或正式Study变更 |

首轮PowerShell创建临时父目录的参数错误，导致pytest 2 passed /34 setup errors；原 .tmp/v040/layout-before-version.xml 保留，不当作库缺陷或合同通过。补建父目录后用 layout-verified 新目录重跑，全部实际执行。普通只读沙箱曾令git diff报告非工作树，真实checkout的只读复核确认worktree正常、core.bare=false；未改Git配置。本机隔离环境无build/wheel，不为本任务添加依赖；包构建/独立安装/CLI交由本PR真实CI，不冒充本机完成。

本机XML、manifest/events、CONTRACT_MUTATIONS及LOCAL_CHECKED在 .tmp/v040 与 .tmp/test-results，不随clone；便携回归与发布说明随PR。最后提交、三项PR检查和实际dev检查以本次PR正文精确链接为准，旧5cff920的绿灯不替代本次版本验收。

本任务仅合dev后停下报告；main仍v0.3.1/7a15c46，合main、v0.4.0 tag和Release交监督助手。开放Issue不自动关，现有分支/数据/Run/provenance保留，不开收费资源，不弹提问窗。后续无新增授权不开展相邻审计或性能工作。

### 75. 三类收敛完成与维护待命边界

本次先同步 origin/dev，确认三类已全部按批准顺序合入 dev `34e95069a3a2814d0fabf4879b1584f1bdbd0758`；分支生命周期规则已随 PR #55 合入 [cicd.md](cicd.md#分支管理2026-10-15)，原工作分支没有未纳入 dev 的提交，无需重复开同题 PR。下表是已完成验收，前文“待 CI / 待集成”为各自记录时点事实。

| **类别** | **已完成内容与 PR** | **实际 dev 验收** |
| --- | --- | --- |
| 第三类 | [PR #52](https://github.com/DreamSoul-AI/RPipe/pull/52)：可整体替换文本复用原子发布，特殊追加 / checkpoint / 锁协议保留 | ea2210b：[Unit](https://github.com/DreamSoul-AI/RPipe/actions/runs/38639077019)、[Package](https://github.com/DreamSoul-AI/RPipe/actions/runs/38639076858) |
| 第二类 | [PR #54](https://github.com/DreamSoul-AI/RPipe/pull/54)：统一私有安全读取与计划字段校验，保留历史 fallback | 945c992：[Unit](https://github.com/DreamSoul-AI/RPipe/actions/runs/38641099712)、[Package](https://github.com/DreamSoul-AI/RPipe/actions/runs/38641099684) |
| 第一类 | [PR #55](https://github.com/DreamSoul-AI/RPipe/pull/55)：仅提炼既有失败保全判断，范围不扩大；含分支管理文档 | 34e9506：[Unit](https://github.com/DreamSoul-AI/RPipe/actions/runs/38643122769)、[Package](https://github.com/DreamSoul-AI/RPipe/actions/runs/38643122705) |

三条 PR 的最终工作提交与上表实际 dev 的所有实际矩阵及聚合均已分别通过。本机对应为第三类 129 passed / 1 POSIX skipped、第二类 107 passed、第一类基线 42 passed / 最终 128 passed；跳过项不计通过。完整 head 与验收链接保留在各 PR 正文，原失败和候选结果不覆盖；本次仅文档核对，不重复运行这些测试。

Issue #36 / #37 / #38 / #41 / #48 / #50 / #53 的修复已在 dev，仍有意保持 open，等待 Enmao 决定是否进入 main 后再处置；不自动关闭，不把 open 当作仍未修复。main 保持已发布 v0.3.1 / `7a15c46`，不合 main、不发布。现有工作分支、Run、数据、provenance 均保留，自动删除设置未改；删除分支或数据仍须单独确认。

当前没有本轮已立项而未完成的正确性缺口；维护进入待命边界：仅响应明确的新任务或有来源、可复现的新 RPipe 缺陷，不再逐个审计相邻路径，不为保持活跃制造修复，不测速 / 改性能 / 查 480 秒 / 跑 GPU，不改文件架构或产物布局，不实施设计第七节能力、不干预下游实验。上述 CPU / 包门不替代 GPU 或完整 Study 科学验收。

本节按 Enmao 的长期授权执行文档提交、工作分支推送及 PR 合 dev，不弹提问窗；如以本地检查替代远端 CI，必须明示实际范围且不伪称远端通过，保留平台强制门。自动审批拒绝则记录准确命令、理由与未执行状态并继续其他工作；删数据 / 删分支 / 强推 / 合 main / 花钱暂停对应操作，在报告列明待确认事项。本次无此类操作。

### 74. 分支管理规则文档更新（随 PR55）

按维护者补充要求，在 cicd.md 新增分支管理节：同工作流只留一条工作分支、同主题续做或从最新dev重新开始并清理旧分支、最终工作提交与实际dev必需CI通过后立即清理对应远程/本地工作分支、长期分支仅main/dev，并建议开启合并后自动删除。同时明确长期分支保护与发布PR头分支保全、本地清理和提交核验责任。

本次只改文档，随当前PR55合dev，不改仓库设置、不删除任何现有分支；已合并分支集中清理等待Enmao明确确认。原三类实现和128项本机结果未变；文档补充后的最终head独立重验必需CI，不用6c61f53旧检查代替。第二类实际dev945c992的Unit38641099712 / Package38641099684全部实际矩阵与聚合已通过，PR54正文已补验收。

### 73. 第一类候选与三类交付边界汇总

第一类先提交42格既有合同 b0ead41，再仅在 runner.py 提取私有 _preserve_success，原表达式原样返回；run 内改为调用该函数。_write_failed_result 的读取、写入、宽容的二次定稿边界、日志、原异常传播、调度失效和重试代码均未动。反向移除 helper 并恢复原调用表达式后，规范化文本与 dev945c992完全一致；raw source hash 另保留，不混同换行规范化与原字节。

改前42 passed；改后冻结、采集拒绝、Runner、调度和 provenance 相邻128 passed /0failed/error/skip，42个基线节点逐一包含于最终批次。不是新 bug，不制造应失败反例，不扩大旧成功保留范围。证据 .tmp/consolidation1/before.zip / baseline.xml / final.xml / LOCAL_CHECKED 不随clone；便携决策表在原 test_freeze_preservation.py。正式 source、tests 都没有新文件，Run结果、数据和provenance未迁移。

| **类别** | **改前 → 改后** | **本机正确性** | **正式集成边界** |
| --- | --- | --- | --- |
| 第三类 / PR52 / B071 | 可替换文本绕过原子发布，index编码失败截断旧证据 → 复用原atomic链，保留LF和mode | 129通过 /1 POSIX跳过；全部15红合同在最终批次 | head eb52c65及实际dev ea2210b全实际CI通过；Issue53仍open |
| 第二类 / PR54 | 分散读取、先信旧mode → 私有三态读取与计划字段校验，保留缺配置历史fallback | 最终107通过，原seed和TypeError候选失败保留；16新合同实际执行 | head41982bf全实际CI通过，合dev945c992，实际dev独立CI另核对 |
| 第一类 / [PR55](https://github.com/DreamSoul-AI/RPipe/pull/55) | 内联保全决策 → 同一私有predicate，范围原样 | 基线42、最终128均通过 | 最终head与实际dev CI待独立验收，不以本机或上两类代替 |

三类均无新模块、产物目录、schema或result含义，特殊协议不搬动；第七节能力不实施，不测速/GPU/正式Study复跑，不合main、不删分支/数据、不自动关闭Issue。最终PR正文给出当前head和实际dev验收链接；旧record的待审/待CI文字保留为对应时点事实，不当作新未修缺陷。

### 72. 第一类：现有保全决策等价提炼，先锁定基线

第二类最终 head 41982bf 的 Unit38640332309 / Package38640332160 / Branch38640332279 全部实际矩阵/聚合通过，经 PR54 正常合入 dev 945c9925c8657b50a7f0906c5cd4fd00b238f1a1，实际 dev CI 独立验收。第三类实际 dev ea2210b 的 Unit38639077019 / Package38639076858 已全部实际通过，DEV_CHECKED 的 SHA256为 e6682607bb41db7ccd5ce6cf6c0298c8d0aef1194d9e2f188edc60e1b48d970e；PR52和Issue53正文已补确切验收链接，Issue53保持open，不重复原六 Issue 评论。

第一类从945c992建立唯一分支 codex/failure-preservation-policy，在既有 test_freeze_preservation.py 先补42格合同：7个阶段位置（含外门）×普通/冻结异常×无旧结果/旧失败/旧成功。基线42 passed，符合此类是等价提炼而非新缺陷的预期；不制造应失败的假反例。保留原错误对象、日志、peer和结果字节；只有 process 或初始/prepare 冻结拒绝且旧结果成功才保全，普通执行失败仍写failed。

### 71. 第二类候选：安全读取复用与类型边界验证

先红合同39d8a22，再补 seed 合同 d0d2fe2；生产仅改既有 readout/status.py 和 logs.py：私有 _read_optional 返回 available / missing / unreadable，日志沿现有依赖方向复用；删除分散的 result / note / event I/O try。activity 的现有可选读取协议与故障回归保留，不反向依赖 readout；严格 load_config / load_result 不变，未新增模块或持久化字段。

_mode 在一次 config 读取后核对 mode / id / seed，以可用计划为先；仅缺失时保留 axes/result 的历史 fallback。冲突或坏配置只降级对应 mode 并在原 error 字段提示，已执行状态、真实 failure、其他格和 peer 不变。其余指标继续用旧 guard，无强转或 schema 放宽。日志正则、排序和 CLI 表头不变，读取不写盘、不清理文件。

首候选102项通过后，新增3项 seed 边界出现2 failed /1 passed（null seed 遮蔽计划0、False 被等同0），修正后最终相邻105 passed /0failed/error/skip。原11项红例、首轮绿和 seed 红例均留在 .tmp/consolidation2，before.zip / XML / LOCAL_CHECKED 不随 clone；原 #37/#48/#50/B068 回归全部在最终批次，不重开旧 Issue。新增三态 helper 的缺失是提炼合同，不冒称原行为回归缺陷。

补充自审发现首候选把日志 TypeError 也带入公共 fallback，会扩大原日志仅 I/O 容错的范围。先补 d4473ed 两项 note / event 编程错误回归，候选2 failed；移除公共 TypeError 捕获后新批次107 passed /0failed/error/skip。原105通过、两项失败和最终 verified.xml 全保留，不拿旧批次掩盖集中处理的反例；RuntimeError 等意外错误仍传播。

正式计划 / 结果 / 数据 / provenance 未改，未运行 GPU 或性能测试。此为第二类独立 PR，最终 head 与实际 dev 必需 CI 尚须逐项验收；第一类未实施，不扩大旧成功保留范围。

### 70. 第二类：三态读取与计划 config 优先合同

第三类 PR52 最终 head eb52c65 全部实际矩阵/聚合通过：Unit38638301970、Package38638301989、Branch38638301958，正常合入 dev ea2210bc27e5456062bdef0ffb331e208c64f838；实际 dev 独立 CI 另验收。PR 汇总 Unit tests 尚 IN_PROGRESS 时拒绝的 ROLLUP_PENDING.json 保留，不拿实际任务已绿提前合并。

从 ea2210b 建立 codex/safe-readout-contracts，按批准设计先补第二类11项合同，基线11 failed：6项为 config 与旧 axes/result 冲突或非法配置 / 身份仍被忽略，5项为计划新增的私有三态 helper 尚不存在。后者是接口提炼合同，不登记为旧代码 bug；config 优先是本设计明确批准的读出规则变更，不把它冒称旧 #37 未修。旧模式 fallback、peer / 原字节保全与严格 loader 均保留回归。

### 69. 第三类：替换文本共用原子发布（B-071 / #53）

先提交红色合同 a7277c1，再补 POSIX script mode 合同 1c56b31，之后生产改动仅五个原文件：index、jobs / scripts、profile / ready marker 改用现有 atomic_write_text；helper 只适配可选 newline 并保留已有 mode，继续调用原 atomic_replace。移除重复 mkdir；不搬模块、不新增产物或布局，不引入 producer 框架。脚本仍 LF，普通文本仍系统原生换行；JSON 字段和默认编码不改。

旧15项合同8 failed /7 passed：真实 index TypeError 截断原文件、四个发布入口绕过提交及架构门失败；两项为新增 newline 适配合同。最终相邻130项为129 passed /1 skipped /0 failed/error，Windows 仅跳过 POSIX mode，待 Ubuntu CI 执行；原129项初轮全通过也保留，不替代最终验证。源码相邻门覆盖共享引用并发、部分写、编码拒绝、checkpoint staging 长名控制及 make/data；无 GPU、下载、正式数据或测速。

架构测试在既有 test_atomic.py 内扫描库源码的直接替换文本写入并测别名 / 负例；Logger 初始化/追加与 AlgorithmTracker 初始化/追加/恢复偏移按精确函数豁免；checkpoint 二进制与 shared 文件锁不属于整体替换文本，原协议和测试保留。未将特殊协议挪到新模块，也不承诺 AST 对任意动态 Python 完整封锁。before.zip / red.xml / final.xml / LOCAL_CHECKED.json 在 .tmp/consolidation3，不随 clone；便携测试随 PR52。

最终 head 与实际 dev 的全部必需 CI 分别验收，完整链接与执行清单回填 PR / Issue；本机通过不称已集成。第一、二类尚未实施，不扩大旧成功保留，main 和原六个 Issue 状态不动。

### 68. 批准后的三类收敛：先合同回归，再最小实现

维护者已审阅并授权按第三类→第二类→第一类实施，每类一个 PR，CI 后合 dev。§67 的待审门禁由本次明确批准解除，结构与磁盘合同仍不改；第一类只提炼 dev 既有保全判断，第三类只约束整体替换文本，特殊写入协议按设计第五（三）保留。现有文档 PR52 复用为第三类工作 PR，避免同任务两个未合分支；先提交文本发布/字节保全/架构负例合同测试，生产代码保持基线，之后再实现。

第三类新增15项合同，基线8 failed /7 passed（2项为 newline 新适配合同）；真实 index 编码失败会截断旧计划，已登记 B-071 / Issue53。四个发布者提交拒绝和架构约束也给出旧链路反例；before.zip 与 red.xml 保留。未新建模块或运行性能/GPU。

### 67. 按最新更正恢复设计待审门禁

维护者明确作废上一条“不必停下等审阅”的指令，恢复设计提交推送后停止汇报、审阅前不动代码。设计正文恢复为 b687342 的文件架构不变方案；§66 / 275ac2b 的继续维护文字仅为当时已被撤销的指令记录，不再生效。当前不继续逐个相邻路径审计，不实施新架构、格式或磁盘布局调整。

更正到达前文档已推送并创建 [PR #52](https://github.com/DreamSoul-AI/RPipe/pull/52)，尚未合入 dev；只读审查命令已结束，没有新复现、代码 / 测试改动或运行验证。本次将等待审阅的约束同步到文档与 PR，并设为草案防止误合；不等待 CI 后自行合入，不合 main、不删分支和数据。推送本次纯文档更正后停下汇报。

### 66. 曾恢复正确性维护（该指令已由 §67 作废）

按维护者最新纠正，设计说明作为参考归档，不再以等审阅为由暂停维护；§64–65 的停止 / 等待文字是当时指令与执行事实，现已被本节取代。文件架构、磁盘布局和 result.json 合同不变的约束继续有效，不实施无失败回归支撑的大重构。已修的六个 Issue 不重复开题或关闭；继续在现有职责范围核查新的真实反例，先复现、记录和提交红色回归，再最小修复与针对性正确性检查。

本轮文档主题提交 c9e0141 / b687342 及本次状态补充一并推送唯一文档工作分支，经 PR→dev 的必需 CI 验证。代码 / 测试 / Study 产物相对基线 d294c1b 不变；文档只做编码、相对链接及差异检查，不以旧代码 CI 宣称架构验收。main、全部已有分支与数据保持；性能、耗时与 GPU 工作继续禁止。

### 65. 按文件架构不变约束修订收敛设计（历史等待状态，后由 §66 取代）

维护者补充明确：文件架构、产物布局和 result.json 的位置 / 格式 / 含义一律不改，模块不搬动。已据此重写[设计说明](artifact_consistency_design.md)：在现有失败写入入口内统一固定保全决策，在现有读取函数内统一三态和字段规则，复用 atomic_write_text → atomic_replace；日志追加、tracker、checkpoint 与资源锁保留原协议。

§64 / c9e0141 中的 attempts、结果指针、格式迁移、计划 generation 和 I/O 职责搬迁方案不再是待实施范围；旧稿仅在 Git 历史保留。本稿把需要结构变化才能提供的完整尝试历史、多文件事务和任意同 Run 多写者互斥列为限制说明，不实施，不把日志说成机器可读历史或把单文件原子替换说成整体事务。替换类文本禁止绕行与特殊协议控制的差异明确待审。

本次只修订设计和记录；检查 UTF-8、相对链接、差异和源码 / 测试相对 dev 不变，不运行代码、GPU 或速度验证。继续同一文档工作分支，等待审阅后才动代码；不推实施 PR、不合 dev / main、不迁移或删除任何产物。

### 64. 产物一致性架构初稿（未实施，后由 §65 修订）

按最新指示停止逐个相邻路径审计，先归纳既有问题并提交[产物一致性设计](artifact_consistency_design.md)。三类目标为：尝试不可变追加、result 根文件作有效结果视图；三态安全读取与计划 config 优先的有类型 Run 视图；唯一原子发布入口及受控 append / 锁。设计列出 #36 / #37 / #38 / #41 / #48 / #50、B-062 / B-067 / B-068 的已修事实、架构欠账、调用方替换、兼容迁移和回归计划，不把已修反例重新登记为 bug。

本轮开始时 fetch 成功、工作区干净。#50 已由 [PR #51](https://github.com/DreamSoul-AI/RPipe/pull/51) 正常合入 dev d294c1b5aa99b4d7148f991f9a4ac8903fe23374；实际 dev 的 [Unit38633933430](https://github.com/DreamSoul-AI/RPipe/actions/runs/38633933430) / [Package38633933432](https://github.com/DreamSoul-AI/RPipe/actions/runs/38633933432) 已全部实际矩阵和聚合通过，不重复开修复 PR 或重派检查。六个 Issue 保持 open，main 仍 v0.3.1 / 7a15c46。

本次仅设计文档和状态记录；文档核对包括相对文件链接、编码、结构、Git 差异与代码基线不变，没有运行代码回归、GPU 或性能检查。上述 CI 只证明此前修复，不能用于宣称新架构通过。attempt 目录、物化结果视图、计划 generation 与写入边界是待审合同，现行 layout / schema 仍有效；审阅通过前不改代码、不迁移历史 Run / 数据 / provenance、不推进实施 PR 或合入 dev / main。保留全部已有分支。

### 63. 相邻正确性审计与 status 指标类型回退（B-070 / #50）

#### （一）审计范围与收尾状态

基于已验收 dev b2390b067abc0d39e347664573a95ad31fe11975，只读检查 readout status/logs、_atomic 写入/清理、launch 的本次重试参数和等待/失败状态、status mode 来源顺序，以及 make/run-one/Runner 的初始拒绝与失败写入。没有运行性能或GPU测试；不是全仓库/全部后端的无缺陷证明。模式合同严格接受小写train/eval/inference，不把无效大小写配置当作受支持模式扩展；checkpoint诊断保全仍按既有合同，不无依据清理。

| **已完成项** | **实际状态** | **等待维护者决定的边界** |
| --- | --- | --- |
| #36/#37/#38/#41 原修复与边界补测 | 已在dev，PR46/47及实际集成CI通过；四条授权验收评论已发布 | Issue保持open；main/v0.3.1未发布这些修复 |
| B-069 / #48 | PR49合入dev b2390b0；最终head ed96c8c的三门及实际dev Unit38629909667 / Package38629909697全部实际矩阵/聚合通过 | 原包门未完成的拒绝证据保留；不把skip或未执行GPU/HF算通过；Issue48仍open |
| 当前新B-070 / #50 | 新鲜合成输入已复现，先开Issue50、再提交红色测试f15e067 | 当前候选及其最终head/实际dev必须独立验收；不合main或自动关Issue |

B-069实际dev执行清单SHA256为636268155bb4d6d185868d539634b0fe9376dd33c1f9291a42fad83536e823c5；main仍7a15c46。旧三个工作分支、全部数据/Run/provenance保留。后续本轮仅使用一个新分支codex/status-metric-shape，不干预下游vendor/实验。

#### （二）新反例与最小方案

公共validate_result/write_result接受成功记录包含metrics=[1]，而list_runs先dict(...)导致TypeError；原生status返回2，正常peer被一起隐藏。列表对[['accuracy',0.9]]则会被当作mapping，显示并未按object记录的accuracy=0.9000。原始API/CLI输出和字节保全见.tmp/audit070/REPRO.json；这不是正式训练失败、结果损坏或性能结论。

先创建[Issue #50](https://github.com/DreamSoul-AI/RPipe/issues/50)，再提交10项portable回归f15e067；旧代码 **7 failed /3 passed**，三个控制为None、空mapping和合法数值mapping。新增原生CLI用例覆盖eval过滤、Run成功状态/peer显示及全部输入字节不变。修复前status.py原字节ZIP/hash、红色XML/日志保留在.tmp/audit070，不随clone提供。

方案仅取消list_runs调用方的dict强转，把原值交给既有_metric_cell的mapping/numeric guard；不改生产者、结果校验/格式、Run状态、mode、聚合或科学门，也不新增schema/解析器/重试/依赖。生产修复0b51ee6仅一行；10项新增节点全包含于绿色批次，与readout全目录/活动提示/原生status相邻门联合 **91 passed，0 failed / error / skipped**。红/绿节点、b2390b0到当前源码逐行仅该处变更与XML/SHA256已核对（Git与工作区原字节均为CRLF，read_text读取后规范化为LF；首次未统一文本比较因此拒绝，原字节hash与NEWLINE_CHECK.json分别保留，不当库失败），保存于.tmp/audit070/LOCAL_CHECKED.json。最终提交及实际集成CI仍分别核对，不以本机通过代替。

### 62. readout 存在性查询与内容读取的错误边界（B-069 / #48）

按新授权在实际 dev f9c6e90 检查四类相邻路径，取得新的逐文件 I/O 反例。readout status 的 _load_result / _note_from_log、logs 的 event_lines 已对内容读取失败做逐文件回退，但 is_file 在 try 外；同一 PermissionError 提前发生时，API 异常、原生 CLI 返回 2，正常 peer 结果或日志一并不可见。与 B-068 activity 已修入口分开，不重复登记 #37 模式问题。

先在唯一新分支 codex/status-readout-errors 提交 portable 红色回归 31d1511；生产原字节 ZIP 与 SHA256 保存在 .tmp/phase069/before.zip / BEFORE.json，新增 13 项为 **5 failed / 8 passed**。五项失败分别定位 result stat、note stat、event stat、原生 status、原生 logs；内容读取拒绝与意外 RuntimeError 控制通过。新 [Issue #48](https://github.com/DreamSoul-AI/RPipe/issues/48) 已公开提交脱敏合成条件，未宣称真实系统 ACL 配置或正式数据损坏。

设计只统一这三个函数各自已有的元数据/读取 I/O 边界；不创建 helper、改异常类、调整模式、改变不可读结果的原计划回退，亦不改 process/调度/科学条件。候选仅移动三个 is_file 到原有 try，异常类别保持原样。新增 13 项均已在绿色批次执行；与 readout 全目录、活动提示及原生 status 相邻门联合 **81 passed，0 failed / error / skipped**。XML、源码前后 hash 和红/绿节点包含关系已核对，保存于 .tmp/phase069/LOCAL_CHECKED.json。最终工作提交和集成提交 CI 分别核对，不以本机通过代替。其他三类审查范围为 Runner 初始拒绝、checkpoint staging 的已有恢复合同以及 CLI 默认/本次参数；未把保留诊断 staging 或正常运行失败写回认作新缺陷，不因此追加改动。

### 61. 四个开放 Issue 补齐修复与验收评论

按用户明确授权，在 #36 / #37 / #38 / #41 各留一条公开评论，分别链接实际生产修复、当前回归文件及已验收 dev f9c6e90 的 Unit38624513622 / Package38624513901；本机 75 项与远端实际矩阵分开，skip 与未执行 GPU/HF 不计通过。发布前逐项读已有评论、核对真实 Git 提交；8269bff 是验证记录，#41 生产修复链接到 PR42 集成 8c15cd8，避免把文档提交冒充代码修复。

实际评论：[#36](https://github.com/DreamSoul-AI/RPipe/issues/36#issuecomment-6177095209)、[#37](https://github.com/DreamSoul-AI/RPipe/issues/37#issuecomment-6177095726)、[#38](https://github.com/DreamSoul-AI/RPipe/issues/38#issuecomment-6177096237)、[#41](https://github.com/DreamSoul-AI/RPipe/issues/41#issuecomment-6177096891)。四个 Issue 保持 open，main 不变；等维护者后续决定 main，再按授权处理关闭。没有发送跨会话消息、覆盖下游源码或开启性能/超时测试。

### 60. 四个开放 Issue 的当前 dev 逐项复验与边界补测

以实际集成 `5ee1214` 为基线，在唯一后续分支 codex/open-issue-correctness 中按 Issue 分开提交。四个原始问题已由此前 PR39/40/42–45 修复，本轮没有发现需要修改生产代码的新反例，因此只补有意义的边界，不重复造修复或改公共合同。

| **Issue** | **原报告行为 → dev 行为** | **本轮新增边界** |
| --- | --- | --- |
| [#36](https://github.com/DreamSoul-AI/RPipe/issues/36) | 初始 freeze 拒绝覆盖旧成功 → 保留旧成功，新的失败仍定稿 | 两次独立原生 run-one 连续拒绝；Study 扩展开/关均核对 result 和非日志 asset 字节不变 |
| [#37](https://github.com/DreamSoul-AI/RPipe/issues/37) | 固定 eval 被显示/筛选为 train → 从实际计划配置读取模式 | base/fixed 的 inference 也经 make→status / 指定模式 / train 过滤核验；全部文件字节不变 |
| [#38](https://github.com/DreamSoul-AI/RPipe/issues/38) | 初次部分写入抛错遗留 staging → finally 清理自己的临时文件 | KeyboardInterrupt 后清理；真实 ASCII 编码拒绝；目标存在/缺失均保留 peer 字节与旧目标 |
| [#41](https://github.com/DreamSoul-AI/RPipe/issues/41) | CLI 无 API 既有重试关闭选项 → launch --no-retry 单次生效 | 一次 opt-out 失败后再显式默认 launch：总计三次尝试，默认重试恢复，jobs 原字节不变 |

本机 Python 3.13.9，使用已有隔离环境，CUDA_VISIBLE_DEVICES 为空、MPLBACKEND=Agg，独立 basetemp `.tmp/e-issues4/p`。四个目标文件分别 11 / 4 / 13 / 4 项；相邻冻结采集 22 项、status 21 项，联合 **75 passed，0 failed / error / skipped**。XML SHA-256：551150e466463fe5a5cdedccc861591fa83ff73c27d5c80519539646704d4ce4。保存每次原生 stdout/stderr、原生 Run/result/log 与 pytest manifest/events；`.tmp/issue-audit-20261015` 和 `.tmp/e-issues4` 不随 clone 提供，回归代码入库。

复验调用：[冻结保全](../../tests/rpipe/flow/test_freeze_preservation.py)、[计划模式](../../tests/rpipe/flow/test_status_planned_modes.py)、[原子文本](../../tests/rpipe/structure/artifact/test_atomic.py)、[CLI 重试](../../tests/rpipe/flow/test_cli_retry.py)。不从 pytest 运行时间作性能结论；没有 GPU、外部数据、下游 vendor 更新或正式 Study 重训。只读检查三个下游最新状态，未取得新的 RPipe 缺陷线索，未向其他对话发送消息。Issue 保持 open，后续 PR 最终提交和实际 dev 的 CI 单独验收，不把本机通过写成已发布。

### 59. PR #46 实际合入 dev 与归档后状态

最终工作提交 `90b7d75323d266c4d272f786c28f4c00959d0432` 的 [Unit](https://github.com/DreamSoul-AI/RPipe/actions/runs/38621366046)、[Package](https://github.com/DreamSoul-AI/RPipe/actions/runs/38621366077)、[Branch](https://github.com/DreamSoul-AI/RPipe/actions/runs/38621366320) 及每个实际子任务全部成功；本机执行清单 checks-90b7d75323d2.json SHA-256 为 f9c96f09ce605bf066255cb035a4a5652f788be5c3e7511182d7da37c3c46f24。核对 clean / 精确 head / 最新 dev 祖先 / MERGEABLE / CLEAN / 仓库不自动删分支后，按维护者最终授权 normal merge [PR #46](https://github.com/DreamSoul-AI/RPipe/pull/46)。

实际 mergeCommit 为 `5ee1214b7ce0dc28b65139ac7b3f9ca16eb8b2ac`。集成提交的 [Unit](https://github.com/DreamSoul-AI/RPipe/actions/runs/38622291540) / [Package](https://github.com/DreamSoul-AI/RPipe/actions/runs/38622291406) 全部实际矩阵和聚合门均成功；DEV_CHECKED.json SHA-256 为 e96bb11ef815777ef13378708ab666f6594be11ab70539bcb86995868aadebbe。未拿 PR 绿灯代替 dev 检查。B-056～B-064、B-066～B-068 已集成，B-065 次数去重在 70231ca 撤回，历史失败/候选和主题提交保留。

origin/main 仍为 `7a15c46b75c5eb951cccd60ed7b9fc3463db90f5`，本地及远端 codex/cache-lock-budget 均保留在最终工作 head；没有删除分支、数据、Run 或来源，没有强推、发布、收费变更。归档快照的“待合入”仅描述记录当时，当前接续已进入四个开放 Issue 的复验。正式 Study 不重新签名，不把本批 CPU 检查称为新科学验收。

### 58. 正确性维护归档与授权进入dev

维护者最终确认允许PR #46转ready、最终CI通过后合dev，撤销早先草案限制；仍不合main、不强推、不删任何分支/数据、不发布、不花钱。重启后先核对工作区干净、源码/测试均已commit/push；B-067并非遗留未提交。为遵守只做正确性，先用独立主题提交撤回B-065次数去重优化，再以文档主题保存[归档快照](maintenance_archive_2026-10-15.md)及handover接续规则。

已完成范围、改前/改后、927c36e实际CI、53项来源正确性和53项并发门、34项status门、历史失败与本机证据边界均在归档文档及§43–57。未完成项明确包括最终归档head CI、PR46合dev/实际dev CI、四开放Issue在当前dev逐项复验及缺口补测；不预先写成全部已合并或新完整科学验收。

本轮不运行GPU/模型/速度测试、不分析原480秒；不覆盖下游vendor，不重写旧Run/provenance。最终合入身份和CI由PR正文/实际GitHub状态核对，归档时的待完成文字只描述当时快照。按最新人类要求保留本地/远端分支，不套用AGENTS合并后删分支的默认规则。

### 57. 按最终范围暂缓 B-065 性能去重

维护者最终限定只做fix/test、不做性能改动，同时授权PR #46转ready并在CI通过后合dev。归档核对发现ad67f74的每次采集次数去重仍在候选差异中，因此以新的普通提交撤回这段优化，不改写旧提交、不强推，也不删除历史证据。

只把source_files恢复为按原声明列表逐项解析/读取，移除dict.fromkeys与本地seen；B-066的库键域/物理身份校验、B-067的mapping验证/冻结错误保全全部保留。对应测试保留重叠路径规范化、hash内容、后续修改检出、读取拒绝、空匹配和越界断言，取消只服务于性能优化的读取次数约束。未改变任何科学门限、配置、Run ID或持久化格式。

旧§52、ad67f74及其红例/计数/性能材料原样保留，不能称为本轮新增测速；§55表是当时草案状态。新的合入候选不含B-065去重优化，后续性能工作仍暂停。本次只运行来源正确性与相邻冻结门回归，最终与归档主题提交共同绑定当前head的必需CI。

### 56. 可选活动提示不能遮蔽正式状态（B-068，草案）

只读复查开放Issues与三个下游对话后，未取得新的RPipe根因反例；dayan的重复配对/归档恢复属于该对话已确认的编排失误，量化在整理本项目结果，output-space仍按研究计划运行，未干预它们。随后主动审查status读取边界，复现可选activity文件使正式状态无法展示的问题。性能与耗时工作继续暂停。

| **条件** | **85d6070 改前** | **候选改后** |
| --- | --- | --- |
| activity.json包含真实非法UTF-8字节 | UnicodeDecodeError直接外溢，CLI即使有成功result也崩溃 | 忽略不可用活动提示，继续读取index/result，旧成功可展示 |
| 查询activity是否存在时PermissionError | is_file在try外，未进入既有IO容错 | 与读取放入同一可选IO错误边界，继续原状态流程 |
| 不存在index且没有有效提示 | 异常遮蔽missing-index诊断 | 保持退出2与missing index提示，不伪报成功 |
| 有效Unicode提示/其他编程错误 | 原有读取或异常行为 | 有效字段与原字节保留，意外RuntimeError仍暴露 |

生产改动仅read_activity：把is_file纳入既有try，增加UnicodeError捕获。已核对唯一生产调用方CLI _execute_status、只读list_runs与既有活动提示用例；announce/clear_activity、索引、来源、Run状态写入和共享输入均不改。沿用坏JSON被忽略的既有合同，不增加重试、警告系统、调度器或依赖。

新增9个unit与4个真实CLI集成，修复前13项中6 failed /7 passed；修复后连同既有status回归共34 passed，无跳过。真实编码坏文件与精准方法替身的存在性/读取拒绝分开记录：非法编码是实际文件读解码，权限边界是隔离故障注入，不改变主机权限。每个集成条件比较index/result/activity原字节及目录成员，无写入或清理；有index显示原成功，无index仍退出2。全部是小型正确性检查，不运行模型、CUDA、下载或训练，不从pytest时间导出速度结论。

.tmp/phase068/before.zip /BEFORE_INDEX.json保存修复前源码和测试，逐字节核对85d6070；red.xml /red.log保留真实失败；green.xml /green.log、final-source.zip与CHECKED.json记录最终源码hash及正确性结果。临时证据不随clone提供；可复现测试为[活动读取单元](../../tests/rpipe/structure/artifact/test_activity.py)与[原生status集成](../../tests/rpipe/flow/test_activity_status.py)。正式Study无已证实受影响记录，未改写既有结果或下游vendor。

本次仍沿用PR #46草案，绑定最终工作提交检查必需CI及所有实际矩阵；最终链接与执行清单放PR正文，不用85d6070旧绿灯替代。dev9324014/main7a15c46不变，不转ready、不合并、不发布。B-058～B-067的逐项集成汇总仍见§55；新增B-068同样只在草案。

### 55. B-058 至 B-067 修复汇总与并发补验

本轮只补正确性测试、提交与开发记录，不修改生产源码，不做 perf 改动、性能采样或耗时比较。B-067 生产修复此前已提交并推送：8a24817，文档补正5bae738；本轮开始时工作区干净，不把已提交修复写成仍未提交。

#### （一）修复汇总与集成状态

逐项核对 Git 祖先关系：下表十项均已进入 codex/cache-lock-budget，即 [PR #46](https://github.com/DreamSoul-AI/RPipe/pull/46) 的草案差异；**十项均未合入 dev 或 main**。dev 为9324014985680a0d186863cd135faa53a2e7bcbf，main 为已发布v0.3.1/7a15c46b75c5eb951cccd60ed7b9fc3463db90f5。dev 中共享输入与#36–38/#41修复来自更早PR，不能与本表草案混为已集成；main仍未发布这些修复。

| **条目** | **改前 → 改后** | **草案修复提交** | **PR #46 / 集成状态** |
| --- | --- | --- | --- |
| B-058 | 事件起点取报告完成时刻 → 使用pytest实际阶段起点 | b20a034 | 已在PR；dev/main未合 |
| B-059 | 清单别名/保留名可进入builder → 构建前校验本机Path等价和相对路径 | 8daebb5（含0bf77af修复） | 已在PR；dev/main未合 |
| B-060 | packed refs/worktree测试来源丢失 → 快速路径加有界Git回退 | 394a068 | 已在PR；dev/main未合 |
| B-061 | Git status失败仍误记dirty → 非零查询返回空可选元数据 | c3b8cef | 已在PR；dev/main未合 |
| B-062 | 同tick引用临时名碰撞 → 每写者短UUID，失败只清理自己的staging | c4d651d（前候选494c064保留） | 已在PR；dev/main未合；本轮补并发失败交错 |
| B-063 | HEAD/ref坏字节或消失使pytest启动失败 → 可选错误隔离并校验SHA形状 | 870da1b | 已在PR；dev/main未合 |
| B-064 | 临时名拼完整目标名令合法长名失败 → 文本/迁移使用短UUID | 11ede41 | 已在PR；dev/main未合 |
| B-065 | 单次来源采集重复读取重叠输入 → 仅当次去重，每次freeze仍重读 | ad67f74；证据补正296fac5 | 历史已在PR；dev/main未合；后续perf工作暂停 |
| B-066 | Study同键覆盖库hash、掩盖库修改 → 保留库键域并核验物理身份 | 80fdce1 | 已在PR；dev/main未合 |
| B-067 | 采集错误绕过冻结保全、覆盖旧成功 → mapping校验和现有拒绝异常，保留cause/旧成功 | 8a24817；文档5bae738 | 已在PR；dev/main未合；本轮补并发拒绝隔离 |

B-065/ad67f74是已经存在的性能改动，本轮不继续优化、测量或扩展，也不将“暂停”写成已撤回该历史提交。PR继续Draft，尚未批准这些差异集成。B-056/B-057也仍在同一草案，本节依本次要求集中汇总B-058～B-067。旧失败、取消和未执行证据保留，既有§45～§54不删减。

#### （二）B-062：失败清理与存活发布者

新增4个确定性交错：默认symlink路径/显式hardlink fallback × 一方失败/双方失败。两线程完成同字节校验并各自产生staging后，令首写者PermissionError退出并完成finally清理；另一写者仍持有自己的staging，确认文件存在、字节完整、与source同物理输入，再成功发布或按计划拒绝。

断言两个临时路径不同，首写者只能清理自己的路径；成功时target是source引用，双方失败时原独立副本保留；source、target及无关peer字节不变，最后只清理各自产生的临时引用。屏障/事件的有限等待仅防挂死，不采集或比较速度，不修改锁、调度或发布实现。仍不承诺任意缺失目标多写者或单Run多写者的新合同。

#### （三）B-067：同Study不同Run同时拒绝

新增4个并发条件：include文件消失/权限拒读 × 两个旧成功Run/一个旧成功加一个无结果Run。两个Runner读取同一份只读来源证据前通过屏障同时进入，使用真实check_frozen、Runner异常处理、日志与结果IO；禁止计算阶段导入。每个Run只有一个执行者/写入者，Context和Logger独立，不共享模型、RNG、sampler或可变状态。

旧成功result逐字节保留；无旧结果的Run各自failed定稿并保留ProvenanceChangedError与原始cause；日志只含本Run标识，来源清单不改写。实际8份落盘结果中6个旧成功保全、2个新failed，不将并发拒绝写成成功模型执行。

#### （四）红例、验收与提交边界

- 先保存B-062基线c3b8cef和B-067基线80fdce1原始文件/hash，再在隔离pytest进程通过临时plugin只恢复link_input、load_provenance/check_frozen；不修改仓库生产文件或下游。8个新增并发用例全部失败，red.xml /red.log /RED_REASONS.json保留。旧B-062确有临时路径撞名、FileExistsError/共享拒绝及关联屏障退出；旧B-067实际覆盖旧成功，不只是异常文本不同。
- 当前草案的新增8项与既有共享引用、原子发布、长名staging、冻结采集等相关回归共53 passed，无跳过；green.xml /green.log保留。没有用早先运行代替本轮执行。
- TEST_SNAPSHOT.zip /SNAPSHOT_INDEX.json保存源码/测试字节；INTEGRATION_MAP.json保存逐项祖先核对；RESULTS.json保存实际8份结果状态/hash。首次证据收集器因pytest缩短临时目录名而拒绝空选择，EVIDENCE_SCAN_FAILURE.json保留，修正目录选择后检查原输出，无重跑/改写结果。临时材料在.tmp/b062-b067-concurrency，不随clone提供；便携回归见[共享引用测试](../../tests/rpipe/structure/artifact/test_shared_references.py)与[冻结拒绝测试](../../tests/rpipe/flow/test_capture_refusal.py)。
- 本轮提交只有测试、record及bugs状态补充，生产src相对5bae738无变化。提交后绑定最终head核对Unit tests / Build package / Branch flow及所有实际子任务，不拿5bae738旧绿灯替代。最终链接与验收清单同步PR正文；PR保持Draft OPEN、无auto-merge，不合并、不发布。

### 54. 暂停性能调查与冻结采集错误保全（B-067，草案）

按最新人类指示，暂停所有速度、耗时、480 秒插桩、profiler、缓存与资源竞争测量。本轮只验证代码正确性；GPU 争用保留为未证实假设，不据此归因或新增全局 GPU 互斥。旧证据保留，后续速度验证另行恢复。未向下游发送消息或干预实验。

#### （一）改前与改后

| **条件** | **80fdce1 / 首候选** | **最终候选** |
| --- | --- | --- |
| 冻结 include 文件消失、读取被拒绝、来源 JSON 损坏 | 普通 FileNotFoundError / PermissionError / JSONDecodeError 越过冻结拒绝合同；实际 Runner 把旧成功改写为 failed | check_frozen 将 OSError / ValueError 转为现有 ProvenanceChangedError，保留 cause；CLI 拒绝，不执行计算，旧成功原字节保全 |
| 合法 JSON 但根、files 或 plan 为列表 | 首候选仍抛 AttributeError，可再次绕过保全 | load_provenance 验证 mapping；结构错误以 ValueError 进入同一冻结拒绝路径 |
| make / freeze 关闭 / 非预期编程异常 | 原输入错误、关闭时不查来源、RuntimeError 直接暴露 | 保留原合同；不改 make 接受行为，不捕获任意异常，不重写旧来源 |
| Study 扩展关闭后的 prepare | 同类采集失败仍可覆盖旧成功 | 真实 prepare 冻结门拒绝；运行时和 System 在测试中隔离，不构造 Data/Model |

只改 provenance.py 的 load_provenance 与 check_frozen；已核对 source_files、write_provenance、provenance_changes、CLI、Runner、prepare、Study process 及 summarize 调用。磁盘根必须是 mapping，files/plan 缺省或 null 沿用旧空映射语义，其余须为 mapping；不改变持久化格式、digest、Run ID、Runner 保全条件或共享输入。异常栈保留原始 cause，非预期 RuntimeError 不被归类为来源变更。

#### （二）可复现证据与正确性检查

新增定向用例使用隔离合成库/Study 与独立 basetemp，不执行模型或 CUDA 计算；prepare 仅对运行时/System 做轻量替身，冻结门和失败定稿使用真实实现。相邻 artifact 读出回归包含 PyTorch 微小 CPU 张量的 checkpoint 比较，不宣称整批未导入 Torch。没有正式输入、下载或训练。pytest 默认报告时间保留在原文件，本节不提取速度指标或比较耗时。

| **证据** | **正确性结果** |
| --- | --- |
| before.zip / source-index.json | 实施前 80fdce1 源码与最初 19 项测试原字节及 SHA-256 |
| red.xml / red.log / red-results.json | 15 failed /4 passed；真实旧成功覆盖为 failed 的三类反例均保留 |
| green.xml / boundary.xml | 首候选19通过；追加关闭扩展的真实prepare后21通过，仅为当时范围 |
| first-candidate.zip / candidate-index.json / shape-red.xml | 追加合法JSON非法结构后7 failed /21 passed；没有用早期绿灯覆盖缺口 |
| final.xml | 34 passed：B-067 的12单元、18真实门/Runner/prepare，加B-066的4个原生门控制 |
| final-core.xml | 214 passed /5 deselected；artifact、Git来源及真实manifest入口的相邻unit content检查；未选中不计通过 |
| final-results.json | 14份B-067实际落盘结果：9份旧成功保留相同字节，5份无旧成功的Runner写failed；CLI无旧结果时不新增结果 |
| CHECKED.json / final-source.zip | 原字节、ZIP CRC、各阶段结果/日志hash、最终生产与测试字节、结果合同核对通过 |

临时证据均在 .tmp/phase067，不随 Git clone 提供；便携回归为 tests/rpipe/structure/artifact/test_freeze_capture.py 与 tests/rpipe/flow/test_capture_refusal.py。没有正式 Study 受影响的已证实事故，不迁移数据、不重签旧 provenance、不重训或修改科学预算。

#### （三）最终提交、验收边界与待合入清单

B-066 的工作提交 80fdce1773fc4510548912661acab8ea058cbc8c：[Unit](https://github.com/DreamSoul-AI/RPipe/actions/runs/38611789823)、[Package](https://github.com/DreamSoul-AI/RPipe/actions/runs/38611789983)、[Branch](https://github.com/DreamSoul-AI/RPipe/actions/runs/38611789914)及实际矩阵最终均成功。早先聚合绿但Ubuntu3.10实际子任务仍IN_PROGRESS时，执行清单正确拒绝；first-check-pr-status.log保留，未靠聚合绿灯签收。最终补报的job116016715215和PR状态快照保存到本轮 prior-matrix-final.json / prior-head-ci.json。因本轮B-067已有工作区修改，没有把旧清单在脏工作区运行并伪报通过；最终提交另核对全部证据与其自身CI。

合入前依次检查：生产差异仅两处边界处理；红例/候选/最终快照与hash可核验；最终本机行为门、实际落盘结果与原生CPU流程/安装矩阵全部成功；绑定最新工作提交核对Unit tests / Build package / Branch flow及所有子任务；PR仍Draft OPEN且无auto-merge；dev9324014、main7a15c46不变。PR #46按人类要求保持草案，不执行合并或发布。

只读复查：开放Issues仍为36/37/38/41，无新增；dayan仍在自己的研究执行，量化正补本项目占用守卫识别范围，output-space正在FP32质量检查，这些反馈未提供新的RPipe根因反例。不将下游自有守卫或研究结果登记为库缺陷。

### 53. 来源键域遮蔽与冻结拒绝保全（B-066，草案）

本轮主动审查source_files，确认Study输入可覆盖库的同名来源键，导致真实库修改/删除被掩盖。候选保留rpipe/库键域，仅允许与对应库路径可核验为同一物理文件的输入复用；不同副本、缺失或不可核验身份均以现有ProvenanceChangedError拒绝。10项新增单元、4项真实冻结门/Runner边界通过，相邻core为202 passed /5 deselected。PR #46继续草案，dev/main未变，不发布或改写旧来源。

#### （一）改前与改后

| **条件** | **296fac5 / 首候选** | **最终候选** |
| --- | --- | --- |
| Study include rpipe/__init__.py是独立文件 | 覆盖库哈希；即使初始字节相同也会隐藏后续库修改 | 拒绝占用库键域，不按相同字节冒充同一输入 |
| 库文件修改后检查旧歧义清单 | legacy-bypass.json实测check_frozen仍放行，记录哈希等于Study副本 | 冻结拒绝，保留旧清单和成功结果，无计算阶段执行 |
| 同一实际文件或硬链接 | 初始用例通过；首候选仍漏掉库路径被删除、Study硬链接保留的反例 | 对应库身份存在且samefile才可复用；修改能检测，删除或身份查询失败拒绝 |
| 尚无对应库文件的Study rpipe/extra.py | 首候选允许，未来库成员变动可能仍被混合键来源掩盖 | 整个rpipe/逻辑键域保留；rpipe_inputs/extra.py等非保留键仍可使用 |

库源码键相对包根加rpipe/，Study键相对Study根，本来共用一个flat mapping。只在已存在键上判断冲突不能保留来源成员身份：库路径删除后，Study硬链接可继续提供相同键/hash，因此最终必须声明保留键域，而不只拒绝当前字典重名。未改schema、digest算法或持久化键格式；无歧义Study的mapping保持。旧歧义Study须将独立输入移至不同相对路径（如rpipe_inputs/）或恢复真实库身份，保留旧证据，按计划用新version/fresh make；不批量迁移正式输入或重签旧Run。

#### （二）真实复现与候选收缩

隔离库/Study各有自己的__init__.py，只在诊断进程调整rpipe.__file__以定位合成库，并以显式synthetic环境/Git stub避免Torch采集；不导入或修改下游vendor，不执行模型。legacy-bypass.json记录库hash从a0e974...变为e3d2db...，check_frozen仍成功，清单值b02cf1...等于Study副本而非库。完整hash、原始字节与诊断路径均保留，不声称正式实验受过该反例影响。

| **阶段证据（均保留）** | **结果与边界** |
| --- | --- |
| 初始单元红 .tmp/phase066/red.xml | 5 failed /2 passed，0.20s；两个同文件控制已通过 |
| 首次混合收集 red-final.xml | 两个新测试basename相同导致1 collection error，0.19s；重命名Flow测试，不删缓存或旧日志；不当库缺陷证据 |
| 实际冻结门/Runner反例 red-gates.xml | 9 failed /2 passed，0.31s；旧launch gate返回True，Runner进入禁止执行的合成阶段 |
| 首候选 green.xml | 11 passed，0.23s；仅说明初始条件，不是最终验收 |
| 追加删除/保留键边界 boundary-red.xml | 2 failed /11 passed，0.27s；库路径真实删除仍被Study硬链接掩盖 |
| 身份拒绝 identity-red.xml | 1 failed /9 deselected，0.14s；PermissionError直接外溢，需保留cause转冻结拒绝 |
| 最终候选首次 final-gates.xml | 1 failed /13 passed，0.29s；实际已正确拒绝缺失身份，但测试只匹配collid文本，调整为library消息断言；保留原失败 |
| 最终 final-gates2.xml | 14 passed，0.25s；10单元+4真实gate/Runner，不启动计算 |
| 相邻 final-core.xml | 202 passed /5 deselected，6.93s；artifact镜像、Git及真实manifest入口，未选integration不计通过 |

实施前的pre-fix-acceptance.zip / acceptance-source-index.json含296fac5源码与当时11项冻结/门测试的原字节和hash；后续扩展边界另有首候选快照；更早pre-fix.zip、收集失败快照、first-candidate.zip和pre-final-boundary.zip均保留。使用互不别名的索引/结果文件名，不重演§52的大小写覆盖。临时证据在.tmp/phase066，不随Git clone提供；便携回归在tests/rpipe/structure/artifact/test_source_keys.py与tests/rpipe/flow/test_source_key_gate.py。

#### （三）最小实现与流程合同

变更仅在source_files写入Study键前检查rpipe/来源身份。相同键域但不同物理文件拒绝；samefile的OSError以cause保留并转为现有ProvenanceChangedError，无新异常或依赖。正常两柱发现、B-065每次调用内去重及跨调用重读、路径边界不变。额外身份IO仅针对保留键输入，不改变共享data/model或Run可变状态。

已核对write_provenance、provenance_changes/check_frozen以及CLI _provenance_gate、prepare、FlowRunner、Study process调用方。复用现有异常使原生launch gate返回False、打印来源冲突；Runner在阶段前拒绝时沿用#36的旧成功字节保全，无旧成功时保留failed/日志合同。4个集成条件覆盖gate/Runner×有无旧成功，验证未进入phase、旧provenance原字节不改、必要日志/错误类型存在；不以退出码代替结果检查。make拒绝时也不覆写已有来源。

最终提交必需CI与实际矩阵在[PR #46](https://github.com/DreamSoul-AI/RPipe/pull/46)绑定核对，不能拿296fac5旧绿灯替代。执行清单检查所有红/候选/收集失败留存、原字节/hash、最终202 core及14门、运行差异只限本函数、PR Draft/open/no-auto-merge、dev/main不变。原480s根因仍未确认；此次是来源完整性修复，不作速度归因。

#### （四）只读下游检查

Issues #36–38/#41无新增反馈，仍按要求open。dayan按该对话用户要求结束当前pair并停止全量扫描，74/216为部分结果，不写成完整通过；其PowerShell/自有停止脚本失败不是RPipe新缺陷。量化的占用守卫因4个活动进程拒绝计算是预期资源边界。output-space新增结论属于研究结果，无新RPipe缺陷。未向其他对话发消息，不干预它们的停止、计时或运行。

### 52. 调用内来源扫描拆解与单次去重（B-065，草案）

本轮按路径/IO/哈希/锁/子进程拆解已冻结的实际 package_sources 函数，并以离线小场景复现 RPipe source_files 的重复工作。B-065 候选把一次采集的重叠声明去重，相关192 passed /5 deselected；PR #46 保持草案，不合入 dev/main。原480s调用内根因仍未闭环，本轮结果不冒充原现场。

#### （一）调用内实测：文件/路径操作多于哈希

取 .tmp/tquant063/s/ 中已冻结 provenance.py 的 package_sources 函数 AST，保留精确函数体和 PACKAGE_ROOT，对实际162文件执行；不导入整个下游业务、不执行模型，不修改下游源码。原始模块SHA-256与输出mapping哈希在 before.json/after.json 中一致。计时覆盖实际函数内部的 glob迭代、文件打开、read_bytes、SHA-256构造和hexdigest；对应原两节点 fixture 中反复执行的来源扫描步骤，不等于整个用例或原480s复现。

| **实际冻结 helper 阶段** | **第一次诊断** | **第二次诊断（函数未改）** |
| --- | --- | --- |
| 函数总墙钟 | 13.700 ms | 13.317 ms |
| read_bytes（包含 open） | 7.250 ms | 7.023 ms |
| 其中打开文件 | 4.119 ms | 3.854 ms |
| rglob迭代（包含内部glob） | 2.281 ms | 2.111 ms |
| SHA-256构造 | 0.453 ms | 0.466 ms |
| hexdigest | 0.174 ms | 0.187 ms |
| subprocess.run / cache_lock 入口调用 | 0 /0 | 0 /0 |

上述阶段是包容计时，不能直接相加；原始记录另含exclusive_wall，残差包括相对路径/排序/字典工作与观测开销。glob的calls记录迭代计时片段（每次next也计一次），不是目录扫描调用次数。计时CPU短段可能受15.625ms粒度影响，不把0或CPU大于短墙钟当物理结论。锁入口仅包裹所选实际函数，不推断整个原用例或其库外等待。

结论限于此函数：路径遍历与文件打开/读取明显多于哈希计算；未发生锁和子进程。§48已测两节点每个package_sources有161次调用、累计约2s，但不能把本轮单调用13ms机械乘到原批次，也不能因此解释所有469.527s。

#### （二）B-065：改前与改后

source_files 会把启用阶段、两柱自动扫描、recipe、重叠include收集到names，之后每项都 _inside → is_file → read/hash；字典最后覆盖重复键，因此重复工作不体现在输出条数。生产调用为write_provenance和provenance_changes，后者用于launch、prepare/Runner及Study process的freeze。

合成场景为32个4KiB源码、2个声明、1个隔离库输入，共35个唯一文件；flow启用，两个相同structure include及recipe别名。仅在诊断进程将rpipe.__file__暂指向小型库输入，__path__/实际函数不变，随后恢复；不改任何仓库或安装内容。三次调用输入相同，无缓存淘汰操作。

| **同一合成输入，每次调用** | **11ede41 原实现** | **候选单次去重** |
| --- | --- | --- |
| 实际文件打开/读取/哈希次数 | 各100 | 各35 |
| Path.resolve 调用次数 | 223 | 95 |
| 总墙钟3次原始值 | 32.285 /32.382 /31.413 ms | 14.209 /14.562 /12.979 ms |
| 总墙钟中位 | 32.285 ms | 14.209 ms |
| 路径解析中位 | 18.051 ms | 7.211 ms |
| read_bytes中位（含打开） | 5.143 ms | 1.662 ms |
| SHA-256构造/hex中位 | 0.368 /0.161 ms | 0.123 /0.060 ms |
| 完整输入mapping | 35项 | 逐项相等 |

采用原生dict.fromkeys保持首次声明顺序，同一次调用用本地seen集合记录已通过_inside边界检查并成功读取的解析Path。只去掉相同字符串的重复解析，以及别名的重复读取；不跨调用缓存路径/内容/hash，不改变库源集合、格式、旧provenance或下游package_sources。去重集合不与库输出键混用，避免无关键碰撞改变现有映射行为。内存仅随本次Study文件数增长；源采集没有新增并发编辑期间的原子快照保证。

这些是活跃无关负载下带观测器的小规模局部计时，修复前后三次顺序执行，非随机交替性能验收；不宣称严格加速比或整批可节省多少秒。确定性收益是少65次读/哈希和128次路径解析。两次冻结helper函数未改，差异作为波动原样保留。未清OS缓存、新进程不称硬件冷启动；未运行Torch、GPU或下载/正式训练。

#### （三）复现、边界和正式验收

真实文件计数回归 .tmp/phase065/red.xml 为2 failed /1 passed，0.20s；flow关闭时两个文件各读2次、helper4次，开启时flow各3次、helper4次。拒绝读取用例修复前已通过。before.zip保全原实现与当时三项测试字节。临时元数据BEFORE.json与计时before.json在Windows上大小写别名，后者覆盖了前者；首次证据清单因KeyError: mode失败，草案代码已推送但未签收。原ZIP及计时字节未改，另存diagnostic-before.json并重建SNAPSHOT_FILES.json，逐字节核对ZIP内生产源码等于11ede41及计时源hash，归档测试AST等于当前前三项；不声称原先独立的测试hash元数据仍存在，FIRST_CHECK_FAILURE.json保留该缺口。之后加空匹配/越界两项；26项相邻门通过，完整artifact镜像与Git/manifest入口最终192 passed /5 deselected，6.06s。5项未被core选中，不计通过；远端CPU Flow单独核验。

五项新增便携回归在tests/rpipe/structure/artifact/test_source_reads.py。两种flow状态均验证单次每文件只读一次、下一次调用必须重新读且修改哈希变化；同时验证PermissionError外溢、缺失glob拒绝、Study外路径拒绝。既有两柱add/edit/delete和路径逃逸用例仍执行。临时完整源快照、分段原始wall/CPU、日志、JSON/JUnit在.tmp/phase065/，不随clone；measured-after.py与已计时候选hash相等，最终源码仅恢复原换行，AST与候选一致。

最终提交的必需CI、实际矩阵及执行清单链接在[PR #46](https://github.com/DreamSoul-AI/RPipe/pull/46)，不得用11ede41旧验收代替。合并前须确认本轮生产差异仅source_files的本地去重、所有红色证据与192通过保留、前后mapping及冻结helper来源不变、PR Draft/open/no-auto-merge和dev/main不变。原480s归因、集成人类授权保持未完成；不重签旧结果、不部署活动源码缓存。

#### （四）可落地结论

B-065只修复RPipe自身已证实的同一调用内重复读取。下游冻结helper内部是单次遍历/读取，161次重复调用是否可安全复用属于业务夹具生命周期设计，不能把此处seen集合扩为跨调用缓存；§50修改/新增/删除反例继续约束后续优化。当前不修改下游业务，也没有依据将旧480s报告为已修复。

### 51. 长文件名 staging 修复与小规模竞争分段诊断（B-064，草案）

本轮主动审查得到 B-064：合法目标文件名在文本写入、共享迁移时因临时名拼接失败；两处改为已有短 UUID 名，未更改格式或实际文件名。本机相关 187 passed / 5 deselected，PR #46 保持草案，dev/main 不变。另完成两个小进程的准备竞争与分段计时；原量化 480 s 的调用内根因仍不可由保留证据唯一确定。

#### （一）改前与改后

| **检查** | **改前（870da1b）** | **改后** |
| --- | --- | --- |
| 245 字符合法 ASCII 目标名 | 原目标能创建，但文本 create/replace 的 staging 超过单组件上限；data/model 同字节迁移亦失败 | 两处使用独立 UUID-only staging；4 项真实文件系统反例通过 |
| 发布失败与并发准备 | 拒绝发布时旧副本/peer 保全已有保证 | 保留保证；新增两线程 prepared_resource 同 revision 只构建一次、相同 manifest 与完整输入 |
| 原 480 s 的解释范围 | 知道外部批次截止，不能从旧 SHA 还原全部 dirty 测试/CPU/栈 | 重新核对 1131 个完成 call 累计 469.527 s；到最后报告 479.017908 s，其余约 9.490908 s。调用外启动不能单独解释累计成本，调用内懒导入/fixture/计算/等待仍混合 |

已核对 atomic_write_text 的 config/result/tracker/activity/provenance/process/readout/manifest 调用方及 CLI shared --migrate → migrate_shared。修复只改两个 staging basename 并删除不再使用的 os 导入，锁顺序、同目录原子 replace、冲突校验、清理、有界 Windows 重试、返回值与目标名保持。未改变 LongPathsEnabled、持久化格式、科学条件、Run 或正式输入；Windows 长名测试显式使用扩展路径，不承诺任意总路径长度。

#### （二）复现、回归与来源

首轮 .tmp/phase064/red.log 因我未先建立 basetemp 父目录而出现 5 setup errors，测试没有执行；保留该次错误，不用作库缺陷证据。建立新父目录后 red2.log / red2.xml 为 4 failed / 1 passed，0.22 s，真实失败为文本 Errno 22、迁移 WinError 123；发布拒绝保全通过。before.zip / BEFORE.json 保存原实现和当时五项测试的原字节与 SHA-256。

首次局部修复后 67 passed / 1.68 s；补充实际 prepared_resource 并发用例后，统一入口对 artifact 镜像及 Git/真实 manifest 入口的最终相关门为 **187 passed / 5 deselected，7.19 s**。5 项 integration 未被 --core 选中，不计通过，远端 CPU Flow 另行验收。所有临时日志、JUnit、微场景和快照在 .tmp/phase064/，不随 Git clone 提供；便携回归为 tests/rpipe/structure/artifact/test_staging_names.py。

#### （三）两个离线进程的分段诊断

micro.py 调用实际本机 cache_lock / prepared_resource，4096 字节合成输入。两个子进程、一个 builder、两个相同 manifest/payload hash，均验证再次调用不重建；用 -X importtime 保留原始导入树，并逐段记录 wall/CPU。未导入 Torch、未触碰下游项目，不下载/训练。仅该诊断子进程设置 OPENBLAS_NUM_THREADS=1、OMP_NUM_THREADS=1；不修改系统或正式科学配置。

| **阶段** | **owner** | **contender** |
| --- | --- | --- |
| import rpipe | 52.532 ms | 49.891 ms |
| import shared | 28.600 ms | 27.558 ms |
| 读取 4 KiB 输入 | 0.127 ms | 0.116 ms |
| 初始化合成 SHA 状态 | 0.022 ms | 0.027 ms |
| 首次合成 digest 操作 | 0.004 ms | 0.004 ms |
| 首次锁获取 | 0.927 ms | 252.655 ms |
| 同进程校验复用（包含再次获取锁） | 1.215 ms | 1.401 ms |

owner 人为持锁 **206.030 ms**（握手及显式 200 ms sleep），builder 写入 **0.679 ms**；整组外层 **0.415655 s**。等待者锁开销因此是受控注入效果，不能叫自然瓶颈/死锁或外推为原 480 s 根因。prepare_first 包含锁与 builder，不能与嵌套阶段相加。短操作 CPU 读数 0 是计时粒度，不是零成本。此为活跃下游负载下的功能诊断，不作吞吐排名；未清 OS 缓存，新进程不称硬件冷启动；合成 digest 不称真实模型 first step。

#### （四）当前可落地结论与合并前检查

原批次 call 外约 9.491 s 只是由报告起止与舍入 call 耗时重建的近似；监控外层起点和最后未完成调用不同，不能拿这个残差反推每阶段成本。当前冻结两节点的已完成分段证据见 §48：首 call 约 7.9 s，内 fixture 约 6.8 s，package_sources 每节点 161 次累计约 2 s，以及依赖首次懒导入；这些包容计时不相加。原批次缺栈/CPU/部分 dirty 测试字节，不能通过继续生成合成负载补回。

可操作优化方向仍是核验不可变夹具来源输入的复用；§49–50 已否证缓存活动源码字节/清单的直接替换，不部署缓存或减少来源检查。当前实做修复为 B-064 的文件名预算，不把它包装为性能修复。以后若原慢批再现，按完整来源、真正 call 起止和有界同步采样采集新现场；本轮不重跑大批次。

合并前可执行检查须核对：最终提交仅声明的两处生产路径改动；保留首轮夹具错误、4 项真红与最终187通过；快照/原始导入栈/hash 完整；微场景 builder=1 和两侧 manifest 相等；PR Draft/open/no-auto-merge，dev/main 不变；最终提交三项必需 CI 与实际矩阵单独核对。原480归因和人类授权集成保持未完成，不自动合并。最新只读下游快照：dayan 28/216 对无失败；量化等待计算窗口；output-space 准备多 prompt/seed 审计，均无新增 RPipe 缺陷。

### 50. 可选 Git 元数据失败隔离与活动源码缓存否证（B-063，草案）

本轮完成 B-063 主动复现、最小修复和相关 108 项本机验证；活动源码缓存的编辑/新增/删除负向控制明确失败，不能直接落地。PR #46 仍为草案，dev/main 不变；最终 head 的必需 CI 另行核对，不使用 c4d651d 的旧验收替代。原 480 s 批次的历史现场缺口仍存在，没有将当前负向控制写成历史慢因闭环。

#### （一）B-063：可选字段不能中断真实测试入口

- 改前：`_git_sha` 只捕获最后 Git 子进程的异常，直接 HEAD / loose ref 的读取在 try 外。非 UTF-8、拒绝读取、文件消失可外溢；无空格的 ref: 会触发 IndexError，任意文本又可直接成为 git_sha。
- 修复：现有直接读取与 Git 查询共用可选错误边界，捕获 OSError / UnicodeError / SubprocessError；安全识别标准 ref 前缀，文件快速返回仅接受 40 / 64 位十六进制形状。正常 loose/detached 快速路径、packed/worktree 的 Git 回退、5 s 截止和显式 GITHUB_SHA 优先级不变。形状检查不证明对象存在，也不替代完整 dirty 来源快照。没有改生产库、科学条件或旧 manifest。
- 调用方是 pytest_configure。新集成回归复制当前 tests/run.py、conftest.py 与配置到独立临时目录，损坏的 .git 仅在该 fixture 内。验证实际测试执行标记、正式 manifest 与一条 passed 事件；空可选来源为 null，显式 CI SHA 则保持原值。不只检查子进程退出码。

修复前 `.tmp/e063-red/` 为 10 failed /10 passed，1.10 s，覆盖 HEAD/ref 真实损坏字节、非法文本、格式及注入读取拒绝/删除；既有布局和两个有效 SHA 快速路径仍通过。`.tmp/e063-session-red/` 为 2 failed /1 passed，1.11 s；两个真实子进程在 pytest_configure 退出 3、尚未执行测试，显式 CI SHA 分支成功。原始日志、JUnit 和独立 basetemp 保留，配置失败不是原 480 s 长调用的证据。

修复后 `.tmp/e063-green/` 联合 108 passed，5.47 s，无跳过；含 12 项新增单元边界、3 项真实入口集成及先前 93 项相关合同。修复前 tests/conftest.py 与当时新增测试的原始字节、逐文件 hash / baseline c4d651d 记录在 `.tmp/phase063/BEFORE.json` 和 b063-before.zip，不随 clone 提供。测试替身只覆盖不可稳定制造的权限/TOCTOU，坏字节与入口配置失败均使用真实文件和进程。

#### （二）缓存影响的明确否证与可落地边界

承接 §49 已测得的原生 9–14 ms / RAM 字节哈希约 0.35 ms，进一步用独立合成源码执行功能控制，记录在 `.tmp/phase063/functional/RESULT.json`，不报告为物理性能测量或正式 Study。

| **变更** | **原生即时扫描 / walk 原型** | **缓存方案的反例** |
| --- | --- | --- |
| 修改 a.py 字节 | 检测到 hash 变化 | 缓存字节仍给出旧 hash |
| 新增 new.py | 清单包含新文件 | 缓存路径清单遗漏新增文件 |
| 删除 sub/b.py | 清单删除该文件 | 缓存路径清单仍尝试读取并抛 FileNotFoundError |

fixture 同时包含隐藏子目录及 .PY 文件，按本机匹配语义比较完整 mapping；原型未证明所有 symlink / junction / 权限语义。结论是不能以“全局缓存 package_sources 字节或路径清单”直接替换活动源码校验，即使缓存版本在静态树上更快，也会丢失必须检测的变化。可落地的当前决定是保持活动源码即时检测；后续若要复用，只能另行设计并验证已确认不可变的来源归档/清单输入，保留变更失效与完整校验责任，不在此自动改下游业务或扩张研究范围。

另准备 `.tmp/phase063/path_probe.py` 比较原生、即时 os.walk、缓存清单、缓存字节，三进程各 20 组交替顺序；启动/运行守卫识别外部 pytest、run-one、--worker 和 watcher。本轮 dayan 已有活动 worker/watcher，截至此记录性能测量未启动；功能否证先独立完成，没有将排队算作通过。守卫最多等待 600 s，若无空闲窗口则保留未执行证据，不干预、停止或重启下游批次。没有清系统缓存，OS 冷盘性能未测；不把首次新进程当硬件冷启动。

#### （三）合并前执行清单

1. 相对 c4d651d 生产 src 不变；变更仅测试记录 helper、新单元/入口集成和相关文档。旧原始失败、候选长名反例、原 480 s 材料与来源全部保留。
2. 可执行清单校验 B-063 的 10 /2 项失败、108 项通过、修复前快照与功能缓存反例；性能批次仍未执行时明确标记，不生成虚假数字。
3. 新 head 必须独立通过四环境 CPU core/Flow、两平台 wheel/sdist 与 Branch flow。PR 最新正文给出最终实际链接、skip 与未执行边界；不能用数量或旧 head 通过替代整体验收。
4. PR Draft/open/no-auto-merge，分支保留；不合并、不直接推 dev/main、不发布、关闭 Issue、删除正式数据或覆盖下游实验。原 480 s 归因与人类集成决定分别保持未闭环/未执行。

### 49. 共享引用发布隔离与源码扫描对照（B-062，草案）

本轮主动审查发现 B-062：同字节副本转共享引用的临时名仍依赖时钟，与之前已修的文本发布入口不同。修复前 2 failed / 3 passed，首轮候选相关 92 项通过，随后长文件名回归发现新反例并修正，最终相关 93 项通过。PR #46 保持草案，未合入 dev/main；最终提交必须单独核对必需 CI，不能使用 c3b8cef 的旧通过替代。

#### （一）复现、改前改后与范围

1. 核对 shared.link_input 的递归 staging、reference_tree 的调用，以及 DataFactory 原始文件布局桥接、显式共享迁移。Factory/迁移的外层锁仍存在，本反例是直接并发调用不可变引用 helper，未声称正式锁保护链已失败或历史数据损坏。
2. 改前临时名为目标名/PID/time_ns；两个线程验证同一已存在的独立副本与源字节相同后，用 barrier 在发布前汇合。冻结 time_ns 时，default 引用和强制 hardlink 路径均共用临时名：分别见 PermissionError、FileExistsError，另一写者因同构测试 barrier 退出。barrier 是测试同步工具，不是生产锁死证据或原 480 s 根因。
3. 最终一行修复把目标/PID/时钟拼接名改为短 uuid4 hex + .link，使用 shared.py 已导入的 UUID，不再携带完整目标名。原子 replace、finally 清理、symlink→hardlink fallback、不同内容拒绝、缓存格式及外层锁都不变。每个写者拥有自己的临时引用，不增加数据副本、不删除源或 peer。
4. 首先新增五项真实文件/线程边界：正常/冻结时钟 × default/hardlink，以及发布 PermissionError 保全源、原副本和 peer。修复前 `.tmp/e062-red/` 为 2 failed / 3 passed，4.25 s；修复后联合共享、原子写、来源及事件回归 `.tmp/e062-green/` 为 92 passed，4.20 s，无跳过。原始日志、JUnit 与独立 basetemp 均保留，不随 clone 提供。
5. 提交前补实际 220 字符合法目标名：Windows 显式使用扩展路径，不修改 LongPathsEnabled；POSIX 用普通临时目录。旧 c3b8cef 的 timestamp 实现成功，首候选 494c064 的“完整目标名+UUID”因临时 basename 过长失败，.tmp/e062-name-red/ 为 1 failed，0.16 s。保留首候选及原始 CI，不把它计为最终验收。改为短 UUID-only basename 后新增该回归，.tmp/e062-final/ 最终 93 passed，4.18 s；源与目标 samefile，源字节/peer/清理不变。没有承诺所有路径、文件系统或自动扩展路径兼容；调用者仍需合法可访问的父目录。

本轮不扩展为任意缺失目标的并发创建合同或可变 Run 多写者协议；模型对象、checkpoint、sampler、RNG 仍独立。没有修改下游 vendor/实验或迁移正式输入。

#### （二）进一步拆分当前源码扫描开销

为核验 §48 的 package_sources 热点，准备 `.tmp/phase062/source_scan_probe.py`，在同一冻结副本及其实际解释器中分开记录模块导入、首次原生调用、glob、read_bytes 与 SHA-256；20 对原生/RAM 字节输入调用交替顺序，逐次比对完整路径→hash mapping。RAM 对照只省略路径扫描和打开/读取，不能据此给活动源码启用缓存。未清 OS 页缓存；父进程来源核对和 fixture 创建本身可能加热文件，首次调用不是硬件冷盘。

量化回归 A 结束后立即开始 B，守卫记录真实活动进程/PID+创建时间并保持等待，未把 A 结束当整机空闲。守卫检测外部 pytest、--worker 和 watch_idle_batch.py；执行过程中若新测试启动，停止自己的诊断而不干预对方。无守卫放行的批次不记为测量完成；guard-wait.json、日志、RUN/DETAIL 与冻结来源校验保留。该守卫仍不覆盖任意非 Python 外部负载。

资源空闲后两个受控进程均完成，外层 1.338476 / 1.334031 s，returncode=0、reason=null，执行前后所有冻结来源 hash 相同。实际模块位于冻结目录，原函数每次都返回相同 162 文件、779639 字节的完整 mapping；两个进程各 20 对，共 40 对交替顺序调用，不称为 80 次独立实验。

| **阶段 / 对照** | **进程 1 秒** | **进程 2 秒** |
| --- | --- | --- |
| 模块导入 | 0.108909 | 0.081222 |
| 原函数首次调用（OS 缓存未清） | 0.015798 | 0.010395 |
| 单独 glob | 0.001987 | 0.001858 |
| 单独 read_bytes（含路径键构造） | 0.012329 | 0.007526 |
| 已有字节 SHA-256 | 0.000391 | 0.000379 |
| 原函数 20 次调用中位数 | 0.013810 | 0.009079 |
| RAM 字节对照 20 次调用中位数 | 0.000358 | 0.000350 |

当前 0.78 MB 小源码树的成本主要在路径遍历、打开/读取与对象构造，纯 SHA-256 低于 0.4 ms。RAM 对照同时省略这些职责，不能把差额全归为磁盘物理读取，也不能由已缓存页上的结果推断冷盘性能。微秒级 process_time 的零值受 Windows 计时粒度影响，不表示没有 CPU 工作。两进程原生中位数差异完整保留，不挑快的一组验收。§48 的 161 次扫描对应静态清单的 26082 次文件哈希访问，说明当前用例会反复执行来源验证；这是需要保持的证据合同，不据此删除校验。

原始 RUN/DETAIL、argv、逐对 wall/CPU、守卫活动 PID+创建时间、解释器和冻结 archive 标识均保留，CHECKED.json 验证两批完成、mapping 断言、导入路径和原始文件 hash。脚本 SHA256 bc5f3c3321d45321bbdc802731687c51caffa0b3538e1ea547f00cd417734a22；红色 pytest.log SHA256 866199bd8312a40a8595c3bb07668fc883f675cac8bf149cc0761bb26212ea2b；绿色日志 e4f59ef61837e59edea60ea7264d53230d08e3f45ae0b51afed3259b84194427。全部 `.tmp` 材料不随 clone 提供；冻结来源仍为 §47 的同一 archive，没有发布下游源码。

原 480 s 批次的历史 dirty 字节与调用内栈/CPU依旧缺失。本轮对当前函数的进一步拆分不自动闭环历史根因，不依靠清缓存、改线程、减少断言或共享可变夹具获得结论。

#### （三）合并前执行清单

1. 相对 c3b8cef 的生产改动仅 shared.py 短 UUID 临时名的一行替换；既有 B-061 两行、其他草案修复不变，git diff --check 无问题。
2. 验证六项引用边界、93 项最终相关通过（并保留前轮 92 项）和两项原始失败，保留各批证据；检查源码扫描诊断的退出码、mapping 相等、冻结来源/脚本/原始记录 hash 与外部资源守卫边界。
3. 必需 CI 绑定当前 PR head：四环境 CPU core/Flow、两平台独立 wheel/sdist、Branch flow 必须实际完成并成功；skip 不算通过。可执行检查结果写 `.tmp/cache-lock-budget/checks-<head>.json`，并在 PR 正文给出最终链接。
4. PR Draft/open/no-auto-merge，保留活动分支；dev/main、发布、Issue 状态、计费、数据/Run/provenance 不变。原 480 s 慢因与人类集成决定仍分别未闭环/未执行，不把已完成旧 CI 或历史 pending 再次派发。

## 2026-10-14

### 48. 分阶段性能定位与 Git 失败来源记录（B-061，草案）

本轮完成 18 组有界合成 Git 测量、4 批已冻结量化节点的函数诊断，发现并修复一个独立来源缺陷 B-061。修复前 3 failed / 8 passed，修复后相关联合回归 87 passed；最终工作提交的必需 CI 以 PR #46 最新 head 为准，不沿用 §47 的旧提交验收。B-056 至 B-061 均仍是草案候选，未合入 dev/main。

#### （一）受控测量与结论

基线为 1ba0877，Windows 11 / Python 3.13.9；RPipe 隔离环境和量化冻结环境仍沿用 §47 的依赖差异。所有测量串行执行；启动及运行期间检查已识别的下游 --worker / watch_idle_batch.py，未暂停或终止下游进程。该守卫不证明整台设备没有其他负载。未下载模型/数据、改变线程科学条件、清理系统缓存或运行正式 Study。

`.tmp/phase061/measure.py` 为本地可复现入口，创建全新 `.tmp/phase061/repositories/`；重复执行须给出另一个全新输出目录，不能覆盖当前证据。每个规模建 40 个额外 ref，10 / 3000 个 tracked 与等量 untracked 合成小文件，依次比较 loose、packed、detached linked worktree。每格 3 个新 Python 进程，每进程重复 3 次 `_git_sha` 和 `capture_git`，共 18 个进程 / 每函数 54 次。分别记录模块导入、每个 Git 命令、测试提交查询、库 Git 来源、库/Study 源码扫描与环境采集。原始 argv、returncode 与秒级时长在 git-stage-results.json；没有用新训练循环代替原生入口。

| **规模（tracked / untracked）** | **布局** | **测试 `_git_sha` 中位 / 最大秒** | **库 `capture_git` 中位 / 最大秒** |
| --- | --- | --- | --- |
| 10 / 10 | loose | 0.000275 / 0.000413 | 0.059897 / 0.073665 |
| 10 / 10 | packed | 0.031256 / 0.034452 | 0.060625 / 0.071389 |
| 10 / 10 | worktree | 0.026086 / 0.033418 | 0.056985 / 0.067707 |
| 3000 / 3000 | loose | 0.000284 / 0.000455 | 0.064061 / 0.072597 |
| 3000 / 3000 | packed | 0.031016 / 0.040995 | 0.061799 / 0.080161 |
| 3000 / 3000 | worktree | 0.029126 / 0.036133 | 0.063746 / 0.384201 |

每格 source_files 的进程间中位数为 0.012806–0.014372 s。`capture_environment()` 在同一进程首次为 2.893815 s，第二次为 0.000018 s，体现依赖首次导入与进程内模块缓存。未清 OS 页缓存，创建 fixture 已可能加热文件；“新进程首次 / 进程内重复”不等于硬件冷盘 / 热盘。这里的“大”只指 3000 个小文件，不能推断超大仓库、网络盘或任意 ref 数量。0.384201 s 的 worktree 首调用反例保留，不只报告中位数。

当前结论：packed/worktree 的测试提交查询增加约一个 Git 子进程启动成本；这组规模下库 Git 查询为几十毫秒，未观察到分钟级等待。不得据此声称所有 Git 负载都很快，或排除原 480 s 调用中的任何历史等待。

#### （二）函数级诊断与观测器核验

`profile_nodes.py`、`profile_manual_nodes.py` 在 §47 的同一 342 文件冻结副本 `.tmp/tquant063/s/` 上运行两个原节点，仍用 tests/run.py 的 C1/C2 CPU integration 过滤；每次新 basetemp / JUnit。未改冻结源码，执行前后逐文件哈希相同，实际 rpipe/quantization 导入路径均指向冻结副本。脚本、插件、argv、JUnit、phases.jsonl、profile 原始 pstats / JSON 全保留在 `.tmp/phase061/`；每批 180 s 外层保护，未触发。

1. cProfile 两批各 2 passed，外层 19.650878 / 19.645906 s。保留原始结果，但首调用某项累计时间 19.3872 s 大于实际 call 11.5558 s，不能据此给函数排占比或相加归因；未把第三方 profiler 失真登记为 RPipe 缺陷。随后改用显式函数和导入 wrapper，单独保留这套观测器，不能挑较快结果覆盖前批。
2. 显式计时两批各 2 passed，外层 15.749357 / 14.411109 s；collection 为 1.243590 / 1.098251 s，setup / teardown 每项不足 0.005 s。第一 call 为 7.883675 / 7.854286 s，CPU 7.8125 / 7.71875 s；第二 call 为 2.887553 / 2.736152 s，CPU 2.859375 / 2.6875 s。
3. 第一 call 的 `test_reduction_nll.fixture` 为 6.808831 / 6.815931 s，其内 `test_dot.trace_fixture` 为 5.432525 / 5.473595 s。`test_trace.fixture` 为 2.008126 / 1.998563 s。这些合成构造在测试 call 内，不是 pytest 的 setup，因此不能靠 setup 报告或单看训练函数定位开销。
4. 两节点合计 `quantization.structure.artifact.provenance.package_sources` 各调用 161 次，累计 2.085731 / 1.981895 s；每次最大 0.017323 / 0.020395 s。这是下游源码扫描/哈希调用，不是 RPipe Git 查询。Qwen3 configuration 的顶层 import_module 首次约 1.828867 / 1.927319 s，modeling 约 0.311228 / 0.385657 s，同进程后续调用接近缓存命中。wrapper 记录的是包含子调用的时长，父 fixture、导入和哈希之间不能直接相加。

可定位的当前开销为合成 fixture 构造、依赖首次懒导入和重复源码扫描/哈希；与 §47 的 CPU/原始栈观测一致，没有在这些样本中重现 480 s 空等。原批完整 dirty 测试字节及调用内 CPU/锁/栈仍缺，历史超时慢因保持未闭环，不将新函数计时倒推为当时唯一原因。没有擅自缓存可变来源或修改下游业务来获得速度提升。

关键本机证据 SHA256：git-stage-results.json 为 3ab639071c18ed6b37d6bab0b9d6d17b34135fe6741cb2eaa55c99d56f160503；manual-summary.json 为 466191834bd292954b38068a9716188677fcaa2b4fe9ad12dca60afa033e4d08；显式计时 runner 为 24b41ec541861a805c8cf27300c16bad5ba89d709213c76e698181824da84a20，插件为 98ff9e18145d8ceaba55f26cd6ee9560715be43400fbe8aeb987824843ad5faf。CHECKED.json 校验四批成功 JUnit、退出码、冻结导入路径及每个原始文件 hash。上述 `.tmp` 材料不随 clone 提供，冻结下游源码不公开上传；结论与主要数字保留在本节。

#### （三）B-061 最小修复与验收清单

审查了生产 capture_git / 调用方 write_provenance、测试 `_git_sha`、tests/run.py 子进程入口、GPU 探针、cache_lock 与 launch_jobs 的启动/等待边界。没有把长计算无默认截止、未验证的取消策略或历史业务 KeyError 自动当成新缺陷；本轮只实施已复现 B-061。

改前：capture_git 检查 HEAD 查询的退出码，却不检查 status；真实临时损坏 index 在 HEAD 成功而 status 失败时被记为 dirty=false，非零且部分 stdout 又被记为 dirty=true。改后：仅增加两行 returncode 判断，status 失败返回既有空 mapping 语义。正常 head/dirty、命令参数、10/30 s 截止、来源格式、freeze 比较规则、Run 结果及调度不变；不改写旧 provenance。

- 修复前 `.tmp/e061-red/` 为 3 failed / 8 passed，1.53 s；真实损坏索引和空/部分失败 stdout 都失败，其他边界通过。只破坏自建 fixture 的索引。
- 修复后 `.tmp/e061-green/` 为 87 passed，3.93 s，无跳过，覆盖新增 11 项、既有来源冻结、事件/Git 元数据、共享准备及原子写边界。新增实际 clean/dirty/packed/worktree、真实损坏索引、两种失败 stdout 与 HEAD/status 各缺命令/超时边界。
- 合并前执行：git diff --check；核对相对 dev 的生产差异只限已声明 helper 与本次 provenance 两行；检查同一工作分支清洁、dev 祖先、PR 最新 head、Draft/open/no-auto-merge；按最新提交跑齐四环境 CPU core/Flow、两平台 wheel/sdist 和 Branch flow，不把本地或旧 head 代替 CI。
- 可执行清单须同时校验本轮 3 项红色基线、87 项绿色联合回归、18 个新进程测量、四批 profile 与其观测器限制；保留 §47 和更早所有失败/取消。原 480 s 归因与人类集成决定仍分别未闭环/未执行。PR 保持草案，不合并、不改 dev/main、不发布或删除数据。

### 47. 当前源码采栈与测试 Git 来源记录（B-060，草案）

本轮已完成：完整当前源码/测试快照、原生CI过滤参数的本地诊断、B-060最小修复和对应回归。RPipe相关诊断58项通过；量化两个合成节点的无watchdog控制通过，同步采样的两次独立运行各2项通过；两次C watchdog原生失败完整保留。最新代码/测试提交394a068的四环境CPU、两平台安装包及Branch flow均通过，本机最终相关联合回归66项通过。所有修复仍只在PR46草案，dev/main不变。原480秒的历史原因仍不能由这些新样本倒推。

#### （1）范围、快照与复跑入口

只读扫描Issues和三个会话无新RPipe反例：dayan依次完成parallel-2、threads-1、ratio-one计时，量化1629项回归通过后推进新对照，output-space为领域数值研究。未给其他会话发消息，不改下游vendor、数据或运行中实验。HTTP504查询失败单独退避，未当作库缺陷。

.tmp/trace480/trace_case.py及其v3/v4版本使用原生tests/run.py，core或C1/C2 integration/e2e的CI过滤参数保持不变，只限定相关节点、添加诊断plugin、新basetemp与JUnit。0.5s资源采样记录PID/创建时间、进程树CPU/RSS/线程；逐call记录真正monotonic起止、进程CPU和模块实际路径。外层180s，CUDA_VISIBLE_DEVICES为空、MPLBACKEND=Agg；OMP/MKL/OpenBLAS未额外覆盖，测试原精度、输入、调用量与断言不改。实际环境Windows11/Python3.13.9、NumPy2.4.4、torch2.11.0+cu130、PyYAML6.0.3；RPipe隔离环境pytest9.1.1，量化环境pytest8.4.2，不是GitHub runner的同一依赖环境。

快照包含完整当前src/tests/config及两个被测用例所需的历史SOURCE.zip，不含数据集/预训练权重。Git HEAD、dirty状态、删除项、逐文件hash、诊断脚本字节和归档校验均保留；文件在捕获前后、运行前后及ZIP内容逐项核对。量化快照是6afad2b加实际dirty/新增源码，不能用HEAD单独冒充旧现场；所有采样变体的342个源文件hash完全相同。下游源码归档不公开，下面的临时材料仅本机保管，不随Git clone提供；本节数字、环境、失败和结论独立入库。

| **材料** | **内容** | **SHA256** |
| --- | --- | --- |
| .tmp/tpipe061/SOURCE_TESTS.zip | 186文件，含82个tests文件；诊断基线8daebb5及当时新增元数据测试 | 3216207c9c0d97e48760f1e0f8fb049700a832ea4660a1f9b54a3cc0c68dda96 |
| .tmp/tquant061/SOURCE_TESTS.zip | 342文件，含73个tests文件；0.25s watchdog脚本 | f78647113564e6780db5e785928583e89ab4610995cc9fce5692b5597e364d59 |
| .tmp/tquant063/SOURCE_TESTS.zip | 同342文件；同步采样脚本v4 | ca10c8ac4bf04c6be765e6013daef5c8c3b0be4403b1e78e30cd2a2b6cc41f05 |
| .tmp/trace480/trace_case_v4.py | 原始采样脚本；同字节副本也在上述归档diagnostic/内 | 803a8bd21d7aa1efa3f63a3d3d34dce692ca165059021b4b0985f3be2ab0df6c |

脚本prepare生成完整快照，run拒绝覆盖批次；SNAPSHOT.json保留spec、源hash及脚本hash。每批启动前查已知校准watcher/worker；若中途出现校准，只终止自己的诊断进程树。两次准备期guard拒绝未启动pytest，证据在.tmp/trace480/guard-rejected.log；后续新增ratio-one批次期间也拒绝。没有暂停/终止下游。这个保护只覆盖已识别的校准进程，不证明排除了所有外部负载。

```powershell
# 本机已冻结的输入；每次换一个不存在的批次名，先确认无活动校准。
& .tmp/v030-20261013/venv/Scripts/python.exe .tmp/trace480/trace_case.py run .tmp/tpipe061 new-core
& .tmp/v030-20261013/venv/Scripts/python.exe .tmp/trace480/trace_case_v4.py run .tmp/tquant063 new-quant
```

这些路径只描述本机隔离环境；异机须恢复相同快照、依赖与可用的Python路径，不能照抄。量化两节点是原慢批次中的reduction NLL完整评分/collect和五视图trace完整评分/collect，均为合成CPU用例、不训练，不替代完整旧批次、真实模型或科学验收。

#### （2）真实执行、失败与采样边界

| **条件与新批次** | **实际结果** | **原始证据** |
| --- | --- | --- |
| RPipe core，0.25s dump_traceback_later | 58 passed，pytest1.65s/外层3.171462s，栈27802B | .tmp/tpipe061/r1/ |
| 量化收集期即启动0.25s watchdog | 原生退出3221225477，9.851128s；未执行test call，不计通过 | .tmp/tquant061/r1/ |
| 同字节量化、关闭周期watchdog，保留fatal handler | 2 passed，pytest11.85s/外层14.070022s | .tmp/tquant061/control-no-watchdog/ |
| 同字节量化，收集后1s watchdog | 首个call中原生退出相同码，12.948206s；未完成call，不计通过 | .tmp/tquant062/r1/ |
| 同字节量化，同步线程每1s dump_traceback，独立重复1 | 2 passed，pytest13.74s/外层17.633124s，11次栈采样、64994B | .tmp/tquant063/r1/ |
| 同字节量化，同步线程每1s dump_traceback，独立重复2 | 2 passed，pytest13.30s/外层16.164850s，11次栈采样、61810B | .tmp/tquant063/r2/ |

两个watchdog失败的原始栈分别涉及DLL加载/pytest断言改写和transformers动态导入/typing；没有据此认定RPipe或CPython的具体根因。同步版本不调用周期C watchdog，保留前两次失败、控制和每个脚本版本。Python采样线程在GIL长期被占用时可能延迟，1s采样也会漏掉更短等待；它不是无扰动或全覆盖分析。其实际观测仍有CPU/文件写入开销，不能按一次更快的结果挑选验收。

两次量化call墙钟/进程CPU：reduction NLL为9.146471/9.062500s和8.810846/8.843750s，trace为3.029232/2.953125s和3.219571/3.203125s。进程CPU包含采样线程、不含其他子进程；Windows计时粒度可使短差值或CPU略大于墙钟，不能强行解释为收益。call收尾观察torch线程数为1。采样进程树RSS峰值851603456/851677184B，可能重复计算共享页，不是独占物理内存。

原始栈观察到transformers/huggingface导入、业务fixture/算法、文件open/mkdir、归档读写/CRC和Linear.forward；分类是采样出现的路径，不是独占耗时百分比。两次call CPU与墙钟接近，支持这组新样本主要在执行本地导入/处理/评分，而不是分钟级空闲。冻结量化v0.3.1副本没有新dev的artifact/shared.py，不能将B-056的新共享锁当作这组调用的原因；仍不能因此排除所有旧代码内等待。RPipe控制中真实Windows句柄拒绝约0.757671s、低CPU，栈定位到已有有界原子替换等待，说明采样能看到该受控等待，并非所有等待都被当作计算。

同步重复1/2的raw stack SHA256分别fe1604172867925513c0a9ab2c2f6b502bec7e49d99376191c88d2786de6ea9c、61b6618b17c5b19b927464b380ebb7503fa957dd5d0b8c713b0327ab70a1775d；RPipe栈为b998c31344145c9703c29c4baae3650f90c059329e897c58240c2ec21d80dc55。RESOURCE.json、pytest.log、calls.jsonl、runtime.json、stacks.txt、JUnit以及.tmp/trace480/SYNC_TRACE_SUMMARY.json保留完整材料与hash。原480s未补录的当时CPU/栈/dirty测试仍缺失；新样本只排查当前路径，不倒推其历史慢因，也不把小样本称作完整性能校准。

#### （3）B-060修复与最终检查

测试先行提交f162eb6的[Unit Tests](https://github.com/DreamSoul-AI/RPipe/actions/runs/38586121972)在四环境均仅packed/worktree两项失败；Windows2 failed/496 passed/2 skipped，Ubuntu2 failed/490 passed/8 skipped，Flow未执行；[Package Check](https://github.com/DreamSoul-AI/RPipe/actions/runs/38586122003)与[Branch flow](https://github.com/DreamSoul-AI/RPipe/actions/runs/38586122146)通过，不把红色整轮算通过。先登记bugs.md和测试合同，再修helper：保留松散ref/detached文件快路径，仅已有.git且未读出HEAD时采用5s有界Git查询；Git不可用/失败/超时返回null，无.git快照不搜索父仓库，GITHUB_SHA优先级不变。

首轮本地.tmp/e060-green/为3 failed/7 passed，0.82s，是新测试替身缺少真实SubprocessError类型；补全替身、不改原断言后.tmp/e060-green2/为10 passed，0.81s。最终结合三个artifact模块、事件和Git元数据的本机联合回归.tmp/e060-all/为66 passed，2.39s，无跳过。原失败和每批独立basetemp/JUnit均保留，不混成一次成功。

代码/测试提交394a068c7fcf7fbb95a7dae8530ca122b2d10da4的[Unit Tests](https://github.com/DreamSoul-AI/RPipe/actions/runs/38588043099)、[Package Check](https://github.com/DreamSoul-AI/RPipe/actions/runs/38588043105)、[Branch flow](https://github.com/DreamSoul-AI/RPipe/actions/runs/38588043112)全部通过。Windows core各498 passed/2 skipped，Ubuntu各492 passed/8 skipped，四环境CPU Flow各74 passed/2 skipped；平台/HF skip不计通过。基线红日志SHA256 e8dbc830498dccb0bbd536d140f5738f1113ed1a08e7263b5e1041594e0a09d7，修复日志496669be96f177082f0e04f410c0ddba78abceda22ad79c9dba0831a5b66df8d，位于.tmp/trace480/。后续整理提交不再改代码/测试，最新PR head仍须单独核对必需CI，不用这个旧head替代。

合并前可执行清单更新：

1. 取PR最新head，对照工作区和origin/dev，确认所有必需检查实际success；运行.tmp/cache-lock-budget/check_draft.py，拒绝旧head/未完成/跳过冒充通过。
2. 检查真实packed/worktree红色、可选Git错误边界及66项本地JUnit；源码只含声明的runtime与测试记录修复，不改预算、RNG、重试默认和正式数据/Run。
3. 核对完整快照、实际模块路径、前后源hash与两次采样的JUnit/call/栈；两次原生失败、无watchdog控制、fixture失败与guard拒绝均保留，互不替代。
4. 原480s性能归因保持未确认；同步采样/GIL限制与不同依赖环境明示，不用这些工程诊断替代研究门。PR46继续Draft/open、无auto-merge，集成待人类另行授权，不发布、不删数据、不改计费配置。

### 46. 共享输入文件别名校验（B-059，草案）

接续用户反馈的限流失败后，先核对干净工作区、DreamSoul-AI origin 并 fetch；本轮 GitHub Issues / PR 查询均成功，没有继续出现 429。#36–#38 / #41 仍为已修复反馈，PR #46 仍 Draft、自动合并关闭；b20a034 的所有必需 CI 已通过。只读复查三个下游会话，没有新的 RPipe 反例；量化 1629 项回归已通过。原 480 秒缺少完整测试字节与调用内栈的缺口仍见 §45，不重复造根因结论或启动大批次。

转查 prepared_resource 的文件合同后复现 B-059。改前：保留名和唯一性用原字符串比较，./manifest.json 进入 builder，正式 manifest 覆盖合成输入后发布 inputs，随后 hash 校验才失败；Windows 的 MANIFEST.JSON 也如此。data / ./data 被接受为两个文件；Windows 的大小写重复、带 drive/root 锚点，以及空名/目录本身也未在 builder 前拒绝。合成复现 .tmp/a51d92a07/repro.json 保存构造是否执行、异常和 inputs 是否已发布，不触碰项目共享数据。

先记录 bugs.md / layout.md 合同，再补用例并在原实现实测：.tmp/e059-red/ 保留日志、JUnit 和独立 basetemp，**8 failed / 5 passed / 31 deselected，0.28 s**。四个 portable 新非法清单和四个 Windows 条件失败，原四个条件及有效路径兼容用例通过；预期红色复现不是质量通过。改后：入口建立 Path 清单，按本机 Path 等价拒绝重复/根 manifest，拒绝 anchor 或无文件名，沿用 .. 检查。原字符串继续作为 manifest 键；不迁移有效缓存、不改变 definition hash、锁、build、发布或任何已完成 Run。

已检索全部调用方，main_exp 和 main_probe 均为固定相对文件名；有效 ./weights 与 nested/manifest.json 的构造、完整 hash、原键名和重复复用都有回归。新增共九个边界条件（四 portable、四 Windows 专用、一个有效兼容）；POSIX 不应假装 Windows 大小写/drive 语义，四项在 Linux 明确 skip，不算通过。入口的 Path 比较不是通用 NTFS 命名或符号链接安全协议，未扩大这个合同。

修复后本机通过 tests/run.py 运行三个 artifact 模块与事件时间戳回归，.tmp/e059-green/ 的新短 basetemp / JUnit / 日志记录 **58 passed，1.50 s，无跳过**。每次执行前只读核对没有 --worker / watch_idle_batch 性能校准；保留旧红色和所有此前失败，不占用下游 GPU、不改外部线程设置或原实验预算。临时证据及 .tmp/test-results/ 不随 clone；正式链接以草案最新 head CI 和正文为准。

b20a034 的 [Unit Tests](https://github.com/DreamSoul-AI/RPipe/actions/runs/38579687816)、[Package Check](https://github.com/DreamSoul-AI/RPipe/actions/runs/38579687789)、[Branch flow](https://github.com/DreamSoul-AI/RPipe/actions/runs/38579687760) 是前一轮验收，不能替代 B-059 新提交。可执行清单沿用 §45 命令和 .tmp/cache-lock-budget/check_draft.py，补充核对 B-059 的红色 JUnit / 本地 58 项，并允许本次共享入口的明确修复差异；最终提交必须单独绑定必需 CI。初次提交 0bf77af 意外把 shared.py 从 LF 转为 CRLF，静态 diff 核对发现后恢复原 LF；语义差异仍为 4 行新增 / 3 行替换。其 Unit Tests 38581784650 与 Package Check 38581784564 在新最终提交前取消，保留记录，不计为验收通过；Branch flow 38581784662 已通过。最终提交重新绑定必需 CI，未 force push。

B-056–B-059 都只在草案，不转 ready、不启用自动合并、不合并、不清理活动分支，不关闭现有 Issue；不触碰 dev/main、发布、系统长路径策略或正式研究来源。

### 45. 原超时时间线、取消边界与本机验收

改前：480 秒记录只有整批归属与后续分批成功，草案缺正预算截止时间线、等待取消及连续获取失败后的所有权断言。改后：只读重建原 1131 个事件并与后续相同节点对齐；补三个锁回归，使用原 tests/run.py 在本机运行相关模块。生产源码、科学门、下游 vendor 和调度保持不变；PR #46 继续草案，不合并。

原记录 started_at 字段在 pytest_runtest_makereport 完成 call 时写入，不能当作真正调用起点。pytest session 开始到最后一项完成为 479.017908 s，已完成 call 的毫秒取整耗时累计 469.527 s；相邻报告时间差扣除后一个 call 耗时，最大余量约 33.835 ms、累计约 3.681223 s。末尾两项仍分别完成 3.037 s、2.452 s 的业务 trace 用例。可确认大部分时间落在已完成的调用区间，未看到一个发生在调用之间的分钟级空洞；调用内部可能包含计算、I/O、子进程或等待，这些汇总不能区分。外部监视器按 0.1 s 轮询整批 480 s 墙钟并终止进程树，实际 480.087342 s / 退出 15；不是 RPipe launch 的超时策略。

后续分批 JUnit 中，原已完成的 1131 个节点全部存在，耗时合计 318.303 s，差异分散在多个业务模块；累计差异最多的模块原 107.780 s、后续 69.829 s。JUnit 包含 setup/teardown，原 events 是 call 毫秒取整；进程、顺序、缓存和负载也未受控，不能将差值直接定义为加速或根因。更关键的是，manifest 的 Git SHA 6afad2b 不包含至少 test_reduction_nll.py、test_dot.py 两个慢用例文件；该 SHA 单独不足以还原当时含新增/未提交测试的工作树。当前源码只作路径参考，不冒充原始字节。监视器也没有记录每线程 CPU、锁 acquire/release、线程栈或创建时间序列。因此直接原因是批次预算耗尽；为何原调用更慢、是否含锁等待仍未闭环，没有可宣称复现的上游死锁。

本机只读分析脚本与 TIMEOUT_TIMELINE.json 放 .tmp/cache-lock-budget/（不随 clone），包含原 events、resource、监视器与两份 JUnit 的 hash。分析自检断言 1131 个成功节点、1491 个分批节点无重名且原节点全部可匹配；未启动原大批次、真实模型或下游训练。旧失败不改写。后续可验证路径：先恢复/冻结完整当时测试字节与环境，再按原 pytest 选择及新 basetemp 复跑；使用 pytest 自带 faulthandler_timeout 和监视器记录进程创建时间、CPU/线程与等待栈。只有获得具体共享调用的可重复等待反例后，才将其缩成公开合成输入并登记缺陷；当前三个锁用例只复现库合同，不是原 480 秒事件的复现。

新增可执行回归位于 test_shared.py：有限 0.1 s 预算用可控 monotonic 验证三次竞争、两次 0.05 s 等待和 TimeoutError cause；等待中的 KeyboardInterrupt 原对象传播，失败句柄关闭后可再获取；另一线程连续三次真实零预算获取失败不释放所有者，所有者 KeyboardInterrupt 后该线程可获取。取消在这里指同步中断，不声称支持跨线程强行退出 context 或异步任务取消。

进一步核对上游测试记录确认 B-058：hook 在阶段完成后调用 datetime.now 写 started_at，字段并非阶段起点。已先登记 bugs.md 和 testing.md 的纠偏合同，新增固定 start=2000-01-01T00:00:00.5Z、duration=1.25 s 的成功 call / 失败 setup 两格，调用真实记录 hook 并检查落盘事件。修复前 .tmp/e058-red/ 两格均 failed（0.05 s），实际记录本次生成时间而非给定起点；原失败保留。最小修复仅将事件字段换为 datetime.fromtimestamp(report.start, timezone.utc)，manifest session 时间、耗时、错误和旧日志均不改。已搜索所有读写方；该修复纠正工程记录，不解释旧调用变慢，也不修改 RPipe 运行时。草案 B-058 未合入 dev。

本地首次选择过长的新 basetemp，47 项中 3 failed / 44 passed，1.65 s；新增三项均通过。失败为缓存 manifest 的路径 291 字符，本机 LongPathsEnabled=0；原 PID/time_ns 格式在同一目录也需 278 字符。保留 .tmp/cache-lock-budget/local-e3500747ec6c4b258fab918064d64e30/ 的日志与 JUnit，没有归因为新 nonce 修复或隐藏失败。缩短到新的 .tmp/l078cf7cf/p 后，同三模块 **47 passed，1.56 s**，无跳过；JUnit 与日志保留 .tmp/l078cf7cf/，原生 events/manifest 在 .tmp/test-results/。执行前检查没有活动 --worker / watch_idle_batch 校准；其他项目 CPU 回归使用各自临时目录，未修改其进程。

可重执行的本机检查（先创建全新短 .tmp 父目录，每次替换示例路径；保留失败，不删除旧目录）：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:CUDA_VISIBLE_DEVICES = ''
New-Item -ItemType Directory .tmp/l-new-check
& .tmp/v030-20261013/venv/Scripts/python.exe tests/run.py --core -- tests/rpipe/structure/artifact/test_shared.py tests/rpipe/structure/artifact/test_atomic.py tests/rpipe/structure/artifact/test_atomic_boundaries.py tests/test_event_timing.py --basetemp .tmp/l-new-check/p --junitxml .tmp/l-new-check/junit.xml
if ($LASTEXITCODE -ne 0) { throw 'local boundary regression failed' }
git diff --check
if ($LASTEXITCODE -ne 0) { throw 'invalid diff' }
git merge-base --is-ancestor origin/dev HEAD
if ($LASTEXITCODE -ne 0) { throw 'dev advanced; reconcile before acceptance' }
```

B-058 修复后将元数据两格与三个 artifact 模块一起执行，.tmp/lafdc49a8/ 保留独立 basetemp、JUnit 与日志，**49 passed，1.49 s，无跳过**。本次未更改旧日志；对应新 events 使用真正 report.start，旧事件按上述完成后时间语义解释。

本机 Python 路径仅描述本次隔离环境，不是异机环境要求。静态检查及远端验收的可执行核对保存在 .tmp/cache-lock-budget/check_draft.py：核对实际 head、工作区、dev 祖先、生产差异、PR 草案/禁用自动合并、最新 head 各项必需 CI 和历史红色任务。最终结果写独立 checks-<head>.json；进行中 CI 不会返回通过。该脚本只读 Git/GitHub，不合并或清理分支，正式依据同时保留 PR 与 Actions 链接。

上轮 b03cf7a 的 [Unit Tests](https://github.com/DreamSoul-AI/RPipe/actions/runs/38576170751)、[Package Check](https://github.com/DreamSoul-AI/RPipe/actions/runs/38576170986)、[Branch flow](https://github.com/DreamSoul-AI/RPipe/actions/runs/38576170759) 全部通过。锁边界提交 f3eb08b 的 [Unit Tests](https://github.com/DreamSoul-AI/RPipe/actions/runs/38578735158)、[Package Check](https://github.com/DreamSoul-AI/RPipe/actions/runs/38578735164)、[Branch flow](https://github.com/DreamSoul-AI/RPipe/actions/runs/38578735162) 全部通过。加入 B-058 后的最终 CI 仍以 PR 最新 head 为准，本机 49 项不能替代四环境与安装包门。剩余项分开标记：原慢调用原因需新现场证据；长路径通用兼容方案仅 brainstorm；草案集成人类决策未执行，dev/main 未包含 B-056/B-057/B-058。

### 44. 草案 #46 的锁错误边界与合并前检查

本轮先核对工作区并 fetch origin；继续使用 codex/cache-lock-budget，不新建分支。开放 #36 / #37 / #38 / #41 及评论均为已修复条件，没有新的复现。只读复查 dayan、quantization 和 denoising speculative decoding（output_space Study 所在会话）：当前分别在比例控制、固定坐标对照和代理舍入研究阶段，没有新的上游 KeyError、锁或调度反例。旧业务 KeyError、外部 480 秒预算耗尽仍按 §40–§43 区分归属，不重复登记为新缺陷；没有向下游发消息或更换 vendor。

改前：草案已验证非法预算和 staging 身份，但未分别断言零等待的异常因果、非竞争系统错误不重试、嵌套异常的外层所有权。改后：新增三个 test_shared 边界回归；零等待忙锁保留原异常为 cause 且不 sleep，EBADF 原对象只抛一次且不包装，两者失败后均能重新取得真实锁；嵌套异常不能提前释放外层锁，外层异常退出后另一线程可立即取得。复用现有 helper，不修改生产代码、默认重试、Run 或数据格式。本机下游仍有校准，实际执行沿用独立 Actions 的四环境 core 和 CPU Flow，不启动竞争本地 pytest；本轮最终提交未完成 CI 前不计通过。

上轮确切工作提交 9c83564f6b229f3c254fb9170c6b5694e6687a21 的 [Unit Tests](https://github.com/DreamSoul-AI/RPipe/actions/runs/38573137667)、[Package Check](https://github.com/DreamSoul-AI/RPipe/actions/runs/38573137647)、[Branch flow](https://github.com/DreamSoul-AI/RPipe/actions/runs/38573137727) 均通过。这些是上轮证据，不代替新增三个用例后的最终提交验收；§43 首轮失败、跳过和已验证复现继续保留。

合并前清单（[PR #46](https://github.com/DreamSoul-AI/RPipe/pull/46) 当前只交付草案）：

1. 核对 PR 最新 head、干净工作区和 origin/dev；按最终 head 确认四个 CPU 环境、两平台 wheel/sdist 及 Branch flow 全部实际 success，不能拿旧 head 或 skipped 代替。
2. 保留 38570617049 的预算失败复现、38571738491 的 Windows/Python 3.10 并发反例及后续全部结果，检查原子写入、锁释放和 CPU Flow 的断言没有放宽。
3. 再核对生产差异仅预算入口校验和临时 nonce；默认 120 秒、有效零预算、重入、清理、Windows 有界重试、正式字节和单 Run 单写者合同均保持。
4. B-056 / B-057 在 dev 9324014 仍未修复；已发布 main v0.3.1 / 7a15c46 也没有本草案修复。下游继续固定旧源码，草案 CI 不等于部署或下游复验；原 480 秒根因仍未确认。
5. 等待人类明确的新集成授权，之前不得转 ready、自动合并、合并或删除活动分支。若以后授权集成，先按 cicd.md 核对最新 dev 和必需检查，再验证实际集成提交，更新 bugs 状态；不自动发布或关闭 Issue。

### 43. 最小启动 / 缓存成本诊断与共享锁预算边界（分步记录）

第一步已完成工程诊断：[独立 Windows CPU 任务](https://github.com/DreamSoul-AI/RPipe/actions/runs/38568650365)，诊断提交 f3885c958b6dde7f4a983f1ece3d4244e1c886f4，生产 src 与 dev 9324014985680a0d186863cd135faa53a2e7bcbf 无差异。Windows Server 2025 / Python 3.13、AMD EPYC 7763、4 个逻辑 CPU；torch 2.14.1、NumPy 2.5.4、PyYAML 6.0.3。CUDA_VISIBLE_DEVICES 为空，OMP / MKL / OpenBLAS 均 1 线程；无 GPU、外部数据、预训练权重或训练。原量化设备为另一硬件和 torch 2.11，不能直接速度校准或推断旧现场根因。

三对新进程采用控制/RPipe、RPipe/控制、控制/RPipe 顺序；“冷”只表示新 Python 进程/尚未导入的模块，不清空 OS 页缓存。每个 RPipe 进程复用原生 main / make / run-one，两个独立 Toy/Stub、module=None 的完整 Flow 分别使用新的临时 Study / version / Run ID，六个 Run 均 succeeded；没有用重复覆盖同一个 Run 冒充热启动。所有计时均包含 -X importtime 观测成本，数据只作诊断，无速度质量门。

| **部件** | **三次或逐调用范围** | **口径与观察** |
| --- | --- | --- |
| CLI 模块首次导入 | 0.1213–0.2575 s | 进程 CPU 0.1250–0.2344 s；不包含后续 make |
| 同进程 help | 2.55–4.75 ms，9 次 | 只测已经导入后的参数入口，不与包含 Flow 的整个新进程作等量速度排名 |
| 同进程首次 / 再次 make | 2.8404–3.6010 s / 0.1028–0.1089 s | 两次都是新 Study，准备量相同；首次进程 CPU 2.7500–3.0938 s。capture_environment 按合同导入 torch / torchvision；首个 importtime 样本中 torch 累计 1.8723 s，不能将嵌套累计时间相加 |
| 首次 / 再次 run-one | 48.32–68.00 ms / 42.54–43.23 ms | 准备环境已在 make 导入，完整六阶段的空模型合同执行，不是训练速度 |
| 2 KiB 原子替换，50 次 | 整批 22.79–25.32 ms | 约每次 0.456–0.506 ms，批次前后句柄均 165 |
| 无竞争共享锁，50 次 | 整批 17.30–21.24 ms | 约每次 0.346–0.425 ms，句柄均 165；不代表竞争等待成本 |
| 64 KiB prepared_resource 初建 / 复用 | 3.54–3.88 ms / 1.60–4.69 ms | 复用包含完整文件 hash 校验；4.69 ms 反例保留，不声称每次都更快 |

空 Python 控制进程 0.0954–0.1036 s；RPipe 整个诊断进程 3.8444–4.8088 s，后者包含两次 make、两次完整 Flow 和所有 I/O，二者工作量不同，不能把差值叫作纯启动开销。首次 make 后句柄从 118 增至 165，此后 help、Flow、写入、加锁和缓存复用没有继续增长；有限样本不证明所有路径无泄漏。Windows process_time 约 15.625 ms 粒度，短项显示 0 不等于没有 CPU 成本；它不含 git 子进程。整机忙率包含本探针，短项有 null / 100% 粗采样，不作为外部负载归因。

结论：本路径可定位的首轮成本主要出现在环境来源捕获触发的真实依赖导入；无竞争文件写入和锁在本样本为毫秒级。未发现分钟级卡住或持续句柄增长。原 480 秒批次最后仍在完成量化用例，缺少当时调用栈、CPU 及进程身份时间序列；本次不同设备的最小诊断不能还原外部负载，也不能宣称已查明该批次变慢原因。本机下游全程未暂停、抢占或执行竞争基准。

原始样本及选取的 importtime 行在上述 Actions 日志；完整本机下载和提取位于 .tmp/cache-lock-budget/startup-baseline.log、STARTUP_BASELINE.json（不随 clone），原始日志 SHA256 c54381bc90943ec088a05019e8ed0db4fe907aaaad62f13b65edc166af61770c。该次运行完整 importtime 树只在 runner 临时文件中，未声称已永久归档；复跑使用[固定诊断脚本](https://github.com/DreamSoul-AI/RPipe/blob/f3885c958b6dde7f4a983f1ece3d4244e1c886f4/tests/_data/startup_diagnostic.py)及同提交的[公共 recipe](https://github.com/DreamSoul-AI/RPipe/blob/f3885c958b6dde7f4a983f1ece3d4244e1c886f4/tests/_data/launch_retry_recipe.py)，放同一临时目录、按上方环境变量及隔离环境执行新输出路径；脚本拒绝覆盖已有工作区。本机副本在 .tmp/cache-lock-budget/startup_diagnostic_fixed.py。一次性脚本与临时工作流从最终草案差异移出，固定 Git 提交和原日志仍保留，不新增常规 CI 基准或诊断框架。

补充验证限制：第二步首个临时任务 [38569905344](https://github.com/DreamSoul-AI/RPipe/actions/runs/38569905344) 因 checkout 浅克隆没有基线 Git 对象，在核对步骤 bad object 后退出，测试 skipped，未作为缺陷复现或通过证据。首轮性能任务也有相同 git diff 错误，但 PowerShell 多命令步骤最终由 Python 成功退出而未阻断；不能仅凭任务绿色声称来源核对通过。已在本机独立比较 9324014:src 与 f3885c9:src 的 Git 树对象，二者均为 e1b4f6f90ba96a6e58b8eb6f71b719252605a5e5；性能样本对应的生产文件未变。后续临时任务先显式 fetch 确切基线并检查每个 Git 退出码，再安装依赖和执行测试；原失败日志保留。
第二步已完成修复前验证：[显式核对来源后的定向回归](https://github.com/DreamSoul-AI/RPipe/actions/runs/38570617049)，生产 src 仍为 9324014；31 项中 26 passed / 5 failed，1.51 s。五个失败分别为 NaN / ±inf / -1 未拒绝，以及 NaN 忙锁被三次 watchdog 截住；原始失败不计作通过。新并发替换、中断后重新准备和 Windows 真正不允许删除共享的句柄测试均通过；后者在 Unicode / 空格路径上经过 0.75 s 有界拒绝、保留旧字节与清理本次 staging，释放句柄后再次写入成功。两个写入线程只检查通用 atomic helper，不把单 Run checkpoint 宣称为支持多写入者。

已先将 B-056 的来源、期望/实际、影响和证据写入 bugs.md，并在 layout.md 明确有限非负预算合同；brainstorm 收纳为本轮选定的低风险准备阶段草案。第三步方案为文件操作前的标准库数值校验；保留原签名、默认 120 秒、零预算、重入、退出释放和有效预算的竞争行为，不改数据/model 或科学门。第三步已在工作分支实现：仅新增 math 导入和两行入口校验，共三行生产差异；所有调用方继续使用原签名。新增八个边界用例沿原测试入口执行，临时远端工作流与一次性基准脚本不在最终差异中。AST / diff 检查通过；本机下游仍在执行，因此不启动本地回归。提交以面向 dev 的草案 PR 交付，最终工作提交由四环境 CPU 和两平台安装包必需 CI 验证，未完成前不计通过；当前 dev 上 B-056 仍开放，即使草案 CI 通过也不等于已集成。

草案首轮 CI 发现第二个真实边界 B-057：[ebcdb8b 的 Unit Tests](https://github.com/DreamSoul-AI/RPipe/actions/runs/38571738491) 在 Windows/Python 3.10 的新增并发原子写用例报 WinError 2（临时源文件不存在），core 1 failed / 471 passed / 2 skipped，流程门未执行；其他三个环境以及两平台包门通过。B-056 五个新预算用例在四环境 core 中均通过；不将整轮质量写成通过。

原子 helper 仍为 dev 的 PID+time_ns 临时名，不能区分同进程同一读数的两线程；这与旧 #38 的初始写失败清理不同，已单独登记 B-057。按 [Python 时钟文档](https://docs.python.org/3.13/library/time.html#time.time_ns)，纳秒单位与实际时钟实现/精度有别，不能当写者身份。保留自然时钟反例，再给相同测试增加固定读数条件，明确检查 staging 独立性；不通过改成串行、跳过 Windows/Python 3.10 或重跑挑通过规避失败。已核对 atomic_write_text 的 config、result、provenance、readout、activity、缓存 manifest 和 checkpoint 分件调用方；改动只涉及临时 nonce，不改变正式 artifact 名、字节或单 Run 单写者合同。
B-057 工作分支采用项目已有的 UUID staging 模式：atomic_write_text 保留 PID / .tmp 后缀与原 finally、replace 和 Windows 重试，只将时钟部分换为随机 uuid4 nonce。来源依据见 [标准库 uuid4](https://docs.python.org/3.13/library/uuid.html#uuid.uuid4)。这与三行 B-056 预算校验组成当前草案的两项生产修复；自然/冻结时钟两个并发条件使本轮新增边界共九项。原 ebcdb8b 的失败和其他环境通过都保留，后续最终提交须完整重新通过必需 CI；草案保持未合并，dev 上 B-056 / B-057 均开放。
### 42. 失败终端输出回归与反馈归属边界

再次核对实际 dev 509a089 的四环境测试及两平台包检查，均为 success，已将 §41 的运行前说明替换为最终工作提交和实际集成提交的完整验收证据，包括最后完成的 Windows/Python 3.10 流程门；没有重新运行旧实验冒充新验收。

#41 原有四格真实 CLI 回归已经验证默认/关闭重试的失败/成功、调用次数、最终错误、日志、jobs 字节和空计划跳过。本轮只扩展原失败用例的终端断言：共享 console 捕获 stdout/stderr 必须包含原始 RuntimeError；关闭重试时不出现二次 FileExistsError，默认重试时该二次错误仍可见。复用现有用例，不增加独立进程或修改生产代码。实际调用链为 CLI → launch_jobs → run-one → Logger.exception，同一错误还须按既有断言保存在 result 与 run.log。

[反馈指南](feedback.md#四归属核验示例)补充三类边界：#38 原子写入清理、#41 CLI 重试入口与业务重复读取的区别、外部 pytest 批次超时的证据要求。量化 480 秒事件不是 #38；业务缓存的重复读取 KeyError 也不是 #41 的上游根因。未为未证实问题新建 B 编号或公开 Issue，未修改下游缓存/调度/监视器。

逐项复查开放 #36 / #37 / #38 / #41 与评论，均已有 dev 修复；src / tests 搜索 TODO / FIXME / XXX 无标记。brainstorm 的尝试历史等候选仍是建议，未自动纳入实施。当前没有新的可本地复现上游缺陷；本轮交付为终端可观察性回归和记录缺口修补。本机性能校准继续，远端必需 CI 负责当前提交的实际执行，未完成的检查不计通过。

### 41. 量化 480 秒超时归属核验与 #38 写入边界补测

本轮只读核对量化项目固定 v0.3.1 模板、原始资源记录、逐项 events 和后续两批 JUnit XML。原 CPU pytest 批次在 480.087342 s 被下游外部资源监视器终止，退出 15；1131 个已完成用例均为量化项目，均 passed，记录耗时合计 469.527 s。当时尚未执行到 tests/rpipe，最后两项仍在完成量化 trace 的计算/采集用例（3.037 s、2.452 s）。这是整批预算耗尽的证据，不能据此归因为 RPipe 调度卡死，也不能把 1131 项局部通过写成完整回归通过。

同源后续串行 A/B 两批分别 530 / 961 项，JUnit 均为零失败、零错误、零跳过；资源监视墙钟分别 146.476927 s / 224.341738 s，退出 0，最终存活 PID 均为空。两份 XML SHA256 分别为 3808787d5318dd8ee184280a992a6939f7efe1053436ba1b0d411cfd60f05837、b8ac30cef61a78f067390964e6445477b590585970e5e81e7f36fecb8fba91c2。直接标为 tests/rpipe 的用例时间合计仅 0.015 s；量化集成用例也调用模板，因此这个目录统计不能当成 RPipe 全部开销。原超时记录曾报告 PID 56508 瞬时存活，随后只读检查未存活；后续批次又出现同号 PID，不能只凭 PID 值推断持续泄漏或相同进程身份。

上述材料仍由下游 Study 报告和本机原始日志保管，未搬入公开仓库，也未修改下游文件。经脱敏的未确认 Issue 草稿保存在 .tmp/feedback-boundary-evidence/quantization-timeout-issue-draft.md（本机临时材料，不随 clone 提供），没有向 GitHub 发布未证实的缺陷。缺少原超时瞬间的调用栈和受控 CPU/线程时间序列，不能精确解释首轮比后续分批更慢的原因；未重跑大回归或改变原科学门与预算。

开放 #36 / #37 / #38 / #41 均已有 dev 修复且没有新评论条件。本轮选择 #38 补充真实写入边界：从 write_result 经过 JSON 编码、UTF-8 文件写入和原子清理，分别验证无旧档/有成功旧档时的真实 UnicodeEncodeError，检查旧成功字节、其他写入者临时文件保留、本次临时文件清除，以及下一次合法写入能成功。此前的回归通过注入部分写入错误验证底层 helper；新增两个集成用例不 mock 文件写入。现有公共修复已覆盖该行为，无需再次修改生产代码。

#41 的最终工作提交 8269bff、集成 8c15cd8 及 Windows/Python 3.13 证据已由 [PR #43](https://github.com/DreamSoul-AI/RPipe/pull/43) 补齐（[§40](#40-原生-launch-显式关闭失败重试41验证通过)），PR #43 实际 dev a1e5b797ca6733dd73cf458f796f85c84d3d76ac 的 [Unit Tests](https://github.com/DreamSoul-AI/RPipe/actions/runs/38560337078) / [Package Check](https://github.com/DreamSoul-AI/RPipe/actions/runs/38560337260) 均通过。未重写原始 v0.3.1 下游报告，也未将其标为新 dev 验收。

[PR #44](https://github.com/DreamSoul-AI/RPipe/pull/44) 最终工作提交 0b3ce37d286c484623791190dece9f7b28469358 的 [Unit Tests](https://github.com/DreamSoul-AI/RPipe/actions/runs/38562868705)、[Package Check](https://github.com/DreamSoul-AI/RPipe/actions/runs/38562868844) 和 [Branch flow](https://github.com/DreamSoul-AI/RPipe/actions/runs/38562868704) 全部成功，已合入 dev 509a0894c68076571f7d75cc7747707465092026；实际集成提交的 [Unit Tests](https://github.com/DreamSoul-AI/RPipe/actions/runs/38563652457) / [Package Check](https://github.com/DreamSoul-AI/RPipe/actions/runs/38563651571) 也全部成功。四个环境 CPU integration 各 74 passed / 2 skipped，Windows core 各 464 passed / 2 skipped，Ubuntu core 各 463 passed / 3 skipped；HF 可选依赖和平台专用 skip 不计通过。最后的实际 dev Windows/Python 3.10 流程门已核对为 success，74 passed / 2 skipped，耗时 137.10 s；不是仍待运行。两平台 wheel/sdist 独立安装验收成功。

旧工作提交 296aa59 的两个工作流在静态检查发现测试模块重名、改为唯一名称后取消，保留原记录，不计作通过；最终 0b3ce37 及实际 dev 509a089 的所有必需门均重新执行。核对最终工作树与集成树一致后，仅清理本任务已合并分支。该轮本机没有并行执行 pytest 或干预下游性能校准；未修改 main、版本、数据、Study 预算或下游 vendor，未关闭 Issue。

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


## 140. 2026-10-16 B-098：整数张量非零容差精度合同（修复前）

基线dev039db15。全仓三类复查在只读compare发现：零容差已有整数精确分支，非零容差仍统一float64。公开compare_runs和CLI，用两个微型CPU checkpoint验证2**60附近delta1/2：scalar遵循数学界，tensor两项错误passed=true/exit0；delta1/atol1正确通过但max_abs_diff误报0。六场景原result/tracker/checkpoint全部hash不变，无GPU/测速。原始REPRO、旧源码/测试ZIP/hash在.tmp/audit098（不随clone）。

先开Issue122，再583c16f测试提交；首轮fixture缺一个映射闭括号，pytest收集错误，原日志/XML/测试独立保留；ff6b92e纠正fixture后，18新节点全部真实执行并失败、164既有节点通过，0error/skip。三形状×绝对界外/界上、负数、int64全范围、相对界外/界上；其中9假通过、9差值错误，不宣称18个错误判定。校验公开API/CLI与JSON、输入/peer原字节；旧失败不覆盖。

设计先于修复：在现有_tree中保留exact零容差整数/bool分支，再仅对两边均为非bool整数张量将tolist交给原_tree→_close的Fraction判断。无需新helper/依赖/模块，不用整数张量减法避免signed溢出，不转换float64丢位；复用原标量算法和输出格式。bool或混合bool、浮点/复数/非有限、形状/空张量、checkpoint归属/恢复及布局不变。全仓调用均在本模块_compare_tree，API/CLI不变。只做正确性，不用下游Study或GPU。

## 141. 2026-10-16 PR121最终和实际dev验收、结束remote清理

PR121最终head09549cd的Branch38818522331/Package38818522318/Unit38818522341全部实际矩阵/聚合成功，PR_CHECKED e8b4ef1e33b4405363154258d408135006a7852bba231add6063e45dcc1e9144。普通merge为dev039db15f4df4f360a57329c3ed5ab2ad92abfda9，实际devPackage38819287687/Unit38819287560全部实际矩阵/聚合成功，DEV_CHECKED SHA256 09f769a2e8168852f9e1a9151108d543268a915a46dabec9d28e3c3ed768d3e6。核对祖先、无开放PR、提交保全及CI后删除结束remote codex/optional-state-stat，本地tip保留；本目标累计22条，远程当时只main/dev。Issue120保持open至下一版。证据.tmp/audit097不随clone，PR/Issue正文已附验收，不重复评论/派发CI。


## 142. 2026-10-16 B-098最小修复、本机正确性验收

补齐56bbfe7：uint64最大值附近两个反例修前均失败，bool容差内/外、小int32界内/外、相同大整数五控制修前通过。连同ff6b92e的18项，共25新增合同、20失败/5控制；原164 compare节点修前通过，非18个错误通过。先f351121设计，再7f3c391只重排原整数分支，新增非零容差且两边非bool整数时_tree(tolist)复用原Fraction算法；生产净增3行、无helper或依赖。原零容差torch.equal及diagnostic、空/形状、bool与混合bool、浮点/复数/非有限路径原字节保持。

readout525 passed、core1507 passed/1既有POSIX executable-mode skip、CPU Flow189 passed；0fail/error，跳过不计通过。25新合同在修前/相关绿/core均核对实际节点，API/CLI/JSON差值、signed两端不溢出、原checkpoint/peer字节保全。红测试Git LF blob可精确重建实际CRLF执行字节，修后断言不变。SOURCE_BYTES_CHECKED确认仅一块替换，其他生产字节保持；LOCAL_CHECKED SHA256 2c6927b3cffc1520b806ab1aefd28e4c85acfafb51268aa223f5f46fe1bfb047。原始ZIP、REPRO、收集错误、两个真实RED及GREEN/CORE/FLOW XML/log都保留.tmp/audit098（不随clone）。

本轮无Study运行或研究结论，不查超时、测速或GPU；无布局/schema/恢复规则改变，不声称混合整数/浮点大数精度或任意稀疏/量化tensor全覆盖。接续同一分支PR到dev，最终提交和实际dev全部实际矩阵/聚合成功前不称集成验收完成。Issue122留至下一版，不合main/tag，结束remote仅在提交保全和实际dev验收后按授权删除。


## 143. 2026-10-16 B-099：有限容差算术溢出（红合同及设计）

基线dev17460b1。公开API/CLI八场景：metrics、history、float64张量、complex128张量的±1.7e308输入均有限，rtol1.5时实际passed=true/exit0，数学期望False；rtol2为正确边界控制。差值3.4e308和界2.55e308都转inf，原scalar比较inf≤inf、tensor比较inf>inf均失真。不是B089/#104的非有限输入身份，不修改已修Infinity/NaN规则。输入/peer每文件hash均保留，原快照/REPRO在.tmp/audit099（不随clone）。

先Issue124，再decd552提交30合同：六入口/shape×相对界外/界上与atol+rtol和界外/界上24项，另复数两分量范数溢出×普通/paired-inf/paired-NaN×界外/界上6项。实际219节点：15失败、204通过（30新中15红失败/15控制，另189旧控制），0error/skip。覆盖API/CLI/输出JSON非有限差值原null、input/peer原字节；不删失败。

设计先于修复：不新建模块/helper/格式，不改容差语义。现有_close对普通有限float仍走原式，仅diff或bound非有限时用现有Fraction重算绝对+相对界；已知输入非有限的原分支先返回。复数仅为有限张量溢出格回退时，在_close用实/虚Fraction计算D²/B²，再以M=D²−A²−R²B²≤0或M²≤4A²R²B²判断，避免溢出与平方根舍入；返回的不可表达差值仍inf。tensor原nonfinite/same/diff/bound保持，只有输入均有限且中间diff/bound非有限的格逐个复用_close，普通格及pairedNaN/inf身份不变；中间NaN差值按超范围inf报告，不改JSON null规则。旧整数/零容差/bool/shape/empty/归属/严格读取不动。只做CPU小合同，无性能/GPU/Study或原结果重写。

## 144. 2026-10-16 PR123最终和实际dev验收、结束remote清理

最终head e2376234d40b5dc45be407d522031ff4e102b8d5 的Branch38850044398/Package38850044255/Unit38850044235全部实际矩阵/聚合成功，PR_CHECKED 7c613f03052760a90e4a22719d5e41849ec7e6843692d7f6b4d4485906052c8b。普通合dev17460b1b579d180dec34a956bfc80d6c34023f34，实际devPackage38850576040/Unit38850576054全部实际矩阵/聚合成功，DEV_CHECKED 3b7d04dad80c16ee1d33813b785835cd137f327c41a4f6c587d00b0260f4dd9f。核对祖先/无开放PR/提交保全后删除结束remote codex/integer-tensor-tolerance，本地保留，累计23条；当时remote仅main/dev，main00e70b1未变。Issue122保留open至下一版，PR/Issue正文已附验收，原证据.tmp/audit098不随clone，无存活验证句柄或待合事项。


## 145. 2026-10-16 B-099修复及补充复数绝对界、本机验收

设计f200686后ba5892a仅在现有_close和_tensor树里补算术溢出回退：有限scalar用已有Fraction比较绝对+相对界；有限complex以范数平方等价式决定；tensor仅输入均有限但diff/bound非有限的格复用同判断，原same/nonfinite/整数/shape/empty及普通有限式保持。生产净增20行，不新增helper/模块/依赖/schema，不改训练计算或产物。返回不可表示差值仍inf，原CLI JSON写null不变；有限输入不能因inf≤inf或inf>inf假通过。

补充fbe1285四项：有限complex两分量模长溢出时，rtol0可产生0×inf=NaN；用nextafter产生可表示微差，覆盖exact drift、绝对界外、绝对界上、相同值。冻结before.zip原比较器模块回放2失败/2控制，明确不是当前HEAD生产红；模块SHA和测试SHA已保存核对，API及CLI实际调用冻结比较器、不模拟数值/I/O。现有修复无需再改便全部通过。原30合同decd552仍逐字前缀相同，不改断言换通过。

初轮readout555/core1537+1skip/Flow189已通过；补4新合同后保留初轮XML/log，更新readout559、core1541+1同一既有POSIX executable-mode skip，0fail/error，34新节点全部执行。生产源码未变、Flow用例未变，189 CPU Flow不重复。额外两个微型连续/转置checkpoint均拒绝界外差异，仅该场景布局控制，不称任意layout全覆盖。原所有Run/result/tracker/checkpoint/peer字节保持。LOCAL_CHECKED SHA256 67db96bec8732148fccd11991fcd17f5418515ebb9392b6c54a4fea930d1c9b4；原ZIP、REPRO、两批红、两轮GREEN/CORE及FLOW、源码三块精确替换与红测试LF/CRLF证明在.tmp/audit099（不随clone）。

后续按唯一分支PR到dev，最终head与实际dev全部实际矩阵/聚合后才完成集成验收；Issue124留至下版。无测速/GPU/原始Study运行或科学结论，不合main/tag/强推/改布局；提交保全后仅清结束remote，不删本地/数据。


## 146. 2026-10-16 PR125最终与实际dev验收、结束remote清理

最终head460741347265dde2c3998382843801752c504fb5 的 Branch38851935543/Package38851935380/Unit38851935399全部实际矩阵/聚合成功；PR_CHECKED SHA256 50248ecc78fe3d9e7b3589bb4cda79e27297118e3cb81c0d15b7ab7dd17f8c2d。普通合 [PR125](https://github.com/DreamSoul-AI/RPipe/pull/125) 到dev82205cbe7fc89a46853eb61c1734d123b7d153f0，实际dev [Package38852496728](https://github.com/DreamSoul-AI/RPipe/actions/runs/38852496728) / [Unit38852496656](https://github.com/DreamSoul-AI/RPipe/actions/runs/38852496656)全部实际矩阵/聚合成功；DEV_CHECKED SHA256 9ca50aa8ee8f3a0af01480167eec251d89db7517be292d1d526ddee3c688e60f。核对祖先、无开放PR、提交保全后删除结束remote `codex/finite-tolerance-overflow`，本地tip保留；累计24条，清理时remote仅main/dev。main00e70b1/v0.4.1不变，Issue124保持open等下一版；PR/Issue原正文已附验收，不重复评论。证据 `.tmp/audit099` 不随clone，所有观察句柄已结束。

## 147. 2026-10-16 单一项目研究计划、完整Study归并与夜间结果更新

按Enmao12:53最新要求，优先交付今晚人工审阅的计划。1ac5293先用git mv原样改名为[研究计划 RPipe.md](<../research/研究计划 RPipe.md>) / [Research Plan RPipe.md](<../research/Research Plan RPipe.md>)，仍在docs/research。现行main_probe、main_exp、Study模板及两份v030-before-local历史PLAN共5份，全文进入同一计划第十一章，英文逐节完整对应；原5个Markdown用git rm移除。第十一章保留每份原路径的固定Git来源和SHA256、全部表格/命令/预算/停止条件/历史授权，明确当前授权优先；没有把历史排班转成新执行任务。

第九章补齐PR117/119/121/123/125对应Issue116/118/120/122/124的反例、红/控制/绿边界、最终与实际dev验收；另归纳夜间PR103/105/107/109/112/113/115。区分当前CPU工程验收与v0.3.0历史科学数字；当前dev完整科学矩阵未重跑。保留文献实际阅读深度与L3全文/引文追踪缺口，列人工审阅→文献→CPU正确性→另获授权后的E3八格60-step→E4八格600-step前置与32 train/32 eval完整预算。shared/logger的89/90不绑定实施，不新增文献全文阅读或GPU任务。

同步AGENTS、README、research、layout/concept/flow及Study/规范入口；历史报告只更新迁移链接，不改实测正文。修正历史快照4处原失效相对链接（中英文共8处），历史recipe入口固定在迁移前dev并说明不能代替旧Run冻结来源。5段中文移除新增锚点、还原标题层级并归一迁移链接后，与原文逐字相等；英译五节标题映射匹配。五份原规范/译本文件与迁移前Git内容保持；运行代码、配置、JSON、Run、图和失败证据零改动，_staging_from_DEM保留。

本机tests/rpipe/test_layout.py实际36 passed，独立basetemp，无fail/error/skip；git diff --check及UTF-8检查通过。链接扫描1651个可取得目标，本次变更新增失效链接0；273条旧引用/锚点或缺本机临时材料的扫描候选未算本批通过，不冒称全库链接零问题。保全映射、原字节、检查与XML在 `.tmp/research-consolidation-20261016`，不会随clone提供；正式计划内固定Git来源可核验。本批文档PR及实际dev CI仍待后续执行，不提前声称合入；无main/tag/性能/GPU/数据删除或运行时布局变更。


## 148. 2026-10-16 B-100：混合整数/浮点比较精度，先红合同

统一计划PR126已正常合dev8c9cb99f3105645457c5207ddf931521e8778224，最终head及实际devPackage38854919745/Unit38854919785全部实际矩阵/聚合成功；DEV_CHECKED SHA256 2f7439a3f61353d952c7414260cfc69445e76313a28b47921614a277f5388a14。结束remote codex/unified-research-plan核对提交保全后已删，本地保留，累计25；main00e70b1不变。完成后三种非连续张量溢出探针均正确拒绝，没有为该假设开Issue。

继续复查确认新的混合数值反例：2**60+1的整数与float(2**60)，数学差1，atol0.5应拒绝；五入口两向共10实际API均false-pass/差0。Issue127后031571a先红，120新合同实际72 failed/48 passed、0error/skip，覆盖signed正负/unsigned、标量/实数/复数tensor、正反向、零及绝对/相对界，并执行API+CLI/JSON与原输入/peer字节检查。设计见artifact_consistency_design.md文末；生产修复尚未实施，原REPRO/ZIP/hash/红结果在.tmp/audit100不随clone。既有开发/科学证据保留，不改布局/main/tag，不跑GPU或性能。


## 149. 2026-10-16 B-100统一修复与极值补充、本机验收

84c44cd设计后c5687b6只改原compare.py：混合int/float按实际表示值用既有Fraction算差和界；浮点非有限身份单独保持，纯int分支不变。有限complex精确范数判据复用实/虚Fraction差，显示norm不先抹掉整数位。tensor保留CPU原值，有限混合非bool整数/实数或复数格与原溢出格共同调用_close，写回判定与差值；纯float/complex、整数、bool、shape/empty、非有限身份与归属/严格读取保持。生产25新增/8删除，无helper/模块/依赖/格式/布局变化，精确CRLF替换与其他字节检查保留。

120项031571a先红72失败/48控制，原断言逐字保留。复核新混合分支后，f3829a9补18项：metrics/history/checkpoint标量×两向×超大整数10**400对有限float/普通整数对inf或NaN；冻结原实现回放6失败/12控制、0error/skip，明确不是当前HEAD红。现有生产修复原样通过，无追加生产变化；有效数字差异应返回CLI1而非文件错误2，超范围显示差值inf/JSON null沿既有格式，非有限身份控制不变。

最终readout697 passed、core1679 passed/1既有Windows上POSIX executable-mode skip、CPU Flow189 passed，0fail/error；138新增全部在最终green/core执行，原输入与peer字节保持。初轮readout679/core1661+1与最终结果均保留；补充只增加readout测试且生产不再变，Flow用例/源码未变不重复执行。LOCAL_CHECKED SHA256 6e16f46d8340a026d1bff0adfbad16134743ba72e6e65f40b8af2aaa91b5ab74；原REPRO/ZIP/hash、真实RED与冻结补充RED、GREEN/CORE/FLOW/XML在.tmp/audit100，不随clone。正式portable测试随Git。集成仍待本批最终head/实际dev全部实际矩阵/聚合，完成后清结束remote，Issue127保持open至下一版本；不合main/tag，不做GPU/性能或完整科学复验。

本机证据checker曾单边把工作源CRLF归一为LF，与Git内CRLF blob比较而拒绝；NEWLINE_CHECK_REJECTED保留。随后原字节核对确认工作文件与c5687b6 Git blob SHA256同为2fe4095710df56da89d59b4650149b1e163d03d7dc54b65dcbf3bba25310bb01，精确相等，未改生产文件；不记为库测试失败。


## 150. 2026-10-16 周期性清理授权更新、PR128验收与完整清单

按Enmao最新清理指示及13:52时间戳澄清：此前保留已合分支/Issue等发版的阶段要求被本轮授权取代。已合dev且提交保全的本地/远程分支清理；未合分支不删，过时项仅列PR说明再定；GPU任务使用中的分支/worktree保留，不删实验数据、结果或provenance，不force push、不动main。后续从本轮清理完成基线起，每累计约5个实际合入dev的PR全面盘点一次；逐PR结束分支仍按CI验收后及时清理。没有修改GitHub自动删除设置或增加定时弹窗。

B100 [PR128](https://github.com/DreamSoul-AI/RPipe/pull/128)最终head38e2a995b3474a82550d7734c682f4b453e80d4b的Branch38856579327/Package38856579274/Unit38856579276全部实际矩阵/聚合成功，PR_CHECKED SHA256 d8ad2e72893d640bd02833446abefdb599b740dc111d0c0100e9ea6d9b263d91。普通合dev2008753c3c2b11321a3da3811102cf5c20b646ff，实际dev [Unit38857214651](https://github.com/DreamSoul-AI/RPipe/actions/runs/38857214651) / [Package38857214660](https://github.com/DreamSoul-AI/RPipe/actions/runs/38857214660)全实际矩阵/聚合成功，DEV_CHECKED SHA256 59ce01011475ad4c94a0742fb8f987f8db91b903cd7e7154a39ac6c0528ba6ca。readout697/core1679+1平台skip/Flow189、138新增执行与原失败边界见§148–149；新原始证据.tmp/audit100不随clone。结束remote codex/mixed-numeric-precision已删除，目标累计26条历史remote删除；本节下表是这次新授权的本地清理清单，不把历史remote重复算本轮。

### （一）本轮删除的36条本地分支

逐条验证tip为当前origin/dev祖先、无工作区占用；普通git branch -d成功，不使用-D。提交均可从dev历史取得；本地main7a15c46与dev9324014未改，remote main00e70b1/v0.4.1未改。当前维护分支codex/periodic-cleanup在盘点后新建且未合，暂保留；本PR合入并核对实际dev CI后再普通删除其本地/远程分支，作为本轮收尾另记报告。

| **删除本地分支** | **已保全tip** | **关联已合PR** |
|---|---|---|
| `codex/activity-hint-fields` | `8ee1913de698` | [#88](https://github.com/DreamSoul-AI/RPipe/pull/88) |
| `codex/artifact-consistency-design` | `eb52c65b9128` | [#52](https://github.com/DreamSoul-AI/RPipe/pull/52) |
| `codex/artifact-contracts-correctness` | `1667f517d858` | [#61](https://github.com/DreamSoul-AI/RPipe/pull/61), [#63](https://github.com/DreamSoul-AI/RPipe/pull/63), [#64](https://github.com/DreamSoul-AI/RPipe/pull/64), [#65](https://github.com/DreamSoul-AI/RPipe/pull/65), [#69](https://github.com/DreamSoul-AI/RPipe/pull/69), [#70](https://github.com/DreamSoul-AI/RPipe/pull/70), [#71](https://github.com/DreamSoul-AI/RPipe/pull/71), [#72](https://github.com/DreamSoul-AI/RPipe/pull/72), [#73](https://github.com/DreamSoul-AI/RPipe/pull/73), [#74](https://github.com/DreamSoul-AI/RPipe/pull/74), [#76](https://github.com/DreamSoul-AI/RPipe/pull/76), [#78](https://github.com/DreamSoul-AI/RPipe/pull/78) |
| `codex/atomic-readonly-cleanup` | `168456adf0fb` | [#107](https://github.com/DreamSoul-AI/RPipe/pull/107) |
| `codex/cache-lock-budget` | `90b7d75323d2` | [#46](https://github.com/DreamSoul-AI/RPipe/pull/46) |
| `codex/compare-checkpoint-boundary` | `ef6084681160` | [#102](https://github.com/DreamSoul-AI/RPipe/pull/102) |
| `codex/compare-nonfinite-contract` | `39f132bfa242` | [#105](https://github.com/DreamSoul-AI/RPipe/pull/105) |
| `codex/declared-batch-contract` | `4536ec497cce` | [#83](https://github.com/DreamSoul-AI/RPipe/pull/83) |
| `codex/failure-preservation-policy` | `bbb2f41fa8af` | [#55](https://github.com/DreamSoul-AI/RPipe/pull/55) |
| `codex/finite-tolerance-overflow` | `460741347265` | [#125](https://github.com/DreamSoul-AI/RPipe/pull/125) |
| `codex/integer-tensor-tolerance` | `e2376234d40b` | [#123](https://github.com/DreamSoul-AI/RPipe/pull/123) |
| `codex/log-reference-contract` | `bf57c88fdf06` | [#96](https://github.com/DreamSoul-AI/RPipe/pull/96) |
| `codex/maintenance-closeout` | `b8648d54296f` | [#56](https://github.com/DreamSoul-AI/RPipe/pull/56) |
| `codex/mixed-numeric-precision` | `38e2a995b347` | [#128](https://github.com/DreamSoul-AI/RPipe/pull/128) |
| `codex/open-issue-correctness` | `8a3f21342ced` | [#47](https://github.com/DreamSoul-AI/RPipe/pull/47) |
| `codex/optional-json-decode` | `000b39a23381` | [#117](https://github.com/DreamSoul-AI/RPipe/pull/117) |
| `codex/optional-state-stat` | `09549cd1f1a0` | [#121](https://github.com/DreamSoul-AI/RPipe/pull/121) |
| `codex/optional-tracker-decode` | `187bb2e56297` | [#100](https://github.com/DreamSoul-AI/RPipe/pull/100) |
| `codex/provenance-declarations` | `2b7c76986950` | [#94](https://github.com/DreamSoul-AI/RPipe/pull/94) |
| `codex/readout-text-cells` | `8dc6a914a6f0` | [#98](https://github.com/DreamSoul-AI/RPipe/pull/98) |
| `codex/report-count-cells` | `07a32762ec77` | [#112](https://github.com/DreamSoul-AI/RPipe/pull/112) |
| `codex/research-plan-masters` | `f303d72328d2` | [#92](https://github.com/DreamSoul-AI/RPipe/pull/92) |
| `codex/result-status-contract` | `37d0262b58fd` | [#87](https://github.com/DreamSoul-AI/RPipe/pull/87) |
| `codex/rpipe-research-plan` | `b631d3a5d813` | [#103](https://github.com/DreamSoul-AI/RPipe/pull/103) |
| `codex/safe-readout-contracts` | `41982bfb62cc` | [#54](https://github.com/DreamSoul-AI/RPipe/pull/54) |
| `codex/saved-plan-read-boundary` | `6863f7ffe5f2` | [#84](https://github.com/DreamSoul-AI/RPipe/pull/84) |
| `codex/scheduling-query-parse` | `fdf4f8fb30a7` | [#119](https://github.com/DreamSoul-AI/RPipe/pull/119) |
| `codex/shared-cleanup-errors` | `9e2f0efbe546` | [#109](https://github.com/DreamSoul-AI/RPipe/pull/109) |
| `codex/status-metric-shape` | `cbc8c0ae13c9` | [#51](https://github.com/DreamSoul-AI/RPipe/pull/51) |
| `codex/status-parser-boundary` | `343012569af6` | [#115](https://github.com/DreamSoul-AI/RPipe/pull/115) |
| `codex/status-readout-errors` | `ed96c8ccbe38` | [#49](https://github.com/DreamSoul-AI/RPipe/pull/49) |
| `codex/test-evidence-unicode` | `335e203cbbc0` | [#113](https://github.com/DreamSoul-AI/RPipe/pull/113) |
| `codex/unified-research-plan` | `b7f527031762` | [#126](https://github.com/DreamSoul-AI/RPipe/pull/126) |
| `codex/v030-local-studies` | `406d7a000bc8` | [#32](https://github.com/DreamSoul-AI/RPipe/pull/32) |
| `codex/v040-layout-release` | `fb15d6efd20f` | [#57](https://github.com/DreamSoul-AI/RPipe/pull/57), [#58](https://github.com/DreamSoul-AI/RPipe/pull/58), [#60](https://github.com/DreamSoul-AI/RPipe/pull/60) |
| `codex/yaml-mapping-roots` | `f4acb936ce66` | [#80](https://github.com/DreamSoul-AI/RPipe/pull/80) |

本轮已删除remote清单：codex/mixed-numeric-precision（PR128验收后，tip38e2a995b347）；其本地tip同时在上表中删除。无其他待清理remote；新维护PR的远端仍为活动分支，合入验收后清理。

### （二）PR、Issue与未合候选

盘点时开放PR为0，所有已处理PR已是MERGED/CLOSED，不重复关闭；未合工作分支为0，无待定删除候选。新维护PR为当前工作，不当过时项关闭。以下26个Issue逐项核对对应PR已合dev、merge commit为当前dev祖先，及独立实际dev Unit/Package所有实际job/聚合成功证据后completed关闭；关闭评论引用对应PR和最新授权，不重复旧验收评论。

| **已关闭Issue** | **对应已合PR** | **已验收实际dev** | **完成范围** |
|---|---|---|---|
| [#77](https://github.com/DreamSoul-AI/RPipe/issues/77) | [#78](https://github.com/DreamSoul-AI/RPipe/pull/78) | `1667f517d858` | 正确性修复已合/CI已验收 |
| [#79](https://github.com/DreamSoul-AI/RPipe/issues/79) | [#80](https://github.com/DreamSoul-AI/RPipe/pull/80) | `f4acb936ce66` | 正确性修复已合/CI已验收 |
| [#81](https://github.com/DreamSoul-AI/RPipe/issues/81) | [#83](https://github.com/DreamSoul-AI/RPipe/pull/83) | `4536ec497cce` | 正确性修复已合/CI已验收 |
| [#82](https://github.com/DreamSoul-AI/RPipe/issues/82) | [#84](https://github.com/DreamSoul-AI/RPipe/pull/84) | `6863f7ffe5f2` | 正确性修复已合/CI已验收 |
| [#85](https://github.com/DreamSoul-AI/RPipe/issues/85) | [#87](https://github.com/DreamSoul-AI/RPipe/pull/87) | `37d0262b58fd` | 正确性修复已合/CI已验收 |
| [#86](https://github.com/DreamSoul-AI/RPipe/issues/86) | [#88](https://github.com/DreamSoul-AI/RPipe/pull/88) | `8ee1913de698` | 正确性修复已合/CI已验收 |
| [#89](https://github.com/DreamSoul-AI/RPipe/issues/89) | [#92](https://github.com/DreamSoul-AI/RPipe/pull/92) | `f303d72328d2` | 初步调研文档交付；实施另授权 |
| [#90](https://github.com/DreamSoul-AI/RPipe/issues/90) | [#92](https://github.com/DreamSoul-AI/RPipe/pull/92) | `f303d72328d2` | 初步调研文档交付；实施另授权 |
| [#91](https://github.com/DreamSoul-AI/RPipe/issues/91) | [#94](https://github.com/DreamSoul-AI/RPipe/pull/94) | `2b7c76986950` | 正确性修复已合/CI已验收 |
| [#93](https://github.com/DreamSoul-AI/RPipe/issues/93) | [#94](https://github.com/DreamSoul-AI/RPipe/pull/94) | `2b7c76986950` | 正确性修复已合/CI已验收 |
| [#95](https://github.com/DreamSoul-AI/RPipe/issues/95) | [#96](https://github.com/DreamSoul-AI/RPipe/pull/96) | `bf57c88fdf06` | 正确性修复已合/CI已验收 |
| [#97](https://github.com/DreamSoul-AI/RPipe/issues/97) | [#98](https://github.com/DreamSoul-AI/RPipe/pull/98) | `8dc6a914a6f0` | 正确性修复已合/CI已验收 |
| [#99](https://github.com/DreamSoul-AI/RPipe/issues/99) | [#100](https://github.com/DreamSoul-AI/RPipe/pull/100) | `187bb2e56297` | 正确性修复已合/CI已验收 |
| [#101](https://github.com/DreamSoul-AI/RPipe/issues/101) | [#102](https://github.com/DreamSoul-AI/RPipe/pull/102) | `ef6084681160` | 正确性修复已合/CI已验收 |
| [#104](https://github.com/DreamSoul-AI/RPipe/issues/104) | [#105](https://github.com/DreamSoul-AI/RPipe/pull/105) | `83615d727122` | 正确性修复已合/CI已验收 |
| [#106](https://github.com/DreamSoul-AI/RPipe/issues/106) | [#107](https://github.com/DreamSoul-AI/RPipe/pull/107) | `3a72bd8cef7e` | 正确性修复已合/CI已验收 |
| [#108](https://github.com/DreamSoul-AI/RPipe/issues/108) | [#109](https://github.com/DreamSoul-AI/RPipe/pull/109) | `fccccc4c28f9` | 正确性修复已合/CI已验收 |
| [#110](https://github.com/DreamSoul-AI/RPipe/issues/110) | [#112](https://github.com/DreamSoul-AI/RPipe/pull/112) | `4974869527bf` | 正确性修复已合/CI已验收 |
| [#111](https://github.com/DreamSoul-AI/RPipe/issues/111) | [#113](https://github.com/DreamSoul-AI/RPipe/pull/113) | `bf24574c3b30` | 正确性修复已合/CI已验收 |
| [#114](https://github.com/DreamSoul-AI/RPipe/issues/114) | [#115](https://github.com/DreamSoul-AI/RPipe/pull/115) | `b28f6a219ce9` | 正确性修复已合/CI已验收 |
| [#116](https://github.com/DreamSoul-AI/RPipe/issues/116) | [#117](https://github.com/DreamSoul-AI/RPipe/pull/117) | `20027afb231b` | 正确性修复已合/CI已验收 |
| [#118](https://github.com/DreamSoul-AI/RPipe/issues/118) | [#119](https://github.com/DreamSoul-AI/RPipe/pull/119) | `4d12719d7eba` | 正确性修复已合/CI已验收 |
| [#120](https://github.com/DreamSoul-AI/RPipe/issues/120) | [#121](https://github.com/DreamSoul-AI/RPipe/pull/121) | `039db15f4df4` | 正确性修复已合/CI已验收 |
| [#122](https://github.com/DreamSoul-AI/RPipe/issues/122) | [#123](https://github.com/DreamSoul-AI/RPipe/pull/123) | `17460b1b579d` | 正确性修复已合/CI已验收 |
| [#124](https://github.com/DreamSoul-AI/RPipe/issues/124) | [#125](https://github.com/DreamSoul-AI/RPipe/pull/125) | `82205cbe7fc8` | 正确性修复已合/CI已验收 |
| [#127](https://github.com/DreamSoul-AI/RPipe/issues/127) | [#128](https://github.com/DreamSoul-AI/RPipe/pull/128) | `2008753c3c2b` | 正确性修复已合/CI已验收 |

89/90只关闭PR92已交付的初步调研任务，不表示批准shared迁移、logger替换、外部服务或新的实验；文档保留缺口、条件性建议和后续准入合同。24项正确性修复仍仅在dev，不冒充main发版；本次没有tag/release。

### （三）worktree、GPU与证据

Git worktree登记仅本物理主工作区；本会话无托管worktree附件，prune --dry-run无废弃条目，所以删除worktree0个，主工作区保留。只读核对进程元数据时检测到GPU Python任务，其cwd/PYTHONPATH为仓库外独立副本；进程、分支/工作区和数据均未触碰。没有执行GPU计算、测速、训练或终止任务。

本轮快照、全部tip/祖先判断、对应CI证据路径与SHA、删除日志、Issue关闭结果、GPU保护核对在.tmp/cleanup-20261016，不随clone；正式清单与公共PR/Issue链接在本节。原Bug索引完整保留于清理前Git commit2008753及各历史记录。AutoResearch额外本机技能正本未找到，本轮按用户明确给出的周期性清理规则执行，不声称读取了其他规范。后续按AGENTS/cicd中累计约5 PR规则继续；运行代码/测试/配置及机器证据格式不改，本机文档验收：5份Study协议全文保全、5份规范/模板原文未改，1647个既存路径链接扫描，变更文档新增坏链接0（273个既有历史/本机证据候选未冒充全仓链接全绿）；布局36 passed/0fail/error/skip，git diff --check通过。本维护PR最终head与实际dev CI仍分别验收。本维护PR合入并清自身分支后作为下一次约5个dev PR清理的累计基线。


## 151. 2026-10-16 周期清理最终验收与自身分支清理

PR129已普通合dev `76f7b3b7ce2bbb6d288c0fc254fdff6c8bcb68e6`。最终head `8b9d0e78126b336fdb76be1836d3261408893d18` 的Branch38859169439/Package38859169741/Unit38859169540与实际dev [Package38859816227](https://github.com/DreamSoul-AI/RPipe/actions/runs/38859816227)/[Unit38859816255](https://github.com/DreamSoul-AI/RPipe/actions/runs/38859816255)全部实际矩阵/聚合成功。DEV_CHECKED SHA256 b9235e907082b859f9de7eef546ac59d2f88e3e642fa0e581637e6d9912fca54。

已在验收后普通删除本维护分支codex/periodic-cleanup本地/远端，tip8b9d0e7保存在dev；加§150前36条本地与remote mixed-numeric-precision，本轮最终合计本地37条、远程2条、废弃worktree0。完整前36条清单与26个Issue/PR在§150；第37条即periodic-cleanup。最新活动codex/runner-log-failure红回归8a351df保留，当前remote仅main/dev；main00e70b1、local main/dev、主工作区、仓库外独立GPU任务和数据结果均未触碰。当前新Issue130不在已处理关闭清单。PR129是下次约5个dev PR全面清理的基线；每批验收后清自身分支规则继续。

正式报告追加于[PR129](https://github.com/DreamSoul-AI/RPipe/pull/129)原正文一次；完整本机清单、祖先/CI/删除核验和报告在.tmp/cleanup-20261016，不随clone；不是新的main发布。

## 152. 2026-10-16 B101 / Issue130 Runner日志失败留证先红与设计

既有flow.md §3要求失败尽力写result并传播原错。PR129清理验收等待期间在隔离Run复现：有效旧success+run.log目录，Logger构造PermissionError发生于run try前，旧success原字节不变。初始合成success缺control/paths被校验拒绝，已保留并更正；此拒绝不是库缺陷。进一步核对调用方prepare/execute/System Logger与现有Runner保全合同，诊断_ensure_logger同样在try外可阻断留证并覆盖原错。

先开Issue130，再在唯一未合分支codex/runner-log-failure提交8a351df正式红：6项启动错误×新旧result，4项collect/process诊断再失败×新旧result，10 failed/0error/skip。每项锁定原异常/peer/旧成功策略；真正目录对象保留。早期草稿6项因冗余mkdir停在夹具，原DRAFT_RED/GREEN和拒绝说明保留，不计正式回归；更正草稿和隔离候选只供设计试验。正式RED.xml/log、原源码ZIP/hash、复现及拒绝在.tmp/audit101，不随clone。

已明确方案后再改生产：原Logger构造与start info移入run既有try，current先None；_log_failure的_ensure_logger出错则返回，外层failed写回和原raise继续。无新helper/依赖/模块，不改变任何文件架构、产物布局、schema、Logger追加协议、最终成功日志时机或初始冻结/process保全条件。正式生产仍待实施；验收覆盖Runner旧42项决策表、12项二次写失败、execute/Logger及CPU Flow，最终head与实际dev分别全部实际CI后合并清理；不GPU/测速/科学重跑。


## 153. 2026-10-16 B101本机修复验收完成，待最终与实际dev CI

9e345c8仅改runner.py：原Logger构造/start info移入既有run try，诊断_ensure_logger失败返回，让failed写回与原异常传播继续。生产6 insertions/3 deletions，原CRLF、_preserve_success、日志追加、最终成功输出、API/schema/layout与所有原红断言保持。无其他生产者、模块、依赖、GPU/性能或科学门改动。

正式RED10 failed/0error/skip；GREEN123 passed，其中10新节点全部执行；core1679 passed+1既有Windows不支持POSIX executable mode bits的skip；原生CPU Flow199 passed，均0fail/error。相邻包含原42项决策表、12项结果写二次失败与Logger/execute/Study调用方；结果可写才保证尽力failed，结果位置本身坏时仍保留原错，不承诺一定落盘。原日志目录/peer与process旧success原字节锁定。LOCAL_CHECKED SHA256 `4f55ec3870e6698ec0bb96b3443a5ebd40e6874262d3663271722b598d0a8633`，XML/log/冻结原源码/初稿夹具拒绝与隔离候选试验证据均在.tmp/audit101，不随clone。

附加核验：Git保存红测试为LF、工作区为CRLF，直接原字节checker的拒绝留存NEWLINE_CHECK_REJECTED.json；双边换行规范化一致且相对8a351df的Git diff为空，未修改红断言，不当成库失败。五份Study协议和五份规范/模板原文保全；链接扫描1649个既存路径，273个旧历史/本机候选保留，本批新增坏链接另核。

上述为本机工程合同验收，不替代最终head与实际dev全部实际矩阵/聚合CI；后续PR正常合dev并验收后关闭Issue130、清自身远端/本地分支。PR129周期基线后本批仅算1个新合入PR，不做无依据额外清理。


## 154. 2026-10-16 B101 / PR131最终与实际dev验收完成

PR131普通合dev `a458eae998e25a908250ed3da887bdd8cd5bf451`；最终head `25601f53b7aed997483e024f3b1783ae2f13cbae` 的Branch38861153918/Package38861153890/Unit38861154021与实际dev [Package38861768629](https://github.com/DreamSoul-AI/RPipe/actions/runs/38861768629)/[Unit38861768569](https://github.com/DreamSoul-AI/RPipe/actions/runs/38861768569)全部实际矩阵/聚合成功。DEV_CHECKED SHA256 f9820e9e02e5d97b8d63a564e833615ffb770cf334512249e3098f0ab70d1b4a。

本机相邻123/core1679+1既有POSIX skip/CPU Flow199通过，10新节点全部执行，原_preserve_success/failed写入等函数源片段保持。Issue130已在验收后关闭，codex/runner-log-failure本地与remote普通删除，tip25601f5保存在dev；本次1+1、累计结束remote28。main00e70b1、local main/dev、GPU任务、数据结果和_staging_from_DEM保留，无tag/main/强推。PR129周期基线后已合1个PR，未到约5批全面盘点触发点。

[PR131](https://github.com/DreamSoul-AI/RPipe/pull/131)原正文已追加实际验收，Issue只关闭反馈一次不重复；正式源码/测试/本机记录在Git，原始证据与报告.tmp/audit101不随clone。此后独立复查确认Issue132，不表示所有正确性工作已结束。

## 155. 2026-10-16 B102 / Issue132原生训练finalizer先红与设计

在已验收a458eae上复查第一类收尾错误：_run_supervised的finally直接flush_state。CPU非stub合成对象首次取批抛原RuntimeError，二次状态落盘PermissionError变成传播异常与result.error，原错仅在traceback chain。iterator调用1次、未yield批次、未执行优化步；algo finally与execute finally各尝试flush。最初stub探针未进入迭代器，已明确拒绝并保留，不作为生产反例。

先Issue132，再1e8c5d1提交22个正式节点：12个unit（RuntimeError/KeyboardInterrupt×编码/发布×Logger可用/拒绝/缺省）、8个Runner原错写回（两原错×两二错×新旧成功）、2个成功体结束flush仍报错控制。20 failed/2 passed、0error/skip，全部执行。初轮2个控制因默认周期checkpoint过早注入，未证明成功体完成；RED_INITIAL/CONTROL_FIXTURE_REJECTED保留，更正checkpoint_period=0仅终点注入后获得本次正式红，不修改生产判定。

设计已明确后再改代码：仅现有finally分成except BaseException/else；前者尽力flush，二次普通异常尽力Logger.exception，日志也失败不遮原错，再原raise；后者直接flush，错误继续传播。BaseException用于保持原finally对终止的收尾尝试与原终止身份，不变成成功或吞新终止。无需helper/模块/依赖/新状态，训练循环、数学、周期内flush、checkpoint/RNG、tracker schema与产物布局不改。

已读TrainAlgorithm原生/legacy调用方与execute/eval_hook/Trainer flush；只有原生既有finally有本轮已确认反例。原始探针/拒绝、正式RED与源码ZIP/hash在.tmp/audit102，不随clone；portable测试已入库。下一步最小生产修复、训练/恢复/checkpoint/Runner相邻和core/CPU Flow，最终head/actualdev全部实际CI分别验收，正常PR合dev后关闭132并普通清本地/remote；不跑GPU、测速、下载或正式Study。


## 156. 2026-10-16 B102本机修复与清理断言补强验收

384bf2c仅在train/__init__.py替换一处原finally，11 insertions/1 deletion。已有训练/终止错误时尽力flush，二次编码/发布错误通过已有Logger尽力留traceback，Logger不可用不盖原错，最后原raise；成功体else仍直接flush并报错。原源码ZIP按CRLF原字节精确单替换验证，除此以外训练体、数值、周期flush、checkpoint/RNG、summary代码字节保持；无helper/依赖/新状态/schema/layout。

正式红1e8c5d1：22项20 failed/2成功路径控制passed、0error/skip；NEW_GREEN22 passed。core1693 passed+1既有Windows不支持POSIX executable mode bits的skip；CPU Flow207 passed，均0fail/error。训练、精确恢复、checkpoint/tracker和Runner冻结/旧成功保全均由本次完整core/Flow覆盖。KeyboardInterrupt为单元内显式捕获的合成终止，不是实际任务取消，也不变成成功；成功路径两控制执行单步CPU合成训练来验证最终写错误仍报错，不是正式Study。

复查测试后a758de4仅增3条tracker.state_path.parent实际staging目录断言，原root/peer及所有红断言保留。冻结原源码函数回放22项20 failed/2控制passed，明确不是HEAD红；当前补强22 passed。完整core/Flow运行于同一384bf2c生产源码、补强前断言集，补强后只重跑受影响22节点；生产未再改，最终CI将执行完整最终断言集。原始RED_INITIAL两控制注入拒绝、无效stub探针、更正正式红、冻结回放和全部原始日志/XML/hash/ZIP均保留.tmp/audit102，不随clone。

LOCAL_CHECKED SHA256 `fd535488c7f66d6b29dff899681bd7916cabd74978c44bbeb8a55d244662144c`；实际模块物理路径和源文件hash已核对。结果可写才尽力生成failed，tracker本身拒绝写入时保留旧state与peer、本写者staging清干净；secondary诊断不能写时仅保证原错误优先，不承诺日志总能落盘。未放宽严格恢复或科学门。待最终head及实际dev全部实际矩阵/聚合分别验收后正常合dev、关闭132并普通清本地/remote；PR129全面清理基线后本批合入将累计2个PR，仍逐批清自身结束分支。


## 157. PR133 实际 dev 验收与已合分支清理（2026-10-16）

PR133 正常合入 dev bc3743e6ef33cc1b87833122a4d070dca2ee6ad3。最终 head ea1d06c 的 Branch38863702787 / Package38863702737 / Unit38863702706，以及实际 dev Package38864260990 / Unit38864260983 全部实际矩阵和聚合成功；Issue132 已验收关闭。DEV_CHECKED SHA256：524566748d28e698049518047387b3c10abc12fb53d5a42a6e4ba469bbcdf5b6。

观察句柄69911因 GitHub unexpected EOF 退出，短超时重读原运行验收成功；不是 CI 失败，未重跑/取消 CI。15:44 最新指令覆盖旧 heartbeat 暂时保留规则，核对提交已纳入 dev 后，普通删除 codex/train-failure-finalizer 本地1条、远程1条；远端仅 main/dev，累计结束远程分支29条。PR129周期清理基线后已合2批。未动 main、GPU 任务、数据、Run、结果或 provenance。原始核验 .tmp/audit102/BRANCH_DELETED_CHECKED.json 不随 clone。此前 pending/保留措辞仅记录当时状态。

## 158. B103 / Issue134：共享锁路径别名重入（2026-10-16）

原生 Windows 两个目录路径 samefile 为真，但普通路径与扩展命名空间路径的 resolve 字符串不同；两个嵌套方向在 timeout=0 均自争用。先 Issue134，42c8b5b 提交四项红回归全部失败，再 afd8ea0 设计，41ce552 只将线程局部重入键改为 pid/st_dev/st_ino 物理目录身份。根路径、锁文件、OS 所有权、超时验证和磁盘布局保持；四项覆盖嵌套方向、内层正常/异常退出、其他线程排斥与释放后重获。

GREEN_FINAL：72 passed；core：1697 passed、1既有Windows POSIX模式位 skip、0 failure/error，新四节点全部执行。CPU Flow 和最终 head/实际 dev CI 尚待验收。证据位于 .tmp/audit103，不随 clone；正式测试随 Git 提供。

首轮 CRLF 替换校验在实际 LF 源文件上拒绝，未写生产；当时 GREEN 仍为旧源码四失败，后按实际 LF 唯一替换再验。自动审批拒绝了可能因切片丢失历史的文档脚本，未执行；改用追加记录和精确替换保全全文，原拒绝说明保留。不把校验拒绝或网络失败算库失败。无 GPU、测速、正式 Study、main/tag 或迁移。


B103 本机补记：CPU Flow 207 passed、0 failure/error/skip；最终 head 与实际 dev CI 仍待验收。


## 159. B103 / PR135 实际 dev 验收及清理（2026-10-16）

PR135 正常合 dev e80724caa06082c5fe313d59c4cf4e54dbf1559e。最终head 1720b0d 的 Branch38871098681 / Package38871098565 / Unit38871098385，以及实际dev Package38871748863 / Unit38871749092 全部实际矩阵与聚合成功。Issue134 验收关闭，核对祖先及干净工作区后普通删除 codex/cache-lock-path-identity 本地1条、remote1条；远端仅main/dev，累计结束remote30条，PR129周期基线后已合3批。无数据/结果/provenance或main改动。

DEV_CHECKED SHA256：d85c429564dcdaf012ace6e8fcf63a6488224270eb45864b64970cc00d0dbdba；最终head证据SHA256：7ab4e4ceecc684088819041594d83688350d2ff4dba23eff87fb7cb93260dd2a。临时原始证据与分支清单在.tmp/audit103，不随clone。范围检查器误以为总共8文件，实际含单独设计文档为9文件；已按实际职责核对1生产/1测试/7文档，拒绝保留，不当库失败。文档审批拒绝后仅追加与精确基线替换，全文历史保留。

## 160. B104 / Issue136：迁移多根锁顺序（2026-10-16）

PR135相邻复查确认：同Study普通/扩展namespace表示，shared.root绝对路径表示及根名相对字典序组合可翻转物理加锁顺序。原生空目录两线程各持前两根再请求第三根，注入零预算立即暴露两侧TimeoutError循环；退出四根可重获。未运行真实Study、下载、测速或等待120秒。

先Issue136，再d3083d2四项正式回归2 failed/2有效controls passed，再3a9267a设计，788a9b4仅shared.py提取物理身份私有函数供重入和排序复用，并先mkdir原本就会创建的四根再按身份加锁。完整shared76 passed，新四节点全部执行；额外实际并发的两个别名迁移均成功，限定1秒等待防挂起而不记录耗时，全部空合成目录。CPU core/Flow与最终/实际dev CI待分别验收。

不改目录、配置/schema、冲突验证、硬链接/发布路径、checkpoint诊断staging、科学门或数值。原始源码、红/绿/并发证据在.tmp/audit104，不随clone；正式测试与合同记录随Git提供。此前B102/B103 pending文字仅历史。


B104本机验收补记：shared76/core1701/CPU Flow207 passed，core另1既有平台skip；新四节点全执行。最终head/实际dev CI待验收。


## 161. B104 / PR137 实际 dev 验收与清理（2026-10-16）

PR137 正常合dev 576292503d1a128fae61410466e87b2f9233f4fd。最终head aa59894的Branch38872977734 / Package38872977777 / Unit38872977687，以及实际dev Package38873780706 / Unit38873780707全部实际矩阵和聚合成功；Issue136验收关闭，codex/shared-migration-lock-order本地/remote各1条普通删除。DEV_CHECKED SHA256：9ff9fd44134fd3740bd367203febf434084d481769bbe5280b0ef87bfe9c6a14。原始CI/清理证据在.tmp/audit104，不随clone；PR137正文保留验收依据。此前pending仅历史。PR129清理基线后已合PR131/133/135/137四批，累计结束remote31条。

## 162. B105 / Issue138：非零退出留证先于诊断输出（2026-10-16）

5762925上的合成Run/Popen替身复现：wait写合法success后返回1，error行stdout抛BrokenPipeError，原先print在失效写回之前，留下run_succeeded=true。先Issue138，e599e4b提交三类stdout错误×retry开关及两个正常控制，正式6 failed/2 passed；3aa02e7设计后02f84e4只移动一行print到既有纠正写回之后。stdout原异常仍传播，不扩大无结果/既有failed、spawn、retry/codes或train/eval屏障语义。peer与config原字节保留。完整调度器88通过，CPU core/Flow及最终/实际dev CI待分别验收。

相邻announce/report/compare已先写再输出，blocked依赖已先failed再输出，Runner诊断二错已有B101/B102合同；本批只改确认的一个错误边界。初次fetch TLS握手失败、Issue查询EOF均留原日志，后短查成功，无重复Issue或CI。目标工具已由paused恢复active，以实际工具状态为准。临时原始源码ZIP、红绿和复现在.tmp/audit105，不随clone；正式测试随Git提供。无GPU/测速/正式Study/main/tag/产物布局修改。

本批合入并实际CI成功后达到PR129后约5批，须完成周期盘点并重置清理基线：当前本地main/dev与唯一未合codex/launch-failure-evidence，远端main/dev；仅一个登记工作目录，无废弃worktree。main/dev与未合分支保留，不触碰独立GPU副本、数据/结果/provenance。工作分支将在两侧CI通过、祖先/tip/干净工作区核对后普通删除，届时实际清单和新基线补入PR验收正文与本机报告。


B105本机验收：调度88/core1709/CPU Flow207 passed，core另1既有平台skip；八新节点全执行。最终head与实际dev CI仍待验收。


## 163. B105 / PR139 实际dev验收与五批周期清理（2026-10-16）

PR139正常合dev90078c81421fd39e746212c16d84bf2b3c9a44de。最终head5af40d1的Branch38876118376 / Package38876118190 / Unit38876118273，以及实际dev Package38876987073 / Unit38876986912全部实际矩阵和聚合成功；Issue138验收关闭。DEV_CHECKED SHA256：e62d34c82d21014c3d76806f09f9976ac7ec4ab200f04872b5bd23dc0b40b689。

PR129后五批PR131/133/135/137/139均已验收，对应runner-log-failure、train-failure-finalizer、cache-lock-path-identity、shared-migration-lock-order、launch-failure-evidence工作分支的本地/remote均逐批普通删除（均有codex/前缀）；本轮实际删除最后1条的本地/remote各1。周期盘点结束仅main/dev、1主登记工作目录、0废弃worktree/额外积压已合分支、0开放PR；新Issue140未修而保留，不关闭。新基线PR139/dev90078c8，后续合入累计0。main、独立GPU副本、数据/结果/Run/provenance及_staging_from_DEM不动。原始清单与核验.tmp/audit105/PERIODIC_CHECKED.json不随clone，PR139正文提供可携带验收记录。此前pending仅历史。

## 164. B106 / Issue140：PNG原子发布与figure所有权（2026-10-16）

90078c8上真实Agg合成输入复现：savefig部分写失败破坏有效旧PNG且遗留1个本次figure，原OSError与peer figure保留。先Issue140，5eef63d提交19项C2回归，正式17 failed/2有效控制passed，0error/skip；首轮18项原红保留。39d9c15设计后fba35b8仅curves.py加入自有figure finally和同目录xb排他staging，经既有atomic_replace发布；保留mode、默认umask、原错误、peer/输入字节与既有路径。不可恢复自有临时清理OS错误留诊断、不遮原错；不是多文件事务或断电保证，不改checkpoint诊断staging/共享硬链接。

19新节点全执行/通过，core1728 passed/1既有Windows POSIX模式位skip，0failure/error；CPU Flow和最终head/实际dev CI待分别验收。冻结旧函数与新函数对含step/epoch/observation及变n标注的同一合成输入，PNG95233字节完全一致，SHA256均2c5c48b4a46b29c3273df0ea601483bd289cdcd811afc23803f2c26eabaff0d2；绘制主体除新增try缩进外逐行原样。此为本环境工程验证，不是跨环境图片金标或新科学实验。调用方仅Study process；既有207项CPU Flow将验证其实际职责。临时原始PNG/ZIP/红绿和哈希在.tmp/audit106，不随clone；正式测试与设计已入库。

本批将渲染/构造/部分保存/取消/替换/创建失败、新旧目标、碰撞、清理拒绝和真实Windows只读目标统一验收；测试在既有flow/process镜像目录新增publication文件声明C2，避免改动原C1用例或叠加两个cost marker。生产没有新模块/依赖、文件夹或schema；坐标/数据/标签/dpi140与最终PNG路径保持，未跑GPU、测速、正式Study，未动main/tag。


B106本机验收：19新节点/core1728/CPU Flow207 passed，core另1既有平台skip；含真实Windows只读清理的19新节点全执行。最终head/实际dev CI待验收。


### B106同批流关闭补充（2026-10-16）

原head9da02b4最终CI已全绿，但相邻故障注入发现临时流close的OSError可替换savefig首错/取消（原错仅在context）。仍属Issue140/PR141，不另开分支/Issue：a095ef0先补6项红回归4 failed/2有效控制passed，e9cae39先补设计，66f6f16显式close保留首错；正常保存后的关闭错误仍严格传播、不发布。BaseException仅为释放取消资源并bare raise，不吞新的终止信号或非IO关闭程序错误。

当前25新节点全绿，补充后core1734 passed/1既有平台skip，最终CPU Flow及新最终head/实际dev CI待验收。原19红/绿、原core/Flow与旧head CI均保留，不冒充补充后的验证。冻结旧函数/当前66f6f16对同一组合成曲线仍95233字节同SHA2562c5c48b4a46b29c3273df0ea601483bd289cdcd811afc23803f2c26eabaff0d2；绘图主体原行仅缩进。补充原始探针CLOSE_REPRO、CLOSE_RED与GREEN_FINAL等在.tmp/audit106，不随clone。


B106含流关闭补充的最终本机验收：25新节点/core1734/CPU Flow207 passed，core另1既有平台skip；25新节点含原生只读清理全执行，PNG字节仍一致。更新后最终head/实际dev CI仍须验收；旧9da02b4绿仅历史。


## 165. B106 / PR141 最终与实际dev验收、分支清理（2026-10-16）

更新head c9e344b的Branch38881441414 / Package38881441317 / Unit38881441470，以及正常合并实际dev c42a78b5835c5fda460b7ad4d0c49e8a1884293e的Package38882276100 / Unit38882275925，全部实际矩阵与聚合成功。旧9da02b4绿仅历史，不代更新head。DEV_CHECKED SHA256：40c8fb6c77d3ab5e9327d1fc34c86155a6035513d3c948ef6669a37ad13d386c。Issue140验收关闭，codex/curve-publication-ownership本地/remote各1条普通删除；源码/25正式节点保全于dev，数据、Run、provenance、main和_staging_from_DEM保持。清理基线仍PR139，已验收累计1；新Issue142保留。原CI/清理清单在.tmp/audit106不随clone，PR141正文提供可携带验收记录。

## 166. B107 / Issue142：共享锁清理IO不替换准备首错（2026-10-16）

真实OS锁加上unlock/close IO注入的六组合探针：四个准备错/取消被OSError替换，两正常体清理控制严格失败；恢复原语后其他线程全部零预算重获。先Issue142，再7614d31正式25项19 failed/6有效控制passed、0error/skip，再4b34097设计，691c240仅shared.py cache_lock复用stdlib ExitStack两个局部退出回调。获取后登记文件close，锁取得后登记release；先release后close，仅已有exc_type时清理OSError不替换首错，没有首错仍严格，非IO程序错误继续暴露，不返回True吞原错。held键/移除、PID/物理身份、预算/重试、OS锁文件和多根排序保持；原获取/预算代码块逐行原样。

完整共享101项通过，25新节点全执行，包含获取错误原cause、嵌套重入、descriptor关闭、不同线程真实重获、prepared_resource原builder错误与诊断staging保留、不可变输入字节及程序错误控制。core/CPU Flow及最终head/实际dev CI待分别验收。临时REPRO/ZIP/red/green在.tmp/audit107不随clone，正式测试/设计随Git提供；不加模块、依赖、资源框架或磁盘schema，不改checkpoint诊断保留，无GPU/测速/正式Study/main/tag。

B107最终本机验收：RED25项19 failed/6有效控制passed，GREEN101、core1759 passed/1既有Windows平台skip、CPU Flow207 passed；0error，全部25新节点执行。LOCAL_CHECKED及原始XML/log在.tmp/audit107（不随clone），最终head/实际dev全部实际CI仍待分别验收。


## 167. B108 / Issue144：四个文本与ledger写入口同批修复（2026-10-16）

前序PR143最终head d9574cf的Branch38884484744/Package38884484539/Unit38884484516全部实际矩阵和聚合成功，正常合dev e589862dbd7348f5425f5c64d56bafa4b883cb3c；实际dev Package38885286451/Unit38885286323待验，未提前关闭Issue142或删该主题分支。工程验收不是发布，不动main/tag。

B108探针在atomic_write_text、Logger._emit、AlgorithmTracker.begin_run/append_jsonl四入口真实部分写入后注入ENOSPC或KeyboardInterrupt，关闭真实descriptor后再注入EIO：12组合中8首错/取消被替换，4正常close控制严格失败，旧target/peer仍保留。先开Issue144，再dcd4e2d提交30正式回归16 failed/14有效控制passed、0error/skip；383d68c设计；f9f5cf6把两个既有Path.write_text高层故障注入调整到实际stream.write，修前99项仍16 failed/82 passed/1既有平台skip，原红与相邻红均留证。

2c33a65仅既有_atomic.py/logger.py/tracker.py共四写入块显式open/close；已有BaseException时尽力关闭，只有close OSError不遮首错，bare raise保留身份/取消；正常写完close严格传播，非IO关闭程序错继续暴露。无新helper/module/依赖或万能I/O层，既有格式/追加/flush/原子发布/模式/临时所有权/诊断staging不变。30新回归全绿，相邻98 passed/1既有Windows POSIX模式skip；core1789 passed/1同一既有skip，CPU Flow与最终head/实际dev CI尚待验收。

冻结旧源码与当前对UTF8默认/LF/CRLF文本、带旧残片的ledger起始标记和指标追加、冻结时间戳日志逐字节相同；tracker state除实际root path外相同。logger/tracker三原写入主体AST相同。原始ZIP/hash/REPRO/RED/ADJACENT_RED/GREEN/CORE和PARITY_CHECKED在.tmp/audit108（不会随clone）；portable测试与设计已入库。本轮无GPU/测速/正式Study/数据下载、main/tag或产物布局变化。


## 168. B107 / PR143 实际dev验收与B108本机收尾（2026-10-16）

PR143实际dev e589862dbd7348f5425f5c64d56bafa4b883cb3c的Package38885286451 / Unit38885286323全部实际矩阵/聚合成功；DEV_CHECKED SHA256 a6240a64cd8d4786653845e9ab28a0421e047fb0047d39078cd9f1d07092db88。最终head d9574cf此前Branch38884484744/Package38884484539/Unit38884484516全绿，PR143正文已追加两侧验收。Issue142验收关闭；核对tip/祖先/唯一worktree后codex/shared-lock-cleanup本地与remote各1条普通删除，当前未合codex/writer-close-errors保留。清理基线仍PR139，实际验收累计2，未达到下一次约5批清理门；无数据、结果、provenance或worktree删除。

B108最终本机验收：30新节点全部执行/通过；相邻98 passed/1既有Windows POSIX模式skip，core1789 passed/1同一既有skip，CPU Flow207 passed，全部0failure/error。LOCAL_CHECKED记录XML/hash、全部新节点执行和原主体AST/有效字节对照。Issue144仍开放，后续新PR最终head与实际dev全部实际CI须分别验收；不把本机通过当远端或发布验收。原始证据在.tmp/audit108不随clone；源码/正式回归/设计和本记录可携带。


## 169. B109 / Issue146：Study comparison PNG原子发布与figure所有权（2026-10-16）

前序PR145最终head f81bd8f的Branch38886373086 / Package38886373046 / Unit38886373084全部实际矩阵及聚合成功，正常合dev fc2b9b6095910aba9c93629c43c6a58b73f19ca9；实际dev Package38887059572 / Unit38887059522待验。Issue144与codex/writer-close-errors分支暂保留，未以head绿代替实际dev验收。

跟踪源码清单显示savefig仅库PNG与main_exp Study comparison两点；库已PR141验收，后者仍直接写最终historical_comparison.png。AST抽取plot_report并用合成report/reference的4探针：render/save/cancel三者各泄漏1个自有figure；部分save/取消将2398字节旧有效PNG变成11字节；正常保存76906字节无泄漏，peer figure全部保留。未调用Study main、科学gate、历史归档/Run或checkpoint，不运行实验。

先Issue146，再6ab3b74正式25项23 failed/2有效正常保存控制passed，0error/skip；70cad05设计，再1db97b1仅改Study helper。与库PNG已批准合同相同：figsize14×5/dpi160不变，先取得figure即进入finally，再同形状axes；同目录UUID.tmp以xb排他取得，向句柄明确PNG保存并关闭后复用atomic_replace；失败保留旧target，只清自有普通tmp/figure，碰撞peer不动。已有保存/取消时close OSError不遮首错，正常close仍严格；旧mode/新umask保持，只读自有临时可恢复mode重试清理。没有新helper、模块、依赖、路径/schema或返回值变化，科学阈值/比较/读取/gate各函数原源码保持。

25新节点及库相邻PNG共67 passed、0fail/error/skip，原生Windows只读控制实际执行；core/CPU Flow与最终head/实际dev全部实际CI待验。冻结旧/新helper对同组合成输入PNG76906字节完全一致，SHA256均68002aeea7067ac878a00fde5d2fdb7c66c23867e93875733f4ff61ef0989141；绘图主体原行仅缩进，其他函数原源码相同。此为同环境工程保全，不是跨环境金标/新科学结果；未来受影响Study正式复跑按新version/实际provenance和原定门限，旧结果不覆写。临时ZIP/hash/REPRO/RED/GREEN/PNG_PARITY和源码清单在.tmp/audit109不随clone；正式测试/设计可携带。


## 170. B108 / PR145 实际dev验收及B109本机收尾（2026-10-16）

PR145实际dev fc2b9b6095910aba9c93629c43c6a58b73f19ca9的Package38887059572 / Unit38887059522全部实际矩阵与聚合成功；DEV_CHECKED SHA256 7dc81633fdb9a8e0a9548afd44e75b3a5f7ab72f1a4b0e75cb5e33b0ee207c93。最终head f81bd8f此前Branch38886373086/Package38886373046/Unit38886373084全绿，两侧验收已追加PR145正文。Issue144验收关闭，核对tip/祖先/唯一worktree后codex/writer-close-errors本地/remote各1条普通删除；当前未合codex/study-plot-publication与Issue146保留。清理基线仍PR139，已验收累计3，未达到下一次约5批门；无数据/结果/provenance/worktree删除。

B109最终本机：25新节点全部执行/通过（含原生Windows只读目标），PNG相邻67 passed，core1814 passed/1既有Windows POSIX模式skip，CPU Flow207 passed；全部0failure/error。LOCAL_CHECKED保留原XML/hash、新节点执行、原绘图/其他科学函数源码保持、同环境PNG76906字节同SHA证据。最终head/实际dev全部实际CI仍须分别验收，不以本机/图像对照代替科学或发布验收。原始.tmp/audit109不随clone；正式回归/设计/本记录随Git提供。


## 171. B110 / Issue148：深层provenance解析拒绝保全旧成功（2026-10-16）

前序PR147最终head87a8e91的Branch38887954486 / Package38887954478 / Unit38887954518全部实际矩阵和聚合成功，正常合dev9a7af374ee6b1354c1c9aea37858059c03c8f329；实际devPackage38888713101仍待验，不能提前关闭Issue146或清其主题分支。

check_frozen只归一化OSError/ValueError。真实深层provenance JSON触发RecursionError时，outer gate与study phases关闭时的native prepare gate都未归为冻结校验拒绝，实际Runner在零execute/data/model调用时覆盖旧succeeded为failed；四个新旧结果探针均确认，坏输入字节不变。首轮2000层在本机Python3.13被接受，仅触发正常changed-provenance拒绝，属于无效反例且脚本/log/目录保留；更正12000层用真实JSON解析、不调递归限额，不能把首轮当库失败。原始证据.tmp/audit110不随clone。

先Issue148，再dadb558正式12项5 failed/7有效控制passed、0error/skip；32f7a54设计，e99691d仅check_frozen既有except元组加RecursionError。仍使用原ProvenanceChangedError from exc，不改strict load_provenance/json/schema/解析限额、freeze真假判据、Runner既有初始/prepare保全决策表或真实phase失败规则，不吞任意RuntimeError。调用方Runner/prepare/launch/Study process继续复用公共门；旧字节只一处替换、其余原样保留，无模块/框架/依赖或路径变更。

12新节点全执行/通过，provenance/freeze相邻204 passed、0fail/error/skip；覆盖outer/native prepare、新旧结果、坏JSON既有控制、freeze关闭不读取、strict loader仍真实RecursionError与程序RuntimeError原身份。core/CPU Flow及最终head/实际dev全部实际CI待验。新回归不执行data/model/algorithm；后续全套既有CPU Toy检查不是正式Study复跑，无GPU/测速、正式Study、main/tag、数据/Run/provenance删除或布局变化。


## 172. B109 / PR147 实际dev验收及B110本机收尾（2026-10-16）

PR147实际dev9a7af374ee6b1354c1c9aea37858059c03c8f329的Package38888713101 / Unit38888713125全部实际矩阵/聚合成功；DEV_CHECKED SHA256 adaa67489558cc5dcb0901043f6e6879fa5ffa1156772eaeea08173791f722d8。最终head87a8e91此前Branch38887954486/Package38887954478/Unit38887954518全绿，两侧验收已追加PR147正文。Issue146验收关闭，核对tip/祖先/唯一worktree后codex/study-plot-publication本地/remote各1条普通删除；未合codex/freeze-parser-refusal与Issue148保留。清理基线仍PR139，已验收累计4；下一次验收达到约5批时做完整本地/远端分支、worktree与已处理PR/Issue盘点。无数据/结果/provenance/worktree删除。

B110最终本机验收：12新节点全部执行，core实际选择其中4项unit、CPU Flow实际选择8项integration；新12及相邻204通过，core1818 passed/1既有Windows POSIX模式skip，CPU Flow215 passed，全部0failure/error。LOCAL_CHECKED保存XML/hash、新节点执行与源码原字节仅一处异常元组替换证明。strict loader/JSON解析限额/schema、freeze真假、真实phase失败和所有调用方原样；原始无效探针/更正证据均保留，不挑选失败记录。最终head与实际dev全部实际CI仍待分别验收，不以本机通过当发布或科学验收。原始.tmp/audit110不随clone，正式测试/设计/记录随Git提供。


## 173. PR149 双侧验收与五批周期清理（2026-10-16）

PR149最终head0c37aada60aab6dc9cb253488766ff012b310650的Branch38889723895 / Package38889723786 / Unit38889723670全部实际矩阵/聚合成功，正常合devafa3a639d85450c24a37244ed92b1182a14228f3；实际devPackage38890448126 / Unit38890448304亦全部实际成功。DEV_CHECKED SHA256 d25336051605d27b04360c924406a7d4e465e5e5731ee3b167dc15823119e111。Issue148验收关闭，PR149正文已追加两侧事实；freeze-parser-refusal本地/remote各1条普通删除。

以PR139/dev90078c8为旧基线，已验收PR141/143/145/147/149共5批，对应Issue140/142/144/146/148均closed。五个已保全主题分支本地和remote均已普通清理：codex/curve-publication-ownership、codex/shared-lock-cleanup、codex/writer-close-errors、codex/study-plot-publication、codex/freeze-parser-refusal。前四的逐次删除证据保留，最后一条本次核对tip/祖先/worktree后删除；没有为了周期盘点重复删除。

完整盘点：本地仅main/dev与活动未合codex/optional-read-boundaries，remote仅main/dev，Git登记唯一primary worktree，无额外/废弃已登记worktree可清，未删除worktree。新分支未合且只在本地累积，保留；PR open清单为0，Issue150/151待修保留。长期分支本地main7a15c46/dev9324014及远端main00e70b1原样，未force push、未动main/tag或数据/Run/provenance/_staging_from_DEM。新的周期基线为PR149/devafa3a639，累计归零；PERIODIC_CHECKED及原清单在.tmp/audit110，不随clone。本轮清单/结论随下个完整里程碑提交，不单开纯文档PR。

## 174. 可选读取与strict readout集中盘点及测试先行（2026-10-16）

承接§81/83既有全仓写入/展示清单，本次以devafa3a639为基线集中核对artifact/Flow process读取与Study对比读取入口；原20项清单后由§176补齐遗漏的第21项Run级回读。表中“源审/既有回归”表示既有合同与已有测试定位，未把未重新执行的故障矩阵宣称为本轮实测；本轮确认反例只有#150/#151两类，留在同一里程碑。原始rg/源码和合成探针保留.tmp/audit111，不依赖正式Study数据。

| **读取点/消费者** | **既有合同与本轮结论** | **证据/后续** |
| --- | --- | --- |
| activity.read_activity / CLI status提示 | 可选；stat/read/Unicode/JSON/Recursion与最低字段 guard已存在，坏提示不改变权威状态 | 源审；test_activity与test_activity_status既有回归 |
| readout.status._read_optional / result/config | 3态；已有OSError/Artifact/JSON/Unicode/Recursion guard，mode冲突及metric-cell边界保持 | 源审；test_status/test_status_planned_modes/test_readout_io |
| readout.logs.event_lines / optional log | 与status共享可选读guard与log_path归属，不写Run或跨目录读 | 源审；test_readout_io与logs既有测试 |
| readout.status.log_path / report._log_links | regular-file/Run归属/Windows命名空间检查；链接URI保持 | 源审；status/report既有归属回归 |
| process._state_history / legacy tracker | 可选；存在性与读取在try，IO/JSON/Unicode/Recursion降级为空，RuntimeError不吞 | 源审；test_aggregate既有状态读取矩阵 |
| process.load_curve_series / JSONL主轨迹 | 可选；逐行坏JSON/Recursion跳过、IO停止读取已得轨迹，start/offset规则不变 | 源审；test_aggregate既有轨迹回归 |
| process._load_result / indexed sibling result | 待修#150；stat在try外，domain/Recursion解析错误阻断peer；应沿已有不可用Run规则，不改变n/n_planned/complete算法 | 6实际process探针，44主红中的unit/process节点 |
| individual.run / Run自身result回读 | §176补项#153；原stat/domain/Recursion逃逸经Runner回退改写原证据；已本地补齐原可选guard，内存优先/数值/Runner策略保持 | 6实际Runner探针、14正式红4失败/10控制；补齐后全绿，见§177 |
| scheduler._run_result / run_succeeded | 可选完成提示已有OSError/ValueError/CorruptArtifactError/Recursion guard；依赖新鲜度stat失败仍false | 源审；scheduler既有回归，不执行launch |
| scheduler.load_launch_plan / jobs缓存 | 可选缓存已有stat/read/JSON/Type/Overflow/Recursion guard及batch核验；None重make合同保持 | 源审；scheduler缓存回归，不触发make/下载 |
| result.load_result / config.load_config | strict边界；非mapping、坏语法/编码应拒绝；不因#150改公共loader | 本轮strict result新控制；既有config/result回归 |
| index.load_index / plan schema及Run目录 | strict计划、形状/目录校验保持；helper允许真实解析错误向消费者传播 | 源审；不改index/schema，CLI分类归#151 |
| report.render_numbers / 当前index-process绑定 | strict；身份/计划不符拒绝，合法partial可写，原子NUMBERS保持 | 源审；test_report既有回归；#151 CLI深层拒绝 |
| compare._load_json / _history / _checkpoint | strict；object、history、checkpoint归属/容差与原数字比较保持 | 源审；test_compare既有回归；#151仅CLI分类 |
| CLI status/logs/report/compare | 待修#151；四语法错误返回2，四深层错误逃逸Recursion；保持strict拒绝、不改输入或旧派生 | 8真实CLI探针、44主红中的20CLI节点 |
| CLI _print_launch_counts | 待修#151同批；语法失败不输出，深层错误却逃逸；应维持可选计数best-effort | 3真实helper探针与4补红节点 |
| provenance.load_provenance / check_frozen | strict读取与冻结拒绝分层；PR149已双侧CI接受，Recursion保留cause，旧成功保全决策不变 | B110正式12节点，未扩宽任意RuntimeError |
| shared.shared_paths / prepared_resource manifest | strict声明与不可变缓存身份/摘要；不把坏manifest当默认可复用输入 | 源审；shared既有回归；不执行builder/下载 |
| System.load_checkpoint / split diagnostics | strict恢复及committed bundle优先；.writing/.incomplete为保全诊断，不套普通tmp清理 | 源审；既有checkpoint恢复合同保持 |
| main_exp curves.read_json/read_tracker/checkpoint_summary/log audits | strict科学证据读取与阈值审计，不能套可选展示降级 | 仅源审，不执行Study门或读取旧Run；PNG保全已PR147验收 |
| historical archive/source exports / source manifest | 原字节、固定Git对象与恢复合同，非optional展示数据 | 仅源审；不加载历史代码或改归档/正式产物 |

同主题B111/#150与B112/#151已先开Issue，8062f71先提交44正式回归12 failed/32有效控制passed；c79b17b补4计数回归1 failed/3有效控制passed，均0error/skip，两套原红分别保留，不冒充单次48红。新18:53频率规则已同步cicd：本地主题节点照常留存，两个根因、清单、验证与文档在一个分支积累到完整里程碑后再统一PR，不立即推单行文档/拆PR。生产尚未修改，按已实施读取合同先记设计再最小修复；正常聚合公式、指标转换、计数与科学门不动。


## 175. B111/#150 与 B112/#151 集中里程碑本机验收（2026-10-16）

§174保留设计时快照，其中“待修/生产尚未修改”表示当时状态；现已按bfc851b设计实施00f3ddf：仅aggregate.py原读取try纳入is_file并补CorruptArtifactError/RecursionError，cli.py五个原异常元组补RecursionError，生产共9行增加/8行删除。可选peer结果不可用时仍沿原缺失策略，n/n_planned/complete和所有聚合公式不变；四strict CLI仍诊断并返回2，可选launch计数仍best-effort。strict公共loader、checkpoint恢复/诊断、科学门与任意RuntimeError传播保持。

正式红分两次保留：8062f71的44项12 failed/32控制passed；c79b17b的4项1 failed/3控制passed，均0error/skip。修复后48新节点全部执行/通过；core1834 passed、1项既有Windows POSIX模式skip、0failure/error，CPU Flow247 passed、0failure/error/skip。新增节点实际分配core16、Flow32，无新节点跳过。SOURCE_CHECKED验证其他aggregate函数原源码相同，CLI除五个异常元组外AST相同；未改变数值算法或正常输出路径。LOCAL_CHECKED保留两套红与三套绿的XML/hash及节点清单。

§174共20项读取边界已逐项标注源审/既有合同与本轮实测；本批两个已确认缺陷均已本地修复，其余未发现新可复现反例，未把源审当成全部故障矩阵实测。原始探针、无效/失败证据、源码ZIP/hash和验证输出在.tmp/audit111，不随clone；正式回归与本记录随Git提供。无正式Study/GPU/测速、布局/schema、数据/Run/provenance改写；本机通过不代替最终head与实际dev全部实际CI。

按18:53规则，两个Issue、完整清单、周期清理和验证文档在唯一codex/optional-read-boundaries分支形成一个完整里程碑，统一推送/PR，不拆纯文档PR。Issue150/151保持open到双侧CI验收；PR149和五批清理已完成，当前清理基线149累计0，见§173。


## 176. B113 / Issue153：补齐遗漏的 Run 级可选结果消费者（2026-10-16）

§174的20项源审遗漏individual.run，先前“未确认其他反例”只表当时范围；本次补为第21项。实际FlowRunner(process-only)、ctx.state无结果时六探针：nonobject/deep/stat Permission三类使原result字节被Runner失败回退改写且没有派生；syntax/read Permission/valid三控制正常，valid accuracy42。Issue153先创建，再14正式回归4 failed/10控制passed、0error/skip（额外覆盖stat IO、missing/Unicode、程序RuntimeError身份、有效内存优先/无需index）。不是训练或正式Study，原始REPRO/ZIP/hash保留.tmp/audit113，不随clone。

设计按既有可选读取合同补individual原try范围和两种已知错误，Runner、metrics公式与公共严格loader保持，纳入已开PR152同一里程碑。原48绿及core1834/Flow247只代表5400249旧head；补齐后将重新验必要本机范围和最终head/实际dev全部实际CI。旧CI保留，不以取消/重跑抹去历史；不增加新PR。


## 177. 三个读取消费者统一验收，仍为一个 PR152（2026-10-16）

B113/#153按70bb75e先红14项4 failed/10有效控制passed→ac7edb2设计→22a4d57最小修复，14项全绿。只在individual.py增加已有domain错误导入，把原is_file纳入原try并补两种已知错误；精确原字节替换证明后续metrics/history/发布源码相同，Runner源码哈希保持。正常内存mapping优先、不需index、正常磁盘accuracy42与history75、source原字节、缺失不造result、程序RuntimeError身份均有实际回归。

补齐后重新执行core1834 passed/1既有Windows POSIX模式skip，CPU Flow261 passed，均0failure/error。三套正式红分别44项12失败、4项1失败、14项4失败，未合并冒充单次红；62新节点全部在新基线实际执行（core16、Flow46），无新skip。此前5400249的48绿/core1834/Flow247与远端CI是旧head证据，不能替代当前最终验收。临时原始探针/被覆盖后结果、原ZIP/hash、红/绿XML和LOCAL_CHECKED在.tmp/audit111与audit113不随clone。

PR152继续同一分支收敛#150/#151/#153，生产总计3个现有文件14行增加/11行删除；完整读取点清单补为21项。未改schema/数值/文件架构或产物布局，不改strict公共loader、checkpoint恢复与科学门。没有另开小PR；一次推送完整补齐里程碑后分别验最终head和实际dev全部实际CI，再关闭三Issue、清自身已合本地/remote分支。周期清理仍基线149累计0；尚未把未合PR152算入。


## 178. PR152 双侧 CI 验收与已合主题清理（2026-10-16）

最终head c91691b266a21cba6c6e2fc59c0c97000a99cd0a的Branch38893739893 / Package38893739576 / Unit38893739634所有实际矩阵与聚合成功，正常合dev f956c635b626bdb2838d3589f0b6a091904925e6。实际dev Package38894477739 / Unit38894477644亦所有实际矩阵与聚合成功；DEV_CHECKED SHA256 a0f0c34bbf9f755b76bfd260adfd2900e3497981f3cb9ea749e5cdae5782765c。三Issue150/151/153已验收关闭并分别引用PR及双侧CI，PR正文追加同一验收，不代表main发布或科学运行授权。

按立即清理指示，在最终head全绿、merge保全和tip/祖先/worktree核对后，codex/optional-read-boundaries本地/remote各1条已普通删除；删除当时实际dev CI尚在运行，现已完成全绿验收。唯一primary worktree先切detached实际dev，随后新建未合本地codex/execute-final-evidence处理新#154；没有删除worktree、main/dev、GPU任务或数据/结果/provenance/_staging_from_DEM。周期基线仍PR149，现累计1个已验收PR152，未到下一次约5批；累计结束remote38。完整清理与CI证据.tmp/audit113不随clone。

原正文精简曾被自动审批拒绝（担心移除完整说明），未执行；保留原正文与追加最终范围，后来只追加验收，不删除远端元数据。该拒绝证据保留，不影响代码/CI事实。验收文档随下一完整里程碑，不单独开纯文档PR。


## 179. B114 / Issue154：通用 execute 最终 tracker 写入假成功（2026-10-16）

在已合dev f956c63复查失败写入合同，实际Runner与真实tracker六合成探针确认2类false-success：算法体正常，最终tracker发布Permission或真实编码TypeError被finally吞掉，结果仍succeeded但tracker旧字节保留；normal与三个primary RuntimeError控制正确。仅合成Algorithm入口调用，无data/model/optimizer/训练/测速。原始REPRO/log、全部合成结果、源码ZIP/hash在.tmp/audit114保留，不随clone。

先Issue154，再aae4b0e正式红24节点6 failed/18有效控制passed、0error/skip。12 unit覆盖成功、原RuntimeError及KeyboardInterrupt/SystemExit与正常/IO/编码finalizer的身份和单次flush；12 integration覆盖实际Runner新旧结果、正常/主错及两类真实写入拒绝，保全旧tracker/peer并拒绝提前成功。按既有execute结束flush要求及原生train收尾合同，设计仅给现有finally区分成功严格与失败尽力；方案见artifact_consistency_design，不改变Runner、科学门/数值/格式或布局。本地主题commit积累，完整里程碑后统一PR，不另开文档小PR。


## 180. B114 最小修复与失败写入边界本机验收（2026-10-16）

按d850ba7设计，82890ce仅execute/run.py增加局部completed标志，并在原finally中区分正常完成后的严格flush与已有异常/终止后的原best-effort flush；8行增加/3行删除。原算法体、ctx返回值/observations、异常日志/failure_logged与Runner策略原源码保持；每条路径仍一次最终flush，不引入模块/依赖/参数/新格式。原生train的严格成功收尾、周期flush、checkpoint/RNG和所有数值算法不改。

24新节点全通过/执行；完整core1846 passed、1既有Windows POSIX模式skip、0failure/error，CPU Flow273 passed、0failure/error/skip；新节点12 core/12 Flow，未跳过。原红24项6失败/18控制保留。LOCAL_CHECKED保存原红/绿/XML/hash及实际节点，SOURCE_CHECKED逐原字节验证仅预期局部替换；六实证、旧tracker/peer/staging和原ZIP均在.tmp/audit114，不随clone。新增测试只调用合成Algorithm入口、真实tracker/Runner，不执行data/model/optimizer；全套既有CPU Toy验证不是正式Study复跑或测速。

复查本批相邻失败写入点，区分已有合同/源审和实际回归：

| **写入/收尾点** | **当前合同/结论** | **验证边界** |
| --- | --- | --- |
| Flow execute最终flush | 本批修复成功体不得吞最终写入拒绝，已有Exception/终止仍保首错，单次flush | 24新回归与六实际探针 |
| native train最终flush | PR133已实施成功严格/已有异常尽力；本批原样 | 源审及本轮core/Flow既有回归 |
| eval_hook周期/末尾flush | 既有直接调用，不吞写入异常 | 源审，未新跑科学评测 |
| Runner failed-result/log fallback | 原错误继续传播，失败落盘尽力；process/冻结旧成功保全保持 | 源码不变，本轮Flow含既有回归 |
| 普通atomic text/Logger/tracker stream close | 首写错不被close IO替换；成功close严格；原子所有权保持 | 源审及本轮core既有回归，不扩大JSONL事务 |
| 库/Study PNG发布 | 仅清自有临时流/图，失败保旧目标/peer，成功流close严格 | 源审及本轮既有合成PNG回归，未执行Study实验 |
| shared cache锁释放及builder | 保首错和诊断building，不让清理产生假成功 | 源审及本轮shared既有回归，不运行builder/下载 |
| checkpoint committed/diagnostic bundle | 严格恢复，writing/incomplete保全不套普通tmp清理 | 源审与已有CPU合同，未运行新恢复实验 |

这些结论不声称全部外部插件或科学矩阵已经实测；当前仅B114有新的已确认反例，统一修在共享Flow边界。连同PR152验收/清理记录在同一主题完整里程碑推送，不另拆文档PR；最终head与实际dev全部实际CI仍待验收。


## 181. B115 / Issue156：两处历史源码归档竞态与写入首错（2026-10-16）

从原始binary写入点交叉盘点确认main_exp.export_source和main_probe.MainArchive.__init__同类exists→write_bytes；shared raw准备由cache_lock保护，未把其固定part按普通临时清理改写。两导出使用真实源码AST、单个合成Git tree/blob、真实竞争线程，八探针两竞态覆盖peer后success；六正常/相同复用/不同拒绝控制正确。MainArchive.ROOTS为空且bootstrap/cwd隔离，无历史源码/Study/下载/data/model/optimizer执行，正式归档未访问。

先Issue156，再0383bd1正式红28项12 failed/16有效控制passed、0error/skip。覆盖创建竞态、相同竞争者完成后重试、四种原写错/终止与close IO、无二次close错控制、成功close失败严格及旧manifest/自有部分诊断bytes。首轮夹具tree变量覆盖为AST Module的28失败无效；INVALID_FIXTURE_RED及原因保留，修正source_tree后才提交正式红，生产未先改。临时探针/全部合成文件/ZIP/hash与红在.tmp/audit115不随clone。

设计仅两处else用xb取得创建所有权，显式沿既有write/close首错合同；失去创建权拒绝、不碰peer或旧manifest，自己失败字节保留。原固定Git ref、序列化/hash/路径、导出内容与数值/科学门均不改。PR155最终head34653e2已全实际CI成功并正常合dev720a13df514efaf8d765c24613c019e664af0b4e，实际dev CI仍运行；#154和已合旧主题待双侧验收清理。当前唯一未合codex/archive-exclusive-creation本地积累，不拆两导出PR，不推单行文档。


## 182. PR155 双侧验收与旧主题清理（2026-10-16）

最终head34653e2705ec056fd946a7eaf6e14fc5c744e910的Branch38895906175 / Package38895906115 / Unit38895906081所有实际矩阵与聚合成功；正常合dev720a13df514efaf8d765c24613c019e664af0b4e。实际dev Package38896655882 / Unit38896656144亦全部实际矩阵/聚合成功，DEV_CHECKED SHA256 d8c8137a19fab24759512fbb145b9fe498437ceb81edad0b4f72e8e66d047757。Issue154已验收关闭并引用PR/两侧CI，PR155正文只追加验收、不删除旧说明，不表示main发布或批准科学运行。

两侧CI成功后核对tip/祖先/唯一worktree及无未保全改动，codex/execute-final-evidence本地/remote各1条普通删除；未合codex/archive-exclusive-creation和Issue156保留。主工作区、main/dev、远端main、_staging_from_DEM原样，worktree/数据/结果/provenance删除0。周期基线仍149，累计2个已验收PR152/155，未达约5批完整盘点；累计结束remote39。原始核验.tmp/audit114不随clone，验收/清理随本完整里程碑合入，不单开文档PR。


## 183. B115 原子所有权/首错集中验收及新准入规则（2026-10-16）

按862db93设计，832c1c9只替换main_exp.export_source和main_probe.MainArchive.__init__各一条write_bytes为xb排他open及既有首错优先的write/close模式，共22行增加/2行删除。抢占失败原生FileExistsError拒绝且不触碰peer/旧manifest；相同winner完成后下一次调用沿原校验复用。自有部分写错/终止保留诊断字节和原错，成功close失败仍拒绝；不宣称多文件归档整体事务或所有竞争者必定成功。

28新节点全部实际执行/通过；完整core1874 passed、1既有Windows POSIX模式skip，CPU Flow273 passed，均0failure/error。新28均在core执行，无新skip；正式红28项12失败/16控制保留，初次无效夹具另存不计库失败。SOURCE_CHECKED精确原字节证明每文件仅替换原write_bytes行，固定Git refs、其他源码/科学门、manifest路径/格式/内容哈希与有效blob字节保持；测试使用合成Git数据和空ROOTS，不加载历史源码或访问正式归档。原ZIP/hash/探针/所有失败与XML在.tmp/audit115，不随clone，正式测试/设计/记录可携带。

所有原始binary写入点复核：这两个不可变归档点本批修复；main_exp raw准备的download/MNIST/CIFAR固定part在现有cache_lock和完整hash验证下，未执行/改写；shared link_input使用内核排他链接/自有staging；checkpoint committed/diagnostic bundle保持严格恢复及诊断保留。除新28外，其余为源审/既有合同和本轮已有回归定位，不声称重新运行原始数据准备或科学门。最终head/实际dev全部实际CI仍待统一里程碑验收。

19:53监督授权的新GPU准入规则随本批记录：桌面/图形/远控按白名单加进程类型判定，Graphics/G或C+G且无/极少CUDA计算占用的Figma、浏览器、dwm、explorer、Office、RayLink/RayLinkCapturer、ToDesk、Sunlogin/向日葵、Parsec、TeamViewer归显示类，不能仅因“未分类身份”阻塞；仅真实CUDA计算占用、实际空闲显存与主机RAM实测峰值+2GB参与准入。未知进程只有真实CUDA计算占用才阻塞并记录告知监督，不杀/挂起任何进程。

静态核对：入库capacity.py只查询GPU/free，不以进程身份拒绝；Study手动capacity入口同样是free检查。找到的.tmp/v030-20261013校准/恢复/telemetry是10-13历史证据，当前不使用，不为套新规则改写历史脚本/原观测。当前会话没有正在使用的进程身份准入脚本可补分类，因此未增加新的GPU框架或宣称已实现分类器；未来GPU准入须先按上述新规则实现/验证白名单+类型和RAM条件，旧脚本不能直接作为新准入依据。本轮只读源码，未调用nvidia-smi/torch.cuda、未启动或干预GPU/性能任务。


## 184. B116 / Issue158：日志墙钟文本颠倒实际事件顺序（2026-10-16）

只读复查六个实际cli.main日志案例，三个原生RFC3339毫秒+偏移反例（DST回拨、UTC跨日、跨偏移）均return0却较晚在前；三个同正/负/UTC偏移控制正确，全输入字节保留。先Issue158，再正式28红8 failed/20有效控制passed、0error/skip，库/CLI两入口及Z不同小数精度也复现。没有执行数据/训练/Study或改日志原文，临时REPRO/ZIP/hash/RED在.tmp/audit116不随clone。

设计只用stdlib datetime规范已识别aware排序键到UTC固定微秒字符串；保留原行与缺失/损坏/无时区/高于微秒文本、转换溢出的原fallback，避免放宽strict loader或截断旧高精度键。Python稳定排序、筛选/路径/IO、状态/数值/科学门保持，不新增模块/依赖/字段/路径。当前唯一未合codex/readout-time-order按18:53规则本地累积，PR157已正常合dev50d5a6bd9d2c4fca76199d7a0dd9cb6d4f1af97b、实际dev CI仍待验；旧已合主题及#156待验收清理。GPU新规则已在§183记录，当前无活动准入/执行任务。


## 185. PR157 双侧验收与归档主题清理（2026-10-16）

最终head10f01de5807d196db84fdc2a274a20c9a65237af的Branch38898015200 / Package38898015174 / Unit38898015143所有实际矩阵与聚合成功，正常合dev50d5a6bd9d2c4fca76199d7a0dd9cb6d4f1af97b。实际dev Package38898741388 / Unit38898741528同样全部实际成功，DEV_CHECKED SHA256 58c2c7f73e20a4a1969f728e490bf054c8438dbac7395c7f46ee298e870fe8be。Issue156验收关闭，PR正文只追加验收，不表示main发布/科学授权。

两侧CI后核对tip/祖先、活动工作区和唯一worktree，codex/archive-exclusive-creation本地/remote各1条普通删除；当前未合codex/readout-time-order与Issue158保留，main/dev和远端main不动，worktree/数据/结果/provenance/_staging_from_DEM删除0。周期基线149累计3（PR152/155/157），未达约5批；累计结束remote40。证据.tmp/audit115不随clone，清理/验收随完整里程碑提交，不另开文档PR。


## 186. B116 原生日志时序本机验收（2026-10-16）

按a183394设计，ba665de仅logs.py增加stdlib datetime导入与原排序键局部处理，11行增加/1行删除。带时区且在datetime精度内的已识别时间转UTC固定微秒字符串比较，避免墙钟文本及小数精度导致颠倒；原行内容、Logger/格式、事件选择、traceback筛选、可选IO/path归属、strict index/CLI错误边界均原字节保持。无时区/缺失/坏时间或转换溢出沿原fallback；高于微秒的旧文本保持原键、不宣称新增其UTC语义，未截断原精度。原稳定排序保持，不加依赖/配置/模块/路径，不改系统时区、状态/数值/科学门。

28新节点全部执行/通过：完整core1902 passed、1既有Windows POSIX模式skip，CPU Flow273 passed，均0failure/error。新28均在core，0新skip；正式红28项8失败/20有效控制保留。LOCAL_CHECKED保存XML/hash与实际节点；SOURCE_CHECKED按原字节确认仅预期导入/排序键变化。六actual CLI原探针与正式library/CLI节点覆盖DST/跨日/跨偏移/Z小数、同偏移、无时区/缺失/坏时间/旧亚微秒/UTC溢出/等时稳定性，输入及目录项保全。证据.tmp/audit116不随clone，正式测试与设计随Git提供。

继续源审其他readout消费者：status提示沿单个Run原追加顺序提取最后事件，不是跨Run排序；report日志链接只做归属映射；compare/轨迹按既有step或observation坐标，不应套墙钟归一化。只event_lines共享排序边界确认新反例，统一修在一处；不宣称整个外部历史日志格式或科学时间轨迹已经验收。新GPU准入规则继续按§183与计划约束，当前不执行GPU探针、准入或性能任务。最终head与实际dev全部实际CI仍须完整里程碑分别验收。


## 187. PR159 双侧验收与时序主题清理（2026-10-16）

最终head33cf72f9ac3bc85106999e6054a268c4829435c6的Branch38899904017 / Package38899904021 / Unit38899904100全部实际矩阵和聚合成功；正常合dev ab5c825ea6016681ba48623cbfcc282d2e781d4e。实际dev Package38901125267 / Unit38901125320亦全部实际成功，DEV_CHECKED SHA256 6c726eaffe276fb390448bd939eff620c65b7f7e2e5c71ec0ab79cf860438ee6。Issue158已验收关闭，PR159正文仅追加验收，不表示main发布或GPU/科学运行授权。

双侧验收后核对tip/祖先/唯一worktree与无未保全改动，普通删除codex/readout-time-order本地、remote各1条。首次ls-remote25秒超时在任何切换/删除前停止，CLEANUP_REFUSAL_1保留；同分支重新核对成功后清理。main/dev及远端main保持，数据/结果/provenance/worktree删除0，_staging_from_DEM保留。周期基线149累计4个已验收PR152/155/157/159，累计结束remote41；尚未达约5批完整盘点阈值。原始证据.tmp/audit116不随clone。

继续静态复查原子文本写入、report/index验证与DataFactory/ModelFactory按调用构造路径，本轮新增确认缺陷0；这是源码复核，不冒称新故障矩阵或科学验证。验收文档在codex/correctness-followup本地积累，按18:53频率规则随下一完整里程碑推送，不为几行文档另开PR。


## 188. B117 / Issue160：同进程数据端点绑定的并发所有权（2026-10-16，已确认实施方案）

`bind_endpoint` 临时修改第三方数据集类的 url / mirrors。DataFactory 的 cache_lock 保护各自资产根，无法保护两个不同根共用的同一个 Python 类。合成类、两个线程与事件调度复现两次上下文读错 URL，退出后还残留国内 URL；不访问网络、真实缓存或 torchvision。先 Issue160，再 9c95667 正式红回归：8项中4 failed/4有效控制 passed，0error/skip，覆盖CIFAR URL/MNIST mirrors、正常/原错退出和同线程嵌套恢复。

方案在既有 bind_endpoint 这一共同边界使用一个进程内可重入锁，持有范围覆盖保存原字段、绑定、整个构造上下文与 finally 恢复。普通互斥锁会破坏现有同线程嵌套，因此用标准库 RLock；不把锁放在每个资产根，也不新增每类锁注册表/依赖/模块。不同进程仍沿既有磁盘 cache_lock，不宣称进程锁替代磁盘锁。端点定义、调用签名、yield endpoint、正常和异常恢复、数据构造内容、下载/hash/布局与科学门保持。锁只解决原本不可并行安全的进程全局类字段，非测速/性能优化。

调用方只有原生 _build_torch_vision 中 train/test 两次构造的同一个上下文；既有调用方无需改。apply_model_origin 是显式进程环境设置，非临时绑定合同，不把本方案扩展成配置新语义。临时原ZIP/hash/probe/红XML在.tmp/audit117不随clone，正式回归按source镜像放tests/rpipe/structure/test_origin.py。生产修改后复查实际Factory路径、首错/嵌套/恢复及相邻origin，再做core/CPU Flow；同分支积累完整里程碑后再PR dev。


## 189. B117 本机并发验证与验证收集纠正（2026-10-16）

实际生产修复是fffd872：origin.py仅加标准库RLock及bind_endpoint全上下文所有权范围，16行增加/11行删除（原主体仅缩进），SOURCE_CHECKED按原字节验证其他内容全保留。55ad528仅修正测试文件名，不是生产修复；该提交标题未精确表达范围，后以独立主题提交保留真实生产回滚点，未重写历史。首次测试收集因新test_origin.py与既有data/test_origin.py同名拒绝，INVALID_COLLECTION_GREEN XML/log保留；改为源镜像test_origin_concurrency.py后13项全部执行/通过，收集拒绝不计库失败。

在真实DataFactory.build入口、两个独立缓存根、合成registry builder中分别调用原ZIP中binder和当前binder：原版两次读错且最终类URL残留；当前版两次读到各自URL、结束恢复原值。probe使用真实cache_lock，不导入torchvision/下载/访问正式数据；REPRO_FACTORY保留。本轮core1910 passed/1既有平台skip，新8全执行，0failure/error；CPU Flow273 passed，0failure/error/skip。未宣称不同后端或真实下载并发均已验收，不做测速。原始证据.tmp/audit117不随clone，正式回归/设计可携带。完整里程碑本地积累，最终head/实际devCI仍待后续验收。


## 190. B118 / Issue162：共享准备的临时环境所有权（2026-10-16）

同类全局状态复查确认prepare_shared_data在独立项目缓存根下并发保存/设置/恢复TQDM_DISABLE，后一个_prepare会看到None/调用方原值，全部退出还残留1。第一次探针误用共享同一根的两个Study，现有cache_lock使人为交错等待超时；INVALID_PROBE_NOTE保留，不计库失败。改用两个独立项目根后实际公共prepare入口/真实cache_lock+合成_prepare稳定复现，无数据下载。先Issue162，再b04a5cd红9项4 failed/5有效控制passed、0error/skip，覆盖初值缺失/自定义/已为1、正常/原错退出和同线程嵌套。

方案仅在prepare.py增加标准库RLock，以独立的进程内环境所有权范围包住原保存、设置、cache_lock内_prepare和finally恢复；必须在原资产根锁之前取得环境锁，维持一致顺序。同线程嵌套仍可重入，异常和返回值原样。与bind_endpoint锁分别保护不同全局对象，不引入通用全局状态框架、配置或路径，不改_prepare/body、下载/hash、数据/结果/科学门。与B117统一归入当前PR161临时进程状态并发里程碑，原已触发CI保留，新增代码后以新最终head重新验收；不因观察超时取消或重跑CI。


## 191. B117/B118 临时进程状态统一里程碑（2026-10-16）

B118按b6edba0设计，以0aaae0b在prepare.py增加RLock及完整保存/设置/资产锁内body/恢复范围，13行增加/10行删除；SOURCE_CHECKED精确证明其他原字节保持。B117/B118分别保护类端点和临时环境变量，不改root锁合同、数据构造、端点内容、下载/hash、marker、返回值、异常或科学门。27项两处及相邻回归全执行/通过，新17项均执行；合批core1919 passed/1既有平台skip，CPU Flow273 passed，0failure/error/skip。两次正式红独立保留：B117为8项4失败/4控制；B118为9项4失败/5控制，不冒称一场17项红执行。

新一轮20:56频率指令已收录cicd：本地正常按主题提交，同主题按一批相关B项完整里程碑后开少量PR；已开的PR161同时收尾#160/#162，不拆PR。先前e8dd7cf的CI只是B117阶段，不作为新增B118后最终head验收。原始失败、无效夹具/收集拒绝、ZIP/hash/XML保留.tmp/audit117与audit118，不随clone；正式回归与记录可携带。无下载/GPU/性能/Study/main/tag/数据删除或布局变化。


## 192. B119 / Issue163：顺序Study配方串用已缓存helper（2026-10-16）

按flow§14.1已支持同目录import及顺序run/子进程一致性，两个独立合成Study在真实load_recipe中导入同名helper，第二个读到第一个值与__file__；新进程第二Study读到自己的值，sys.path两种方式均正确恢复，问题在sys.modules。先Issue163，再5c8f5fd提交12项正式红：8 failed/4单Study控制passed、0error/skip；模块/包、加载时/注册时import、第一次注册成功/原错退出的8个串用反例均执行。原源码ZIP/hash/probe/新进程对照/红XML在.tmp/audit119不随clone；未运行数据/GPU/下载/正式Study。

当前生产代码未改，正在比对既有独立Study阶段命名空间与recipe绝对import合同；修复必须避免清空第三方全局模块或破坏注册后的运行对象，先明确临时模块缓存的所有权/恢复范围，再实现。不是等待人工批准，也不以文档替代修复。唯一未合codex/study-loader-isolation本地积累，按20:56规则与相关加载隔离问题形成完整里程碑后再PR，不立即另开小PR。PR161最终head d42072f的Branch38904767623/Package38904767492/Unit38904767906已全实际CI成功，正常合dev8480d795d0cb09714deb9bdd4bd79f41bdc5b2a9，实际devCI仍在执行；#160/#162与已合主题待双侧验收清理。


## 193. PR161 双侧验收与第5批周期盘点（2026-10-16）

最终head d42072fd2f1b5ec5392c1f239e6c0cb150215d67的Branch38904767623/Package38904767492/Unit38904767906全部实际矩阵/聚合成功，正常合dev8480d795d0cb09714deb9bdd4bd79f41bdc5b2a9；实际dev Package38905658240/Unit38905658071亦全部实际成功，DEV_CHECKED SHA256 5005764c673784eecc110d4f459b10762616b0eaa70c1429bcf7d2d4cf604a0e。#160/#162均验收关闭，PR正文仅追加验收；不表示main发布或科学运行授权。

双侧CI后核对tip/祖先、当前活动分支/唯一worktree及无未保全改动，普通删除codex/correctness-followup本地/remote各1条，tip d42072f已保全在dev。当前未合codex/study-loader-isolation及本地红回归保留，main/dev/远端main不动；worktree/数据/结果/provenance/_staging_from_DEM删除0。累计结束remote42。

从149基线完整盘点已验收PR152/155/157/159/161共5批，五个merge均为实际dev祖先；对应已处理Issue150/151/153/154/156/158/160/162均已CLOSED，未完成163/164保持OPEN。实际远端仅main/dev，实际本地仅main/dev和活动未合study-loader-isolation，唯一主worktree，无废弃worktree可清；开放PR0。未合保留清单及理由随下个里程碑PR说明提供，不删除。新周期基线161/dev8480d79，计数归0；原始PERIODIC_CHECKED及验收/清理证据.tmp/audit118不随clone，记录本地积累随下个完整里程碑合入，不另开文档PR。

## 194. B120 / Issue164：配方缺失模块身份使标准dataclass失败（2026-10-16）

load_recipe从spec构造ModuleType后未将其注册sys.modules即exec；合法的future annotations+标准库dataclass在读取定义模块字典时抛AttributeError。合成配方真实load_recipe复现；已核对历史路径别名导入的源码与当前工作区samefile为真，不把另一克隆当证据。先Issue164，再5bb02f5正式8红4 failed/4 eager注解控制passed、0error/skip，覆盖根/嵌套recipe与直接/模块别名dataclass；ClassVar字段解析也保留期望。无数据/GPU/外网/正式Study。原ZIP/hash/traceback/红XML在.tmp/audit120不随clone。

B119已补ace8485边界：18正式红14 failed/4控制，补既有外部模块原样保留和注册返回函数仍引用各Study实现。此前12红8失败/4控制保留，不互相替代。两个加载问题在同一未合本地分支积累，当前生产recipe.py尚未改，不立即另开PR。


## 195. B119/B120 配方临时导入状态所有权方案（2026-10-16，实施方案）

目标同时修同目录helper串用和执行时缺失recipe模块身份，保留现有load_recipe/apply_recipe调用、路径/来源清单/注册时机与正常异常行为。只在原recipe.py复用标准库contextmanager/RLock建立私有临时导入上下文，不新增架构模块、公开配置或产物目录。

进入load/register现有sys.path范围时，在同一个可重入进程锁内登记当前目录可导入的本地顶层模块/包名；仅暂存并移开这些名字及子模块的现有cache项，执行结束清理该范围新项、原对象原样恢复，路径亦恢复。内建/标准库名字保持原Python优先级，不批量清空sys.modules或第三方cache。其它非冲突第三方模块不进入所有权范围。锁在当前模块保存/移开/导入/恢复全程持有，支持同线程嵌套。

load_recipe在上述上下文内执行spec前，将自己的既有模块名临时注册sys.modules；无论exec/校验成败都恢复此前同名项（包括不存在），使dataclass可解析定义模块。load后返回ModuleType、注册后已捕获的函数/模块全局引用继续有效；测试明确核对两个Study注册函数各自返回原值和既有外部模块身份。register时的同目录import同样走该上下文。保留当前load使用recipe父目录、register使用Study根的路径行为；不把此修复扩展为新公共namespace API、清空外部模块、任意后端的序列化保证或跨进程解释器改造。

按先红与方案后改生产，完整测试须包含正常/原错、模块/包、load/register、外部cache保全、注册引用和dataclass根/嵌套/eager/future，以及已有Study hooks/来源冻结/core/CPU Flow。现有Study phase私有namespace只作设计参考，不为了复用反向依赖flow层或搬文件。当前仍本地积累，不立即新PR；进一步反例保留并修合同，不能以测试全绿替代边界核对。


## 196. 配方加载候选验证边界（2026-10-16，本地未推送）

初版私有上下文26项通过后，register内部定义future dataclass仍失败；6e9fff7先补该边界4红2失败/2 eager控制，再将模块身份保存/注册/恢复统一放进load/register共用上下文。另一次未提交候选复用了name作为循环变量和配方模块身份，误改隔离pytest进程cache而使报告中断，GREEN_COMPLETE.log及INTERMEDIATE_CANDIDATE_FAILURE保留；这是本轮候选实现错误，不计原始库红测试或CI失败，未提交/推送。纠正为module_name后30项通过，再补所有既有非本地cache项身份保全检查，防止以绿用例掩盖外部cache污染。未改正式Run/数据或外部解释器。


## 197. B119候选晚导入回归：不接受31项基线替代新边界（2026-10-16）

候选da6d0b0在31项定向测试、core1950 passed/1既有skip、CPU Flow273 passed上通过，但随后的注册对象生命周期复查确认：配方顶层已导入helper，注册的builder稍后执行时再次from helper import，会因上下文已清cache而ModuleNotFoundError。4a53165新增模块/包2红均failed、0error/skip；这是新候选引入的支持行为回归，不计原始14/4红，也不以先前全绿接受候选。两个新节点不在此前CORE/FLOW收集清单中，当前33项边界未通过，不能冒称整体全绿。候选及红测试只在本地，未推送、未开新PR、未影响dev8480d79。

§195暂存/恢复设计须继续覆盖注册函数晚导入的生命周期，不能简单清掉本地cache就作为最终实现，也不能改成禁止已支持的late import或粗暴清空外部模块。下一步先扩展私有导入绑定/模块所有权方案，保留这两个反例及原始预期，再修代码并重新执行当前全部节点/相关Flow。不是外部审批/网络阻塞，不因实现复杂而暂停；继续按20:56在当前唯一未合分支合批本地积累，不立即推PR。原始与候选失败、XML/hash均在.tmp/audit119/120不随clone。


## 198. 加载隔离生命周期与checkpoint兼容方案修正（2026-10-16，本地里程碑）

隔离namespace原型在.tmp中通过现有33项（非生产验收），但真实System save/load探针发现必要兼容边界：原裸helper类的model._extra_state可原生写读，namespace原型注册成功后读旧bundle却ModuleNotFoundError。首次探针误把device传SystemConfig构造器，另存无效夹具；改用既有from_mapping后复现。随后正式Flow prepare→System旧状态/新状态换Study根恢复两回归均红，不能靠只通过纯import测试接受namespace方案。

修正方案仍只改既有文件内容、不增模块/产物布局：recipe.py按物理Study根复用独立私有包，标准库source loader给该包的Python helper绑定本地import；普通import/from/包相对导入与注册后的晚导入都解析该Study，不清外部裸名cache。每Study固定包/recipe键复用，不按每Run无限增新包；不同Study物理根分离。finder只处理该私有前缀，不触碰第三方模块的__getattr__；load失败恢复此前recipe模块项，正常Python模块生命周期承载注册对象和dataclass定义模块。既有Run对象仍由Factory逐次构造，不把module代码cache当data/model/RNG对象cache。

私有类模块身份会影响含自定义Python对象的pickle全局引用，因此同步提供限定当前Study的兼容Unpickler：裸本地helper或历史私有包前缀映射当前Study同名helper/recipe类，非本地/标准库走原pickle查找。system_api新增私有桥，仅在标准Run assets位置且该Study recipe包已加载时提供reader；System.load_checkpoint全部已有torch.load点和compare checkpoint reader通过此桥取可选pickle_module，默认无recipe时kwargs/行为保持。System层不反向import make，仍经API跨层。Flow既有prepare注册顺序保持，不新增配置/公开选项/Run字段/序列化版本，不重写旧checkpoint；原torch.save、piece/bundle/diagnostic路径及恢复/科学门不改。

该桥须验证旧裸helper对象、不同物理根的新私有类型对象，以及外部cache保全；比较reader同步兼容，不把修改限于刚好测试的System入口。外部绕过RPipe直接反序列化自定义类仍需加载其定义代码，不能宣称任意plugin/宿主均可无上下文反序列化；native RPipe读取合同必须兼容，未通过之前不接受整批、不推PR。当前按20:56同分支本地积累，进一步反例先保留再修，任何旧绿色检查不代替新边界。


## 199. namespace候选的兼容与所有权反例（2026-10-16，本地）

命名空间原型先通过33项，但旧helper自定义model._extra_state的真实System兼容探针失败；39cacdb补Flow prepare→System旧类/迁移后类两正式红均failed。当前候选的私有recipe模块曾用合法helper名_recipe，7e7ad8f红复现其遮蔽Study现有_recipe.py；内部身份改为非Python标识符<recipe>，不新增用户文件名保留限制，标准注解反射与pickle回归验证可用。

进一步给合成Study加未被配方导入的torch.py，并在原checkpoint同时保存Tensor，暴露兼容reader按文件名重解释全局torch._utils：6项都红，RED_PICKLE_OWNERSHIP保留。已限定bare引用的已加载模块所有权与完整module spec，未被namespace接管的既有全局模块走标准Unpickler；不是把所有同名文件当pickle类来源。外部缺失类按原严格错误，不吞恢复失败。system_api桥只查询已加载recipe模块，不因普通System IO额外安装recipe import机制。

当前49项定向节点全通过（43源镜像unit、6实际Flow prepare/System/compare集成）：含legacy bundle、pieces/model.pt/meta.pt、payload.pt，自定义类和Tensor、搬到不同Study根后的类型归属、只读compare reader、helper内部晚导入/包相对导入、相同Study basename不同项目、重复加载模块键不逐Run增长、失败替换保全旧module、注解反射/pickle、双线程及原错退出。源码尚未提交生产新修复；不是完整core/Flow或CI验收，旧31项全绿不能替代本批。当前仅本地合批，不开新PR。


## 200. B119/B120模块生命周期与原生reader统一候选（2026-10-16，本地未推送）

现有4个源码文件内完成：recipe按物理Study根私有包隔离，定义模块和helper保持标准模块生命周期；source loader仅给私有模块绑定本地导入，外部import仍调用当前Python hook（a892cea先补钩子被绕过的候选红）。内部recipe身份使用<recipe>，不遮蔽合法_recipe.py；每Study固定键，重复4次加载键数不增长。源码热重载不在本合同内，不把代码cache当Run可变对象cache。

system_api私有桥只检查已加载recipe及标准Run assets，System原4处torch.load与compare1处读取使用可选reader；无recipe时参数保持原样。兼容裸helper和跨根私有类，保留已加载外部cache及完整global module spec，避免未使用的torch.py误接管Tensor解码。torch.save、bundle/pieces/诊断布局、payload键/协议、恢复指纹和科学门原字节保持；没有迁移/覆写旧数据。输入checkpoint字节保全加入实际Flow回归。

当前50项定向节点通过（44 unit+6 integration），0fail/error/skip；legacy bundle/pieces/meta.pt/payload.pt、Tensor与自定义_extra_state、跨根恢复/compare，以及late import/相对包/同basename/模块数量/失败替换/反射/pickle/并发均执行。此为新候选本机定向结果，完整core/Flow与更大兼容范围仍待检，不宣称已合dev或关闭163/164。只本地积累，20:56频率规则下不另开PR。


## 201. B119/B120当前完整本机验收（2026-10-16，仍未推送）

当前源码ac65eb1/041e8c1及aa75064边界测试后重新执行完整检查：core1964 passed、1既有Windows POSIX模式skip，CPU Flow279 passed，均0failure/error。新增51节点全部实际执行（45 unit+6 integration）、0新skip；这次CORE_CURRENT/FLOW_CURRENT包含全部当前边界，不用旧31节点/旧候选结果替代。LOCAL_CURRENT_CHECKED保留XML/hash及节点归属；FINAL_SCOPE51全通过。Runtime优先级候选6红、外部import hook红、旧/迁移checkpoint红、晚导入红和所有无效夹具/报告中断仍保留，不冒称它们是原始同一场红测试。

原字节核对SOURCE_CURRENT_CHECKED：只4个既有源码文件变更。recipe的phase校验、RecipeContext字段、路径拒绝等原字节保持；System.save_checkpoint全部原字节保持，load只在4处既有torch.load加可选reader参数；compare只加同类reader，整个数学树、容差/非有限/整数精度、格式和错误门其他字节保持；system_api既有导出/build签名保持，仅附私有桥。无新源模块/配置/Run字段/产物路径，未改producer协议、checkpoint布局/恢复指纹或科学门。正常代码helper按Study模块缓存，不承诺热改源码自动reload；原生reader已验证旧helper和迁移后私有类，外部直接反序列化自定义类仍需要定义代码可导入。

全部仅本地codex/study-loader-isolation主题提交，当前open163/164保留；本批未开PR、未跑最终head/实际devCI、未合dev/main、未打tag。接受的dev仍8480d79/PR161，清理基线161计数0。按20:56相关B项完整里程碑和每日少量PR规则继续本地收敛，不以多次验证拆更多PR；后续流转仍须最终head/实际dev全实际CI验收后再关闭Issue和清已合分支。原始.tmp/audit119/120不随clone，正式回归/设计/记录可携带；无GPU/测速/下载/正式Study或数据/结果删除。


## 202. PR165验收前补齐B121 / #166动态导入候选回归（2026-10-16）

B119/#163、B120/#164完整本机检查后统一开PR165，最终候选d385c410在实际CI执行期间继续审查相邻入口，发现recipe在register内用importlib.import_module加载同目录helper：dev8480d79独立子进程成功返回37，候选报ModuleNotFoundError。PR未合并、Issue未关闭，未将任何旧head绿色结果当作本批完成。此为加载隔离候选引入的回归，与原B119/B120失败分开；登记#166后f32416d先提交10节点红回归，8失败/2非本地对照通过，8f37360记录同批设计，不拆新PR。

9166cbd只在原recipe.py增加14行/替换1行：当前Study私有代码导入importlib时持有局部模块视图，其import_module沿同一归属判定解析本地绝对名，其他名/相对名委托当前原生函数；不修改全局importlib及其cache。importlib子模块仍是原模块，现有标准运行时与外部导入hook继续委托。加载、register、注册后、helper内动态导入×2取用方式均覆盖；未引入源模块/依赖/目录或checkpoint写入改变。视图并非原生importlib对象本身，属于私有recipe执行环境，不宣称全局模块身份被替换。

当前源码重新完整检查：core1974 passed/1既有Windows POSIX模式skip，CPU Flow279 passed，均0failure/error；当前新增61（55 unit+6 integration）全部执行、0新skip。目标两个文件67 passed，含6既有Flow节点，不冒称67都是新增。原始动态探针的前后子进程、RED8失败、TARGET_GREEN、CORE、FLOW、hash和SOURCE_CHECKED保存在.tmp/audit121及.tmp/audit119，不随clone；两个探针驱动的import sys遗漏均发生在子进程调用之前，单独保留、不算库失败。修复后独立子进程返回37。

SOURCE_CHECKED核对相对于上个完整候选只有recipe动态绑定改变，其余3个生产文件hash完全不变；此前System保存、compare数值/格式、API已有字节及文件架构/磁盘产物布局合同仍保持。没有数据迁移、GPU、测速、下载、正式Study/科学结论、main/tag/发布或数据/结果删除。

三项作为同一完整加载里程碑继续PR165普通推送，不amend/force push。待新最终head的全部实际矩阵/聚合CI，再正常合dev并另验实际dev；目前163/164/166都OPEN，已接受基线仍PR161/dev8480d79，清理基线161计数0。旧候选CI即使通过亦不可替代当前head。验收后的收尾记录随下一里程碑入库，不为几行文档再拆PR。


## 203. PR165双侧实际CI验收、三Issue关闭与已合主题清理（2026-10-16）

完整加载里程碑B119/#163、B120/#164、B121/#166统一[PR165](https://github.com/DreamSoul-AI/RPipe/pull/165)。最终head3a74728e34c8f5f27216e27dd3fe7b50e5b7ccd8的[Branch38917963279](https://github.com/DreamSoul-AI/RPipe/actions/runs/38917963279)、[Package38917963440](https://github.com/DreamSoul-AI/RPipe/actions/runs/38917963440)、[Unit38917963881](https://github.com/DreamSoul-AI/RPipe/actions/runs/38917963881)及所有实际矩阵/聚合均success。严格PR_CHECKED SHA256 c73dafef661ecb02c06e7cdbedc4913c68e0f90455b1db8d778a47be762261c7，未采用已被动态导入反例否决的d385c410检查替代当前head。

使用expected-head正常merge到实际dev70a8750e98df9cb941e6c8f525d51f9d41cec77b，未squash/force push。实际dev的[Package38918867862](https://github.com/DreamSoul-AI/RPipe/actions/runs/38918867862)、[Unit38918867923](https://github.com/DreamSoul-AI/RPipe/actions/runs/38918867923)及所有实际矩阵/聚合亦success；DEV_CHECKED SHA256 1560ec68260bc7a864ef2ff4e99265c9a002dc638a015ed95d6aec42d74cbc56。PR正文保留原材料并追加真实验收；163/164/166各关闭并一次引用同批验收，不重复评论。关闭不是main发布或新实施授权。

本机当前core1974 passed/1既有Windows POSIX模式skip、CPU Flow279 passed；新增61（55 unit+6 integration）全执行/0新skip，目标67还包含6既有Flow节点。原始红、候选回归、两个驱动import sys遗漏、ZIP/hash/XML及严格checker日志保留.tmp/audit119/120/121，不随clone；正式portable测试/方案和工程记录随Git提供。相邻复核recipe/phase、atomic文本、共享锁、checkpoint诊断、Tracker追加与status/log/report/index，除本批已修动态入口外未确认新反例；检索清单仅是源码搜索，不冒称穷尽所有IO路径，也不代替既有403组全仓清单。

双侧验收后核对实际远端tip=3a74728、其为dev祖先、工作区只有保留的_staging_from_DEM、仅一个primary worktree、没有命令引用RPipe的活动Python进程。先detach到实际dev，再普通删除以下已合分支：

| **范围** | **删除分支** | **提交保全位置** |
| --- | --- | --- |
| 本地 | codex/study-loader-isolation | dev70a8750包含tip3a74728 |
| 远端origin | codex/study-loader-isolation | dev70a8750包含tip3a74728 |

删除后远端与本地均不存在该主题；main7a15c46、本地dev9324014、origin/main00e70b1原值保持，没有worktree、数据、Run、结果或provenance删除。remote累计结束43条。清理基线仍PR161/dev8480d79，已验收累计1/约5；本批即时清理完成，不重复触发全面第5批清理。实际开放Issue快照为空，下一缺陷编号B122，不为活跃制造Issue。

从实际dev建立唯一活动未合分支codex/correctness-next-milestone，继续三类正确性复查；本节收尾及后续同类内容先本地按主题积累，随下一个完整里程碑统一PR，不为几行文档另开PR。目标保持active，无决策阻塞；无GPU/测速/下载/正式Study/main/tag/发布/数据删除或文件架构与产物布局更改。


## 204. B122 / #167末条Runner日志失败留证复现与红回归（2026-10-16）

PR165验收/清理后，工作区65ed4e6仅多收尾文档；fetch正常、开放Issue快照为空，继续第一类边界复查。实际FlowRunner/Logger/result IO，合成collect/process无计算body，末条stdout拒绝×新旧结果四探针全部原异常传播但无error诊断；collect旧succeeded残留、新Run无failed，process旧成功仍保留但无诊断，新Run同样无failed。原始REPRO.json/log、runner.py ZIP/hash在.tmp/audit122（不随clone），无GPU/训练/下载/测速或正式Study执行。

先开#167，再4ac6883提交portable12节点红回归：真实末条stdout/file-write两拒绝×collect/process×新旧结果8失败，正常末条4对照通过，0error/skip；phase body均实际调用一次、真实peer/原异常身份与process旧成功原字节均锁定。并非再开已修B101启动Issue，根因是同一Runner的最后一条日志未纳入既有失败处理。

按已批准第一类合同只缩进原语句进入try，保持所有阶段完成后输出时机，_preserve_success/原except/failed IO完全原样，不引入更多保全范围或吞日志错误。实施方案见artifact_consistency_design§204；当前先本地同主题合批，不单bug开PR。接受基线dev70a8750，清理基线161计数1，活动codex/correctness-next-milestone保留；不动main/tag/布局/数据结果，下一编号B123。


## 205. B122本机修复与完整验证（2026-10-16，同主题本地积累）

761700f只将FlowRunner末条logger.info('succeeded')原语句缩进到既有try中；除这一处移动外，_preserve_success、整个except/_log_failure/_write_failed_result、阶段/hook顺序与输出时机等原字节完全保持。SOURCE_CHECKED与原zip/hash锁定原生源文件仅runner.py两行diff；未改result/schema、磁盘路径、日志格式、算法/Tracker/checkpoint或科学门，不增加helper/模块/依赖。

RED12项8失败/4正常对照，GREEN同文件49 passed，全部12新节点执行；当前完整core1974 passed/1既有Windows POSIX模式skip、CPU Flow291 passed，均0failure/error，新增12全执行/0skip。不是用只跑目标49替代完整当前集合。独立basetemp和XML/hash/ZIP/原始REPRO均在.tmp/audit122，不随clone；正式回归已入库。完整检查进程52264/58084均exit0终结，不保留伪活动观察句柄。

复查相邻launch打印/失败定稿、prepare结束日志及Runner调用方，尚未确认需另记的新反例；没有为活跃拆新Issue/PR。#167保持OPEN，本机通过尚不能代替最终head/实际devCI。当前codex/correctness-next-milestone仅本地积累（含PR165收尾记录），按20:56将相关失败留证工作汇成完整里程碑再统一PR；未推送或开PR，接受dev仍70a8750，清理基线161计数1，remote累计43。继续三类复查，下一编号B123；目标active，无决策阻塞，无GPU/测速/下载/正式Study/main/tag/数据删除或文件架构/产物布局变化。


## 206. B123 / #168：中断日志尾行与下一条事件分隔（2026-10-16，同一失败留证批次）

真实Logger在无行尾的旧run.log后追加ERROR，将两条native header连成一行；event_lines将新错误读成旧flow内容，_note_from_log(errors_only=True)得到'-'。原始字节未丢但事件身份丢失，原生合成Run/Logger/index/readout确认，无训练或科学运行。#168后66d3526先提交14节点红6失败/8有效控制，后补单独CR有效行尾对照，完整16红仍6失败/10控制；ASCII/UTF8半字节、同一Logger/重新打开和真实只读回读均覆盖。

最小方案在现有Logger._emit：stdout输出仍原位置；以单独rb句柄检查可读文件最后一个字节，仅非空且既非LF也非CR时，在同一次原a文本write的record前加入一个newline。保留全部旧字节，不重写/截断/迁移文件。原append模式、编码/newline映射、权限/umask、write/flush/close及首错规则保持；有效空/LF/CRLF/CR日志原字节不变。二进制尾读不解码破损UTF8，不改变status/logs的解析或科学门。

尾读仅用于可读尾行修复：OSError时继续原append行为，不能因为新增read要求阻断原来可写但不可读的日志。此时无法保证中断尾行修复，明确限制与回归保护，不把可选探查失败当作写入成功证据；原write/flush/close失败依然严格。保留constructor不重开既有证据、peer、合法记录stdout/文件对应、其他调用方及Tracker恢复协议。不是新日志框架，不承诺任意多写者同Run事务。

与B122/#167同一codex/correctness-next-milestone本地主题积累，先红/设计/最小生产修复，再合批当前core/CPU Flow及最终head/实际devCI；不单独开一个bug PR，不等审批、不改文件架构/产物布局或运行GPU/测速。


## 207. B122/B123同主题日志失败与原生读回里程碑，本机完整验证（2026-10-16）

B123/#168的原生Logger/index/logs/status探针确认：中断半行和新ERROR连成一行，错误内容被归到旧flow且errors_only为'-'。先66d3526提交14红6失败/8控制，再6588d28补有效单CR控制得16红6失败/10控制；3ad90d3在原Logger源码、已有B122 Runner修复上补四项真实联动（末条写部分字节/flush后抛错→Runner定稿→native readout），完整RED_COUPLED20红10失败/10控制。该红不是把B122的12红也放回同一生产基线；所有独立红场次与原候选/旧数据均保留。

97a1443仅在既有Logger._emit加可选二进制尾读，已知非空非LF/CR才在原a文本write前加分隔；其余write/flush/close首错规则、stdout时机、append模式/权限/编码/newline及全部其他方法原字节不变。不能读尾时保持原append-only能力，不承诺此限制下修复半行；正常empty/LF/CRLF/CR、构造不重开、peer及旧半行原字节均保留。有效日志记录格式未变，不改readout解析、Tracker、checkpoint、schema或科学门；没有通用I/O框架/新文件。

与B122合批定向298 passed，其中32新节点全部实际执行（14 unit+18 integration）；当前完整core1988 passed/1既有Windows POSIX模式skip、CPU Flow297 passed，0failure/error，新增32无skip。完整进程57962/10745均exit0终结。新32不是298的全部；B122原12红8失败/4控制与B123最终20红10失败/10控制是各自真实生产状态，未伪称统一一场32红。

SOURCE_CHECKED精确原字节比较：src只Runner末条info缩进1增1删、Logger可选尾读/原追加11增1替换，其余源码/原异常处理和持久化格式不变。原始REPRO、源码ZIP/hash、各场红/绿、XML、源码证明在.tmp/audit122/123，不随clone；portable回归/设计/记录和双语研究计划随Git提供，无科学运行或结论新增。

目前167/168仍OPEN，接受dev70a8750/PR165，当前两B是同一个完整“末条失败→半行保全→独立error事件读回”里程碑，不单bug或几行文档拆PR。本机验收后统一PR，仍须新最终head/实际dev全部实际矩阵/聚合验收再关闭及清分支。清理基线161计数1、remote累计43；下一编号B124。无GPU/测速/下载/正式Study/main/tag/数据/结果/架构或产物布局变化，不等待设计审批。


## 208. PR169日志失败/原生读回里程碑双侧验收与已合主题清理（2026-10-16）

B122/#167与B123/#168统一[PR169](https://github.com/DreamSoul-AI/RPipe/pull/169)，最终head05341f85691d1ad41aa54fe4a7a70f55a96b14e6的[Branch38923564342](https://github.com/DreamSoul-AI/RPipe/actions/runs/38923564342)、[Package38923564221](https://github.com/DreamSoul-AI/RPipe/actions/runs/38923564221)、[Unit38923564299](https://github.com/DreamSoul-AI/RPipe/actions/runs/38923564299)及所有实际矩阵/聚合均success。PR_CHECKED SHA256 d5d2a52a47560e1f9303adba1a0bd714e5eee1d908c5a52e139918174ad66c4b。

expected-head正常merge到实际devbef7de966a0cea0e86b1e2392222ddf8bfeff920，实际dev的[Package38924513921](https://github.com/DreamSoul-AI/RPipe/actions/runs/38924513921)、[Unit38924513792](https://github.com/DreamSoul-AI/RPipe/actions/runs/38924513792)及所有实际矩阵/聚合也success。DEV_CHECKED SHA256 905d10c06fef037b85a3a20de90b32f4818f11025246c5d4698fbd310ba3a6e7；不是仅靠PR绿色验集成。167/168各一次验收评论后关闭，不表示main/tag/发布。PR原正文保留，追加真实验收与清理事实，不重复评论。

本机相邻298/core1988 passed+1既有Windows POSIX模式skip/CPU Flow297 passed，32新全部实际执行（14 unit+18 integration），0新skip。原始独立红/原Logger与已修Runner联动红、原ZIP/hash/REPRO、XML与严格checker证据均保留.tmp/audit122/123，不随clone；正式回归/设计/记录已入库。只改两个现有生产文件，原日志有效记录/权限/首错与process旧成功例外、Tracker/checkpoint/readout解析/数值门/架构/产物布局保持；不可读尾保留原追加行为，不宣称此时能够修复半行。没有GPU/测速/下载/正式Study/科学结论或数据迁移。

双侧验收后核对本地/实际远端tip=05341f8且为dev祖先、工作区只保留_staging_from_DEM、一个primary worktree、无命令引用RPipe的活动Python进程；先detach实际dev，再普通删除以下已合主题：

| **范围** | **删除分支** | **提交保全位置** |
| --- | --- | --- |
| 本地 | codex/correctness-next-milestone | devbef7de9包含tip05341f8 |
| origin远端 | codex/correctness-next-milestone | devbef7de9包含tip05341f8 |

删除后本地与远端均无此主题，main7a15c46、本地dev9324014、origin/main00e70b1原值保持；没有worktree、Run、数据、结果或provenance删除。remote累计结束44条。周期清理基线仍PR161/dev8480d79，已验收PR165/169共2/约5，本批即时主题清理完成，不重复触发全面盘点。

最新实际开放Issue快照为空。唯一活动未合codex/correctness-review-next从实际dev建立，继续三类正确性复查，下一缺陷B124；先确认反例再开Issue/红回归，不为活跃制造任务。本节收尾与后续相关工作本地主题积累，随下个完整里程碑合批，不单独文档PR。20:56后PR165为加载/兼容里程碑、PR169为日志失败/读回里程碑，未按每bug拆PR；继续控制每日少量完整批次。目标active，无决策阻塞；不合main、不打tag、不force push，不动GPU任务/数据/结果与文件架构/产物布局。


## 209. B124 / #170：合法多行消息逐行保留Logger原生记录身份（2026-10-16，本地设计）

原生Logger.info的合法多行字符串仅第一行带完整header，后续[error]正文变成独立错误事件，errors_only读出一个从未由Logger.error发出的错误。REPRO使用真实Logger/index/logs/status，原始ZIP/hash/log保留.tmp/audit124（不随clone）；事件闭集与有效Run身份仍按既有合同，不把无效event或Run配置扩展成支持能力。

先开#170，8ca7187提交14项正式红回归11失败/3正常对照：info/warning/error、exception多行上下文/异常值、report extra六条真实调用路径，加CRLF与Unicode行分隔；原生Runner collect→Logger→logs/status三条集成确认事件不能被正文改名。普通/空消息和原f-string对象只格式化一次是兼容对照。均合成CPU/IO，无训练/GPU/数据或性能运行。

方案沿已有exception traceback逐行_emit规则：只在原_emit外加一层消息行循环，使用原f-string转换一次后splitlines，空消息保留一次原记录；每行复用原完整emit主体。正常单行记录字节/事件/级别/Run身份/编码/newline/权限、末条失败边界、可读尾行分隔及readable限制、write/flush/close首错规则均保持。多行内容不压平/转义、不通过修改只读parser猜事件，逐行保留文本及原拥有事件/级别；失败仍传播，不吞消息或放宽科学门。

不新增helper/模块/文件/schema/配置或磁盘位置，不改变Tracker/checkpoint与各Study方案。代码只做一层循环及既有主体机械缩进，精确字节证明除缩进外主体不变。当前先本地原分支积累字段边界主题，不立即为单bug开PR；本机定向和完整core/CPU Flow后，后续完整里程碑仍须最终head/实际dev全部实际CI再验收，不等待设计批准。


## 210. B125 / #171：可选activity在两个展示点共用单行投影（2026-10-16，本地设计）

合法非空Unicode phase及字符串detail包含换行/制表符时，announce原样打印成额外行列，status同样在权威summary/header前输出多行提示。真实announce/read_activity/index/CLI探针已复现：输入mapping合法、统计仍planned=0、原activity字节未改，错误仅在可选提示的显示身份，不称为Run结果损坏。原始REPRO/log/ZIP/hash在.tmp/audit125（不随clone）。

#171后c2dfbdb先提交18红16失败/2正常对照：phase/detail×LF/CRLF/Unicode行分隔/tab的producer8项，status的phase/detail×LF/tab×缺index/正常peer8项，以及正常Unicode/空detail2控制；原mapping、peer与全部输入字节/目录清单锁定。

最小方案在现有activity.py内加私有_format_activity，仅将已拼接提示的splitlines用空格拼回、tab换空格；announce与CLI两输出点复用。不能反向import readout（现有设计第四节），不搬_fmt到新模块；此helper只服务当前两个重复显示点，不是新框架/公开schema。read_activity及announce持久化主体原字节保持，自定义phase/Unicode/None或缺省detail规则、正常行空格和stdout时机保持。CLI继续原detail-or-empty及rstrip约定，producer正常空detail继续原尾空格。

提示只做显示投影，不改存储值、不拒绝合法字符串、不增加phase枚举/ownership/TTL协议；有效hint缺index仍exit0，无hint缺index仍exit2，mode/正式table及peer不变。与B124/#170作为同一字段边界里程碑在原分支本地积累，不单项PR。先定向再当前core/CPU Flow；后续完整里程碑仍须最终head/实际dev全实际CI，Issue未验收不关，无需等待设计批准。


## 211. B126 / #172：原生日志按已知Run身份定位真正event字段（2026-10-17，本地设计）

只读两函数在整行search所选tag，会把有效Run名字或正文中的[error]当事件：真实warning/split提示替换原错误note，合法单行Run目录name [error] Traceback marker使原生ERROR/FLOW整行被误当traceback过滤。真实Logger/Tracker/index/result/status/logs探针均复现，原始证据.tmp/audit126不随clone。此问题在单行消息也存在，独立于B124；没有扩展无效mode/event或多行log路径。

#172后4fcca9c先提交28项红回归20失败/8所选事件控制：普通/空格/含tag的合法Run组件，默认与自定义受限log路径，native warn/split/flow/error加metric/time/ckpt/resume；原生Runner/peer、错误note、默认独立note、过滤政策与全部输入字节锁定。

最小方案在现有status.py共享_event_match：识别原生时间/级别头后，以当前计划run_dir或原生'-'身份的完整字面前缀找到实际event起点，所选_EVENT只在该处match；已确认原生owner而event不在原集合时返回None，不能再从payload寻找。未知/legacy行继续原search，不猜旧文本、不重写文件。logs复用此house helper；status列表把已有run_dir传给_note_from_log，自定义log无需路径猜测；独立默认note可按固定runs/id/assets/logs路径字面取得上下文，缺上下文沿原兼容回退。

原选择集合、traceback抑制、timestamp/sub-microsecond排序、path guard、严格index/config/result/数值门与行输出原字节保持；不新增模块/文件/schema或改变Run目录/日志格式，不扩大为任意坏文本认证器。仅两个既有readout文件，先红后改，再与B124/B125统一当前本机core/Flow及后续完整里程碑最终head/实际devCI；不单bug PR、不等待设计审批。


## 212. B127 / #173：traceback抑制只匹配标准行形态（2026-10-17，本地设计）

两reader的startswith('Traceback')/startswith('File ')会丢失独立业务ERROR，例如File cache missing、File "data.csv" is missing、Traceback writing failed。原生Logger/index/result/status/logs五探针前三普通错误均note为'-'/日志无行，标准Traceback头和File帧两控制同样过滤但本来应如此。原始ZIP/hash/REPRO在.tmp/audit127不随clone；发生在B126事件字段修复后，是独立的正文分类过宽，不改原生owner匹配。

#173后正式18红12失败/6控制：六种普通或近似File/Traceback消息×native/legacy；标准Traceback头、带/不带函数名的quoted File/数字line三形态×2控制。原生Runner/peer与所有输入字节/目录清单锁定，非科学运行。

最小方案：在既有status.py共用_traceback_detail，只将完全相同的标准Traceback头或File "...", line 数字（可带, in 函数）的整行识别为细节；logs复用此helper。其他含File/Traceback首词的正文仍是原error记录。保留原选择事件、native/legacy匹配、时间排序、文件/只读边界与其他逻辑，不做traceback状态重建、不增加结构化格式或重写旧日志。与标准细节形态完全相同的非结构化文本仍按原显示策略略过，不承诺无法辨别的来源归属。

仅两个现有readout文件，把共同规则收敛一次而非在两个入口散改；先红后最小修，再与B124/B125/B126统一当前core/CPU Flow和完整里程碑最终head/实际devCI，未验收四Issue不关闭，不单bug PR、不等待设计批准。不改schema/布局、生产者或数值/科学门，不测速/跑GPU/Study。


## 213. B124–B127字段/事件身份里程碑与展示点清单（2026-10-17，本机完整验收）

2026-10-16夜间至17日按同主题收敛四个确认反例，而非逐bug开PR：B124/#170合法多行消息丢完整记录身份；B125/#171 activity提示增加额外行列；B126/#172 Run名字/正文tag覆盖实际原生事件字段；B127/#173普通File/Traceback错误被过宽过滤。分别先开Issue、提交独立红、记录设计再最小修：14红11失败/3控制（8ca7187）、18红16失败/2控制（c2dfbdb）、28红20失败/8控制（4fcca9c）、18红12失败/6控制（2d7295f）。这些是四个真实生产状态，不冒称统一一场78红；原失败与每阶段目标绿保留。

38d1e09仅给原Logger._emit加消息行循环、其余主体机械缩进；1a4cce4只在原activity内共用小formatter并接入两个现有打印点；62d0f47/a94622d在原status/logs共用原生字段定位和标准traceback形态规则。没有新生产文件/依赖/公开接口/配置/schema/路径，不搬代码或新增I/O框架。旧不完整头的多行日志不重写、不猜归属；原legacy兼容回退保留，标准细节完全同形文本仍按原噪声策略；既有不可读尾保留原append-only能力而不保证半行修复。

### 当前字段展示点核对

| **既有入口** | **字段边界与当前状态** | **本批依据/限制** |
| --- | --- | --- |
| Logger._emit，info/warning/error/exception/report | 本机已修；多行内容每行保留完整级别/Run/event，不压平或改名 | B124六调用路径、LF/CRLF/Unicode分隔、空/普通/一次f-string控制；原每条IO主体保持 |
| activity.announce与CLI status提示行 | 本机已修；两输出点共用单行投影，JSON/mapping未规范化 | B125缺index/正常peer、phase/detail/tab/行分隔、正常Unicode与空detail控制 |
| status._note_from_log与logs.event_lines | 本机已修；原生header按已知Run身份定位event，正文/Run名tag不能抢占 | B126三类合法Run组件、默认/自定义受限log、所选/忽略事件、peer和原字节；legacy不猜 |
| 两reader的traceback细节过滤 | 本机已修；仅标准头/quoted File数字line形态抑制，普通首词错误保留 | B127 native/legacy六普通与三标准形态，完全同形非结构化文本不可辨别 |
| status.format_status/mode/metric与path guard | 既有代码未改，本次相邻/完整回归通过 | SOURCE_CHECKED锁定原字节，表头/列/统计/严格config/输入路径合同不变 |
| report.render_numbers/text/Markdown/link和计划绑定 | 既有代码未改，现有回归通过 | 不更改格式、有限数/计数、计划/Run身份门，不借显示修复放宽科学校验 |
| compare.format_compare与严格数值比较 | 既有代码未改，现有回归通过 | key已有_text_cell；其余detail来自内部类型化报告，不把任意坏report当支持输入 |
| activity.read_activity | 既有代码原字节未改 | 最低字段/Unicode/自定义phase/None与缺省detail、坏JSON/IO回退保持，reader无反向readout依赖 |
| 原子写/共享锁/Tracker/checkpoint写协议 | 源码未改，既有故障/取消/恢复回归通过 | 不承诺多文件或任意同Run多写者事务；不做性能/实验运行 |

该清单是本轮字段边界复核，承接原record§83–87的403组全仓I/O清单，不将源码检索当作所有未来输入已穷尽。CLI其他路径/错误自由文本保持原功能，不凭无协议的新假设制造缺陷。

当前合批相邻420 passed；完整core2009 passed/1既有Windows POSIX模式skip、CPU Flow354 passed，均0failure/error；新增78全执行（21 unit+57 integration）、0新skip。Core/Flow进程40258/60788均exit0终结。B124/B125早先core2009+1/Flow308已单独保存，只证明当时两个修复，不能代替当前包含两reader改动及新增46集成节点的完整集合。

SOURCE_CHECKED核对仅五个现有生产文件：Logger原主体逐行字节相同仅加循环缩进；activity reader/JSON写前缀、CLI其他功能、status的mode/metric/path/表/统计及logs排序/格式原字节保持；不改其余源码、numeric/Tracker/checkpoint或任何科学门。原REPRO/ZIP/hash/各场红绿/XML/源码证明在.tmp/audit124/125/126/127，不随clone；portable回归/方案/双语计划随Git提供。

四Issue170/171/172/173仍OPEN，尚未推送/开本批PR；按完整字段/事件身份里程碑一次流转，最终head/实际dev都需全部实际矩阵/聚合验收后再关闭及清理。接受dev仍PR169/bef7de9，清理基线161计数2、remote累计44；下一编号B128。无决策阻塞，不等待设计审批；无GPU/测速/下载/正式Study/科学数字或结论新增，不合main/tag/发布、不force push、不删Run/数据/结果/provenance、不改文件架构和产物布局。
