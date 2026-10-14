# 产物证据、只读视图与统一写入设计

## 一、摘要

- **状态：待审阅，未实施。** 本轮只提交设计与状态记录，停止逐个相邻路径审计；审阅通过前不修改生产代码、测试合同或现有产物。
- **基线：** dev `d294c1b5aa99b4d7148f991f9a4ac8903fe23374`。#50 已经由 [PR #51](https://github.com/DreamSoul-AI/RPipe/pull/51) 合入，无需再建重复修复 PR。最终工作提交 `cbc8c0a` 与实际 dev 的 CPU / 包必需 CI 已通过，实际 dev 依据为 [Unit Tests](https://github.com/DreamSoul-AI/RPipe/actions/runs/38633933430) / [Package Check](https://github.com/DreamSoul-AI/RPipe/actions/runs/38633933432)。这些是既有修复证据，不是本设计的验收。
- **目标：** 把失败写回、展示容错和原子发布收敛为三项公共合同，消除各调用方重复判断异常类型、旧成功和临时名的职责。
- **范围：** 仅代码正确性；不测速、不做性能改动、不调查 480 秒事件、不运行 GPU、不更改研究计划和科学门，不迁移正式数据或修改历史 provenance。

| **类别** | **当前局部处理** | **拟统一结构** | **仍未完成** |
| --- | --- | --- | --- |
| 失败写入与证据保全 | Runner 在部分阶段保留旧成功，调度仍通过写 failed 表达失效；重试共用结果位置 | 每次尝试独立追加不可变记录；`result.json` 是最近有效结果的可重建视图 | 尝试账本、结果引用、依赖身份、崩溃恢复及旧格式适配均未实施 |
| status / logs 只读展示 | 各文件分别捕获错误，mode 在 axes / result / config 间回退，字段类型 guard 分散 | 文件安全读取三态；先从计划 config 建立有类型 Run 视图，再补充结果及诊断 | 统一读取层、类型视图、冲突规则与兼容回归均未实施 |
| 原子写与并发 | 文本已有公共 helper，引用 / checkpoint / 日志等仍管理各自部分写入生命周期 | 一个原子发布入口，流式追加与锁也由同一 I/O 模块封装；架构测试禁止业务直接写文件 | 调用点迁移、二进制 / 外部序列化适配、禁止绕行测试均未实施 |

上述是架构缺口，不表示已修复 Issue 又出现了新反例。#36 / #37 / #38 / #41 / #48 / #50 均保持 open，等待维护者决定 main；它们的已知反例已在 dev 修复。main 仍为已发布 v0.3.1 / `7a15c46`。

## 二、依据、归类与边界

### （一）已修事实与设计欠账

| **问题** | **真实问题与 dev 已有处理** | **归入的公共合同** |
| --- | --- | --- |
| [#36](https://github.com/DreamSoul-AI/RPipe/issues/36) | 初始冻结拒绝曾把旧 succeeded 改成 failed；现有 Runner 保全特定初始拒绝，相关调度门已有回归 | 尝试记录与结果视图分离；拒绝不改变旧证据 |
| [#41](https://github.com/DreamSoul-AI/RPipe/issues/41) | CLI 缺少 API 已支持的关闭重试选项，第二次错误可能成为最终错误；现有 `--no-retry` 仅作用于本次 launch，默认重试保留 | 每次重试独立记录首错 / 次错；选项不进入持久化全局状态。它不是所有失败覆盖的同一根因 |
| B-067 | 冻结采集 I/O / 结构错误曾绕过旧成功保全；现有 `ProvenanceChangedError` 边界保留 cause | 所有 admission 拒绝都记录本次事实，不靠异常类型决定是否覆盖旧结果 |
| [#37](https://github.com/DreamSoul-AI/RPipe/issues/37) | pending 的固定 / 基底 eval、inference 曾被误判 train；现有 fallback 读生成 config | 计划 config 是 mode、seed、身份的首要来源 |
| [#48](https://github.com/DreamSoul-AI/RPipe/issues/48) / B-069 | result / log 存在性查询曾在容错边界外，单个文件拒绝使整表失败；已移入原 try | 元数据、读取、解码与形状检查构成完整的单文件读取边界 |
| [#50](https://github.com/DreamSoul-AI/RPipe/issues/50) / B-070 | 提前 `dict(metrics)` 曾使非对象指标崩溃或误显示为数字；已复用 metric-cell guard | 有类型视图只接受有效字段，不强制转换畸形输入 |
| B-068 | 可选 activity 的编码 / stat 错误曾遮蔽正式状态；已隔离可选读取失败 | 可选提示与正式状态独立，提示不可提升成功等级 |
| [#38](https://github.com/DreamSoul-AI/RPipe/issues/38) | 初次部分文本写失败曾漏清临时文件；现有 finally 覆盖写入和替换 | 从创建 staging 到发布 / 失败清理的完整生命周期只定义一次 |
| B-062 | 同 tick 的直接共享引用发布临时名可能碰撞；已改短 UUID 并补失败交错回归 | 公共发布入口持有独占 staging，失败不碰其他写者的临时文件 |
| B-057 / B-064 | 文本临时名碰撞及拼接目标名导致合法长文件名失败；已修 | 同一入口统一随机短名、目标组件预算及有界 Windows 替换处理 |

历史证据见 [record §40–63](record.md)、[归档](maintenance_archive_2026-10-15.md) 与各 Issue / PR。B-062、B-067、B-068 经 [PR #46](https://github.com/DreamSoul-AI/RPipe/pull/46)，后续边界经 [PR #47](https://github.com/DreamSoul-AI/RPipe/pull/47)，B-069 经 [PR #49](https://github.com/DreamSoul-AI/RPipe/pull/49)，B-070 经 PR #51 集成。B-065 性能去重已撤回，不恢复。

### （二）不改变的边界

1. Run ID 仍由配置身份决定，独立科学复跑仍使用新 version / Run ID；attempt 仅记录同一 Run 的启动、拒绝、重试和恢复，不提供混合不同来源轨迹的许可。
2. `shared/data`、`shared/model` 仍只共享不可变输入；模型对象、checkpoint、tracker、sampler / RNG 等可变状态按 Run 和实际尝试隔离。
3. status / logs 是展示；调度、恢复、process、数值门继续严格验证输入。不得把安全读取的 `unreadable` 转为“没有结果，所以可以重新训练”。
4. 保留全部历史成功、失败、取消和原字节，不自动清理旧 Run / 账本 / 诊断。当前实现不回退；旧补丁只有在对应公共合同和原反例全部通过后才替换。

## 三、第一类：尝试账本与结果视图

### （一）统一目录与数据职责

拟在 artifact 内增加尝试记录职责，不新建数据库或外部调度器。目录草案：

```text
runs/<run_id>/
  config.yaml                     # 生成配置，身份仍使用当前 CONFIG_NAME 合同
  result.json                     # 可重建结果视图，不是唯一证据
  attempts/<attempt_id>/
    start.json                    # 排他创建、不可变：操作、Run、来源、配置 hash、父尝试
    result.json                   # 本次有效结果快照；未到 write 阶段时可不存在
    outcome.json                  # 至多一次：成功 / 失败 / 拒绝 / 中断及原始错误引用
    assets/                       # 本次可变输出，包含日志、tracker、checkpoint
```

`attempt_id` 使用随机短身份；不使用纳秒时钟作为唯一性或先后顺序。复用现有 shared.cache_lock 的本机文件锁机制建立统一 Run 写者租约；当前还没有覆盖这些入口的统一 Run 租约。锁内分配递增执行序号、预留目录并提交记录。并发 launch / 直接 run-one 必须在此处相遇：同一 Run 只有一个执行者；其他请求以独立请求身份追加 busy 拒绝记录，不能覆盖、等待后偷偷重复执行。busy 拒绝记录没有执行序号，不修改持锁者、执行指针或根视图；它的独立请求目录由公共 I/O 层排他创建，待展示时作为请求诊断读取。若记录也无法写入则 stderr 明示。

`start.json` 和 `outcome.json` 是同一次尝试的开始 / 终态记录，各自只写一次；并非反复覆盖一个 attempts.json 大字典。开始记录使进程在终态之前消失仍有证据；终态由拥有租约的执行者提交。恢复者取得锁后方可对未终结尝试追加 interrupted 终态，不能只凭 PID、时间或 activity.json 判断原执行者已死亡。拒绝请求只能写自己的记录，不能终结持锁者。

记录至少包含格式版本、Run ID、attempt ID / 顺序、命令种类、阶段、来源 / 配置摘要、父重试 ID、错误类型和脱敏摘要、结果相对引用及 hash。日志保留原始堆栈；JSON 不嵌入配置全文、凭据或无限异常对象。未知 / 未能采集的来源显式标未知，不伪造旧身份。

attempts、计划 generation、结果视图和运行资产必须从来源 / 计划文件采集排除；写诊断不能让下一次冻结检查产生自我变化。config / 实验 version 的 hash 算法和历史来源清单原字节保持不变。

没有 Run 身份的 Study make / launch 总门拒绝，写 Study 级 `attempts/<id>/`，范围为 study，不给矩阵所有 Run 虚构一次执行。单个 Run 的 admission 拒绝写本 Run 尝试。make 只记录计划展开操作，不制造训练结果。

### （二）`result.json` 的含义与兼容投影

账本里的本次结果快照是事实源；Run 根 `result.json` 只指向最近有效结果，损坏或丢失后可由明确的修复命令重建，status 不负责修复。

为兼容当前直接读取 `status / control / metrics / paths` 的调用方，第一阶段采用带引用的物化视图：保留这些旧字段，新增独立格式标记、`result_ref`、结果 hash、`latest_attempt_ref`。旧字段只是所引用不可变快照的投影，不再独立接受任意写入。新 resolver 校验投影与引用一致；不一致时严格调用方拒绝，展示侧仅标不可用。未知格式版本拒绝，不猜旧格式。

“最近有效结果”与“最近一次尝试”分开：

| **情况** | **账本** | **结果视图 / 当前执行状态** |
| --- | --- | --- |
| 首次执行失败，没有旧成功 | 追加失败及原始原因 | 可投影有效 failed 终态，不伪造 metrics 或 succeeded |
| 已有成功，下一次 admission 拒绝或执行失败 | 新增 rejected / failed，旧快照和资产保留 | 旧成功仍可查；最新尝试明确失败，不能只展示旧 succeeded 作为本次通过 |
| 允许重试 | 每次各一条开始与终态，并引用前次 | 首错、次错都在；CLI 默认只汇总，不用第二次错误替换第一条 |
| 成功定稿 | 结果快照、终态先发布，视图后更新 | 视图引用该次有效结果，执行状态与所要求阶段分别判定 |
| 子进程非零退出或 spawn 失败 | 父调度器核对本次 attempt 后补自己的故障事实，避免双重终结 | 不再用 `_fail_result` 重写历史成功 |

调度 `run_succeeded` 不再只看投影 status 和 result 文件 mtime，而核对当前计划身份、所需阶段终态、train→eval 依赖引用。eval 记录实际采用的 train attempt / checkpoint hash；新的 train 尝试已 admission 后，旧 eval 保留但标 stale，不再把旧文件写 failed。admission 拒绝不制造新的训练版本，也不覆盖旧有效训练；本次命令仍失败退出。合法成功、最新尝试失败、依赖已过期必须同时可表达。

阶段职责需要一起调整：`flow.write` 发布本次不可变结果快照；`process` 从显式快照引用读取，不依赖提前覆盖根视图；Runner 提交终态并刷新视图。现有 process 失败保留已完成计算结果的合同继续保留：计算结果可有效、process 阶段和本次命令仍失败，两者分别记录，不写成完整 Flow 通过。缺失所要求阶段的部分运行不能定稿成完整成功。

### （三）崩溃、并发与失败边界

1. start、结果快照、outcome、根视图不是一个多文件原子事务。每一步都可故障；以已完整提交的不可变记录为事实源，先记录后投影视图，绝不先把 succeeded 写到根文件。
2. outcome 已落盘但视图更新失败，命令明确报“证据已提交、视图待修复”，保留原始异常；不能再次执行计算来修复视图。只读层发现该状态给诊断，严格消费者暂拒绝调度 / 聚合，显式修复仅重建视图。
3. result 写完而 outcome 缺失，不得算完整成功；取得租约的恢复流程先记录 interrupted / 待核验，再按 checkpoint 原合同决定能否另开恢复尝试。缺少 sampler / RNG 能力不能宣称精确续跑。
4. 磁盘满或目录无权限导致开始记录不能提交时，禁止进入计算；终态记录也失败时保留最初业务异常，在 stderr 明示记录失败。不得递归生成“记录失败的记录”或吞掉原错误。
5. 所有日志 / checkpoint 路径绑定本次 attempt。父子进程通过显式 attempt ID 和所有权验证共用同一尝试，不能 launch、run-one、Runner 各建一个无关联尝试。不同 launch 无权重用该 ID。

### （四）替换现有局部补丁

| **当前落点** | **迁移后职责** |
| --- | --- |
| `FlowRunner._write_failed_result` 的 preserve_success 分支 | 统一提交失败终态；删除“某阶段 / 某异常才不覆盖”的决策，日志和原异常传播保留 |
| `check_frozen` 的 B-067 错误归类 | 继续表达冻结拒绝和 cause；不再承担证据保全开关的职责，不删来源校验 |
| `schedule._fail_result`、rerun / sibling eval 失效写回 | 追加本次故障 / 依赖状态；由严格资格判断排除 stale，不改旧结果 |
| `flow.write` / `write_result` | 内部转向本次快照发布；兼容 API 在明确 Run 上建立事务，不能静默绕过账本覆盖根视图 |
| `launch --no-retry` | 保留当前每次调用作用域和默认行为；只决定是否增加下一次尝试，不改磁盘配置 |
| `make` / CLI admission | 写操作尝试与诊断；拒绝时不替换已有 index、provenance、Run config 或结果 |

## 四、第二类：统一安全读取与有类型 Run 视图

### （一）单文件仅三态

在 `artifact/readout/` 内收敛安全读取职责；使用标准库类型，不引入验证框架。底层读取接收路径和明确的解码 / 形状检查器，返回一个带判别状态的对象：

| **状态** | **条件** | **返回内容** |
| --- | --- | --- |
| `available`（可用） | 文件读取、解码和最低结构校验成功 | 有效值、路径；结果引用可带来源身份 |
| `missing`（缺失） | 打开时确认目标不存在 | 空值及路径；不是执行失败 |
| `unreadable`（读不了） | 不是预期文件、权限 / I/O 错误、编码 / JSON / YAML 错误、根结构非法 | 空值、脱敏原因码、异常类和路径 |

不用 `exists / is_file` 的成功作为下一次 read 的保证；元数据检查如需存在，须和打开 / 解码处在同一错误边界。只捕获已列明的文件 / 解码 / 验证错误，意外 RuntimeError 等编程缺陷继续暴露。坏编码替换是否允许由日志文本解码器明确决定；JSON / config 不用替换字符伪造成功解析。

三态描述文件是否可消费，不把 succeeded / failed / pending 混成文件状态；metrics 子字段非法也不把整个合法结果都降成缺失。`available` 表示可形成部分视图，不表示它已通过科学或调度校验。

### （二）计划优先的有类型视图

每个 Run 先以 index 的身份 / 引用定位计划 config，生成最小 `RunView`：id、seed、mode、factors、执行状态、结果状态、metric、note、诊断与引用。字段校验独立：metrics 只接受对象里的有限数值，不把 bool 当数值、不把列表对强转字典；不可用单元格为 `-`。

身份、seed、mode 优先取生成 config，并和 index 约束核对。result 只补结果状态、指标、资产与错误；不能悄悄改变计划 mode。config 可读而与 index / result 冲突时标 identity_conflict，受影响格为 `-`，严格消费者拒绝，不选一个值掩盖冲突。

历史只有 index / result 而缺 config 时，兼容适配器可用已知的轴 mode 或合法 result.control.algorithm.mode，标明来源是 legacy；都未知则 mode 为 unknown，显示 `-`，不默认 train。只在 config 可用且其合法默认规则允许时推导默认 train。不得猜测损坏 config 的值。已知 mode 过滤保持；未知 mode 不混入指定 train / eval / inference，未过滤视图仍列出并诊断，避免无声消失。

最新尝试与最近有效结果分别填格；在 UI 中旧成功不得遮住最新失败。result / note / activity 读不了时，仅相关格 `-` 并带原因，正常 peer、合法 mode 和 seed 仍可见；读不了不能变成 failed 训练或成功训练。

### （三）status / logs 与严格读取的分工

- status、logs、activity 使用同一三态读取层和错误原因表达；删掉各自散落的 is_file / try / 强转补丁。三态对象不承担日志内容正则筛选、表格排版或算法配置建构。
- logs 合并可用文件；不可用 Run 返回结构化诊断，CLI 把诊断写 stderr 或专门诊断区，不伪造训练日志行。既有合法日志行和顺序保持；部分不可用的展示不使整张表消失。
- Study index 缺失 / 根结构不可用导致无法确定计划集合时，保留明确 Study 级错误与退出 2；只有某个可定位 Run 坏时隔离该行。不能把坏 index 解释为空矩阵。有效 activity 仍只是提示，不替代正式 index。
- `load_config`、`load_result` 的严格公共合同不因展示而吞异常。process / compare / 调度通过严格 resolver 复用解码和身份规则，在必需文件不可用时拒绝或报告 partial；不把 UI 的 `-` 当计算输入。
- “有类型视图”只为读出建立 dataclass / 明确字段，不实例化 Data / Model / System，不启动子进程 / GPU，不写回或自动修复任何文件。

### （四）旧补丁如何退出

#37 的多源 `_mode` 改为计划优先解析与显式 legacy 分支；#48 的三个 I/O try 统一委托三态读取；#50 的 metric guard 保留为字段验证规则并移入 RunView 构建处；B-068 的可选活动读取委托同层。删除重复捕获前保留原有 native CLI 红例，另加 config / result 冲突、未知 mode 和 peer 保全用例。异常类型归类可以合并，原错误信息和退出边界不可消失。

## 五、第三类：一个原子发布入口及受控写入

### （一）公共入口与所有权

以现有 `artifact/_atomic.py` 为唯一物理写入模块，扩展为一个原子发布实现；示意接口 `atomic_write(target, producer, *, replace)`。producer 接收该实现独占创建的 staging 路径或句柄，负责文本 / 二进制 / 第三方序列化，不能直接拿正式 target 写入。`atomic_write_text` 等原有 API 只作兼容适配，不再各自实现 staging / replace。

公共实现负责同目录 staging、短随机名、排他预留、完整关闭后发布、现有有界 Windows sharing-error 重试、失败保留原 target 及清理本写者 staging。唯一随机名仍须独占创建，不能把 UUID 等同于可忽略碰撞；冲突只重分配本次临时名。`replace=False` 负责不可变记录的禁止覆盖，不得用“先 exists 后 replace”假装排他发布；由同一 Run / 资源锁内的预留和提交合同保证，并测试重复提交拒绝。`replace=True` 只用于明确定义的可变视图 / 指针。

本地支持文件系统内的文件可见性原子性不等于断电持久性，也不等于多个文件 / 目录的事务。跨卷操作先拒绝，不自动退化成覆盖拷贝；不承诺网络文件系统、任意 Windows 长路径或全机锁。路径校验、同目录原则和必要 flush 的责任在公共层，实际支持边界由跨平台正确性测试确认。

### （二）日志、锁、外部序列化不是绕行许可

“禁止别处直接写文件”覆盖库内业务源码和仓库维护的 Study / 工具写入；统一物理层仍须提供两种不能伪装成整文件替换的操作：

1. 日志 / tracker 通过同模块的受控 append sink；保留追加、flush、partial-line 隔离和 jsonl_offset / keep_until 恢复语义。append 不宣称整文件原子，单 Run / attempt 写者规则和记录边界必须验证；不为统一而每次重写整份历史。
2. 锁文件的排他创建、文件锁及必要 marker 也封装在该模块；不能拿原子 replace 一个新的 inode 替代持锁文件。业务模块只请求持有锁，不直接 `open('a+b')`。

这两者不是第二个原子发布入口；它们是同一模块内显式区分的物理 I/O 合同。非写入的读取不受禁止规则影响。目录创建、链接和删除也经明确的生命周期方法管理，不允许业务层另造 staging 清理器。

`torch.save`、图生成、第三方下载 / 缓存适配器在公共入口授予的 staging 中输出，关闭 / 校验后由唯一入口发布；不得以第三方 API 为理由写正式 Run target。既有项目共享锁、manifest 的 ready 校验保留：原子替换不能代替资源构建锁或多文件完整性验证。

checkpoint 以已提交 bundle 为恢复依据，`.writing / .incomplete` 的诊断保全和旧目录读取合同保留。公共层通过显式的诊断保全策略接管这些 staging，不把它们当普通文本临时文件删除。checkpoint 组件镜像只能在 bundle 提交协议内更新；不能将多次原子 replace 宣称整个目录原子。未结束的历史诊断只读兼容，新尝试使用隔离目录，不覆盖旧尝试资产。

### （三）迁移调用清单

| **调用方** | **当前写入方式 / 职责** | **统一后** |
| --- | --- | --- |
| config、result、provenance、activity、report / compare、process JSON | 多数已调用 atomic_write_text | 保留序列化，委托唯一发布实现；result 另外遵守尝试账本 |
| `artifact/index.py`、`make/schedule.py` | index、jobs、Bash / PowerShell 脚本有直接文本写 | 由统一入口发布，保留编码、脚本 LF 与可执行属性约定 |
| `shared.py` | 锁、引用临时名、迁移 staging 与 atomic_replace 分散 | 锁 / 临时所有权 / link producer 委托统一模块，保留 hardlink / symlink / fallback 与源文件保全 |
| `system/factory.py` | torch.save、checkpoint 镜像、marker 和 bundle 发布 | 序列化只写受控 staging，恢复协议和诊断保全不变 |
| `system/logger.py`、`algorithm/tracker.py` | 初始化与流式追加 / state 快照 | 受控 append sink 与统一快照发布；不改变曲线计数和恢复偏移 |
| `data/prepare.py`、`data/profile.py` | marker / profile 直接写 | 统一发布；builder 第三方输出只能写构建 staging |
| Study 扩展、仓库脚本、图导出 | 当前调用和第三方保存接口需逐个列入迁移清单 | 明确属于正式产物还是隔离测试 fixture，正式写入统一委托；不得把 studies 整目录豁免 |

make 的 config / index / provenance / jobs 是多文件一致性问题，不能只把各 write_text 替换就宣称解决。拟先完成全量内存预校验与冻结拒绝，再写同一 plan generation 的不可变快照及 manifest，最后原子发布计划视图；消费方必须核对相同 generation。与已冻结 Run config 冲突时仍拒绝并要求新身份，不就地改旧配置。中途失败只留下未发布 generation，不让新旧计划混用；旧平铺文件通过明确兼容投影读取。具体布局在实施前同步 layout / make 合同。

### （四）架构测试禁止直接写入

建立仓库源码写入清单并收敛为唯一准入点，测试按 AST 和实际调用验证，不仅匹配 `write_text` 字符串：

1. 禁止物理层之外的 `Path.write_text / write_bytes / touch / replace / rename`、写模式 open / io.open / os.open、写 fd、shutil 拷贝 / 移动及已知第三方 save / savefig 等直接写正式路径。追踪导入别名和常用 Path 实例，不误禁 `str.replace`。
2. 兼容 wrapper 必须只委托物理层；每种第三方 serializer 的 producer 单独登记具体函数和 staging 参数约束，不给整个模块开放任意写 target 权限。原有未迁移点逐项归零后开启硬门，不留下无限期 wildcard allowlist。
3. 运行时故障注入验证 serializer 实际目标均在当前 staging / attempt 内，正式目录只能由公共入口发布。静态扫描无法证明任意动态 Python / 第三方内部行为，因此组合实际路径断言和 package smoke，不能声称 AST 是完整沙箱。
4. `tests/_data` 构建及 pytest 临时 fixture 可在明确的测试准备区写入；测试需能制造坏字节和部分文件，不能误把它们移除。测试仍须覆盖“业务回调直接写正式 target”会被架构门拒绝的负例。

## 六、兼容、迁移与交付顺序

### （一）格式与既有证据

本方案引入独立的 artifact 格式版本，不借用实验 config 的 version 字段。旧 schema 的读兼容保留；新 schema 的未知字段 / 版本有明确拒绝合同，不把 format version 加进旧 Run ID hash 或重签 provenance。

历史只有平铺 `result.json / assets` 的 Run：只读时通过 legacy adapter 展示，不自动创建 attempts。首次显式写入升级前，在持有 Run 租约且没有活动旧执行者时，把原 config / result / 相关可变资产逐字节复制为只写一次的 legacy 快照并核验 hash，保留原文件；没有原始日志就标来源未知，不推断成功次数。不能用硬链接保护未来仍会被原地写的可变文件。

无法确认静默、磁盘空间不足或快照失败时拒绝升级，不删除源、不继续新尝试；必要时让操作者使用新 version 的 fresh make。当前任务不执行此迁移。共享不可变大输入仅记引用，不跨 Run 复制数据。历史声明、checkpoint 恢复身份和 provenance 原字节不改。

新写入启用前，调度、读出、process、恢复、Study 扩展和正式数字导出必须一起适配；不能只发布 pointer 而让旧 train/eval 依赖判断照常跑。兼容投影照顾离线老读者，但老程序无法理解新 attempts / stale 语义，不支持新旧 writer 混用。回退先停止写入并保留整套账本，只读旧投影不能成为恢复旧 writer 的许可。

### （二）替换顺序与合入条件

1. 审阅本设计；确认 attempts 目录、结果视图、计划优先规则、统一写入范围。当前 [layout](../code/layout.md) 明确“不再增加 attempt 子目录”，因此本提案是有意的磁盘合同变更，不能默默当内部 refactor；审阅前不修改权威现行规范。
2. 审阅后先提交新格式 / legacy adapter / 失败边界的契约测试，再实现统一 I/O 与安全读取，并逐项迁移调用方。旧局部保护在相应红例变绿前保留，不能先删后补。
3. 接入尝试账本、父子单写者、计划 generation、依赖引用、process / resume 与视图重建，再启用新格式写入。每个阶段都明确旧格式读写能力和未启用开关，不让混合生产状态依赖隐式默认。
4. 原反例与新架构合同通过后，才删除 preserve_success 特例、散落 I/O try、重复 staging 和直接写入；不删除语义仍必要的冻结验证、字段 guard、日志、重试作用域和恢复诊断。
5. 工作分支经 PR 进入 dev，分别验收最终工作提交及实际 dev 全部实际 CI；main、发布、Issue 关闭另等维护者决定。当前只提交待审设计，不开实施 PR、不合入 dev / main。

应同步的正式规范是 concept / layout / structure / flow、README 接入、testing、受影响 Study 接口说明；本次不把它们改成已实现状态。这里给出目标合同，具体导出名称、最小 dataclass 字段与文件名在审阅后的实现 PR 绑定确认，不增配置框架或新依赖。

## 七、回归测试计划与验收

所有新实现先有能区分旧 / 新行为的测试；既有失败证据原样保留。使用隔离临时目录、独立 basetemp，无 GPU、外网、正式 Study 或速度比较。

| **测试组 / 拟沿用位置** | **关键断言** |
| --- | --- |
| artifact/result 与新 attempts 镜像测试 | 先成功后拒绝 / 失败 / 重试，每条记录只写一次；首错和次错可追；旧快照及资产原字节不变；根视图解析与引用 hash 一致 |
| `flow/test_freeze_preservation.py`、`test_capture_refusal.py` | 覆盖扩展开 / 关、来源 I/O / 结构拒绝、make / launch / direct run-one；拒绝无计算、作用域正确、旧成功与 peer 保全 |
| make/schedule 与 CLI 集成 | spawn 失败、子进程未定稿非零退出、重试关闭后下一调用默认恢复；同一 attempt 无双重终结；train 新代后旧 eval 保留但不合格；实际依赖引用替代 mtime |
| 新账本提交故障矩阵 | start、快照、outcome、视图各提交点前后故障；磁盘满 / 权限、进程 kill、视图滞后；有证据不重复计算、无完整终态不冒充完整成功 |
| readout/status / logs 与 activity | available / missing / unreadable、stat/read 竞态、坏编码 / 非对象、非法 metrics、config 身份冲突；单格回退、peer 不消失、输入和目录成员不变 |
| readout 的计划 / legacy 类型视图 | 固定 train / eval / inference、config 缺失但合法历史轴 / result、完全未知、无效 mode、过滤与来源诊断；未知不猜 train、不放宽科学 schema |
| atomic / shared_references | 固定随机碰撞、双线程 / 双进程发布与失败交错、replace / create-only、合法长 basename、真实编码失败 / 部分写 / KeyboardInterrupt；只清自己的普通 staging，旧 target / peer hash 不变 |
| checkpoint / tracker / logger | bundle 旧新选择、诊断 staging 保全、截断末行、恢复 offset 与 RNG / sampler 合同；原 CPU 微小计算数值门保留，不增加测速 |
| make 计划提交与架构约束 | 半写 generation 不可消费；旧 index / config / provenance 不混代；直接写入负例被拒，所有正式发布和 serializer staging 可核对 |
| legacy / 安装后离线 CLI | 旧 fixture 不迁移可读、显式升级复制保全、未知格式拒绝；Toy/Stub make→launch→process→status/logs/report，包含部分失败与依赖屏障 |

架构验收至少满足：没有绕行正式写入；所有已修反例继续通过；单文件不可用不使其他 Run 消失；新的失败无法覆盖旧证据；过期成功无法被调度当作本次通过；中断可诊断但不伪称精确恢复。正确性 CI 沿用 [cicd.md](cicd.md)：Ubuntu / Windows × Python 3.10 / 3.13 core 与 CPU Flow、两平台独立 wheel / sdist。skip、未执行与通过分别记录，不能用数量代替合同。

## 八、本次交付与待审事项

本次已完成现状核对、归类、统一结构、旧补丁替换清单、兼容 / 迁移方案与回归计划。**没有实现上述新架构，没有运行新实验或回写历史结果。** 原有 dev 修复继续有效，main 不变。

审阅重点是：attempt 是否作为正式证据层；兼容物化视图及最新尝试与有效结果双状态；计划 config 优先及 legacy 冲突规则；统一物理 I/O 对 append / checkpoint 的显式边界；旧格式升级和禁止混用 writer。以上均为本设计待审合同，不作为已经批准的实施任务。
