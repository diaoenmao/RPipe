# 产物证据、只读读取与原子写入收敛设计

## 一、摘要与硬约束

- **状态：设计参考，未实施；不阻塞独立的正确性修复。** 本修订依据维护者补充指令，取代 `c9e0141` 中的尝试账本、结果视图和格式迁移方案；旧稿只保留在 Git 历史中，不作为实施计划。
- **文件架构和产物布局不改。** 不增加文件、目录、模块或格式版本，不移动模块；`result.json` 的位置、格式、含义与现有读写合同保持不变。函数内的内存类型、判断和私有复用可以收敛，不引入持久化新结构。
- **实施范围：** 在现有函数和文件内归纳保全规则、读取边界及原子发布职责。先保留既有修复，再用统一规则替代重复判断，不恢复逐个相邻路径审计。
- **当前只改文档。** 不动生产代码、测试或历史产物；不测速、改性能、调查超时、运行 GPU、迁移数据或重签 provenance。不自动实施设计中的行为调整；按最新授权继续有失败回归支撑的独立正确性修复。

基线为 dev `d294c1b5aa99b4d7148f991f9a4ac8903fe23374`。#50 已由 [PR #51](https://github.com/DreamSoul-AI/RPipe/pull/51) 合入，最终工作提交和实际 dev 的必需 CI 已通过；实际 dev 依据为 [Unit Tests](https://github.com/DreamSoul-AI/RPipe/actions/runs/38633933430) / [Package Check](https://github.com/DreamSoul-AI/RPipe/actions/runs/38633933432)。不重复开 #50 修复 PR。这些 CI 只证明已有修复，不证明本设计已实施。

| **类别** | **当前分散处理** | **本修订的收敛方式** | **不采用的结构变更** |
| --- | --- | --- | --- |
| 失败写入与证据保全 | Runner 特定拒绝保留成功；调度失效、阶段失败和重试各有职责 | 在现有失败写入 / 调度入口统一“允许写什么、何时拒绝写”的规则，错误继续追加到原 run.log | 无 attempts 目录、不可变账本、结果指针或双版本 schema |
| status / logs 安全读取 | stat、打开、解码及类型 guard 分散 | 在现有读取函数内统一三态与字段校验；Run 行先由计划 config 建立内存中的有类型值，再补结果 | 无新读取模块、磁盘 RunView、字段迁移或新结果语义 |
| 原子写与并发 | 文本已有 helper，其他调用仍有重复 staging / 直接文本发布 | 复用 `_atomic.py` 的 `atomic_write_text` → `atomic_replace`，统一替换类文本发布及临时文件生命周期 | 不新造万能 I/O 层，不搬 logger / tracker / checkpoint，不增加锁目录 |

## 二、问题归类与已完成状态

| **问题** | **dev 已修事实** | **本设计要归纳的共同规则** |
| --- | --- | --- |
| [#36](https://github.com/DreamSoul-AI/RPipe/issues/36) | 初始冻结拒绝不覆盖旧成功 | 计算前拒绝与运行中失败分开；拒绝不写旧结果 |
| [#41](https://github.com/DreamSoul-AI/RPipe/issues/41) | `--no-retry` 投影现有 API，默认行为保留且不跨调用残留 | 首错先记录原日志；重试选项只属于本次 launch。此 Issue 本身不是所有覆盖问题的同一根因 |
| B-067 | 冻结采集 I/O / 结构错误统一为 `ProvenanceChangedError`，保留 cause | 冻结拒绝不依赖底层异常恰好是哪一类；不能因错误分类遗漏走错失败写入 |
| [#37](https://github.com/DreamSoul-AI/RPipe/issues/37) | 固定 / 基底 eval、inference 的 pending mode 可读生成 config | 计划值优先，缺失与损坏不混淆，不猜 train |
| [#48](https://github.com/DreamSoul-AI/RPipe/issues/48) / B-069 | 三处 stat 已纳入原有 I/O try | stat / read / decode 是同一文件的一个错误边界 |
| [#50](https://github.com/DreamSoul-AI/RPipe/issues/50) / B-070 | 不再提前 `dict(metrics)`，复用单格 guard | 非对象不强转；坏格不遮蔽 peer 或其他合法字段 |
| B-068 | activity 的 stat / 编码失败被隔离 | 可选提示不可用不遮蔽正式结果，不能代替 index |
| [#38](https://github.com/DreamSoul-AI/RPipe/issues/38) | 初次部分文本写入进入 finally 清理 | staging 创建、写入、发布、失败清理是一套完整规则 |
| B-062 | 共享引用使用短 UUID 临时名，补齐失败交错回归 | 每个发布者仅处理自己的 staging，保留旧 target 和 peer |
| B-057 / B-064 | 文本碰撞及合法长 basename 的 staging 问题已修 | 不再各造基于时钟或拼完整目标名的临时文件方案 |

这些已知反例均已在 dev 修复，不能重新登记为开放根因。B-062 / B-067 / B-068 经 [PR #46](https://github.com/DreamSoul-AI/RPipe/pull/46)，后续边界经 [PR #47](https://github.com/DreamSoul-AI/RPipe/pull/47)，B-069 经 [PR #49](https://github.com/DreamSoul-AI/RPipe/pull/49)，B-070 经 PR #51 集成。证据见 [record §40–65](record.md) 和 [维护归档](maintenance_archive_2026-10-15.md)。六个 Issue 保持 open；已发布 main 仍 v0.3.1 / `7a15c46`。B-065 性能去重已撤回，不恢复。

## 三、第一类：现有写入入口统一证据保全规则

### （一）结果与诊断的现有含义不变

`result.json` 继续是当前合同下的一份 Run 结果，字段仍由 `validate_result / encode_result / load_result` 管理；不改成指针、账本或“最近有效结果视图”，不添加 attempt ID / history / format_version 等字段。已完成 Run、失败和诊断仍留在原路径。

错误追加到现有 `assets/logs/run.log` 并输出终端；首个业务错误必须先记录，重试产生的第二个错误另追加，不覆盖原日志。不能把文本日志说成完整结构化尝试历史，也不能仅凭旧 succeeded 文件推断本次命令成功。

### （二）统一决策表

在 `FlowRunner._write_failed_result`、`schedule._fail_result` 与调用它们的现有函数内，按下表收敛。可在现有 `artifact/result/io.py` 内复用一个私有判断函数，但不新建文件、移动业务模块或把语义塞进 `_atomic.py`；既有 `write_result` 仍明确校验并写入，不静默忽略调用方要求的写入。

| **事件** | **result 处理** | **诊断与调用结果** |
| --- | --- | --- |
| 初始来源冻结拒绝，有可读旧成功 | 保留 result 原字节；计算、旧资产与 provenance 不动 | 原日志追加阶段 / 原异常，命令仍失败 |
| 初始拒绝，无旧成功，进入现有 Runner 失败定稿路径 | 按现行合同写 failed；CLI 总门原本不写结果的仍不新增 | 保留原异常及 cause，不虚构执行过阶段 |
| 普通执行阶段失败 | 维持原有 failed 定稿语义，不因“保全”改为保留 succeeded | 首错先记日志，退出仍失败；不宣称保留了历次结果快照 |
| process 失败，原计算成功已写入 | 维持现行 process 失败保留成功的规则 | process 的异常与命令失败仍可见，不称完整 Flow 通过 |
| 合法 rerun / train→eval 失效 | 继续现有失效写回与屏障；必须先过相关拒绝门 | 不把失效写成科学计算失败，不因旧成功跳过本次失败 |
| 旧 result 不可读或新结果无法编码 / 写入 | 有效诊断与原错误不得被二次异常遮蔽；本次文件写失败不得破坏 target | 在现有日志 / stderr 说明记录失败，不递归写错误、不拿 unreadable 当成功 |

本表统一现行边界，不扩大成“任何失败都保留旧成功”。`write_result` 里的语义校验、Runner 的阶段信息、freeze 的原始 cause、调度的返回码和依赖检查都保留。保全判断应使用同一次读取所得值，不能多个调用方重复读取后得出冲突结论。

### （三）替换旧局部判断

1. Runner 保留日志与异常传播，现有 `preserve_success` 对应的规则收成上述固定表；B-067 的错误归类继续作为来源拒绝合同，不取消冻结检查。
2. 调度入口在任何旧结果失效写入前完成现有来源 / 计划检查；实际允许的 rerun 继续按原合同失效，不能留下错误 succeeded 被下游使用。若发现新反例，先写回归，不用通用“保留成功”掩盖它。
3. `--no-retry` 与默认重试均保持当前作用域和次数；不把 retry_failed 写入 jobs / config，也不从上次 CLI 调用继承。
4. make 的配置 / 来源拒绝先验证再写，调用既有原子函数。单文件失败保全与多文件事务是不同保证；不新增 plan generation 或承诺 make 整体回滚。

## 四、第二类：现有读取函数内统一安全读取

### （一）三个内存状态

三态只在现有函数内部表示，可用简单的判别值与已解码对象，不持久化新文件，也不改变现有 API 返回字段。

| **读取状态** | **判定** | **展示处理** |
| --- | --- | --- |
| 可用 | 打开、解码和最低结构检查成功 | 使用合法字段；各子字段仍独立校验 |
| 缺失 | 打开时目标确实不存在 | 沿用计划信息与既有 pending / 缺失提示，不伪造 failed |
| 读不了 | 权限 / I/O、不是预期文件、坏编码、解析或根结构错误 | 该文件提供不了的格为 `-`，保留已有诊断位置与正常 peer |

`is_file` 如需使用，必须与打开 / 解码放在同一错误边界；存在性检查不是后续读取安全的保证。只捕获明确的文件 / 解码 / 结构异常，非预期 RuntimeError 等继续暴露。日志保留现行文本容错规则，JSON / config 不用替换字符假造可用值。

### （二）有类型值先来自计划，再补结果

`list_runs / _mode / _metric_cell` 内先由 index 定位现有生成 config，核对身份、seed、mode，形成仅在内存中使用的明确字段；不构建 Data / Model / System。结果再补状态、metrics、错误与资产引用，不能悄悄改写计划 mode。

- config 可用时，按其合法默认规则读 mode，并核对 index / result。冲突在已有 error / note 中诊断，受影响格为 `-`，不能默选一个值消除冲突；严格执行与聚合路径继续拒绝非法身份。
- 历史 config 缺失时，保留已知合法 axes / result.control 的兼容读取；来源都未知时沿用 unknown，不从损坏配置猜 train。此优先级是内存读出规则收敛，须以现有合法结果与过滤回归证明兼容，不修改磁盘格式。
- metrics 不做 `dict(...)` 强转；对象子字段通过原有数值 guard，非法格为 `-`。不得为统一读取而放宽科学 schema 或把 bool / 列表对变成数字。
- 一个 Run 的文件出错只影响该行相关格；config 仍能给出的 mode / seed 与 peer 继续显示。缺失或损坏的 Study index 不能当空矩阵，保留 Study 级明确失败。

### （三）函数内复用与旧补丁替换

`status._load_result / _note_from_log` 与 `logs.event_lines` 采用同一 stat→read→decode 边界；如果需要共享文本读取，可放在既有 `readout/status.py` 的私有函数中，`logs.py` 沿用现有依赖方向复用。`activity.read_activity` 在原函数内遵守同一规则，不能为了共用 helper 反向依赖 readout 或制造循环导入。

#48 的重复 try、#50 的字段 guard、B-068 的可选提示隔离，收敛的是规则与真正重复的代码；必要的字段验证不删除。`load_config / load_result` 的严格读取不改成全局吞错；process / compare / 调度不能把展示回退的 `-` 当真实输入。status / logs 不写盘、不自动修复、不清理文件，现有表头 / 输出字段和有效日志排序保留。

## 五、第三类：复用现有原子写函数

### （一）现有职责不搬动

继续使用 `_atomic.py` 的两层调用链：`atomic_write_text` 管文本 staging / 清理，`atomic_replace` 管最终替换和既有 Windows 有界重试。不另加万能 `atomic_write(producer)` 接口，不把 logger、tracker、锁、checkpoint 的业务实现搬入此模块。

替换类文本产物统一调用 `atomic_write_text`：config、result、provenance、activity、process、readout 导出已有调用保留；index、jobs、生成脚本、profile / marker 中可复用的直接文本发布，逐项改为现有入口。编码、换行、返回路径及脚本属性必须保持；若某处现有签名不能保持字节语义，只在原函数 / 同一 `_atomic.py` 内评估最小适配，未有回归不得直接替换。

### （二）原子与并发保全规则

1. 临时名使用已修的短随机方案，不拼完整目标名，不用时钟当写者身份。临时创建需避免占用其他写者文件；故障测试验证真实所有权，而不只检查命名风格。
2. 写入与替换均在 finally 的范围内；失败保留旧 target，只清理本写者的普通临时文件，peer staging / 源文件不动。KeyboardInterrupt、编码失败、替换拒绝都要覆盖。
3. shared 的引用 / 迁移保留本文件内逻辑和资源锁，最终发布复用 `atomic_replace`；不新建引用子目录、改变 hardlink / symlink / fallback 语义或扩大无锁多写者支持。B-062 的原并发反例必须继续通过。
4. logger / tracker 的原路径追加和恢复 offset 不改成整文件替换；checkpoint 的 bundle / 分件与 `.writing / .incomplete` 保全合同不改、不清理诊断。锁文件不能以替换 inode 的方式“原子更新”。
5. 单文件替换不是多文件事务、全机互斥或断电持久性保证。既有单 Run 单写者合同不因复用原子函数而扩大；不增加 Run 锁文件或靠旧 activity 猜独占。

### （三）架构回归的可执行边界

测试禁止业务路径绕开现有原子入口发布**可整体替换的文本产物**，覆盖导入别名、Path.write_text / 写模式 open 等明确模式，同时给真实写入注入失败检查 target 字节。不能只用 grep 字符串或误禁 `str.replace`。

日志追加、tracker、checkpoint 二进制、现有 shared 文件锁与受控第三方保存仍在原模块；逐个函数列明其不可直接替换的理由与已有发布保护，不能把整个目录豁免。测试须包含绕过 atomic 的负例，以及合法追加 / checkpoint 不被误判的控制例；隔离测试 fixture 必须仍能制造坏字节、部分写和故障。

原指令“其他地方绝不直接写任何文件”的字面绝对保证，涉及 append / torch.save / 文件锁时不能仅复用现有文本函数做到。这里明确收窄为替换类文本发布的硬门，其余保持现有协议并做正确性回归；不通过搬模块、新增 I/O 框架或删除功能冒充全仓统一。该差异待审，不自动扩展实施范围。

## 六、兼容与无需迁移的交付顺序

1. **无需产物迁移。** 所有目录、文件名、schema、Run ID、version、结果含义和 provenance 原字节不变；不存在升级时创建 legacy 快照、重签、重排日志或复制正式数据的步骤。
2. **不改变公共返回合同。** 内存三态与类型检查止于现有读取函数边界；公共 API、CLI 表格和严格执行失败规则由原用例锁定。旧合法输入按同一路径继续读取，未知 / 非法输入只按已批准规则降级。
3. **实施须有失败回归支撑。** 在现有测试文件先补对应合同回归，再在对应原文件内收敛规则；每步只移除已被公共规则覆盖的重复判断。无匹配保证时保留保护，不先删后补。
4. **开发流不变。** 实施仍按工作分支→PR→dev，最终 head 和实际 dev 的全部实际 CI 分别验收；本设计不触发整体重构；文档可经 PR 合 dev，随后继续逐项有红色回归支撑的正确性修复。不合 main、不删分支或数据。

正式代码说明需要在实施时同步相关行为，现有 [layout](../code/layout.md)、[structure](../code/structure.md)、[flow](../code/flow.md) 的文件架构继续有效。本轮不把待审设计写成已实现合同。

## 七、确实需要结构变化才能提供的能力（仅说明，不实施）

| **能力 / 旧稿方案** | **为什么现有合同不够** | **本轮处理** |
| --- | --- | --- |
| 保存每次尝试的完整机器可读结果和独立资产 | 一个保持现有含义的 result 文件与共享可变 assets 无法同时保存全部历史；仅追加日志不能替代快照 | 不建 attempts，不加字段，不承诺该能力；继续原错误日志保全和现行失败定稿 |
| result 仅作有效结果指针、按明确训练代引用 eval | 会改变结果含义、schema、依赖身份与读取方；不是函数内部去重 | 撤回实施方案；现有 train→eval 屏障和新 version 规则保留 |
| make 多文件全有或全无的持久化提交 | 逐文件原子替换不能保证 config / index / provenance / jobs 整体事务；需新提交标识、布局或格式协议 | 不建 generation / manifest；保留预验证和单文件保证，不宣称跨文件原子 |
| 任意两个进程同 Run 写入的全局互斥 | 原有单写者约束与 shared 资源锁不等于统一 Run 锁 | 不增锁文件、租约或调度器；不扩大并发支持范围 |
| 所有直接文件写入均归一个万能入口 | 日志追加、恢复偏移、锁 inode 和 checkpoint 诊断具有不同合同，强行迁移会改变职责或协议 | 不搬模块，不做框架；复用现有 atomic 链，按五（三）的明确边界测试 |

这些是限制说明，不是认定结构必须立即改变，也不是待自动执行的任务。确需上述保证时，另给具体反例和无法在现有合同内解决的理由，由维护者决定；当前不动手。

## 八、正确性回归计划

测试围绕固定合同归组，不继续以新编号逐个审计。位置沿用现有测试文件；隔离临时目录、独立 basetemp，无 GPU / 外网 / 正式 Study / 性能测量。

| **测试组** | **关键断言** |
| --- | --- |
| result、Runner 与冻结拒绝 | 旧成功 / 无旧成功 × 外层门 / prepare × 来源 I/O / 结构拒绝；拒绝不计算，旧字节与 peer 保全，已有 failed 定稿语义不变 |
| make / launch / run-one | 拒绝发生在失效写回前；正常 rerun 和 sibling eval 失效不被新保全规则绕过；spawn / 非零退出保留原诊断与退出语义 |
| CLI 重试作用域 | 默认重试与 no-retry 次数保持；首错、次错按序追加原日志；第二次独立 CLI 调用不继承上次选项，不改 jobs 字节 |
| status / logs / activity | 三态 × stat / read / decode；缺文件与坏文件分开；异常 RuntimeError 不吞，单格回退不遮蔽 peer，目录成员 / 文件字节不变 |
| 内存类型与 mode | config 优先、合法历史 fallback、冲突 / unknown、train / eval / inference 过滤；非法 metrics 不强转，已有合法行 / 表头兼容 |
| atomic / shared_references | 双线程 / 双进程失败交错、随机名冲突、长 basename、部分写 / 编码错误 / KeyboardInterrupt；旧 target / 源 / peer 不变，只清自身普通 staging |
| logger / tracker / checkpoint 控制 | 追加、partial-line、恢复 offset 与 bundle / 诊断选择保持，既有 CPU 微小数值回归继续通过 |
| 架构约束与安装后 CLI | 替换文本绕行负例被拒，特殊协议精确控制；离线 Toy/Stub make→launch→process→status/logs/report；产物路径和合法 JSON 结构与原合同一致 |

正确性验收沿用 [testing.md](testing.md)、[tests/README](../../tests/README.md)、[cicd.md](cicd.md)：Ubuntu / Windows × Python 3.10 / 3.13 core 与 CPU Flow，以及两平台 wheel / sdist 安装。skip、未执行和通过分开记录；旧 CI 不替代实施后的验收。

## 九、本次状态

设计已按“不改文件架构和磁盘合同”修订；`c9e0141` 的结构方案被本稿明确取代。当前仅提交文档，代码、测试、正式数据、Run、结果与来源均未变。第一类固定决策表、第二类读出优先级兼容，以及第三类替换文本硬门与特殊写入协议仍是设计参考，不能据此无红例地大改。维护工作继续，不以等待设计反馈阻塞新的可复现正确性修复。
