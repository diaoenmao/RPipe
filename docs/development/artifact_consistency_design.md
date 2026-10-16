# 产物证据、只读读取与原子写入收敛设计

## 一、摘要与硬约束

- **状态：已实施。** 三类已按第三类→第二类→第一类分别经 PR #52 / #54 / #55 合入 dev，后续同类修复继续按已批准合同推进，不再等待设计批准。 本修订依据维护者补充指令，取代 `c9e0141` 中的尝试账本、结果视图和格式迁移方案；旧稿只保留在 Git 历史中，不作为实施计划。
- **文件架构和产物布局不改。** 不增加文件、目录、模块或格式版本，不移动模块；`result.json` 的位置、格式、含义与现有读写合同保持不变。函数内的内存类型、判断和私有复用可以收敛，不引入持久化新结构。
- **实施范围：** 在现有函数和文件内归纳保全规则、读取边界及原子发布职责。先保留既有修复，再用统一规则替代重复判断，不恢复逐个相邻路径审计。
- **实施约束：** 每类先补合同回归；第一类仅提炼现有私有判断，不扩大旧成功保留范围。第三类架构测试仅覆盖可整体替换文本，特殊协议按第五（三）精确豁免。不测速、改性能、调查超时、运行 GPU、迁移数据或重签 provenance。

初始设计的历史基线为 dev `d294c1b5aa99b4d7148f991f9a4ac8903fe23374`。#50 已由 [PR #51](https://github.com/DreamSoul-AI/RPipe/pull/51) 合入，最终工作提交和实际 dev 的必需 CI 已通过；实际 dev 依据为 [Unit Tests](https://github.com/DreamSoul-AI/RPipe/actions/runs/38633933430) / [Package Check](https://github.com/DreamSoul-AI/RPipe/actions/runs/38633933432)。不重复开 #50 修复 PR。上述历史 CI 是初始基线证据；三类实施与后续修复的已合 PR、提交和验收见第十节及各补充节。

| **类别** | **初始分散处理** | **已实施的收敛方式** | **不采用的结构变更** |
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

这些已知反例均已在 dev 修复，不能重新登记为开放根因。B-062 / B-067 / B-068 经 [PR #46](https://github.com/DreamSoul-AI/RPipe/pull/46)，后续边界经 [PR #47](https://github.com/DreamSoul-AI/RPipe/pull/47)，B-069 经 [PR #49](https://github.com/DreamSoul-AI/RPipe/pull/49)，B-070 经 PR #51 集成。证据见 [record §40–65](record.md) 和 [维护归档](maintenance_archive_2026-10-15.md)。这六个 Issue 已随 [v0.4.0](https://github.com/DreamSoul-AI/RPipe/releases/tag/v0.4.0) 发布关闭；当前 main 为 v0.4.1 / `00e70b1`。后续已修 Issue 按各自版本关闭，不因本文重复处理。B-065 性能去重已撤回，不恢复。

## 三、第一类：现有写入入口统一证据保全规则

**已实施：** [PR #55](https://github.com/DreamSoul-AI/RPipe/pull/55)；后文为持续有效的合同，非待审任务。

### （一）结果与诊断的现有含义不变

`result.json` 继续是当前合同下的一份 Run 结果，字段仍由 `validate_result / encode_result / load_result` 管理；不改成指针、账本或“最近有效结果视图”，不添加 attempt ID / history / format_version 等字段。已完成 Run、失败和诊断仍留在原路径。

错误追加到现有 `assets/logs/run.log` 并输出终端；首个业务错误必须先记录，重试产生的第二个错误另追加，不覆盖原日志。不能把文本日志说成完整结构化尝试历史，也不能仅凭旧 succeeded 文件推断本次命令成功。

### （二）统一决策表

已在既有 `flow/runner.py` 提炼私有 `_preserve_success`，原表达式等价返回；`FlowRunner._write_failed_result`、`schedule._fail_result` 及调用方保持下表各自职责。不新建文件、移动业务模块或把语义塞进 `_atomic.py`；既有 `write_result` 仍明确校验并写入，不静默忽略调用方要求的写入。

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

**已实施：** [PR #54](https://github.com/DreamSoul-AI/RPipe/pull/54)；后文为持续有效的合同，非待审任务。

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

`status._read_optional / _note_from_log` 与 `logs.event_lines` 已采用同一 stat→read→decode 边界；共享读取已在既有 `readout/status.py` 的私有 `_read_optional` 中实现，`logs.py` 沿用现有依赖方向复用。`activity.read_activity` 在原函数内遵守同一规则，不能为了共用 helper 反向依赖 readout 或制造循环导入。

#48 的重复 try、#50 的字段 guard、B-068 的可选提示隔离，收敛的是规则与真正重复的代码；必要的字段验证不删除。`load_config / load_result` 的严格读取不改成全局吞错；process / compare / 调度不能把展示回退的 `-` 当真实输入。status / logs 不写盘、不自动修复、不清理文件，现有表头 / 输出字段和有效日志排序保留。

## 五、第三类：复用现有原子写函数

**已实施：** [PR #52](https://github.com/DreamSoul-AI/RPipe/pull/52)；后文为持续有效的合同，非待审任务。

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

原指令“其他地方绝不直接写任何文件”的字面绝对保证，涉及 append / torch.save / 文件锁时不能仅复用现有文本函数做到。这里明确收窄为替换类文本发布的硬门，其余保持现有协议并做正确性回归；不通过搬模块、新增 I/O 框架或删除功能冒充全仓统一。该边界已获本轮授权，不扩展为万能 I/O 框架。

## 六、兼容与无需迁移的交付顺序

1. **无需产物迁移。** 所有目录、文件名、schema、Run ID、version、结果含义和 provenance 原字节不变；不存在升级时创建 legacy 快照、重签、重排日志或复制正式数据的步骤。
2. **不改变公共返回合同。** 内存三态与类型检查止于现有读取函数边界；公共 API、CLI 表格和严格执行失败规则由原用例锁定。旧合法输入按同一路径继续读取，未知 / 非法输入只按已批准规则降级。
3. **实施顺序：第三类→第二类→第一类。** 在现有测试文件先补三组合同回归，再在对应原文件内收敛规则；每步只移除已被公共规则覆盖的重复判断。无匹配保证时保留保护，不先删后补。
4. **开发流不变。** 实施仍按工作分支→PR→dev，最终 head 和实际 dev 的全部实际 CI 分别验收；每类独立 PR，CI 通过后合 dev；后续正确性批次不合 main；已结束工作分支按 [cicd](cicd.md) 与当前明确授权清理，不删除数据。

三类已实施合同已同步至 [layout](../code/layout.md)、[structure](../code/structure.md)、[flow](../code/flow.md)；本文件记录已批准并落地的规则，不是待审或暂停实施的入口。文件架构和产物布局继续不变，第七节限制保持。

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

测试围绕固定合同归组；后续真实反例先登记 Issue 并补回归，不把编号当作独立设计或设计批准等待项。位置沿用现有测试文件；隔离临时目录、独立 basetemp，无 GPU / 外网 / 正式 Study / 性能测量。

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

**已实施。** 本设计已按“不改文件架构和磁盘合同”完成三类收敛；`c9e0141` 的结构方案已撤回。PR #52 / #54 / #55 及后续 PR 是实际实施记录；旧成功保留范围、特殊协议和第七节不实施约束继续有效。后续正确性修复直接按已批准合同和失败回归推进，状态以确切提交、实际 dev CI 与开发记录为准，不重新等待设计批准。

## 十、实施进度

**三类均已实施并合入 dev，最终 head 与实际 dev 的全部实际 CI 已验收。** 完整改前/改后和旧失败见 [record §69–75](record.md)，历史候选或待验文字不表示当前待审。

| **类别** | **已合 PR** | **实际 dev merge commit** |
| --- | --- | --- |
| 第三类：原子文本发布 | [PR #52](https://github.com/DreamSoul-AI/RPipe/pull/52) | `ea2210bc27e5456062bdef0ffb331e208c64f838` |
| 第二类：三态安全读取 | [PR #54](https://github.com/DreamSoul-AI/RPipe/pull/54) | `945c9925c8657b50a7f0906c5cd4fd00b238f1a1` |
| 第一类：等价保全 predicate | [PR #55](https://github.com/DreamSoul-AI/RPipe/pull/55) | `34e95069a3a2814d0fabf4879b1584f1bdbd0758` |

第一类仅提炼原判断，42格合同与相邻128项锁定原成功保留范围；分支规则随 PR #55 合入。

特殊追加、checkpoint、锁协议保持，第七节能力未实施；没有布局、schema或 result 含义变化，未测速或运行 GPU。上述阶段先合 dev，后按 v0.4.0 / v0.4.1 发布流程进入 main；当前后续正确性工作仍只合 dev，本轮不修改 main。


## 十一、2026-10-15 全仓同类正确性接续

按最新授权继续三类系统核对。目录与磁盘合同唯一依据仍为 layout.md；不恢复结构性账本或改变已有成功保全范围。#53 原 index 反例已修且随 v0.4.0 发布，本轮先重验原回归，再补齐此前仅覆盖库包而漏掉的 Study 发布边界。

第三类本批：受跟踪的 Study recipe / flow / structure 中，可整体替换的报告、数据/源码 manifest 和 probe JSON 全部复用已有 atomic_write_text。main_probe 固定 `.writing` 是可复现的 peer 所有权缺陷（B-072）；测试先失败再修。数据下载的二进制 partial、原始 Git blob、OBSERVATIONS.pt、模型/checkpoint 不纳入文本转码，直接字节合同保持。precision.main.execute 的 worker.log 是 subprocess 实时写入流，按函数精确豁免，不作为整体替换报告。

只在隔离合成输入上运行回归；不会触发性能入口、下载数据或计算循环，也不修改旧 provenance。更改源码的后续研究须使用新 version / make 并重新过原门。第二类与第一类核对采用已有三态/保全判断，确认新反例后再补合同，不为制造工作扩大异常吞咽或保留范围。

第二类本批（B-073 / #62）：status 与数字表复用 report.py 既有私有 _fmt，统一有限数显示边界；浮点转换溢出/非有限值显示占位，status 继续寻找可用 fallback。report 对 metrics 先确认 dict，不把列表对强转为对象。只改显示，不放宽严格读取、计划一致性、compare 或 process 科学门；无新增函数层/文件/schema。22项合同先红20失败/2通过，正常peer和原字节由回归锁定。

第一类本批只补合同：六阶段原异常 × 失败结果编码/提交再失败12项，验证原异常身份、原错日志、旧result/peer和本写者清理，与原42项决策表共同锁定既有predicate。原规则全部通过，无新增生产反例，不改Runner/schedule/make的失效语义。全仓归纳及严格compare/process、流式/二进制/锁豁免边界见record §81；不把单文件原子等同多文件事务或任意同Run并发。

本轮第三类PR61与第二类PR63均已普通合dev，最终head和实际dev全部实际CI分别成功；#53/#62 后续已随 [v0.4.1](https://github.com/DreamSoul-AI/RPipe/releases/tag/v0.4.1) 发布关闭，旧 v0.4.0 六 Issue 不重开。第一类新增合同全部通过，生产规则不改。具体源码/红绿/验收链接见record §79—81；main与layout不变。

## 十二、2026-10-15 逐点清单后的三类边界补齐

以record §83全仓清单为准，所有固定产物位置/schema继续不变。第三类先修B-074：短随机名仍须排他取得所有权，失败不unlink未创建的peer；atomic_write_text先排他创建并记录owned，shared引用去掉递归晋升，软链→硬链创建与原子replace仍在原函数，迁移仅成功link后清自己。第二类B-076维持compare严格错误语义：坏容器拒绝，不降级成空；合法null兼容，整数不用float丢精度，有限容差按数学值比较，普通浮点比较逻辑不变；输出在CLI既有try内。第一类B-075只改日志/曲线构造器的append创建，目录错误与恢复offset不变，不扩大Runner旧成功范围。

三类分别先红合同、主题 PR 与最终/实际 dev CI；该阶段 Issue 后续已按 v0.4.1 发布关闭，后续批次按各自发版状态处理。无性能/GPU/原始结果覆盖，不增加账本、锁、目录、模块或第七节能力。

本节三类补齐已按PR69（第三类）、PR70（第二类）、PR71（第一类）普通合dev，最终head与实际dev全实际CPU/包CI分别成功；完整403组清单与验收见record §83—87。三类只在既有入口内部收敛，43项新增合同覆盖原反例和有效控制；既有布局/schema、专用协议与第七节不实施边界保持，该阶段 GitHub Issue 已随 v0.4.1 关闭，后续已修 Issue 仍在其下一版关闭。


## 既有文本表格的单元格合同补充（2026-10-16，B-086 / #97）

合法 factor 的 key/value 和 report metric key 允许多行文本，不应通过限制配置或修改持久化字节修显示问题。现有 status TSV 和 NUMBERS Markdown 是两种视图；格式化时统一将单元格内的 Python splitlines 行界及 tab 转为显示空格，保持普通空格与数字原样。Markdown 再先转义原反斜线、后转义 pipe，避免偶数反斜线让分隔符重新生效。只在现有 report.py 放私有纯文本函数，status 复用（已有 status→report formatter 依赖），不增加模块或输出格式。

status 对标题及全部9个单元格统一应用单行合同，替换 error/note 仅处理 LF/tab 的重复局部逻辑；report 对 factor、metric name/text、seed、Run ID 应用 Markdown 单元格合同，标题仅保持单行。日志链接仍复用 B-085 的 URI 规则，数字格式器与指标白名单不变。只写派生 NUMBERS 的原入口不变，reader 不改 index/process/result、peer 和人工结论。

验收先提交公共渲染红测试：factor key/value、report 自定义 metric、CR/LF/CRLF、tab、NEL/Unicode LS、pipe 与一/二反斜线 pipe及普通控制；校验固定行列、原数值/peer/链接/计数和输入精确字节。初轮误将自定义 metric 纳入 status 既有白名单的10测试假设已撤出，原始证据留存；更正后 56 项 39 failed / 17 passed、无 error / skip 是修复前基线，旧失败保留。**已实施：** [PR #98](https://github.com/DreamSoul-AI/RPipe/pull/98) 已普通合 dev `8dc6a914a6f0e0f42a2353f43e73505880672087`，最终 head 与实际 dev 全部实际 CI 已通过，见 [record §114](record.md)。此补充不包含通用富文本清洗、HTML框架、配置schema扩展或本设计第七节结构性能力。


### compare checkpoint 引用与显示合同（B-088 / #101）

layout.md已规定checkpoint为每Run的可变状态，compare的参数名称应指向该Run的assets/checkpoints。保留原regular-file/missing/目录/IO预检，再在既有_checkpoint内检查已解析的checkpoint根在已解析Run内、已解析文件在该根内，之后才调用torch.load。合法nested、自身绝对、内部文件/目录别名继续可用；缺失可选checkpoint仍按原missing，读取/程序异常不扩大捕获。保持原生IO路径，只对归属比较复用已有_log_path_key的Windowsnamespace键，不复制路径框架/移模块。

format_compare复用现有_text_cell，仅把动态key显示成单行；JSON原key、比较数值/容差、缺失判定和原输出位置不变。训练checkpoint保存/恢复与strict pickle语义不受影响；不建目录、搬产物或自动迁数据。错误外部引用由CLI既有ValueError边界返回2，旧--out不覆盖。

先Issue101，隔离实际三种外部引用可使完整比较错误true、正常不同权重control false；Unicode LS合法stem使显示5行而非4。再冻结原字节17节点10 failed/7 passed、0error/skip，包含API/CLI父路径/绝对/文件alias/根alias、合法nested/自身abs/内部alias/missing/普通名称/Windowsextendedown控制。原始快照与原始REPRO/REPRO2/导入path/hash/JUnit留.tmp/audit088（不随clone）；正式红测试先提交再改代码。record§111的compare key固定判断不完整，以本轮真实证据明确更正，不把旧结论当新验收。

**已实施：** B-088 / #101 经 [PR #102](https://github.com/DreamSoul-AI/RPipe/pull/102) 普通合 dev `ef6084681160a68588e0b2a359ab7cc07c398879`；checkpoint 归属预检与 compare 单行显示合同均已落地，最终 head 和实际 dev 全部实际 CI 已通过。上述 17 节点 10 failed / 7 passed 明确为修复前证据，不再表示未修或等待批准；后续验收见 [record](record.md)。第七节能力仍不实施，不修改文件架构与产物布局。


### 混合整数数值的比较精度（B-100 / #127）

dev8c9cb99上，整数2**60+1与实际可表示浮点float(2**60)差1；metrics/history/checkpoint标量和整数-实数/复数tensor共五入口、两向十个公开API反例均错误通过并显示差0。不是#122同类整数张量容差或#124有限运算溢出；这里在减法/张量提升前已经丢失整数位。统一在现有compare边界保全表示值，不改生产者/格式/容差定义。

Issue127先开，031571a先提交120个公开API+CLI/输出JSON回归：五入口×正负×两向，另unsigned实数/复数两入口正向值×两向；各覆盖零容差、绝对界外/界上、相对界外/界上。实际72失败/48有效边界控制通过，0error/skip；每项均先执行API和CLI再断言，输入与peer原字节保持。红XML/log、原REPRO/原源码ZIP与SHA在.tmp/audit100，不随clone。

实施方案：整数/整数既有Fraction分支保持；现有_close对混合int/float先处理唯一浮点的非有限身份，再用现有Fraction按表示值计算差及绝对+相对界，输出可表示差转float、超范围仍inf/原JSON null。现有有限complex精确分支保留平方范数判据，用精确实/虚差分量计算显示norm，避免先把整数转complex后报告差0。Python complex标量原严格比较不扩展。

tensor保留原CPU输入再转现有float64/complex128供普通格和非有限身份处理。只在一个输入为非bool整数、另一个为float/complex时，将该有限格与原溢出格一并复用_close；从原输入item取值，同时写回passed格和显示diff。纯float/complex、纯整数、bool、shape/empty、配对NaN/inf、checkpoint归属及严格读写合同保持。无需helper/模块/依赖/新schema，不改变Run产物/科学门；本批只CPU合同，不测速/GPU。


### Runner日志失败留证（B-101 / #130）

第一类相邻复查确认：Logger构造/首条start info在run既有try之外，日志路径为目录或IO被拒时，即便result可写仍不进入failed回写；旧成功保留成误导状态。_log_failure中的_ensure_logger也在诊断try之外，阶段原错可被二次日志初始化错误替换并阻断结果留证。先开Issue130，再以8a351df提交10项正式红回归，10 failed/0error/skip；六启动节点（初始化拒绝/首条输出拒绝/真实目录×新旧result）与四诊断节点（collect/process×新旧result）均实际执行。

实施方案：将原构造与首条输出移入run既有try，current仍先None；_log_failure对_ensure_logger再失败直接返回，让外层_write_failed_result和原bare raise继续。既有日志exception→error fallback、failure_logged去重、_preserve_success初始冻结拒绝/process规则、result/schema/路径、append日志与最终succeeded输出时机保持。不引入新helper/模块/logger框架，不改文件架构和产物布局。

原草稿六节点被夹具重复mkdir阻断的证据保留且排除；更正草稿10红与隔离候选10绿只供方案验证，不替代正式入库红/绿。正式RED.xml/log、修复前源码ZIP/hash和最初无效success夹具拒绝在.tmp/audit101（不随clone）；portable回归已入库。后续先验证现有42项保全决策表、12项结果写二次失败及Runner/execute/Logger相邻合同，再CPU Flow和最终head/actualdev全部实际CI；不跑GPU、性能或Study。


### 原生训练结束flush的异常优先级（B-102 / #132）

第一类继续复查：_run_supervised的finally无条件flush_state，训练迭代器先失败、tracker编码/原子发布再失败时，后者成为对外异常和result.error；前者仅留在traceback chain。execute外层best-effort flush无法还原已经被替换的异常。首个stub探针走_run_stub未执行迭代器，作为无效夹具保留；更正非stub零批次/零优化步CPU探针确认原错确已执行。

Issue132后，1e8c5d1先提交22项合同：unit12覆盖RuntimeError/KeyboardInterrupt原错、编码/发布二错、Logger可用/拒绝/缺省；Runner8覆盖两种原错×两种二错×新旧成功；另2项正常训练体完成后最终flush拒绝的控制。正式20 failed/2 passed、0error/skip。初轮两控制在默认周期checkpoint过早注入导致夹具失败，原始RED_INITIAL与说明保留，更正checkpoint_period=0只在终点注入；未放宽生产判据。

实施方案：仅把原finally拆成except BaseException与else。已有异常时在嵌套try尽力flush；二次普通Exception时使用现有Logger.exception尽力记traceback，Logger本身再失败不遮原错，最后原bare raise。成功体的else仍直接flush并传播错误。捕获BaseException仅为保留原finally在终止时的flush动作和原异常身份，原终止不转为成功或新failed；不吞新的终止信号。不用sys.exc_info或共享状态推断失败，不加helper/模块/格式/路径，不改数值、checkpoint/RNG或周期内flush。

相邻调用方已核对TrainAlgorithm.run/legacy run、execute的finally，以及eval_hook/Trainer回调；当前只在原生训练已有finally确认同类反例，其他路径无反例不修改。正式红、原始探针/拒绝、源码ZIP/hash在.tmp/audit102，不随clone。后续训练/恢复/checkpoint/Runner相邻、CPU core/Flow及最终head/实际dev全部实际CI分别验收后合dev、关闭Issue、普通清理本地/remote；无main/tag/GPU/测速或正式Study。
