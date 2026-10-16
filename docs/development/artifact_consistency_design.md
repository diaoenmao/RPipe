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


## B103 / #134：共享锁路径别名的重入身份（2026-10-16）

已复现：Windows 普通 drive 路径与扩展命名空间路径经 resolve 后仍有不同字符串，samefile 确认同一目录；两个嵌套方向在 timeout=0 下均因同线程再次取得同一 OS 锁而失败。42c8b5b 先提交四项原生 Windows 回归，四项全部失败；覆盖两个方向及内层正常/异常退出，并要求其他线程仍被排除、外层退出后可重获。

最小方案：保留 resolve、mkdir 与全部 OS 锁和预算流程，只将线程局部重入表的字符串身份改为目录 stat 的 st_dev/st_ino 物理身份（与标准库 samefile 一致），同时保留 pid 隔离。原 root 仍用于全部文件操作和错误信息；不改变共享输入目录、锁文件、磁盘布局或持久化格式，不引入跨层 helper。调用方 prepared_resource、migrate_shared、data/model factory、data prepare 均复用这个共享根实现。验证既有进程/线程排斥、死亡持有者释放、取消与零预算合同；不运行 GPU、正式 Study 或测速。


## B104 / #136：多根迁移锁按同一物理身份排序（2026-10-16）

继 PR135 的同线程重入验收后，原生空目录探针发现 migrate_shared 的 key=str 排序受 Windows namespace 拼写影响。同一 Study 两种表示在 shared.root 为普通/扩展绝对路径时，可交换私有根与共享根的取得顺序。真实 OS 锁、两线程各持前两根后请求第三根，零等待预算注入使两侧立即暴露循环，四根退出后均可重获；不作测速或运行真实迁移。Issue136 后四项排序回归先提交，2 failed/2 controls passed。

方案：在 shared.py 内将 PR135 已有 stat 二元身份提取为最小私有函数，cache_lock 的 pid/物理键保持；migrate_shared 仍只锁既有四个根，先确保这些原本就会 mkdir 的目录存在，再按同一 st_dev/st_ino 身份排序取得 OS 锁。排序不依赖 namespace/大小写或声明名称；同物理身份的重复根只重入，不改变数据迁移、冲突验证、硬链接、staging 或文件布局。跨目录并发更名不属于共享不可变输入根合同。保留既有失败、权限拒绝与取消行为，不增加全局互斥、模块、依赖或磁盘格式。


## B105 / #138：非零退出先留证，再输出诊断（2026-10-16）

在5762925上复现：子进程替身写出合法succeeded结果后返回1，schedule错误行stdout再抛BrokenPipeError，原实现先print导致既有失效写回完全未执行，run_succeeded仍真。没有真实子进程、训练或数据。Issue138后e599e4b先提交八项：BrokenPipe/Permission/UnicodeEncode三类×retry开关6 failed，正常输出两控制passed；peer与config原字节保持。

最小方案：仅把非零退出分支的print移到已有success→failed纠正之后。stdout原异常仍按原方式传播，不吞掉异常、不增加新logging策略；retry和codes、已有failed结果/无结果的处理、spawn语义、train/eval屏障和磁盘格式保持。不把诊断失败当成功；若纠正写入本身拒绝，原有严格写错误仍传播。相邻blocked依赖已先写failed再输出，announce/report/compare已先发布再输出，Runner日志二错已有B101/B102合同，不重复改动。当前设计已实施后按本机与最终/实际dev全部实际CI验收。


## B106 / #140：PNG单文件原子发布与自有figure释放（2026-10-16）

90078c8上已复现：有效旧PNG被savefig部分写覆盖，原OSError传播但新figure遗留。Issue140后5eef63d先提交19项C2回归：构造axes/绘制/部分保存/取消/替换/排他创建失败×新旧目标、UUID碰撞、清理拒绝、两项有效PNG控制及真实Windows只读目标；正式17 failed/2控制passed、0error/skip，首轮18项原红亦保留。测试放既有flow/process镜像目录的test_curves_publication.py，单独C2避免给原C1文件叠加两个cost marker，不新增生产模块。

实施方案沿用已批准第三类合同：只改curves.py，在原figsize下取得figure后立即进入finally所有权范围，再以等价fig.subplots取得axes；绘制主体原样缩进，不改数据、坐标/标注、dpi140或路径。短UUID的同目录.tmp用xb排他取得所有权，savefig向该文件句柄以显式PNG格式写入，关闭后复用既有atomic_replace发布。目标原mode沿用；新目标沿用普通open的umask。临时创建失败不unlink peer，自有文件才best-effort清理；PermissionError只对自有独立临时文件恢复0o600并重试一次，其余OS拒绝留诊断，不遮原保存/替换异常。最外finally始终只关闭本次取得的figure，peer figure保留。现有checkpoint诊断staging、共享硬链接和日志追加均不改；不加通用I/O框架、文件夹或schema。

验证：19新节点全部执行、旧图/缺目标/输入/peer bytes和mode，PNG有效控制及原绘图合同；本机另用冻结旧函数与新函数对相同合成输入比较PNG原字节。随后core、CPU Flow及最终head/实际dev全部实际CI分别验收。源ZIP/失败、原始PNG和哈希在.tmp/audit106，不随clone；无GPU、测速、正式Study或main/tag。


### B106同批补充：临时流关闭IO二错（2026-10-16）

9da02b4相邻故障注入确认：with临时流的__exit__/close再抛OSError，会替换savefig已经抛出的RuntimeError或KeyboardInterrupt。a095ef0先提交6项补充回归（两种首错×新旧目标4 failed，成功保存后close错误的2控制passed），与最初19项红分开保存；仍属Issue140/PR141，不另起主题。

仅把临时流上下文改为显式取得/关闭：保存已有BaseException时尽力close，只有关闭OSError不替换首错，随后bare raise；保存正常时close仍严格传播，未关闭成功不发布。捕获BaseException只为取消也释放资源且原样重抛，不吞新的终止信号或关闭中的非IO程序错误。PNG绘制/权限/所有权规则不变。原head9da02b4的CI快照保留，新代码应更新同一PR并重新接受最终head的实际CI，不能沿用旧head绿作为验收。


## B107 / #142：共享锁释放与关闭的首错保全（2026-10-16）

实际OS锁取得后注入unlock/close IO，准备RuntimeError或KeyboardInterrupt均被OSError替换；不同线程随后仍能零预算重获。Issue142后7614d31先提交25项正式回归：19 failed/6有效控制passed，0error/skip。覆盖两种/同时清理故障、首错/取消/正常体、重入、获取busy/坏descriptor后close、prepared_resource诊断staging及非IO程序错误；未假授锁，无训练/测速。

最小方案只改既有shared.py的cache_lock：复用已导入的stdlib ExitStack，在文件取得后登记close退出回调，取得OS锁后再登记release回调；逆序先release、后close。回调只在已有exc_type时不让自身OSError替换首错；没有首错则照常抛出，两个清理故障保留先发生的unlock错；非IO程序错误不吞。回调不返回True，不抑制原异常/取消。获取失败不登记unlock，仍保留Timeout的cause与原预算。held移除时机、物理root/PID键、OS原语/死持有者释放、模式与锁文件、多根排序和输入staging保全保持。不加资源框架、模块、路径/schema或共享状态推断。

native Windows/Posix各用原生获取，仅注入cleanup/acquire故障；失败/取消和普通严格控制均要验证descriptor关闭及另一线程真实重获、不可变输入字节和builder staging保留。随后共享相邻、core/CPU Flow与最终head/实际dev全部实际CI分别验收。原始探针/ZIP/red在.tmp/audit107不随clone，正式回归/设计随Git提供；无GPU/main/tag或正式Study。


## B108 / #144：普通文本及ledger流关闭的首错保全（2026-10-16）

PR143最终head全部实际CI通过并正常合dev e589862d，实际dev CI尚在运行；对应锁主题分支保留至验收，不混入本批。Issue144后dcd4e2d先提交30项正式红回归：16 failed/14有效控制passed，0error/skip。现有atomic_write_text、Logger._emit、AlgorithmTracker.begin_run/append_jsonl四入口，真实部分写入/flush或编码错误和取消遭文件上下文close OSError替换；原错仅在context。12探针中8首错/取消重现、4正常close严格控制，原始证据在.tmp/audit108，不随clone。

沿用已实施第一/第三类合同：只改既有_atomic.py、logger.py、tracker.py的上述四个写入块，显式取得句柄；正常体完成后close严格传播，已有BaseException时尽力close，仅关闭OSError不遮原错，随后bare raise；非IO程序错误继续暴露。不新增通用I/O层、helper/module或依赖，不用异常context恢复原错。打开模式、编码/newline、追加顺序、flush、JSON/日志字节、目标权限/umask、原子替换、自有tmp清理和checkpoint诊断staging均保持。原子文本改用同参数Path.open/write，两个旧测试从Path.write_text高层注入调整为同一真实文件write的部分写入注入，仍验证原错身份、旧目标和peer不变；先在未修生产上验证这些既有相邻合同，再修。

30新回归覆盖新旧文本目标、实际UnicodeEncodeError、日志write/flush、tracker起始标记/指标追加write/flush、OSError与KeyboardInterrupt原身份、正常close失败、非IO关闭程序错、实际descriptor关闭、旧ledger前缀/peer字节、失败不提交tracker state。测试共用既有tests/conftest.py的局部故障fixture，未改元数据筛选/测试框架；三测试文件沿用模块原标签。随后相邻/core/CPU Flow、最终head和实际dev全部实际CI验收，正常关闭Issue与清本地/remote主题分支；无main/tag/GPU/测速/正式Study或布局改变。


## B109 / #146：Study comparison PNG剩余直写点（2026-10-16）

跟踪源码savefig清单中，库curves写入已由PR141验收，但studies/main_exp/flow/process/curves.py的plot_report仍直接写historical_comparison.png且仅成功后close。实际Study process入口run.py→curves.main调用该helper。仅AST抽取helper、用合成report/reference的4探针：渲染/部分保存/取消各泄漏1个自有figure，后两者把2398字节旧有效PNG覆盖成11字节；正常保存76906字节且不泄漏；peer figure均保留。不运行Study main/门限、数据/模型/历史归档、checkpoint或训练。

Issue146后6ab3b74先提交25正式节点23 failed/2正常控制passed，0error/skip。采用库PNG已批准合同的最小本地实现：保持原figsize=(14,5)、dpi160和1×2 axes，先取得figure即进入finally所有权范围，再fig.subplots；绘图主体仅缩进、原行保持。同目录UUID.tmp以xb排他取得，向句柄以显式PNG格式savefig、关闭后既有atomic_replace；正常close失败严格、不发布，已有保存错误/取消时close OSError不遮首错。保留原目标mode、新目标umask；只有自有普通tmp尽力清理，只读自有tmp可恢复0o600重试一次；未取得所有权的碰撞不删，peer figure保留。不抽新通用I/O层/模块/依赖，不改变helper返回None、文件架构、最终路径/格式、科学门或阈值、数据与坐标/标注。

测试在既有flow/process镜像目录AST读取当前Study代码，不依赖正式Study输入/结果，不启动main。覆盖新旧目标、axes/render/save/cancel/replace/create、碰撞、清理拒绝、原生Windows只读、close二错和正常PNG/图像数据控制；25新节点均需执行，随后相邻PNG、core/CPU Flow与最终head/实际dev全部实际CI。冻结修前与修后对同一组合成输入比较PNG字节，确认科学比较/读取/gate函数原字节块保持。正式未来受影响Study复跑按新version/实际provenance和既定前置门，不将本次图像工程检查冒充科学实验或重写旧结果。原始探针/ZIP/hash/红绿在.tmp/audit109不随clone；无GPU/测速/main/tag/数据删除。


## B110 / #148：不可解析provenance的冻结拒绝分类（2026-10-16）

实际JSON解析12000层metadata数组触发RecursionError，check_frozen仅归一化OSError/ValueError使其逃逸；实际Runner outer gate与study phases关闭时的native prepare gate均在零data/model构建、零execute调用时，把旧succeeded结果覆盖为failed。新Run也写failed但缺少既有冻结拒绝类型。初2000层被本机Python3.13接受，只触发普通changed-provenance拒绝，作为无效反例原脚本/log保留；更正12000层不调递归限额，原始深层JSON/四新旧路径证据在.tmp/audit110，不随clone。

Issue148后dadb558先提交12项正式回归5 failed/7有效控制passed、0error/skip；覆盖strict check_frozen、outer/native prepare新旧结果与普通坏JSON对照、freeze关闭不读坏文件、strict load_provenance仍抛真实RecursionError、非递归RuntimeError程序错不归一化。方案只在既有check_frozen异常元组补RecursionError，仍通过原raise ProvenanceChangedError from exc保留cause；不改strict loader/json解析/递归限额、schema、freeze真假判据、Runner现有保全决策表或真实业务phase失败规则。调用方Runner/prepare/launch/Study process继续使用同一公共边界，不复制guard或增加模块/依赖/新路径。新测试没有执行data/model/algorithm，仅native prepare到既有冻结门即拒绝。

随后provenance/freeze相邻、core/CPU Flow和最终head/实际dev全部实际CI分别验收；保留坏输入原字节及新旧结果边界，不以吞程序错误或放宽冻结检查通过。无GPU/测速/正式Study、main/tag、数据/Run/provenance删除或布局变化。


## B111 / #150 与 B112 / #151：读取边界集中里程碑（2026-10-16）

遵循18:53新频率规则，两个同类缺陷及§174系统读取点清单在唯一codex/optional-read-boundaries分支本地累积，完成全里程碑后才统一PR；不按单行文档或单个反例拆PR。已先Issue150/151，再8062f71提交44正式节点12 failed/32有效控制passed、0error/skip；c79b17b补4计数显示节点1 failed/3有效控制passed、0error/skip，两份原红分开保留。

可选结果读取：_load_result已将缺失、语法损坏或打开/读取IO作为不可用Run，但stat在try外、非object的CorruptArtifactError和JSON RecursionError逃逸，阻断有效peer。方案只把原is_file查询纳入原try，并沿scheduler._run_result现有模式补CorruptArtifactError/RecursionError；不可用行仍status None/metrics空，n仍仅有效succeeded，n_planned仍包含所有计划Run，complete仍false，peer均值/标准差公式不改。strict load_result原样拒绝，正常结果/计数、指标转换/科学门及所有Run/index原字节不改；不把坏metrics mapping或非有限数归一化，本批不改聚合公式。

严格CLI：status/logs/report/compare四个既有错误出口只补已知RecursionError，仍拒绝坏权威index/process/result、返回2并输出原诊断，不把strict loader降级为默认数据，不写旧NUMBERS/compare结果。可选_print_launch_counts则在其既有best-effort元组补同一解析错误，保留语法失败不输出、有效计数照常输出。非递归RuntimeError程序错误仍传播。仅改既有aggregate.py与cli.py，不加通用helper/框架/模块/依赖、schema或路径，不修改checkpoint/RNG恢复、Study科学审计或JSON限额。

44主回归覆盖6解码/缺失/有效结果、stat/read的permission/IO/disappeared/编程错误、strict公共loader、实际process partial/complete、四CLI的深层/语法/非object、正常输出和编程错误；补4计数显示深层/语法/正常/程序错。全部用合成临时输入，不跑data/model/algorithm或正式Study；compare正常控制仅CPU序列化空checkpoint字典。随后可选读取及strict readout相邻、core/CPU Flow、最终head与实际dev全部实际CI分别验收。原探针/源码ZIP/hash、44红与4补红在.tmp/audit111不随clone；不删除原失败/校验拒绝。


本机实施与验收：00f3ddf仅修改上述两个现有文件，9行增加/8行删除；48新节点全部执行/通过，core1834 passed/1既有平台skip，CPU Flow247 passed。其他aggregate函数原源码及CLI五个异常元组之外AST均保持。§174清单中的两个待修项已本地实施，最终head/实际dev CI仍待统一里程碑PR验收；不是待设计批准。详record§175，原红/绿/hash在.tmp/audit111不随clone。


## B113 / #153：同一读取里程碑补齐 Run 级回读（2026-10-16，已批准合同内实施）

继续源审发现individual.run的独立读取入口未列入§174原20项清单。实际FlowRunner(process-only)六探针中，非object、深JSON和stat PermissionError三类使派生失败且经既有Runner失败回退改写原结果；语法/读Permission/有效输入三个控制按预期保全。正式14项先红4 failed/10有效控制passed，0error/skip，含新增stat IO；原始坏文件及被覆盖后证据在.tmp/audit113保留，不随clone。

沿Run process已有缺失/坏语法/read IO的可选降级合同，只把is_file移入原try并补CorruptArtifactError/RecursionError，保留ctx.state有效mapping优先、无index要求、原metrics转换与history读取。严格公共loader和Runner通用失败/旧成功规则原样，不吞任意RuntimeError，不改路径/schema/恢复/科学门。此为#150相邻遗漏消费者，加入同一PR152，不另开小PR；旧head CI不取消，不能代替补齐后的最终head验收。


B113已实施22a4d57：14新回归全执行/通过，重新core1834 passed/1既有平台skip、CPU Flow261 passed。连同B111/B112共62新节点实际执行，21项清单已补齐遗漏入口；仍在同一PR152，最后head与实际dev CI分别验收。旧head证据保留，非待设计批准；详record§176–177。


## B114 / #154：Flow execute 最终写入与成功判据（2026-10-16，按已批准失败写入合同实施）

Flow通用execute在finally无条件吞掉tracker.flush_state的Exception，覆盖了原生train之外的成功收尾合同。实际Runner execute→collect→summarize→write、真实AlgorithmTracker与合成Algorithm入口六探针中，正常算法体加最终发布拒绝/真实JSON编码拒绝仍定稿succeeded且旧空tracker保留，两个反例；正常发布与三种已有body失败控制正确。无data/model/optimizer/训练或科学运行。Issue154后aae4b0e先提交24正式节点6 failed/18有效控制passed、0error/skip。

最小方案：在现有execute局部记录正常体（含现有info日志）是否完成；finally中正常完成后的flush严格传播，已有算法/日志错误或终止时仍用原best-effort flush。这样沿原Runner错误留证阻止后续collect/summarize/write成功，不改变Runner通用失败策略；原execute异常日志、failure_logged、返回值/observations和单次flush次数保持。无新helper/模块/依赖、格式/路径/数值/周期flush或checkpoint/RNG改动；原生train已实施的成功严格/失败保首错规则不动。候选单行文档不推送，完整失败写入里程碑一次PR→dev。


B114已本地实施82890ce，8行增加/3行删除；24新节点全执行/通过，core1846 passed/1既有平台skip，CPU Flow273 passed。SOURCE_CHECKED精确原字节替换证明算法/日志/Runner/数值等边界未变；最终head/实际dev全部实际CI待完整里程碑验收。详record§180；当前不是待设计批准。


## B115 / #156：不可变历史源码的排他创建与首错保全（2026-10-16）

main_exp.export_source和main_probe.MainArchive.__init__均以exists→mkdir→write_bytes写固定归档。八个隔离合成Git blob探针用真实线程落盘复现两处竞态覆盖；六个全新/既有相同/既有不同控制正常。此为确定性交错，不声称正式归档曾损坏；未加载任何历史源码、bootstrap或科学运行。正式0383bd1先红28项12 failed/16有效控制passed、0error/skip：四竞态（含相同peer的竞争失败后可重试复用）和八个原写错被close IO替换反例。最初夹具Git tree闭包被AST变量覆盖的28失败为无效，单独保留，不当库失败。

沿原“不同已有源码不得覆盖”合同，两处现有else分支改为xb排他创建，不新增文件/模块/依赖/通用IO层。失去创建所有权时原生FileExistsError拒绝，不删除或截断peer、不写新manifest；胜者完成后相同内容按原exists/read校验复用。自己已获所有权的部分写入/终止保留诊断字节，close仅OSError不得替换原错；成功体close失败严格传播，manifest只有全部blob正常后才按既有原子writer写。既有相同/不同判定、原Git批协议/固定commit、manifest键/hash/文件名和所有归档路径保持，不改科学算法/阈值、checkpoint/RNG或恢复合同。两处统一一个主题里程碑，后续先本机/CI验收再合dev；不执行或覆盖正式Study归档/数据。


B115本机验收：28新节点全执行/通过，core1874 passed/1既有平台skip、CPU Flow273 passed，0failure/error。两文件精确原字节证明仅替换各一条write_bytes，22增/2删；原红12失败/16控制、无效夹具/初稿拒绝均保留.tmp/audit115不随clone。无正式Study或归档执行，最终head/实际dev CI待完整里程碑验收，详record§183。


## B116 / #158：只读日志按实际时间排序（2026-10-16）

event_lines文档与既有测试要求timestamp order，Logger原生输出RFC3339毫秒及UTC偏移。原排序键直接用墙钟字符串，六个实际CLI探针中DST回拨/跨日/跨偏移三反例返回0却晚事件在前；三个同偏移控制正确，输入/索引/日志字节全不变。Issue158先于正式28红8 failed/20有效控制passed、0error/skip，补库/CLI两入口与Z/不同小数精度。

只改logs.py排序键：已识别且带时区、在datetime精度内的时间用标准库转UTC，并以固定微秒精度字符串比较，不用浮点epoch、不改原行文本/Logger或输入格式。缺失、坏/无时区时间、UTC转换溢出继续原文本/9999 fallback；高于微秒的旧文本保持原排序键，避免datetime截断造成新的顺序损失，不宣称为其新增UTC语义。相同键仍沿Python原稳定排序。event选择/traceback筛选、path归属与可选IO guard、strict index/CLI拒绝、状态/聚合/科学门均原样。无新helper/依赖/配置/路径，不设全局时区。

整批回归/源码保全/本机与最终head/实际dev CI验收后统一PR，不因几行设计另开PR。无GPU/测速/正式Study、数据/Run/provenance改写或布局变化。


B116已本地实施ba665de：28新节点全执行/通过，core1902 passed/1既有平台skip、CPU Flow273 passed，0failure/error；原字节证明仅导入/排序键变动。超微秒旧文本不新增UTC排序合同，fallback保留，严格恢复/科学门不变。最终head/实际dev CI待完整里程碑验收，详record§186；原证据.tmp/audit116不随clone。


## B117 / Issue160：同进程数据端点绑定的并发所有权（2026-10-16，已确认实施方案）

`bind_endpoint` 临时修改第三方数据集类的 url / mirrors。DataFactory 的 cache_lock 保护各自资产根，无法保护两个不同根共用的同一个 Python 类。合成类、两个线程与事件调度复现两次上下文读错 URL，退出后还残留国内 URL；不访问网络、真实缓存或 torchvision。先 Issue160，再 9c95667 正式红回归：8项中4 failed/4有效控制 passed，0error/skip，覆盖CIFAR URL/MNIST mirrors、正常/原错退出和同线程嵌套恢复。

方案在既有 bind_endpoint 这一共同边界使用一个进程内可重入锁，持有范围覆盖保存原字段、绑定、整个构造上下文与 finally 恢复。普通互斥锁会破坏现有同线程嵌套，因此用标准库 RLock；不把锁放在每个资产根，也不新增每类锁注册表/依赖/模块。不同进程仍沿既有磁盘 cache_lock，不宣称进程锁替代磁盘锁。端点定义、调用签名、yield endpoint、正常和异常恢复、数据构造内容、下载/hash/布局与科学门保持。锁只解决原本不可并行安全的进程全局类字段，非测速/性能优化。

调用方只有原生 _build_torch_vision 中 train/test 两次构造的同一个上下文；既有调用方无需改。apply_model_origin 是显式进程环境设置，非临时绑定合同，不把本方案扩展成配置新语义。临时原ZIP/hash/probe/红XML在.tmp/audit117不随clone，正式回归按source镜像放tests/rpipe/structure/test_origin.py。生产修改后复查实际Factory路径、首错/嵌套/恢复及相邻origin，再做core/CPU Flow；同分支积累完整里程碑后再PR dev。

实施补记：fffd872已按方案实施；测试改名test_origin_concurrency.py避免pytest同名模块收集冲突，首次收集拒绝保留。新旧端点13全通过，core1910 passed/1既有skip、CPU Flow273 passed，新8全部执行；实际Factory双根原版反例/当前恢复已核对。最终head/实际devCI待完整里程碑验收。


## 190. B118 / Issue162：共享准备的临时环境所有权（2026-10-16）

同类全局状态复查确认prepare_shared_data在独立项目缓存根下并发保存/设置/恢复TQDM_DISABLE，后一个_prepare会看到None/调用方原值，全部退出还残留1。第一次探针误用共享同一根的两个Study，现有cache_lock使人为交错等待超时；INVALID_PROBE_NOTE保留，不计库失败。改用两个独立项目根后实际公共prepare入口/真实cache_lock+合成_prepare稳定复现，无数据下载。先Issue162，再b04a5cd红9项4 failed/5有效控制passed、0error/skip，覆盖初值缺失/自定义/已为1、正常/原错退出和同线程嵌套。

方案仅在prepare.py增加标准库RLock，以独立的进程内环境所有权范围包住原保存、设置、cache_lock内_prepare和finally恢复；必须在原资产根锁之前取得环境锁，维持一致顺序。同线程嵌套仍可重入，异常和返回值原样。与bind_endpoint锁分别保护不同全局对象，不引入通用全局状态框架、配置或路径，不改_prepare/body、下载/hash、数据/结果/科学门。与B117统一归入当前PR161临时进程状态并发里程碑，原已触发CI保留，新增代码后以新最终head重新验收；不因观察超时取消或重跑CI。

B118实施补记：0aaae0b已按方案实施；与B117统一里程碑的新旧相邻27项全执行/通过，合批core1919 passed/1既有skip、CPU Flow273 passed，新17全执行。两个原始生产文件各自精确字节核对，仅锁和原上下文范围改变；不改body/端点/下载/hash/格式/布局或科学门。最终head与实际devCI仍待分别验收。


## 195. B119/B120 配方临时导入状态所有权方案（2026-10-16，实施方案）

目标同时修同目录helper串用和执行时缺失recipe模块身份，保留现有load_recipe/apply_recipe调用、路径/来源清单/注册时机与正常异常行为。只在原recipe.py复用标准库contextmanager/RLock建立私有临时导入上下文，不新增架构模块、公开配置或产物目录。

进入load/register现有sys.path范围时，在同一个可重入进程锁内登记当前目录可导入的本地顶层模块/包名；仅暂存并移开这些名字及子模块的现有cache项，执行结束清理该范围新项、原对象原样恢复，路径亦恢复。内建/标准库名字保持原Python优先级，不批量清空sys.modules或第三方cache。其它非冲突第三方模块不进入所有权范围。锁在当前模块保存/移开/导入/恢复全程持有，支持同线程嵌套。

load_recipe在上述上下文内执行spec前，将自己的既有模块名临时注册sys.modules；无论exec/校验成败都恢复此前同名项（包括不存在），使dataclass可解析定义模块。load后返回ModuleType、注册后已捕获的函数/模块全局引用继续有效；测试明确核对两个Study注册函数各自返回原值和既有外部模块身份。register时的同目录import同样走该上下文。保留当前load使用recipe父目录、register使用Study根的路径行为；不把此修复扩展为新公共namespace API、清空外部模块、任意后端的序列化保证或跨进程解释器改造。

按先红与方案后改生产，完整测试须包含正常/原错、模块/包、load/register、外部cache保全、注册引用和dataclass根/嵌套/eager/future，以及已有Study hooks/来源冻结/core/CPU Flow。现有Study phase私有namespace只作设计参考，不为了复用反向依赖flow层或搬文件。当前仍本地积累，不立即新PR；进一步反例保留并修合同，不能以测试全绿替代边界核对。

实施前边界补充：模块身份须在load exec与register调用两个上下文均可见，不只加载时；中间候选26项通过后，register内dataclass新4项仍2失败/2控制，此反例RED_REGISTER保留。统一由同一私有上下文保存/注册/恢复当前ModuleType，保留每次上下文前的同名cache项。生产提交在此修正和验证之后。


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

当前实施/验收补记（record§201）：§195的短暂清cache候选已由§198/200的模块生命周期与兼容reader方案替代；当前core1964 passed/1既有平台skip、CPU Flow279 passed，新51节点全部执行/通过。只4个既有源码文件，生产写入/数学门/布局原字节核对保持。这里只是本机验收，尚无本批PR/最终head/实际devCI，不表示163/164已关闭或已发布，也不等待人工设计批准。


## 202. B121 / #166：动态导入继续绑定所属Study（2026-10-16，PR165候选回归）

最终验收前复查确认：recipe在register内用importlib.import_module加载同目录helper，实际dev8480d79的独立子进程返回37，d385c410候选报ModuleNotFoundError。原导入隔离方案仅绑定Python import语句，漏掉标准动态入口；PR165不合并，保持同一分支补齐，不单独拆PR。最初探针驱动漏import sys、未调用库，单独保留且不算库失败。

方案：复用每Study私有包的导入归属判定，在recipe/helper导入importlib时提供仅当前私有代码持有的模块视图，import_module将绝对本地名映射到自己的私有空间；相对名和非本地名委托当前原生importlib.import_module。视图复制原模块属性，不修改sys.modules['importlib']或其函数，不改变其他调用方；importlib子模块继续使用原模块。Python import语句、标准库/已加载运行时优先级、外部导入hook、旧/迁移checkpoint读取合同不变。动态导入类与普通导入类使用同一稳定身份。

先提交10节点红回归：2种importlib取用方式×加载/注册/注册后/辅助模块4时机为8失败，2个非本地json对照通过。修复后运行当前全部新增集合与core/CPU Flow，再推同一个PR165重验最终head；此前d385c410的CI即使通过也不能替代新head。原始红/前后子进程/无效驱动证据在.tmp/audit121和.tmp/audit119（不随clone），portable回归已入库。不改变文件架构/产物布局，不等待设计审批。


2026-10-16本机实施：9166cbd在同一recipe文件补动态入口绑定；当前core1974 passed/1既有skip、CPU Flow279 passed，新增61全执行。该importlib视图仅属于私有代码执行环境，不替换全局模块身份；原生子模块与当前外部动态函数继续委托。最终PR165新head/实际devCI仍待验，不是等待设计批准，详record§202。


2026-10-16已实施并验收：B119/#163、B120/#164、B121/#166统一PR165，最终head3a74728与实际dev70a8750全部实际矩阵/聚合CI均成功；三Issue已关闭且已合主题本地/remote清理。动态入口、私有模块生命周期、旧/迁移checkpoint兼容的候选反例与方案修正均保留；这不是待审/待修状态，不改变第七节未实施能力。详record§203，后续三类复查仍先确认真实反例再同主题合批。


## 204. B122 / #167：Runner末条日志纳入同一失败留证边界（2026-10-16）

第一类新反例：最后logger.info('succeeded')位于run既有try/except之外。真实Logger的末条stdout或文件write被拒时，原异常虽传播，却绕过_log_failure/_write_failed_result；普通collect新Run无failed、旧Run残留succeeded，process旧成功按原规则应保留但也没有原错诊断。四个原生Runner/Logger合成探针确认stdout路径，正式12节点（stdout/file-write/正常×collect/process×新旧结果）先红8失败/4正常对照，4ac6883已提交。与B101启动日志处不同：原修复保留末条时机，本批保持该时机，只补末条异常范围。

最小方案：将末条info原语句缩进到现有try内、仍在全部阶段/Study hook之后；已有except原样记录并按当前最后phase调用_preserve_success。process旧成功仍逐字节保全；其他普通阶段仍failed，首错对象/既有诊断fallback/failed写二错规则不变。没有新catch、helper、phase名、依赖、结果字段、路径、文件或目录。日志本身失败仍对调用方失败，不吞错或伪造成功。只在同主题下一里程碑分支积累，先本机定向+core/CPU Flow，后续完整批次一个PR验最终head/实际dev；不为单个相邻bug立刻开PR。


2026-10-16本机已实施B122：761700f仅原末条日志缩进到既有try，原捕获/保全规则及全部其他源码原字节不变。Runner49/core1974 passed+1既有skip/CPU Flow291 passed，新增12全执行。#167仍OPEN，等待同主题完整里程碑最终head/实际devCI，不是等待设计批准；不单bug立即开PR。详record§205。
