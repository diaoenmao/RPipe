> **RPipe 研究计划：研究执行与证据一致性的可验证合同**
> 版本：1.2；更新：2026-10-16；中文为正本，[English](<Research Plan RPipe.md>) 逐章对应。
> 当前阶段：完成计划重做和定向文献初筛；继续代码正确性，尚未执行本计划的新科学实验。
> 当前实现基线：dev `2008753c3c2b11321a3da3811102cf5c20b646ff`；已发布对照为 v0.4.1，不将 dev 冒充发布版。
> 本轮零 GPU、零测速、零训练、零数据下载；保持既有文件架构与产物布局。

## 本次人工审阅摘要（2026-10-16，按12:53更新）

- **已完成：** 三类收敛及夜间PR103–125；尤其PR117/119/121/123/125最终与实际dev全部实际CI已通过。原始失败、错误假设与科学负结果保留。
- **计划归一：** 本中文正本与完整英文译本是同一项目计划；五份Study/历史/模板方案已在第十一章完整归并，原独立PLAN.md移除，报告/JSON/图/运行产物不动。
- **研究结论：** 支持已测工程合同；不证明当前dev完整科学复验或外部优越性。文献仍为定向初筛，阅读范围见第二章及文献证据。
- **未完成与下一步：** [第九章缺口/实验顺序](#gaps-next)列出CPU正确性与文献优先项；E3/E4须另获资源/运行授权。当前不测速、不用GPU、不合main或打tag。
- **阅读导航：** [夜间修复](#overnight-correctness)、[完整Study计划](#study-plan-inventory)、[文献依据](literature.md)。

# 一、灵感与初步问题

## （一）课题、来源与价值

研究类型为**系统与应用研究，结合可复现性方法调研和有限范围历史实现复现**。RPipe 是研究项目的源码模板与执行底座，不提出新的分类器、优化器或量化算法。研究对象是“声明—执行—持久化—只读展示—人工结论”之间的证据链。

实践触发来自已经有失败回归的三类问题：失败写入可能覆盖旧证据，坏元数据可能拖垮或误导整组展示，并发发布可能碰撞或暴露不完整文本。最新 compare 反例表明，只比较到相同数值还不够：若两侧误读同一个 Run 外 checkpoint，甚至会得到错误的完整通过。来源见 [产物一致性设计](../development/artifact_consistency_design.md)、[开发记录](../development/record.md)及 [#101](https://github.com/DreamSoul-AI/RPipe/issues/101)。这些是本仓库的工程观察，不代表其他框架普遍存在同样缺陷。

初步问题：在固定计算与磁盘合同下，能否通过少量统一的失败保全、可选读取和原子发布规则，使正常结果可追溯、错误可定位、局部损坏不制造错误成功？选题价值是减少研究判断中的证据混淆；研究新颖性与外部推广性尚未证明。

## （二）约束与责任

- 负责人：Enmao 负责范围、资源与研究方向；Agent 负责实现、检索、证据整理和 PR；GitHub CI 验证工程合同，不代替研究者判断。
- 今晚交付：本中文计划、完整英译、文献证据入口和适用规范，通过工作 PR 合 dev。Study 原协议全文保留在第十一章，历史结果留在各 Study，不把事后经验伪装成其预注册方案。
- 当前允许：文献与代码只读核查、合成 CPU 正确性回归、必要 CI、已授权的结束远程分支清理。具体配置与落盘始终以 [layout.md](../code/layout.md) 为准。
- 当前不执行：GPU/长训练、速度/耗时研究、历史 480s 追查、新目录或结果格式、共享路径/logger 迁移、发布到 main 或打标签。四个任务共用一张 24GB GPU，未来科学实验须另立资源安排；本计划不解锁设备使用。

# 二、文献调研

采用 [研究计划规范](<standards/研究计划规范 Research Plan Standard.md>)和[模板](<standards/研究计划模板 Research Plan Template.md>)，检索/阅读/索引依据新挂入的[文献调研规范中文原件](<standards/文献调研规范 Literature Review Standard.md>)。本仓库适配与实际检索过程见 [literature.md](literature.md)，不把复制规范当成已完成系统综述。

- 信息截止与实际检索日：2026-10-16；本轮为问题驱动的定向初筛，非穷尽性或系统综述。
- 概念组：ML reproducibility / experiment provenance；failure evidence / artifact integrity；workflow / optional metadata / atomic publication。以论文原页、作者公开版本和官方版本化文档为准。
- 纳入：直接解释复现证据、研究设计、实现/评价/文档因素的来源；可核对版本的工作流方法资料。排除：与本问题无关的模型排行榜、二手摘要、仅营销材料；重复版本归并。
- 待补：异机构件可复用性、外部工作流在相同故障集上的对照、更新文献及引文追踪。未检索到不能证明空白。

| **文献与定位** | **问题与阅读范围** | **数据与模型** | **算法** | **指标与实验条件** | **本课题用途与限制** |
|---|---|---|---|---|---|
| [L1 Pineau 等，JMLR 2021](literature.md#l1) | 官方摘要；复现计划的组成 | 研究流程，不借用其模型数字 | 代码、复现挑战、报告检查项 | 本轮未复现其项目调查 | 作为记录与检验分工的参考，不能据此声称 RPipe 已获外部复现 |
| [L2 Gundersen 等，arXiv v2 2023](literature.md#l2) | 摘要及研究设计/实现/评价/文档相关章节的定向阅读 | 复现因素分类 | 从设计选择到结论的分析框架 | 本轮不移用跨论文数字 | 支持区分运行、分析、结论与适用范围；不充当本系统效果实测 |
| [L3 Snakemake，2021 v1](literature.md#l3) | 官方检索摘录初筛；直读受限 | 工作流系统；配置未复现 | 可追溯工作流路线候选 | 尚无同故障集/同配置对照 | 后续外部基线候选，当前不做能力或性能优劣结论 |
| [L4 PyTorch 2.9 文档](literature.md#l4) | Reproducibility 页面 | 平台、运行时及 RNG 条件 | 随机性与确定性控制 | 实验依赖仍按具体 Study 固定 | 不把相同 seed 当作跨版本/平台等价证明；本轮不改精度设置 |

路线判断为**继续并收窄到可验证合同**：已有真实反例支持开展工程失效机制研究；文献支持说明复现条件和证据边界，不足以证明本模板优于既有系统。跨项目共享 [#89](https://github.com/DreamSoul-AI/RPipe/issues/89) 与监控/logger [#90](https://github.com/DreamSoul-AI/RPipe/issues/90) 是独立探索，不绑定为本研究前置条件。

# 三、研究问题与假设

| **编号** | **问题与假设** | **支持条件** | **反驳或不足以判断** |
|---|---|---|---|
| RQ1 / H1 | 既有冻结拒绝和 process 失败保全规则统一后，受保护的旧成功证据是否保持原字节？ | 预先声明的保全场景中旧 result/peer 哈希不变，原异常可取得 | 任一保护文件改变即反驳该场景；不把普通 execute 失败扩成保留旧成功 |
| RQ2 / H2 | 按现有读取函数统一三态、类型与引用归属，能否让局部坏输入不造成 peer 丢失或错误成功？ | 可选坏输入按原规则降级；严格前置输入明确拒绝；越界 compare 拒绝且不覆盖旧输出 | 表格丢行、伪指标、串读或错误通过即反例；仅“没有异常”不足以支持 |
| RQ3 / H3 | 可整体替换文本经既有原子入口发布，是否保留旧/新完整字节及清理职责？ | 受控失败与并发交错下最终仅为合法完整版本，拒绝发布不损坏 peer | 撕裂文本、撞名覆盖、删除不属于本调用的临时文件即反例 |
| RQ4 / H4 | 来源明确、条件对齐的历史原码对照，能否检验执行模板未改变已选计算？ | 将来在当前基线按相应 Study 的原门完成完整对照 | 旧版通过不能证明当前版；缺 raw/权重/运行条件时只能判证据不足 |

六方面衔接：文献确定证据边界；合成输入与既有 Study 限定数据；状态机与神经网络分别承担系统模型与科学载荷；统一合同改变的是执行/证据算法；指标检查不变量和已登记数值门；配对旧/新实现与故障注入区分系统变化和算法变化。

贡献目标是可解释、可回归的工程合同及证据材料。当前不声称新的机器学习方法、普适错误率下降、统计显著性、外部独立复现或系统性能优势。

# 四、数据

| **集合** | **来源与单位** | **版本、划分与质量检查** | **位置与限制** |
|---|---|---|---|
| D1 合同输入 | 测试自建 JSON/YAML/文本、路径/别名、微小 CPU 张量；单位是故障场景与合法 peer | 按故障家族划分开发案例、合法控制和后续扩展边界；保存红/绿测试字节，避免只保留通过样本 | `tests/_data/` 与源码镜像测试；运行输出 `.tmp/`，不随 clone 提供 |
| D2 短对照证据 | main_probe 的成对原码/当前链；单位是数据集×模型的配对 Run | 历史精度、seed、输入/参数/RNG/调度条件见原计划；新源码须新 version、原门复验 | [计划](<研究计划 RPipe.md#study-main-probe>)、[报告](../../studies/main_probe/docs/STUDY_REPORT.md)；本轮只读 |
| D3 长曲线证据 | main_exp 的跨 seed 曲线与自身 best eval | 历史训练/中断/新身份分别登记；不能用新计划重标旧来源 | [计划](<研究计划 RPipe.md#study-main-exp>)、[报告](../../studies/main_exp/docs/STUDY_REPORT.md)；原始制品不随 clone 提供 |
| D4 文献证据 | 原论文页面、官方文档和实际阅读笔记；单位为固定版本文献 | DOI/arXiv/版本去重，作者主张、工程实测与本计划判断分开 | [检索与证据表](literature.md)；不伪造本地 PDF 或已读全文 |

本次不新增人类受试者或私有数据收集。未来使用 MNIST/CIFAR10 时，按原 Study 核对来源、版本、划分及许可/分发条件，本轮未重新核验数据许可且不下载数据。共享不可变输入仍在项目根 `shared/{data,model}`；模型对象、checkpoint、sampler/RNG 与可变缓存属于各 Run。

故障输入不作为训练/测试样本独立同分布抽样；定向回归集也不是真实故障分布。后续新增边界案例须明确哪些见过开发过程，不能将调试同一案例后的通过率解释成未知缺陷泛化能力。科学数据的 train/development/test 及泄漏检查沿相应 Study，不能因系统计划而重划已有实验数据。

# 五、模型

系统模型为 Study/Experiment/Run 的状态与证据关系：Study 声明产生 index 和 Run config；执行写 result 与 assets；process 产生派生汇总；status/logs/report/compare 读取已有证据，人工报告形成结论。该模型是分析视角，不新增磁盘字段或类。

科学载荷沿用 main_probe/main_exp 的 linear、MLP、CNN、ResNet18 及其原配方。初始化、权重、参数结构、预处理与许可依据各 Study 的归档来源，不在此另设模型或虚构参数量。固定部分为相同数据/模型/算法/精度与预算；研究变化部分为已经有失败合同支撑的系统边界。

基线有三层：同一故障输入下的修复前/后实现；最小直接文件操作或原判断分支的局部对照；固定历史 RPipe 原码的科学配对。外部工作流仅列为待补文献/协议基线，当前不假定其功能、数据与本仓库可直接横比。

# 六、算法

1. 从文献与真实失败提出范围明确的问题，登记原始输入、期望、实际和来源；先提交能失败的回归。
2. 在既有函数内复用共同决策、读取或原子写入口；保存原错误类型与传播，拒绝无失败支撑的大重构。
3. 对同输入进行修复前/后配对，核对受保护字节、peer、状态、引用归属和数值语义；不以“程序能跑”替代合同。
4. 通过针对性及必要 CPU 流程检查，再以 PR 进入 dev；最终工作提交与实际 dev 分别核对实际 CI。结束远程分支按授权核对后清理。
5. 科学比较若将来获准执行，沿原 `prepare → execute → collect → summarize → write → process`，在定稿前完成影响成功判断的门；派生 process 失败不改写成功 result。

实现复用 `src/rpipe/`；失败保全的私有决策不扩大旧成功保留范围。可选读取的空值降级不能套到严格 index/process 或训练 checkpoint 恢复。原子替换只覆盖可整体替换文本；日志追加、checkpoint 诊断 staging、锁的豁免理由见[已批准设计](../development/artifact_consistency_design.md)。不新增尝试目录，不改变 result.json 的位置、格式和含义。

官方/第三方/自研来源分开：两项历史 Study 的对照来自本仓库历史原码（短对照 `98648f3`，长配方 `4ccb28d`），属于项目自身实现复现；不是对文献 L1–L4 的算法或训练复现。精确来源、归档及运行条件以各 Study 的已有记录为准。

本研究不新增损失函数、超参数搜索或性能算法。确定性案例按预定输入和屏障交错验证；浮点预算、seed、RNG、sampler 与精度在科学实验中沿原计划固定。出现来源漂移、输入不可取得、错误成功或旧证据改变时停止该批判定，保留失败并修正，不放宽门限。

# 七、指标

以下是分析口径，不新增 result schema、CLI 选项或自动指标字段。

| **角色与名称** | **定义、单位与方向** | **数据与聚合** | **实现与条件** | **判据及依据** |
|---|---|---|---|---|
| 主：证据保全违反数 | 保护集合内前后字节不同的文件数，越少越好 | 每个适用拒绝/失败案例分别列明文件，不把普通失败混入保护集 | 原字节/哈希断言，保留 peer 和原异常 | 枚举案例必须为 0；既有保全合同 |
| 主：错误通过数 | 应拒绝的越界/失配比较却报告通过的案例数 | 每个案例与合法控制成对核对 | 公共 API 与 CLI，必要时断言禁止加载外部 payload | 必须为 0；不能只看 exit code 或总通过率 |
| 主：展示结构/归属违反数 | 丢 peer、伪指标、断行增列、串读引用的案例数 | 分 TSV/Markdown/compare 和严格/可选前置条件 | 行列、原键/指标、状态及输出保全断言 | 对声明合同必须为 0；坏输入不自动修写 |
| 主：发布不变量违反数 | 不完整最终文本、撞名误覆盖、非本调用清理等事件数 | 按明确交错/故障点分别报告，不作概率推断 | 既有原子入口，日志/锁/checkpoint按各自协议 | 适用合同必须为 0；不声称任意文件系统断电保证 |
| 辅助：工程覆盖状态 | 每个必要节点实际执行/通过/失败/跳过/未执行 | 按版本、平台、Python、case 分开 | JUnit、manifest、CI 实际 job | 必需门全通过；平台 skip 不计通过，不以数量证明稳定 |
| 条件性科学指标 | 参数/输入/RNG/调度一致性、Loss/正确计数、曲线与模型次序 | 沿各 Study 原口径，不跨版本混算 | [main_probe 原门](<研究计划 RPipe.md#study-main-probe>)、[main_exp 原门](<研究计划 RPipe.md#study-main-exp>) | 保持原门；当前基线未执行则明确未验收 |

本轮无吞吐、延迟、运行时间或成本排名指标。确定性合同不做 p 值；并发用受控屏障/故障点验证，并保留每次重复，不估计总体事故概率。未来随机科学对照的独立单位为独立 Run/seed，重复次数、样本量与统计方案在具体 Study 明确，不能将同一轨迹的多个点当独立重复。

# 八、实验设计与执行顺序

| **实验与问题** | **方法及对照** | **因素与控制** | **数据、指标与重复** | **预算与判读** |
|---|---|---|---|---|
| E0 文献与来源核验 | 原论文/官方文档与仓库事实分层 | 固定实际阅读版本和日期 | D4；先分类/证据地图，再补引文 | 本轮已做初筛，不宣称系统综述完成 |
| E1 三类合同回归 | 同一原字节输入的旧/新实现与合法 peer | 每次改变一个失败边界；固定正常输出/错误合同 | D1，H1–H3；全部必要边界逐项执行，跨 OS 分列 | 当前允许 CPU；红测试先于修复，不以多次挑通过验收 |
| E2 原生 CPU 流程与安装 | 固定当前源码的 make/launch/process/readout/包安装 | Toy/Stub、小输入、独立 basetemp、GPU隐藏 | 既有测试门；最终 head 与实际 dev 分别验收 | 必要本地检查+现有 CI；不是科学精度验证 |
| E3 数值探针 | 原码 vs 当前链，沿原短预算配方 | 数据、初始化、精度、RNG、评测、作用位置相同 | D2；原完整矩阵及原门，新 version | **本轮不执行**，未来明确授权/资源/输入齐备后安排 |
| E4 长曲线 | 历史配方 vs 当前链与原估读曲线门 | 固定训练预算/seed集合/完整评价，保留中断身份 | D3；不能由 E1/E2 或短前缀替代 | **本轮不执行**，无自动解锁长训练 |

## （一）基线复现与后续安排

当前基线复现范围是**历史文档与已提交证据核对**，不是重新训练，也不是对外部论文的训练复现。旧版 `v0.3.0 / 804b3823` 的两项正式 Study 报告记录了当时的完整验收；新 dev 有后续修改，旧结果不能标记为当前基线结果。若未来执行 E3/E4，先检查实际归档、raw/划分、环境与权重来源，再沿原前置门与停止规则，不因资源不足缩短后仍称完整复现。

| **目标与条件** | **计划目标及依据** | **既有报告值/出处** | **当前 dev 新实测** | **判断** |
|---|---|---|---|---|
| 历史短计算对照 | 原完整配对矩阵，原逐段数值门 | 仅链接 [main_probe 报告](../../studies/main_probe/docs/STUDY_REPORT.md) | 未执行 | 只能沿旧版本解释，不证明当前数值等价 |
| 历史长曲线与自身 best | 原全预算/四 seed 与原曲线门 | 仅链接 [main_exp 报告](../../studies/main_exp/docs/STUDY_REPORT.md) | 未执行 | 不把历史图或估读一致视为原权重逐位相同 |
| 当前三类工程合同 | 枚举案例的字节/状态/引用/发布不变量 | [开发记录](../development/record.md)与对应红/绿提交 | 已有工程回归；见第九章的证据范围 | 支持已测合同，不外推全部研究正确性 |

## （二）消融覆盖与组成

不进行量化、scaling、旋转或模型训练消融；模板中的这些例子不适用于本轮系统问题。H1–H3 的机制对照是回放故障案例时比较旧局部行为与统一规则，固定数据、操作及观察点。不得在真实 Study 中删除已修规则来“做消融”；对照放隔离副本/测试输入，输出 `.tmp/`，保留原始失败。

## （三）执行安排与停止规则

顺序为 E0 → 对已有真实缺口执行 E1 → E2；E3/E4 作为单独的条件性工作。文档批次只做文档/来源/链接/布局一致性检查及仓库必需 CI，不为了填满计划重训。CPU 案例使用测试既有 C1/C2 成本类别和独立临时目录，不新增测速或资源校准。

已知混杂包括：故障集来自已见问题、合成文件不等于所有文件系统故障、同一实现团队不等于独立复现、不同平台/精度/依赖不自动可比、旧报告引用的忽略资产不随 clone 取得。多任务共用 GPU 的竞争可能影响历史计时，但本轮不验证该推断，也不据此给历史慢因定案。

DOE/OR/BO/贝叶斯实验设计本轮不采用；无需为合同修复搭搜索系统。若将来要优化配置或资源，另设目标、预算、搜索/最终评价隔离及授权，不能混作 H1–H4 已通过。

# 九、执行记录与结论

本章记录**已有事实与本计划新执行状态**，不是把修复后的经验包装成事前预测。正式科学数字、图、命令与运行结论仍以对应 Study 为唯一正式位置。

| **记录** | **实际证据位置** | **状态** | **能支持与不能支持的判断** |
|---|---|---|---|
| 两项历史科学 Study | 本计划第十一章原协议、各 Study 的 STUDY_REPORT 与正式 JSON | 旧版本已有验收；当前基线未重跑 | 仅支持其记录的来源与条件，不外推本 dev |
| 失败保全/安全读取/原子发布三类收敛 | [已批准设计](../development/artifact_consistency_design.md)、record 与对应 PR | 已有多批回归；新反例继续登记 | 验证有限合同；不是全仓“无 bug”证明 |
| B-086 文本视图 | [PR98](https://github.com/DreamSoul-AI/RPipe/pull/98)、record §111–114 | 最终 head/实际 dev 已验收 | 前置 56 节点及对照支持视图合同；不等于科学结论 |
| B-087 可选 tracker 状态 | [PR100](https://github.com/DreamSoul-AI/RPipe/pull/100)、record §113–115 | 最终 head/实际 dev 已验收 | 前置 24 节点支持该解码边界，严格训练恢复未改变 |
| B-088 compare 引用 | [PR102](https://github.com/DreamSoul-AI/RPipe/pull/102)、record §116 | 最终 head 与实际 dev ef60846 CI 全成功；结束远程已清理 | 前置 17 节点/合法控制与零加载断言；不能拿旧“key固定”判断作证 |
| 本计划与文献初筛 | 本文件、[英译](<Research Plan RPipe.md>)、[证据表](literature.md) | 本轮文档重做；未执行新科学实验 | 提供下一阶段问题/协议，假设不因写入计划而获支持 |

失败与负结果：保留所有原红、收集错误、错误测试假设、候选失败与取消/未执行；其身份与正式通过分开。对历史科学门的失败不放宽容差，也不以新权重替代失踪旧反例。

当前结论：继续以三类统一合同做正确性；当前工程验收与历史科学证据各有边界。未知项包括当前代码完整科学复验、历史原始全部权重/seed可得性、历史自定义sampler精确恢复、外部独立复现和完整文献覆盖。shared/logger 议题保持独立调研，不因计划落库实施。

<a id="overnight-correctness"></a>
## （一）截至本次人工审阅的夜间 PR、结果与结论

夜间五批修复的验收截止来源为 dev `82205cbe7fc89a46853eb61c1734d123b7d153f0`（PR125），main 仍为已发布 v0.4.1 / `00e70b1`。下表五批最终 head 和各自实际 dev 全部实际矩阵/聚合均已通过；本机 skip 不算通过。按2026-10-16最新清理指示，已修且验收完成的 Issue 已关闭；关闭不等同发布。

| **已合 PR / Issue** | **真实缺口** | **实际 dev 与独立 CI** | **本机 CPU 检查口径** | **结论及保留边界** |
|---|---|---|---|---|
| [PR #117](https://github.com/DreamSoul-AI/RPipe/pull/117) / [Issue #116](https://github.com/DreamSoul-AI/RPipe/issues/116) | 可选 JSON 的整数解析限制/深递归 | `20027afb231b`；[Package Check](https://github.com/DreamSoul-AI/RPipe/actions/runs/38815660139) / [Unit Tests](https://github.com/DreamSoul-AI/RPipe/actions/runs/38815660146) | core 1432 passed / 1 平台 skip；CPU Flow 189 passed | 可选提示/state/JSONL保留有效peer/轨迹；严格loader不弱化。 |
| [PR #119](https://github.com/DreamSoul-AI/RPipe/pull/119) / [Issue #118](https://github.com/DreamSoul-AI/RPipe/issues/118) | 调度保守查询与已实施设计状态 | `4d12719d7eba`；[Package Check](https://github.com/DreamSoul-AI/RPipe/actions/runs/38817527778) / [Unit Tests](https://github.com/DreamSoul-AI/RPipe/actions/runs/38817527562) | core 1452 passed / 1 平台 skip；CPU Flow 189 passed | 坏result不猜成功、坏jobs缓存请求make；设计不再待审。 |
| [PR #121](https://github.com/DreamSoul-AI/RPipe/pull/121) / [Issue #120](https://github.com/DreamSoul-AI/RPipe/issues/120) | 可选state元数据 I/O 边界 | `039db15f4df4`；[Package Check](https://github.com/DreamSoul-AI/RPipe/actions/runs/38819287687) / [Unit Tests](https://github.com/DreamSoul-AI/RPipe/actions/runs/38819287560) | core 1482 passed / 1 平台 skip；CPU Flow 189 passed | stat拒绝不遮蔽合法JSONL；普通RuntimeError仍传播。 |
| [PR #123](https://github.com/DreamSoul-AI/RPipe/pull/123) / [Issue #122](https://github.com/DreamSoul-AI/RPipe/issues/122) | 整数张量非零容差丢位 | `17460b1b579d`；[Package Check](https://github.com/DreamSoul-AI/RPipe/actions/runs/38850576040) / [Unit Tests](https://github.com/DreamSoul-AI/RPipe/actions/runs/38850576054) | core 1507 passed / 1 平台 skip；CPU Flow 189 passed | 2**60附近的int64及uint64不再因float64丢位假通过；bool/零容差保持。 |
| [PR #125](https://github.com/DreamSoul-AI/RPipe/pull/125) / [Issue #124](https://github.com/DreamSoul-AI/RPipe/issues/124) | 有限容差算术溢出 | `82205cbe7fc8`；[Package Check](https://github.com/DreamSoul-AI/RPipe/actions/runs/38852496728) / [Unit Tests](https://github.com/DreamSoul-AI/RPipe/actions/runs/38852496656) | core 1541 passed / 1 平台 skip；CPU Flow 189 passed | 有限diff/bound同时inf或复数0×inf=NaN不再假通过；不是训练精度改动。 |

#117–#125 中的 #118/#120/#122/#124 是对应 Issue，#117/#119/#121/#123/#125 是修复 PR，不能把相邻编号都称为 PR。PR117 的27项新边界、PR119 的20项、PR121 原2项归成32实例（净增30）、PR123 的25项，以及PR125 的34项都保留各自红/控制/绿证据。PR125 的30项先红15失败/15控制，另4项为冻结原比较器回放2失败/2控制，明确不是当前 HEAD 红。最终本机 readout559/core1541+1既有POSIX skip/CPU Flow189；这些是工程正确性，不是新的模型结果。原始材料在 `.tmp/audit095`–`.tmp/audit099`，不随 clone 提供；公共 PR/CI 和 [record §132–145](../development/record.md) 可随仓库核验。

更早夜间工作也已集成：[PR103](https://github.com/DreamSoul-AI/RPipe/pull/103) 建立双语计划与原规范入口；[PR105](https://github.com/DreamSoul-AI/RPipe/pull/105) 统一非有限值身份；[PR107](https://github.com/DreamSoul-AI/RPipe/pull/107) 清本写者只读文本 staging；[PR109](https://github.com/DreamSoul-AI/RPipe/pull/109) 保留 shared 发布原异常；[PR112](https://github.com/DreamSoul-AI/RPipe/pull/112) 计数格类型防护；[PR113](https://github.com/DreamSoul-AI/RPipe/pull/113) 保留含不成对 Unicode 的测试失败证据；[PR115](https://github.com/DreamSoul-AI/RPipe/pull/115) 隔离可选 YAML 递归解析失败。全部是按已有三类合同做修复/证据，不引入尝试目录、统一新文件格式、GPU互斥或新logger，也没有重新执行长实验。B065性能去重仍已撤回，不恢复。

历史科学结果的实际边界：main_probe 在 v0.3.0 / `804b3823` 的八格60-step完整验收通过，参数最大差0、train/test Loss最大差4.44e-16；后续共享输入只做了报告中列出的CPU单格/两格检查，不是新完整八格。main_exp 同一发布来源的八格600-step、32 train各80000-step及32自身best eval、固定曲线门通过；旧图估读最大终点均值差0.254999个百分点，不等于全部原始历史权重逐位相同。当前 dev 的上述两套完整科学矩阵均未重跑。数字的正式来源仍为 [probe报告](../../studies/main_probe/docs/STUDY_REPORT.md) 与 [exp报告](../../studies/main_exp/docs/STUDY_REPORT.md)，原 Run/JSON/图和失败不迁移、不重写。

2026-10-16后续正确性补充：[PR128](https://github.com/DreamSoul-AI/RPipe/pull/128)已正常合dev2008753，最终head及实际dev [Unit38857214651](https://github.com/DreamSoul-AI/RPipe/actions/runs/38857214651)/[Package38857214660](https://github.com/DreamSoul-AI/RPipe/actions/runs/38857214660)全部实际矩阵/聚合成功。B100混合数值精度138新合同全部执行，最终本机readout697/core1679+1既有平台skip/CPU Flow189通过；原红与冻结极值补充证据区分，见[record§148–150](../development/record.md)。Issue127及其他已处理Issue按最新清理授权关闭，代码只在dev、未发布main；这是工程验收，不是新科学实验或文献结论。

<a id="gaps-next"></a>
## （二）未完成项、今晚审阅与下一步实验

| **顺序 / 问题** | **下一步与进入条件** | **固定对照、预算和判读** | **当前状态 / 停止条件** |
|---|---|---|---|
| P0 人工审阅 | Enmao 审本计划中英文、五个完整Study章节、夜间修复与缺口；指出方向变化再改计划 | 单一项目计划；历史授权与当前授权分开 | 本轮文档整理；不是新的实验验收 |
| P1 E0 文献证据 | 先补L3可读原文/相关工作流语义，再做定向引文追踪；记录版本、实际阅读、纳排依据 | 先确定同故障/同语义协议，未做跨工具实验不称外部优越性 | L1摘要/L2定向章节/L3摘录/L4整页的既有阅读沿用；本轮未新增网络检索或全文阅读 |
| P2 E1/E2 正确性 | 只接续真实可复现反例；先Issue/红合同、最小统一修复、本机CPU及最终head/实际dev CI | 三类合同、原输入字节与peer、普通错误不吞、严格前置输入不降级；独立basetemp | 已修且验收完成的Issue按最新授权关闭；无新反例不制造代码/空PR；不查480s或测速 |
| P3 E3 main_probe | 另获明确GPU/新运行授权后，核对原代码归档、raw/Stats、环境、实际FP32及来源；新version/make | 八组合×seed0，连续60 steps，step30/60完整test10000，各自Loss-best eval，原atol1e-6/rtol1e-5及Loss≤1e-6/整数严格门 | 当前未执行，不把v0.3.0旧结果算当前验收；任何缺格/来源或数值异常即停止，不放宽门 |
| P4 E4 main_exp | P3及自身八格600-step/eval200前置门通过、另获全预算/资源授权后才启动；新version且保留旧身份 | 八模型/数据组合×4seed，32连续80000-step train+32自身best eval，400完整test点；原TARGET/REFERENCE_CURVES估读门 | 当前未执行；200-step两格不解锁；自定义sampler精确resume未验收，训练中断换新身份从头，不续写旧证据 |
| 独立探索 | 跨项目shared [#89](https://github.com/DreamSoul-AI/RPipe/issues/89)、TensorBoard/legacy logger [#90](https://github.com/DreamSoul-AI/RPipe/issues/90) 继续独立文献/协议调研 | 不绑定三类修复，不先迁目录或替换logger | 89/90初步调研交付已完成并关闭；实施仍未授权，shared仍项目根不可变输入、Run可变状态独立 |

未完成的研究能力还包括：当前源码完整科学复验、历史全部原始权重/seed的可取得性、custom sampler精确恢复、异机/独立外部复现和系统文献覆盖。不得由工程通过推断这些已完成。四任务共享24GB GPU只作为已知资源约束；本轮不申请、不占用、不终止GPU任务。未来排程须先确认授权、实际设备资源和同机任务，不能把历史吞吐或旧缓存状态称为新的冷启动/速度证据。

# 十、GitHub 使用与计划变更

本计划是本仓库唯一宏观计划正本；英文只是对应译本。位置沿既有 `docs/research/`，不再并行创建 `docs/research-plan.md`；每个项目只维护本中文计划及完整英译；当前、模板与历史 Study 计划全文统一在第十一章，原独立计划已迁入并移除，不再生成平行 PLAN.md。研究工作流程仍见 [research.md](../development/research.md)，目录以 [layout.md](../code/layout.md) 为准。

后续 Issue 说明关联 RQ/H、输入、失败预测、验证与产物；PR 保留红/绿、实际检查及边界，普通合 dev。已修且验收完成的工程 Issue 按最新清理授权关闭，仍有实际反例或未完成验收的 Issue 保留；#89/#90 仅初步调研交付已完成并关闭，实施未获授权。清理本地/远程分支必须核对已纳入 dev、实际验收与当前活动分支及 GPU 使用保护；未合分支保留，每约 5 个合入 dev 的 PR 全面盘点。不得因本计划发布新 tag/main 或删除数据。

| **日期** | **变更与原因** | **影响** |
|---|---|---|
| 2026-10-16 | 首次按 DEM 更新模板建立 RPipe 宏观中文计划、英译与文献证据入口；补挂可读取的文献规范 | 对齐 RQ、六方面、分层证据和执行状态；不改历史 Study 方案/结果、运行代码或授权 |

规范来源、实际阅读范围与检索限制见 [literature.md](literature.md)；两份研究计划正本仍保持 [standards/README](standards/README.md) 记录的原字节。

<a id="study-plan-inventory"></a>
# 十一、各 Study 的完整计划与历史范围

2026-10-16 按 Enmao 12:53 指示收敛：五份原文件完整并入本章，英文逐节对应。当前方向与授权以第八/九章为准，本章保留方案、历史限制和模板内容，不把历史排班写成今晚执行任务。历史快照只比现行文件少一段2026-10-13后续授权，仍各自完整保留；原字节可由迁移前Git commit核验。机器执行/容量/恢复JSON是证据，不是另一个研究计划，保持原字节与位置。原快照中四处失效相对链接已按实际目标修正；历史 recipe 链接明确固定在迁移前 dev，仅作可取得的代码入口，不替代旧 Run 的冻结来源。

<a id="study-main-probe"></a>
## （一）main_probe：完整短探针协议

以下完整保留原方案；其日期、旧授权、校准/排班文字均是历史范围，不解锁当前 GPU/测速/训练。命令及代码路径仍以原 Study 为工作目录；可点击链接已改为当前可达入口。

[迁移前原文件](https://github.com/DreamSoul-AI/RPipe/blob/82205cbe7fc89a46853eb61c1734d123b7d153f0/studies/main_probe/docs/PLAN.md) · SHA-256 `f8b389ae118a16dc6787945aaf13809549f1bac42fcf333ef470a152f669abbf`.

<!-- BEGIN study-main-probe -->
<a id="study-main-probe--main_probe-计划"></a>
### main_probe 计划

2026-10-13 发布后新增授权：以固定 v0.3.0 / `804b3823` 在本机重新准备、验证、校准并完整执行八个成对组合，见 [本轮执行记录](../../studies/main_probe/docs/v030_execution.md)。下文短实验条目保留当时范围；配方、预算和验收门保持不变。

<a id="study-main-probe--一目标"></a>
#### 一、目标

在同一设备、同一确定性条件下，比较固定 main `98648f3a5c7db7dccf3ca806410d5b6fdee9484c` 的原代码与当前 RPipe 的 60-step 计算是否逐段一致，确认 RPipe 改写没有改变数值行为。这是短预算探针，不代替 [main_exp](../../studies/main_exp/README.md) 的长曲线复现。

<a id="study-main-probe--二配方"></a>
#### 二、配方

MNIST / CIFAR10 × linear / mlp / cnn / resnet18，seed0；全量 train / test，batch250 / test1000，连续 60 optimizer steps，step30 / 60 完整 test；SGD lr0.1、momentum0.9、Nesterov、wd0.0005，无梯度裁剪，cosine T_max60；当前无 BN CNN、原生 Factory 和 Kornia 模型前增强。全 train 统计由归档原 Stats(dim1) 按顺序 batch250 重算。两边都用 deterministic=true、benchmark=false、`CUBLAS_WORKSPACE_CONFIG=:4096:8`。

2026-10-11 的 version03 另显式设置 cudnn_allow_tf32=false、matmul_allow_tf32=false。两侧共享精度配置，recipe 拒绝未声明的精度；门限不变。version02 的通过证据保留在 [版本报告](../../studies/main_probe/docs/versions/main-probe-flow-20261010-02/STUDY_REPORT.md)，不能代替新源码 / 精度基线的八格实测。变更依据为 main_exp 的 [设备数值诊断](../../studies/main_exp/docs/EVAL_DIAGNOSTIC.json)。

<a id="study-main-probe--三统一-flow-入口"></a>
#### 三、统一 Flow 入口

专用计算与注册在 `structure/algorithm/`，六阶段实现在 `flow/<phase>/run.py`；各 `__init__.py` 仅导出，不写阶段实现。collect 的数值比较 helper 在 `flow/collect/comparison.py`，prepare 的数据准备在 `flow/prepare/data.py`；recipe 只校验并注册专用 source。两柱 Python 源码自动进入 provenance，不再手列源码 include。

`rpipe make / launch studies/main_probe` 展开八个成对 Run，每个 Run 只计算自己的 data/model 组合，包含两侧训练和各自 best 独立评测。Algorithm source 为 `main_probe`，recipe 注册本 Study 的 Data 和成对 Algorithm；CPU 准备由 Study `prepare.before(ctx)` 在库初始化后、recipe 注册前完成，以便原版全 train Stats 在 Data / Model 构造之前落盘。工作区位于该 Run 的 `assets/probe/`，不得共享可变原版 config。

- prepare：归档固定原代码、复制该数据集 raw、重算 Stats，检查该模型 CPU 初始化与前向一致性；之后恢复本 Run 的运行时 seed。普通 make 仅展开声明，不执行探针准备。
- execute：成对 Algorithm 先执行原版，再用库 prepare 构造的当前 Data / Model / System / Tracker 执行当前版；恢复准备时记录的初始化 RNG，保持原版先、当前版后及 seed0 语义。每个 Run 只处理一个组合，不调用独立脚本或启动全矩阵。
- collect：从 execute 的 OBSERVATIONS.pt 收集输入、RNG、参数、optimizer、scheduler、完整样本计数及独立 eval 对比。summarize 拼接报告与证据路径，拒绝未通过的数值门，保留失败证据。
- write：在库 result 定稿前投影两侧已有观测，调用库 compare，写完整比较报告；附加门失败写 failed，成功后库登记完整资产清单并定稿。不增加训练次数。
- process：Run 阶段检查本组合；Study 阶段只汇总当前 index 中的八组证据。缺组、重复、失败或不同来源不能算完整通过。

失败保留原始证据并交由 Flow 写 failed 状态。不接受 checkpoint resume；重试须使用新的 Run 目录，不能覆盖此前探针工作区。

<a id="study-main-probe--四门限"></a>
#### 四、门限

step30 / 60 浮点参数与 SGD momentum 状态 atol1e-6 / rtol1e-5，整数 buffer 与 optimizer 配置严格一致；train / test Loss 绝对差≤1e-6，正确样本数相同；初始化参数、CPU 与 CUDA RNG、训练样本索引、实际输入、各段 scheduler 一致。两边按 test Loss 选 best 且选择与权重一致；独立 eval 正确样本数相同、Loss 差≤1e-6。只有八个组合全部通过且来源不变才算完整通过。

<a id="study-main-probe--五验证边界"></a>
#### 五、验证边界

<a id="study-main-probe--2026-10-13-发布前完整短探针"></a>
##### 2026-10-13 发布前完整短探针

用户授权持续缺陷修复和发布准备，本次执行八组合60-step完整短探针，不启动main_exp长训。先逐字节保存可取得的version05声明、index / provenance / process及报告，旧Run不删除；使用main-probe-flow-20261013-06重新make并串行launch。完整raw缓存现已校验，来源与实际FP32条件不变、原数值门不放宽；八组齐全才通过Study终验。

CPU 合同门使用伪计算与微小张量验证阶段职责、正常定稿、数值失败和库 compare 失败，不能证明正式数值一致性。正式复跑使用新的 version / Run ID，保留当前基线、数据、设备与失败证据。此前数值证据不能替代本版本重跑验收；历史授权与当前限定范围分开，不能自动重启完整矩阵。

<a id="study-main-probe--2026-10-12-本轮短实验"></a>
##### 2026-10-12 本轮短实验

本轮仅运行 main-probe-flow-20261012-05 的 MNIST/linear 与 MNIST/cnn 两个 seed0 成对 Run，沿用上述 60-step 配方、完整 test、FP32 和固定数值门，不改预算或门限。先保存可取得的 version03 报告 / 声明快照，再重新 make；本工作区缺失旧 index / provenance / process，不重建历史记录。首次 version04 因缺 evaluate 在 prepare 失败，原 Run 与快照保留；补齐隔离依赖后换新 version05 复验。原计划第二组为 CIFAR10/cnn，但本机 raw 缺失且下载缓慢；完成 MNIST 校验后改用 MNIST/cnn 控制本轮成本，CIFAR10 不在本次实测覆盖内。使用 `rpipe run-one` 顺序执行这两个新 Run，不 launch 其余六组。随后显式 process 验证子集被判为 incomplete，预期该终验拒绝而非八组通过。原始证据保留在新 Run，逐段数字与本轮结论写入 Study 报告。
<!-- END study-main-probe -->

<a id="study-main-exp"></a>
## （二）main_exp：完整长曲线协议

以下完整保留原方案；其日期、旧授权、校准/排班文字均是历史范围，不解锁当前 GPU/测速/训练。命令及代码路径仍以原 Study 为工作目录；可点击链接已改为当前可达入口。

[迁移前原文件](https://github.com/DreamSoul-AI/RPipe/blob/82205cbe7fc89a46853eb61c1734d123b7d153f0/studies/main_exp/docs/PLAN.md) · SHA-256 `6ba2017b66ac2a422fd5753da284296042fe036a69f136e56d17270517c9800a`.

<!-- BEGIN study-main-exp -->
<a id="study-main-exp--main_exp-计划"></a>
### main_exp 计划

2026-10-13 发布后新增授权：以固定 v0.3.0 / `804b3823` 在本机重新准备、验证、校准并在前置门通过后执行完整矩阵，见 [本轮执行记录](../../studies/main_exp/docs/v030_execution.md)。下文发布准备阶段的短实验限制保留历史语境；配方、预算和验收门保持不变。

<a id="study-main-exp--一目标"></a>
#### 一、目标

用历史 main 提交 `4ccb28d0496110253e9f8e3f3df658853f07996b` 的配方，在当前 RPipe 训练循环上重跑 MNIST / CIFAR10 × linear / mlp / cnn / resnet18 × seed0–3 的 80000-step 曲线，与 main README 的两张原图比较。原运行记录没有找回，所以分别验收三件事：配方能完整重跑、同环境下原代码与当前链计算一致、四 seed 均值曲线接近原图。

<a id="study-main-exp--二配方"></a>
#### 二、配方

1. 全部训练数据，完整 10000 张 test。数据、模型、旧 CNN 的四层 BN、CPU torchvision 增强、常量统计和采样顺序直接复用归档代码，由 [recipe.py](../../studies/main_exp/recipe.py) 注册为 `historical_4ccb28d` source。
2. 连续 80000 optimizer steps，每 200 step 完整 test，共 400 个曲线点。train/test batch250，SGD lr0.01、momentum0.9、Nesterov、wd0.0005、clip1、cosine T_max80000，见 [experiment_config.yaml](../../studies/main_exp/experiment_config.yaml)。
3. best 按 RPipe 的全程 test Accuracy 选择；归档代码的旧 best 比较缺陷不移植。曲线取训练内 test history。
4. 确定性条件：deterministic=true、benchmark=false、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、CPU 2 线程。2026-10-11 的新 version03 显式设置 cudnn_allow_tf32=false、matmul_allow_tf32=false，两侧前置对照、训练与独立评测一致；version02 的精度未显式固定且独立评测一致性失败，完整证据保留。新旧精度基线不混用，变更与停止失败的恢复条件见 [eval_recovery.md](../../studies/main_exp/docs/eval_recovery.md)。
5. 训练必须连续。通用原生 resume 已按单进程合同本机补齐 RNG / 数据位置，但本 Study 的自定义 sampler / 数据适配仍未验收精确恢复，继续拒绝中断 train 续跑，换新 `version` 从头跑。

<a id="study-main-exp--三门限"></a>
#### 三、门限

1. preflight：同设备原代码与当前链 600-step / eval200 八格对照全部通过才可 launch。参数 atol1e-6 / rtol1e-5，整数 buffer 一致，Loss 差≤1e-6，正确样本数一致，scheduler、RNG 与实际输入一致；scheduler 与采样总预算仍为 80000。
2. 终验：32 train + 32 eval 全部成功，四 seed 在每个评测点对齐。门限预先固定在 [TARGET.md](../../studies/main_exp/docs/TARGET.md) 与 [REFERENCE_CURVES.json](../../studies/main_exp/docs/REFERENCE_CURVES.json)：均值终点与末 50 点均值距原图估读 ≤0.25 个百分点（MNIST）/ ≤1.00（CIFAR10）；七个锚点 MAE≤0.25 / 1.00、最大偏差≤0.60 / 2.00；末段模型次序与时间波动门。结果出来后不改门限。
3. 原图估读的误差和原始 seed 集合未知，结论限于这组固定估读门。

<a id="study-main-exp--四执行"></a>
#### 四、执行

<a id="study-main-exp--2026-10-13-校准与精度复验"></a>
##### 2026-10-13 校准与精度复验

先用固定源码快照完成 CIFAR10 的 CNN / ResNet18 / 混合任务各三个 seed、800-step、round1/2/3 容量测量，再用新短程 CNN best 做16次显式 FP32和16次 TF32固定权重全 test，覆盖串行与并发；保留逐样本预测 / 输入 / 模型哈希及可取得的 kernel trace。新权重不冒充旧 B-023 的76800-step权重；旧 checkpoint 未找回，旧反例及其 JSON 不覆盖。

容量采用既有 system.config.vram_bytes，库默认经验系数与 safety=0.5 不变。main_exp 的后续 fresh make 使用4.5GiB单进程预算，保守统一覆盖配方的模型；这限制轻型任务的自动并行数，也不保证其他设备 / 配方可沿用。重跑前仍须新的600-step八格前置门。此次只做前置门与短校准，不启动完整矩阵、不覆盖旧 version03。

命令见 [Study 入口](../../studies/main_exp/README.md)。同组并行数由 `rpipe launch` 按显存装箱，必要时用 `--round N`；train 全部结束后再跑 eval。报告写在 `docs/STUDY_REPORT.md`，数字表由 `rpipe report` 生成。

专用 Data / Model 与归档支持放 `structure/`；`flow/prepare/run.py` 核对来源和实际运行条件，`flow/process/run.py` 只在 Study scope 调用 `curves.py`。`__init__.py` 仅导出。raw 准备使用 `flow/prepare_data.py`，速度校准使用 recipe 调用 `flow/execute/speed.py`；手动工具不自动作为 hook，两柱源码自动冻结。

<a id="study-main-exp--2026-10-12-本轮短实验"></a>
##### 2026-10-12 本轮短实验

仅用现有 `recipe.preflight` 做 MNIST/linear、seed0、200-step / 完整 test 的同设备原版与当前链前缀对照，保持 sampler / scheduler 总预算 80000、batch250、原精度与数值门。独立验证标识为 main-exp-flow-smoke-20261012-01，输出到本 Study `shared/validation/` 的新目录，逐段数字写入 `docs/`。这不是正式 64-Run 实验，不改 version03 的声明、index、provenance、Run 或 `PREFLIGHT.json`，也不触发正式 Flow launch；现有 loader / scope 合同检查另行复验。200-step 单组不满足 600-step 八格前置门，不能解锁长实验。以后正式复跑仍需新 version、重新 make 和完整前置门。
<!-- END study-main-exp -->

<a id="study-template"></a>
## （三）新 Study 章节填写骨架

以下完整保留原方案；其日期、旧授权、校准/排班文字均是历史范围，不解锁当前 GPU/测速/训练。命令及代码路径仍以原 Study 为工作目录；可点击链接已改为当前可达入口。

[迁移前原文件](https://github.com/DreamSoul-AI/RPipe/blob/82205cbe7fc89a46853eb61c1734d123b7d153f0/studies/_template/docs/PLAN.md) · SHA-256 `cf5e709413dfc15136af1f2c8e0438ff8dad23bae8dd2fb1a769ef6da4f76925`.

<!-- BEGIN study-template -->
<a id="study-template--study-plan-my_study"></a>
### Study Plan: my_study

> 按[研究工作要求](../development/research.md)补齐本轮问题、假设、文献/数据/模型/算法/指标/实验设计；可引用已明确的宏观方案，不适用项说明理由。下方是执行骨架，不是完整研究计划。排班见[根 README 的 Study 指南 §3](../../README.md#3-高效率实验设计)，跑完写同目录 `STUDY_REPORT.md`。

<a id="study-template--1-研究问题"></a>
#### 1. 研究问题

（这一轮要比什么？成功长什么样？）

<a id="study-template--2-study--experiment--run"></a>
#### 2. Study / Experiment / Run

- Study 名：`my_study`
- Experiment：由 `axes` 展开（不含 seed）
- Run：每个 Experiment × `seeds`

<a id="study-template--3-变量轴"></a>
#### 3. 变量轴

| **轴** | **字段** | **取值** |
|----|------|------|
| （因素） | | |

<a id="study-template--4-固定条件"></a>
#### 4. 固定条件

| **项** | **取值** |
|----|------|
| `seeds` | |
| `system.device` | `cuda` |

<a id="study-template--5-高效率排班3"></a>
#### 5. 高效率排班（§3）

- **同类一组：** 同一 `model.name`、相近耗时的 Run 一起并行；不要把 resnet 和 linear 塞进同一 `wait`。
- **`wait` 闸门：** 组末必须等本组进程退出、显存释放完，才开下一组。不 wait 下一波会挤进来，容易 OOM。
- **吃满 GPU：** `python -m rpipe make studies/<name> --num-gpus 1` 默认 `--round auto`；看 **make** 打印的 `pack N waits`（launch 不重印）。conservative 墙钟是各组最慢条加总。
- **依赖：** 有 `algorithm.mode: eval` 时先全部 train `wait` 完，再 eval。
- **error：** 单条失败记 `run_id`；本波结束后对未 succeeded 的格子 `resume: latest` 再跑。train 重试成功后才放行依赖 eval；最终失败则依赖 eval 不算成功。
- **seed ≠ 并发。**

```bash
python -m rpipe make studies/<name> --num-gpus 1 --init-gpu 0
python -m rpipe launch studies/<name> --num-gpus 1 --init-gpu 0
```

<a id="study-template--51-时长预估"></a>
##### 5.1 时长预估

`make` 之后、`launch` 之前填。每一行是一条 Run 的预估秒数。同一 `wait` 并行，组墙钟取该组最慢的一条；整轮是各组相加。不含显存。

| **wait** | **factors** | **mode** | **seed** | **id** | **est** |
|---:|---|---|---:|---|---:|
|  |  |  |  |  |  |
| | 整轮 | | | |  |

<a id="study-template--6-成功标准"></a>
#### 6. 成功标准

- 按本轮研究问题明确数值、质量或其他证据判据；下面的执行和产物条件不能单独证明研究质量通过
- 计划的 Run 均 `status: succeeded`
- `docs/figures/learning_curves.png`
- `STUDY_REPORT.md` 嵌图，并写清实际 `pack` 排班

<a id="study-template--7-执行范围与验证边界"></a>
#### 7. 执行范围与验证边界

填写本轮允许执行的规模、资源和预算，列出未覆盖场景。说明失败证据与重复运行如何保留，短探针和长实验分别使用哪些验收条件。
<!-- END study-template -->

<a id="archive-main-probe-v030"></a>
## （四）main_probe：v0.3.0 本机复验前历史计划全文

以下完整保留原方案；其日期、旧授权、校准/排班文字均是历史范围，不解锁当前 GPU/测速/训练。命令及代码路径仍以原 Study 为工作目录；可点击链接已改为当前可达入口。

[迁移前原文件](https://github.com/DreamSoul-AI/RPipe/blob/82205cbe7fc89a46853eb61c1734d123b7d153f0/studies/main_probe/docs/versions/v030-before-local-20261013/docs/PLAN.md) · SHA-256 `49675c7d7fbcc82e918d1e15b976aecb03569a4037ad3eec7e99dc07634a28e3`.

<!-- BEGIN archive-main-probe-v030 -->
<a id="archive-main-probe-v030--main_probe-计划"></a>
### main_probe 计划

<a id="archive-main-probe-v030--一目标"></a>
#### 一、目标

在同一设备、同一确定性条件下，比较固定 main `98648f3a5c7db7dccf3ca806410d5b6fdee9484c` 的原代码与当前 RPipe 的 60-step 计算是否逐段一致，确认 RPipe 改写没有改变数值行为。这是短预算探针，不代替 [main_exp](../../studies/main_exp/README.md) 的长曲线复现。

<a id="archive-main-probe-v030--二配方"></a>
#### 二、配方

MNIST / CIFAR10 × linear / mlp / cnn / resnet18，seed0；全量 train / test，batch250 / test1000，连续 60 optimizer steps，step30 / 60 完整 test；SGD lr0.1、momentum0.9、Nesterov、wd0.0005，无梯度裁剪，cosine T_max60；当前无 BN CNN、原生 Factory 和 Kornia 模型前增强。全 train 统计由归档原 Stats(dim1) 按顺序 batch250 重算。两边都用 deterministic=true、benchmark=false、`CUBLAS_WORKSPACE_CONFIG=:4096:8`。

2026-10-11 的 version03 另显式设置 cudnn_allow_tf32=false、matmul_allow_tf32=false。两侧共享精度配置，recipe 拒绝未声明的精度；门限不变。version02 的通过证据保留在 [版本报告](../../studies/main_probe/docs/versions/main-probe-flow-20261010-02/STUDY_REPORT.md)，不能代替新源码 / 精度基线的八格实测。变更依据为 main_exp 的 [设备数值诊断](../../studies/main_exp/docs/EVAL_DIAGNOSTIC.json)。

<a id="archive-main-probe-v030--三统一-flow-入口"></a>
#### 三、统一 Flow 入口

专用计算与注册在 `structure/algorithm/`，六阶段实现在 `flow/<phase>/run.py`；各 `__init__.py` 仅导出，不写阶段实现。collect 的数值比较 helper 在 `flow/collect/comparison.py`，prepare 的数据准备在 `flow/prepare/data.py`；recipe 只校验并注册专用 source。两柱 Python 源码自动进入 provenance，不再手列源码 include。

`rpipe make / launch studies/main_probe` 展开八个成对 Run，每个 Run 只计算自己的 data/model 组合，包含两侧训练和各自 best 独立评测。Algorithm source 为 `main_probe`，recipe 注册本 Study 的 Data 和成对 Algorithm；CPU 准备由 Study `prepare.before(ctx)` 在库初始化后、recipe 注册前完成，以便原版全 train Stats 在 Data / Model 构造之前落盘。工作区位于该 Run 的 `assets/probe/`，不得共享可变原版 config。

- prepare：归档固定原代码、复制该数据集 raw、重算 Stats，检查该模型 CPU 初始化与前向一致性；之后恢复本 Run 的运行时 seed。普通 make 仅展开声明，不执行探针准备。
- execute：成对 Algorithm 先执行原版，再用库 prepare 构造的当前 Data / Model / System / Tracker 执行当前版；恢复准备时记录的初始化 RNG，保持原版先、当前版后及 seed0 语义。每个 Run 只处理一个组合，不调用独立脚本或启动全矩阵。
- collect：从 execute 的 OBSERVATIONS.pt 收集输入、RNG、参数、optimizer、scheduler、完整样本计数及独立 eval 对比。summarize 拼接报告与证据路径，拒绝未通过的数值门，保留失败证据。
- write：在库 result 定稿前投影两侧已有观测，调用库 compare，写完整比较报告；附加门失败写 failed，成功后库登记完整资产清单并定稿。不增加训练次数。
- process：Run 阶段检查本组合；Study 阶段只汇总当前 index 中的八组证据。缺组、重复、失败或不同来源不能算完整通过。

失败保留原始证据并交由 Flow 写 failed 状态。不接受 checkpoint resume；重试须使用新的 Run 目录，不能覆盖此前探针工作区。

<a id="archive-main-probe-v030--四门限"></a>
#### 四、门限

step30 / 60 浮点参数与 SGD momentum 状态 atol1e-6 / rtol1e-5，整数 buffer 与 optimizer 配置严格一致；train / test Loss 绝对差≤1e-6，正确样本数相同；初始化参数、CPU 与 CUDA RNG、训练样本索引、实际输入、各段 scheduler 一致。两边按 test Loss 选 best 且选择与权重一致；独立 eval 正确样本数相同、Loss 差≤1e-6。只有八个组合全部通过且来源不变才算完整通过。

<a id="archive-main-probe-v030--五验证边界"></a>
#### 五、验证边界

<a id="archive-main-probe-v030--2026-10-13-发布前完整短探针"></a>
##### 2026-10-13 发布前完整短探针

用户授权持续缺陷修复和发布准备，本次执行八组合60-step完整短探针，不启动main_exp长训。先逐字节保存可取得的version05声明、index / provenance / process及报告，旧Run不删除；使用main-probe-flow-20261013-06重新make并串行launch。完整raw缓存现已校验，来源与实际FP32条件不变、原数值门不放宽；八组齐全才通过Study终验。

CPU 合同门使用伪计算与微小张量验证阶段职责、正常定稿、数值失败和库 compare 失败，不能证明正式数值一致性。正式复跑使用新的 version / Run ID，保留当前基线、数据、设备与失败证据。此前数值证据不能替代本版本重跑验收；历史授权与当前限定范围分开，不能自动重启完整矩阵。

<a id="archive-main-probe-v030--2026-10-12-本轮短实验"></a>
##### 2026-10-12 本轮短实验

本轮仅运行 main-probe-flow-20261012-05 的 MNIST/linear 与 MNIST/cnn 两个 seed0 成对 Run，沿用上述 60-step 配方、完整 test、FP32 和固定数值门，不改预算或门限。先保存可取得的 version03 报告 / 声明快照，再重新 make；本工作区缺失旧 index / provenance / process，不重建历史记录。首次 version04 因缺 evaluate 在 prepare 失败，原 Run 与快照保留；补齐隔离依赖后换新 version05 复验。原计划第二组为 CIFAR10/cnn，但本机 raw 缺失且下载缓慢；完成 MNIST 校验后改用 MNIST/cnn 控制本轮成本，CIFAR10 不在本次实测覆盖内。使用 `rpipe run-one` 顺序执行这两个新 Run，不 launch 其余六组。随后显式 process 验证子集被判为 incomplete，预期该终验拒绝而非八组通过。原始证据保留在新 Run，逐段数字与本轮结论写入 Study 报告。
<!-- END archive-main-probe-v030 -->

<a id="archive-main-exp-v030"></a>
## （五）main_exp：v0.3.0 本机复验前历史计划全文

以下完整保留原方案；其日期、旧授权、校准/排班文字均是历史范围，不解锁当前 GPU/测速/训练。命令及代码路径仍以原 Study 为工作目录；可点击链接已改为当前可达入口。

[迁移前原文件](https://github.com/DreamSoul-AI/RPipe/blob/82205cbe7fc89a46853eb61c1734d123b7d153f0/studies/main_exp/docs/versions/v030-before-local-20261013/docs/PLAN.md) · SHA-256 `257f348f1151f24c902145d2d386506e295217da4c3131cb72c4096c1cea832c`.

<!-- BEGIN archive-main-exp-v030 -->
<a id="archive-main-exp-v030--main_exp-计划"></a>
### main_exp 计划

<a id="archive-main-exp-v030--一目标"></a>
#### 一、目标

用历史 main 提交 `4ccb28d0496110253e9f8e3f3df658853f07996b` 的配方，在当前 RPipe 训练循环上重跑 MNIST / CIFAR10 × linear / mlp / cnn / resnet18 × seed0–3 的 80000-step 曲线，与 main README 的两张原图比较。原运行记录没有找回，所以分别验收三件事：配方能完整重跑、同环境下原代码与当前链计算一致、四 seed 均值曲线接近原图。

<a id="archive-main-exp-v030--二配方"></a>
#### 二、配方

1. 全部训练数据，完整 10000 张 test。数据、模型、旧 CNN 的四层 BN、CPU torchvision 增强、常量统计和采样顺序直接复用归档代码，由 [recipe.py](https://github.com/DreamSoul-AI/RPipe/blob/82205cbe7fc89a46853eb61c1734d123b7d153f0/studies/main_exp/recipe.py) 注册为 `historical_4ccb28d` source。
2. 连续 80000 optimizer steps，每 200 step 完整 test，共 400 个曲线点。train/test batch250，SGD lr0.01、momentum0.9、Nesterov、wd0.0005、clip1、cosine T_max80000，见 [experiment_config.yaml](../../studies/main_exp/docs/versions/v030-before-local-20261013/experiment_config.yaml)。
3. best 按 RPipe 的全程 test Accuracy 选择；归档代码的旧 best 比较缺陷不移植。曲线取训练内 test history。
4. 确定性条件：deterministic=true、benchmark=false、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、CPU 2 线程。2026-10-11 的新 version03 显式设置 cudnn_allow_tf32=false、matmul_allow_tf32=false，两侧前置对照、训练与独立评测一致；version02 的精度未显式固定且独立评测一致性失败，完整证据保留。新旧精度基线不混用，变更与停止失败的恢复条件见 [eval_recovery.md](../../studies/main_exp/docs/versions/v030-before-local-20261013/docs/eval_recovery.md)。
5. 训练必须连续。通用原生 resume 已按单进程合同本机补齐 RNG / 数据位置，但本 Study 的自定义 sampler / 数据适配仍未验收精确恢复，继续拒绝中断 train 续跑，换新 `version` 从头跑。

<a id="archive-main-exp-v030--三门限"></a>
#### 三、门限

1. preflight：同设备原代码与当前链 600-step / eval200 八格对照全部通过才可 launch。参数 atol1e-6 / rtol1e-5，整数 buffer 一致，Loss 差≤1e-6，正确样本数一致，scheduler、RNG 与实际输入一致；scheduler 与采样总预算仍为 80000。
2. 终验：32 train + 32 eval 全部成功，四 seed 在每个评测点对齐。门限预先固定在 [TARGET.md](../../studies/main_exp/docs/versions/v030-before-local-20261013/docs/TARGET.md) 与 [REFERENCE_CURVES.json](../../studies/main_exp/docs/versions/v030-before-local-20261013/docs/REFERENCE_CURVES.json)：均值终点与末 50 点均值距原图估读 ≤0.25 个百分点（MNIST）/ ≤1.00（CIFAR10）；七个锚点 MAE≤0.25 / 1.00、最大偏差≤0.60 / 2.00；末段模型次序与时间波动门。结果出来后不改门限。
3. 原图估读的误差和原始 seed 集合未知，结论限于这组固定估读门。

<a id="archive-main-exp-v030--四执行"></a>
#### 四、执行

<a id="archive-main-exp-v030--2026-10-13-校准与精度复验"></a>
##### 2026-10-13 校准与精度复验

先用固定源码快照完成 CIFAR10 的 CNN / ResNet18 / 混合任务各三个 seed、800-step、round1/2/3 容量测量，再用新短程 CNN best 做16次显式 FP32和16次 TF32固定权重全 test，覆盖串行与并发；保留逐样本预测 / 输入 / 模型哈希及可取得的 kernel trace。新权重不冒充旧 B-023 的76800-step权重；旧 checkpoint 未找回，旧反例及其 JSON 不覆盖。

容量采用既有 system.config.vram_bytes，库默认经验系数与 safety=0.5 不变。main_exp 的后续 fresh make 使用4.5GiB单进程预算，保守统一覆盖配方的模型；这限制轻型任务的自动并行数，也不保证其他设备 / 配方可沿用。重跑前仍须新的600-step八格前置门。此次只做前置门与短校准，不启动完整矩阵、不覆盖旧 version03。

命令见 [Study 入口](../../studies/main_exp/docs/versions/v030-before-local-20261013/README.md)。同组并行数由 `rpipe launch` 按显存装箱，必要时用 `--round N`；train 全部结束后再跑 eval。报告写在 `docs/STUDY_REPORT.md`，数字表由 `rpipe report` 生成。

专用 Data / Model 与归档支持放 `structure/`；`flow/prepare/run.py` 核对来源和实际运行条件，`flow/process/run.py` 只在 Study scope 调用 `curves.py`。`__init__.py` 仅导出。raw 准备使用 `flow/prepare_data.py`，速度校准使用 recipe 调用 `flow/execute/speed.py`；手动工具不自动作为 hook，两柱源码自动冻结。

<a id="archive-main-exp-v030--2026-10-12-本轮短实验"></a>
##### 2026-10-12 本轮短实验

仅用现有 `recipe.preflight` 做 MNIST/linear、seed0、200-step / 完整 test 的同设备原版与当前链前缀对照，保持 sampler / scheduler 总预算 80000、batch250、原精度与数值门。独立验证标识为 main-exp-flow-smoke-20261012-01，输出到本 Study `shared/validation/` 的新目录，逐段数字写入 `docs/`。这不是正式 64-Run 实验，不改 version03 的声明、index、provenance、Run 或 `PREFLIGHT.json`，也不触发正式 Flow launch；现有 loader / scope 合同检查另行复验。200-step 单组不满足 600-step 八格前置门，不能解锁长实验。以后正式复跑仍需新 version、重新 make 和完整前置门。
<!-- END archive-main-exp-v030 -->
