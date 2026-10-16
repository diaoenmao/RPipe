# RPipe 研究计划文献证据

## 一、摘要与适用边界

本页服务于 [RPipe 中文研究计划](<研究计划 RPipe.md>)及其 [English counterpart](<Research Plan RPipe.md>)：用定向初筛界定复现证据和系统合同，不声称穷尽研究空白。2026-10-16 实际检索并阅读四条来源，深度分别为摘要、定向章节、官方检索摘录和版本化文档；L3 全文直读受限，ACM 候选未核验，不用于结论。没有下载 PDF、运行外部实现、安装 skill 或产生新的科学数字。

判断：已有故障支持继续研究三个合同家族；现有文献只提供方法论与条件边界，尚不足以宣称 RPipe 优于外部工作流。下一步以真实失效案例的可验证合同为主，补外部基线协议和引文追踪，不能用相关论文数量代替判断。正式实测数字仍在对应 Study。

## 二、规范、检索与筛选

### （一）实际采用的材料

- [研究计划规范](<standards/研究计划规范 Research Plan Standard.md>)和[模板](<standards/研究计划模板 Research Plan Template.md>)：完整阅读；随 Git 的 2026-10-16 交付版本为准，来源/字节哈希见 [standards/README](standards/README.md)。不以本机较旧 DEM 同名版本替换。
- [文献调研规范中文原件](<standards/文献调研规范 Literature Review Standard.md>)：本轮完整读取 623 行，按原字节复制，共 34809 字节，SHA-256 `2605cf562aee59b2023ecb0954c3639658958c1874ddce79a623dd3df63489a8`。实际来源是可读本机 DEM `文档/规范/研发/研究/文献调研规范 Literature Review Standard.md`，不是声称 Office 直读成功。保留原文内部相对引用；缺失的论文库治理依赖见正本入口，不复制整个 DEM 文档树。
- 本地 `D:/GitHub/DreamSoul/Knowledge` 提交 `0d7c68daa2d105a5282d15da2944075f2abe29f0`：完整阅读 `skills/literature-search/SKILL.md`、`templates/research-note.md`、`wiki/research/literature-search/README.md`。采用查询组、去重、证据地图与阅读深度记录；不采用“引用越多越好”作为质量判据，不声称已安装或运行工具。该本机阅读记录不会随 clone 提供，正式可执行要求已写在本页；历史 knowledge 参考链接仍见[研究工作要求](../development/research.md#六复用-knowledge-的研究-skills)。本轮未阅读 paper-analysis、通用调研规范或论文库治理全文。

### （二）范围与实际查询

信息截止日、检索日和访问日均为 2026-10-16。围绕执行与证据一致性构建概念组：reproducibility / provenance、failure evidence / integrity、workflow / optional metadata / atomic publication。实际执行以下四组网页查询，随后直读候选原页；没有逐数据库检索、系统筛全或统计完整命中总数。

| **查询** | **来源限制与目的** |
|---|---|
| `site.jmlr.org Improving Reproducibility in Machine Learning Research Pineau 2021 20-303` | 原期刊页，核对研究复现组织与记录职责 |
| `site.arxiv.org 2204.07610 Sources of Irreproducibility Machine Learning Review` | 作者公开版本，区分复现因素与结论层次 |
| `site.acm.org publications policies artifact review badging current functional reusable` | 候选构件评价政策；原页访问失败，未采用 |
| `site.f1000research.com Snakemake sustainable data analysis 10 33 2021` | 原论文工作流路线，只有官方摘录可读 |

PyTorch 文档为根据运行条件问题定向打开的官方来源，固定到 2.9 页面，不把 `/stable/` 的未来变化当本轮阅读版本。检索结果中的二手摘要、论坛和无关论文未用作证据；同一来源多个版本只算一条。纳入依据是与问题直接相关且能核对身份/阅读范围，非引用量、场所排名或资料条数。本轮未核验分区/排名，不编造排名字段。

## 三、路线与证据地图

| **路线** | **代表来源** | **能回答什么** | **本计划判断与缺口** |
|---|---|---|---|
| 复现组织与报告 | L1 | 代码、检查项与复现活动的角色 | 对应 RQ1/RQ4 的来源与验收分工；组织建议不等于当前代码通过 |
| 从设计到结论的失效因素 | L2 | 研究设计、实现、评价与文档的条件 | 对应全部 RQ 的效度边界；必须保留原始失败，不能只记成功 |
| 可追溯工作流系统 | L3 | 提供后续外部协议基线候选 | 阅读不足，不比较功能覆盖/效果，更不排名速度 |
| 运行时随机性条件 | L4 | 平台/版本与随机性的约束 | 对应 RQ4；原 Study 条件对齐不能由相同 seed 替代 |

三个工程合同的具体判断来自仓库红回归与实际实现，不把 L1–L4 写成它们的实证验证。外部工作流是否在同一故障协议下提供相同保证仍待核查；当前的差距是证据缺口，不能写成已证明的方法缺口。

## 四、来源证据表

### L1

**身份与版本：** Joelle Pineau 等，*Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)*，JMLR 22(164):1–20，2021。[原期刊页面](https://jmlr.org/papers/v22/20-303.html)。稳定定位为期刊卷/文号 `22/20-303`，不将不同入口重复计数。

**实际阅读：** 官方摘要；未精读全文、附录或二级引用。作者介绍代码、挑战和检查项构成的复现计划。本轮不借用项目调查数字、模型结果或统计结论。

**六方面提取：** 问题为研究复现；数据/模型、算法细节及指标/实验条件未作全文提取。用途是区分报告职责与独立验证（RQ1/RQ4），不据此判断 RPipe 的外部复现效果。下一步若采用其评价方法，先补读相应正文及量化口径。

### L2

**身份与版本：** Odd Erik Gundersen、Kevin Coakley、Christine Kirkpatrick、Yolanda Gil，*Sources of Irreproducibility in Machine Learning: A Review*。[arXiv 身份页](https://arxiv.org/abs/2204.07610)，本轮固定读取 [v2 HTML](https://arxiv.org/html/2204.07610v2)，2023-04-14；v1 为 2022 年，不将 v2 写成 2022 版本。

**实际阅读：** 摘要、§2–4、§6、§8–9 与设计/实现/评价/文档相关内容的定向阅读；未逐篇核验其引文或精读全部附录。作者将复现问题分解为多个研究环节。

**六方面提取：** 对象是方法论分类；本轮不转录其数据/模型数字或跨论文指标。计划据此将执行、分析、结论及条件分开，保留反例与证据不足（RQ1–RQ4）；这是本计划的应用判断，不是作者对 RPipe 的评价。

### L3

**身份与版本：** Felix Mölder 等，*Sustainable data analysis with Snakemake*，F1000Research 10:33，2021，固定 v1。[原文 v1](https://f1000research.com/articles/10-33/v1)，[DOI 10.12688/f1000research.29032.1](https://doi.org/10.12688/f1000research.29032.1)。

**实际阅读：** 搜索返回的官方结论与软件/数据可用性摘录。随后原页及其他版本直读返回 403/访问错误，因此只记为**初筛**。其可追溯/文档/可移植工作流路线可列候选；完整方法、基准数据/模型、指标和实验条件未提取、未复现。

**用途与限制：** 后续外部基线候选（RQ1–RQ3），当前不能推出其缺少或拥有本仓库同一故障保证，不能比较效果/性能。先取得并阅读所用版本，再制定相同输入与语义的对照协议；本轮不安装 Snakemake。

### L4

**身份与版本：** PyTorch 官方文档 2.9，*Reproducibility*。[版本化页面](https://docs.pytorch.org/docs/2.9/notes/randomness.html)。页面标记最近更新 2024-11-26；本轮读取 2.9 URL，不把该日期当 PyTorch 版本发布日期。

**实际阅读：** 整个文档页面；未逐个阅读其外部链接。文档说明相同 seed 仍不能保证不同版本/平台乃至 CPU/GPU 的完整等价，并讨论随机性控制。

**六方面提取：** 这是运行条件参考，不是本课题模型、数据或指标实验。用于 RQ4 的条件登记；具体 Study 使用的 PyTorch/精度仍以其环境与原计划为准，本轮不因文档版本而升级运行时，不设置新性能选项。

## 五、失败、未读与继续顺序

| **事项** | **已取得的证据** | **处置与后续** |
|---|---|---|
| ACM Artifact Review and Badging 候选 | [原页](https://www.acm.org/publications/policies/artifact-review-and-badging-current)直读失败；无可用正文 | 未采用其条款/标签含义，不从搜索片段推断政策；需要时另核验 |
| L3 全文访问 | 原页直读受限，只有官方搜索摘录 | 保留初筛等级；不冒充全文阅读或已实现外部基线 |
| 文献完整性 | 四条定向来源，无逐库/系统引文追踪 | 后续按用途补强，不称完成系统综述或证明新颖性 |
| 原件依赖 | DEM 相对路径包含论文库治理等文件 | 保留原字节，缺失依赖显式登记；不伪造、全量迁移或额外建库 |
| PDF/本地论文库 | 本轮未下载 PDF，也未建立正式 PDF 索引 | 不编造文件、PDF 哈希、已读页码或材料获取成功 |

优先补读 L3 和与真实合同直接相关的外部工作流语义，再做前向/后向引文追踪与更新文献筛选。只有选择了可执行外部比较后才细化数据、模型、指标与公平条件；未授权的训练、共享/logger 迁移、性能测量不由文献任务自动启动。新来源追加版本、实际阅读范围、支持/反驳关系及路线变更理由。

英文读者从 [English plan, Chapter II](<Research Plan RPipe.md#ii-literature-review>)进入；本证据表与文献规范目前以中文为正本，未声称有整份文献规范英译。研究计划及其规范/模板已有对应完整英文。
