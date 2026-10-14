# 2026-10-15 正确性维护归档与接续状态

## 一、摘要

本文件记录PR #46合入前的归档快照：此前源码、测试和记录已提交并推送至codex/cache-lock-budget，基准工作提交为927c36e5f86c1ada49a2768d632cda66018e0b56；随后按最终“只做正确性”要求单独撤回B-065次数去重优化，再提交本状态文档。没有重写历史、强推、删除分支或数据。

维护者最终授权[PR #46](https://github.com/DreamSoul-AI/RPipe/pull/46)从工作分支转ready，在最终提交全部必需CI成功后合入dev。此前保持草案/不合dev的文字均为旧阶段要求。main仍为已发布v0.3.1/7a15c46b75c5eb951cccd60ed7b9fc3463db90f5，不合main或发布。

本文不预先声明合并已经发生：实际mergeCommit、最终工作提交CI及合入后的dev CI，以PR和GitHub Actions为权威；接续者应先查PR当前状态，已MERGED则不要再次执行合并。归档时dev为9324014985680a0d186863cd135faa53a2e7bcbf。

## 二、已完成

| **范围** | **改前 → 改后** | **状态与证据** |
| --- | --- | --- |
| B-056 / B-057 | 非法共享锁预算、同tick文本staging碰撞 → 入口校验与写者独立短UUID | 源码与回归已提交；详细红例/验证保留record §43–45 |
| B-058 / B-059 | 测试事件起点错误、共享清单别名可进入builder → 正确阶段起点与构建前验证 | 已提交；record §45–46 |
| B-060 / B-061 / B-063 | packed/worktree、失败Git查询、损坏可选元数据可能误记或中断 → 有界回退与失败隔离 | 已提交；record §47–48、§50 |
| B-062 / B-064 | 共享引用staging碰撞、完整目标名撑破临时名预算 → 独立短名、失败只清理自己 | 已提交；record §49、§51、§55 |
| B-066 / B-067 | Study遮蔽库来源、冻结读取错误覆盖旧成功 → 库键域校验、冻结拒绝保全 | 已提交；record §53–55 |
| B-068 | 可选活动提示坏编码/权限错误遮蔽正式status → 忽略不可用hint并保留index退出合同 | 已提交；record §56 |
| B-065 | 历史草案新增单次读取次数去重 → 按最终范围撤回该优化，保留正确性校验 | 独立撤回主题提交；record §57；不进入本轮正确性修复成果 |

927c36e的[Unit](https://github.com/DreamSoul-AI/RPipe/actions/runs/38618519093)、[Package](https://github.com/DreamSoul-AI/RPipe/actions/runs/38618518990)、[Branch](https://github.com/DreamSoul-AI/RPipe/actions/runs/38618518984)及所有实际矩阵均已通过。执行清单checks-927c36e5f86c.json SHA-256为c2426cdef6af14c1144774045a2e74d784720835e51c790b21eee70665ab0ed5；它是该提交的证据，不能代替后续归档提交或实际dev集成验收。

高风险并发补验已提交85d6070：B-062与B-067新增8项在旧函数全部失败，当前相关53项通过；旧成功结果逐字节保留、失败写者不删除另一写者staging，Run仍单写者。撤回B-065优化后，来源/冻结相关53项正确性回归也通过；没有性能采样或比较。B-068新增13项与status相关34项通过。各批次原红例、候选和最终结果分别保留，不互相替代。

所有非忽略源码/测试改动按主题提交；本机.tmp、venv、数据、Run、checkpoint、index/process/provenance依现有忽略规则保留在原位置，不强行加入Git，也不承诺它们会随clone取得。旧源码与开放缺陷全文可从[归档前提交](https://github.com/DreamSoul-AI/RPipe/blob/927c36e5f86c1ada49a2768d632cda66018e0b56/docs/development/bugs.md)恢复，历史分支全部保留。

## 三、尚未完成或明确不执行

1. 归档文档主题提交之后，须核对最终head的全部Unit tests / Build package / Branch flow及实际子任务，再合PR #46到dev；合入后独立核对实际dev的Unit/Package，不能只看工作分支通过。
2. 合入后逐项核对四个开放Issue：#36初始冻结拒绝保全、#37 pending模式显示/过滤、#38部分文本写失败清理、#41 CLI重试关闭。它们已有PR39/40/42–45中的修复与回归，后续在当前dev逐项复验、补有意义的缺口，不重复造修复或把业务KeyError当成库问题。Issue不自动关闭。
3. 不测速、不做性能改动、不查480秒问题；原根因未确认，GPU争用仅假设。本批不运行GPU任务，四个任务共享24GB GPU的计时不能作为本批性能结论。
4. 当前代码基线没有新完整main_probe八格或main_exp八格600-step/32条80000-step科学验收。历史v0.3.0完整结果仍有效于其804b3823来源，不冒充新dev验收；[main_exp报告](../../studies/main_exp/docs/STUDY_REPORT.md)与[main_probe报告](../../studies/main_probe/docs/STUDY_REPORT.md)保留原范围。本次最终指令先完成四个Issue正确性维护，不自动重训已完成大规模研究。
5. 不更新下游vendor或干预它们的运行，不改数值门/正式预算；共享不可变data/model继续复用，Run模型对象、checkpoint、sampler、RNG及可变状态独立。不合main、不发布、不花钱，不删除任何数据或分支。

## 四、接续入口

工程事实见[record.md](record.md) §43–58，原始失败与限定范围不删；当前缺陷看[bugs.md](bugs.md)，流程看[cicd.md](cicd.md)，本地测试看[testing.md](testing.md)。临时归档输入、源码/结果hash与执行清单在.tmp/archive-20261015及此前phase目录，不随Git clone提供。

## 五、归档后的集成状态（2026-10-15 补记）

上文保留合入前快照；PR #46 现已按最终授权 normal merge 到 dev `5ee1214b7ce0dc28b65139ac7b3f9ca16eb8b2ac`，最终工作 head `90b7d75` 的全部必需检查与实际矩阵通过。实际 dev CI 和四个 Issue 的逐项复验见 [record §59–60](record.md#59-pr-46-实际合入-dev-与归档后状态)，后续 PR / CI 当前状态不由上文历史“未完成”推断。main、历史分支与数据保留，性能工作仍暂停。
