# Bugs

> 这里只保留尚未解决的缺陷。编号只增不复用；修复并验证后移除条目，开发证据留在 [record.md](record.md)，实验依据留在对应 Study 报告。正常差异或决定不修的事项不列为开放缺陷。权威仍是 [concept.md](../code/concept.md) → [layout.md](../code/layout.md) → 代码。

更新日期：2026-10-16。

## 开放

当前开放Issue快照为空。B122/#167、B123/#168已由[PR169](https://github.com/DreamSoul-AI/RPipe/pull/169)合devbef7de9，最终head/实际dev全部实际矩阵/聚合成功，已验收关闭并清理已合主题本地/remote；原始独立红/联动红及半行/首错证据保留，详[record§204–208](record.md)。不可读日志尾仍沿原追加行为，不保证该限制下半行修复。下一编号B124，继续三类复查；未确认的假设不列为缺陷。

B119/#163、B120/#164、B121/#166已PR165双侧全实际CI验收并关闭，已合主题本地+remote清理，详[record§203](record.md)。

B117/#160、B118/#162已PR161双侧全部实际CI验收关闭且主题本地/remote清理，第5批周期盘点见[record§193](record.md)。

B116/#158已由PR159双侧全部实际CI验收关闭并清本地/remote，见[record§187](record.md)。

B115/#156已PR157双侧全部实际CI验收关闭并清本地/remote，见[record§185](record.md)。B114/#154由PR155验收关闭，见[record§182](record.md)。

B101/#130已PR131合dev并验收关闭，正式记录与分支清理见[record§154](record.md)。

前一轮已关闭26个已处理Issue（24正确性+2初步调研），完整清单与旧反例见[record§150–151](record.md)及[清理前索引](https://github.com/DreamSoul-AI/RPipe/blob/2008753c3c2b11321a3da3811102cf5c20b646ff/docs/development/bugs.md)。关闭不等同main发布或授权探索后续实施。


## 暂缓（不进入正确性修复队列）

### B-065：单次来源清单采集对重叠输入反复解析、读取和哈希

- 状态：deferred；按最终仅做正确性的要求已用新普通提交撤回ad67f74次数去重优化，原复现与性能记录仅作历史保留，不进入本轮修复队列。见[范围收缩记录](record.md#57-按最终范围暂缓-b-065-性能去重)。
- 来源：source_files 经 study_phase_files、flow/structure 自动扫描、recipe 与 provenance.include 收集同一路径，最终 mapping 覆盖重复键，但此前每次均执行 _inside/读取/哈希。生产调用方为 write_provenance 和 provenance_changes，后者供 launch/prepare/Runner/Study process 冻结检查。
- 期望：同一次清单采集每个解析后 Study 文件只读取/哈希一次；重复原始字符串不反复解析；新一次调用仍完整发现文件并重读，保持变化检测、路径边界和读取错误。
- 实际：两项真实读取计数用例失败，flow 开关关闭时两个 flow 文件各读2次、helper4次，开启时 flow各3次、helper4次。32个合成源文件、2个声明及1个隔离库文件共35个唯一输入，被读取/哈希100次；三次小场景结果相同。
- 证据：.tmp/phase065/red.log/red.xml 为 2 failed /1 passed，0.20 s；before.json/日志逐层记录 total/resolve/glob/open/read_bytes/hash，源码原字节在before.zip；BEFORE.json/before.json大小写别名覆盖的原hash索引缺口及事后重建SNAPSHOT_FILES.json已在record §52如实记录，不随clone。当前量化冻结 helper 的162文件单调用仅诊断其内部阶段，不等于原480s现场，也不是本问题的同一函数。
- 最小方案：仅在一次 source_files 调用内按首次出现去重原始声明，再按已验证的解析 Path 去重 Study 文件；不缓存跨调用内容/清单，不减少任何独立 freeze 检查，不改库文件集合、格式、Run或下游业务。补修改后下一次调用、读取拒绝、空匹配和越界回归，保留 add/edit/delete 旧门。

## 集成与历史说明

[归档快照](maintenance_archive_2026-10-15.md)明确区分合入前与后续状态；PR #46 的最终工作提交 `90b7d75323d266c4d272f786c28f4c00959d0432` 与实际 dev `5ee1214` 分别验证，不能互相替代。归档前条目中的 open / 草案文字不再代表当前代码状态。B-053（跨 Study 输入共享）和 B-054（关闭扩展时的初始冻结拒绝保全）此前经 [PR #39](https://github.com/DreamSoul-AI/RPipe/pull/39) 合入 dev `614d0a6`，证据见 [集成记录](record.md#39-共享与三个反馈修复合入-dev)。

B-055 的原生 launch 重试关闭入口已通过 [PR #42](https://github.com/DreamSoul-AI/RPipe/pull/42) 合入 dev `8c15cd84f8a47d5bfeb80edfa69bc5c370998e74`。最终工作提交 `8269bff` 的三项必需检查及实际 dev 的 [Unit Tests](https://github.com/DreamSoul-AI/RPipe/actions/runs/38557561151) / [Package Check](https://github.com/DreamSoul-AI/RPipe/actions/runs/38557561212) 均通过，包含 Windows / Python 3.13；开发证据见 [记录 §40](record.md#40-原生-launch-显式关闭失败重试41验证通过)。默认重试仍保留；#41在v0.4.0发布后经本次明确授权关闭，而非默认自动关闭。

原v0.3.1未包含这些修复；截至v0.4.0，#36—#38及#41已在main发布并经明确授权关闭。历史dev验证与各自发布版本仍分开记录。没有开放软件条目不代表所有后端、设备或研究条件均已验收。

此前撤回的范围外反例保留历史事实，未自动恢复为本目标的实施队列；接续以新的实际反馈和已确认设计为准。旧收缩依据见[当时记录](record.md#32-按用户要求收缩到本机适配与三个下游-issue)。

B-002、B-021 / B-024及B-025至B-029的既有修复证据仍在 [record.md](record.md)。B-022 / B-023 的代表性容量、精度合同与探索边界保留各自报告；本轮新的设备 / 并发结果见 [本轮执行记录](../../studies/main_exp/docs/v030_execution.md)。未实测支持范围不是已通过，缺环境依赖的导入失败也不自动列为库缺陷。

已使用至 B-102，编号不复用；下次新增问题从 **B-103** 起编号。历史段落的旧“下一编号”仅表当时状态。

新增条目一条一事，记录编号、open / investigating状态、复现条件、期望 / 实际、影响落点、方案和实际验证证据。


## 2026-10-16 最新状态补记

B102/#132 已经 PR133 合入 dev 并全 CI 验收关闭，上文待验收文字仅历史。B103/[Issue134](https://github.com/DreamSoul-AI/RPipe/issues/134) 为当前开放项：共享锁路径别名自争用，四红后最小修复，本机共享72/core1697+1既有skip通过，待Flow及最终/实际dev CI。详[record§157–158](record.md)。


B103 本机补记：CPU Flow 207 passed、0 failure/error/skip；最终 head 与实际 dev CI 仍待验收。


## 2026-10-16 B104 当前开放项

B103/#134 已PR135实际dev验收关闭并普通清分支；当前B104/[Issue136](https://github.com/DreamSoul-AI/RPipe/issues/136)为迁移多根锁顺序受namespace拼写影响，2红/2控制后最小修复，共享76通过，待完整本机及CI验收。详[record§159–160](record.md)。


B104本机验收补记：shared76/core1701/CPU Flow207 passed，core另1既有平台skip；新四节点全执行。最终head/实际dev CI待验收。


## 2026-10-16 B105 当前开放项

B104/#136已PR137实际dev全CI验收关闭。B105/[Issue138](https://github.com/DreamSoul-AI/RPipe/issues/138)：stdout诊断拒绝会跳过非零子进程退出后的既有success失效写回。6红/2控制后最小顺序修复，调度88通过，待完整本机及CI。详[record§161–162](record.md)。


B105本机验收：调度88/core1709/CPU Flow207 passed，core另1既有平台skip；八新节点全执行。最终head与实际dev CI仍待验收。


## 2026-10-16 B106 当前开放项

B105/#138已PR139实际dev全CI验收关闭并完成周期清理。B106/[Issue140](https://github.com/DreamSoul-AI/RPipe/issues/140)：savefig直接写旧PNG时部分失败破坏旧图、遗留自有figure；19项先红17/2，最小局部原子发布/释放修复已本机19绿、core1728+1既有skip。待Flow与两侧CI；见[record§163–164](record.md)。


B106本机验收：19新节点/core1728/CPU Flow207 passed，core另1既有平台skip；含真实Windows只读清理的19新节点全执行。最终head/实际dev CI待验收。


B106同批补充确认关闭IO二错可遮原保存错/取消；6项先红4/2后修复，当前25新节点绿、core1734+1既有skip，最终Flow/新headCI待验收。仍仅Issue140/PR141。


B106含流关闭补充的最终本机验收：25新节点/core1734/CPU Flow207 passed，core另1既有平台skip；25新节点含原生只读清理全执行，PNG字节仍一致。更新后最终head/实际dev CI仍须验收；旧9da02b4绿仅历史。


## 2026-10-16 B107 当前开放项

B106/#140已PR141更新head/实际dev全CI验收关闭并清分支。B107/[Issue142](https://github.com/DreamSoul-AI/RPipe/issues/142)：共享锁unlock/close IO可遮准备首错/取消；25项先红19/6后最小退出回调修复，共享101通过，待core/Flow/两侧CI。见[record§165–166](record.md)。

B107 / Issue142本机补充：共享101、core1759 passed/1既有平台skip、CPU Flow207 passed，25新回归全部执行；最终head/实际dev CI待验收，见record §166。


## 2026-10-16 B108 当前开放项

[Issue144](https://github.com/DreamSoul-AI/RPipe/issues/144)：四个文本/日志/tracker写入口的close IO替换首错或取消。dcd4e2d先30项16红/14控制，383d68c设计、2c33a65最小四写入块修复；30新节点全绿、相邻98 passed/1既有平台skip、core1789 passed/1既有skip；Flow和最终head/实际dev CI待验。原写入格式与字节通过冻结旧源码对照，临时原证据.tmp/audit108不随clone，见[record §167](record.md)。前序Issue142已PR143合dev，实际dev验收尚待完成，不冒充关闭。


B107/#142已PR143最终head与实际dev全CI验收关闭；主题本地/remote已清。B108/#144最终本机30新回归全执行通过、相邻98+1既有平台skip、core1789+1同一skip、CPU Flow207 passed；0failure/error，最终head与实际dev CI待验，保持open，见record §168。


## 2026-10-16 B109 当前开放项

[Issue146](https://github.com/DreamSoul-AI/RPipe/issues/146)：剩余Study plot_report直接保存PNG，部分保存/取消覆盖旧图且render/save/cancel泄漏自有figure。6ab3b74先25项23红/2正常控制，70cad05设计，1db97b1仅现有helper按已实施PNG所有权合同修复；新25与相邻PNG67全绿，core/Flow/最终head/实际dev待验。有效PNG76906字节旧新同SHA、绘制主体/科学函数保持；只用AST+合成输入，不跑Study，见record §169。前序PR145已head全CI合dev，Issue144仍待实际dev验收。


B108/#144已PR145最终head与实际dev全CI验收关闭，主题本地/remote普通删除。B109/#146本机25新节点全执行、PNG相邻67/core1814+1既有平台skip/CPU Flow207 passed；0failure/error，最终head/实际dev CI待验，保持open。见record §170。


## 2026-10-16 B110 当前开放项

[Issue148](https://github.com/DreamSoul-AI/RPipe/issues/148)：深层JSON的RecursionError逃逸冻结拒绝分类，使outer/native prepare零构建仍覆盖旧成功。初2000层无效反例保留，更正12000层真实解析四新旧路径确认；dadb558先12项5红/7控制、32f7a54设计、e99691d单行既有except补RecursionError，12新节点与相邻204全绿。strict loader/限额/schema和真实phase失败规则保持，core/Flow/两侧CI待验。见record §171。


B109/#146已PR147最终head与实际dev全CI验收关闭，主题本地/remote普通删除。B110/#148最终本机12新节点全部执行（core4/Flow8），相邻204、core1818 passed/1既有平台skip、CPU Flow215 passed；0failure/error，最终head/实际devCI待验，保持open。见record §172。
