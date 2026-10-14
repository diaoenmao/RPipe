# Bugs

> 这里只保留尚未解决的缺陷。编号只增不复用；修复并验证后移除条目，开发证据留在 [record.md](record.md)，实验依据留在对应 Study 报告。正常差异或决定不修的事项不列为开放缺陷。权威仍是 [concept.md](../code/concept.md) → [layout.md](../code/layout.md) → 代码。

更新日期：2026-10-14。

## 开放

### B-056：共享准备锁接受非法超时预算，NaN 可绕过竞争截止

- 状态：open，已复现；当前 dev 9324014 尚未修复。按本轮授权准备修复草案，不自动合并。
- 来源：本轮审查 shared.cache_lock 的等待边界，不是量化 480 秒事件或下游 KeyError 的根因结论。
- 条件：Windows Server 2025 / Python 3.13.15，生产 src 与 dev 9324014 相同。直接调用 cache_lock(root, timeout=NaN / ±inf / -1)；NaN 忙锁通过固定 EACCES 和三次循环 watchdog 注入，避免真实无限挂起。
- 期望：timeout 有限且非负，非法值在创建目录、锁文件或等待前抛 ValueError；合法 0 不等待竞争，默认 120 秒不变。
- 实际：四种非法值在无竞争时均可进入临界区；NaN 忙锁不能满足 monotonic >= deadline，被 watchdog 截住。+inf 同样没有有限截止，负预算也未校验；没有声称默认有限预算会失效。
- 证据：[修复前有界回归](https://github.com/DreamSoul-AI/RPipe/actions/runs/38570617049)，提交 7fba2b0f51642b2441ffbe760169e340d03e35dc，5 failed / 26 passed，1.51 s。原日志 SHA256 9241e3708dcb4040314eb3fbfe63820928db1306a9957449d4fe332ddaa38bab；本机 .tmp/cache-lock-budget/boundary-baseline-verified.log 不随 clone 提供。
- 方案：标准库 math.isfinite 提前校验，不新增调度、重试、锁所有者账本或缓存格式。已核对各 Factory、prepare、迁移使用默认 / 0 / 20 等有限非负值。
- 验证：非法值与 watchdog 回归、既有零预算和退出释放锁均须通过。并发写、失败 staging 重建、Windows Unicode 路径的真实句柄拒绝在修复前已通过。最终工作提交 CI 另行核对，不把预期红色任务或未执行项算通过。见 [记录 §43](record.md#43-最小启动--缓存成本诊断与共享锁预算边界分步记录)。

当前 dev 新确认 B-056，修复候选在工作分支，尚未合入 dev；见下方条目。B-053（跨Study输入共享）和B-054（关闭扩展时的初始冻结拒绝保全）已通过[PR #39](https://github.com/DreamSoul-AI/RPipe/pull/39)合入dev `614d0a6`；工作提交07bcf2e的Unit tests / Build package / Branch flow及实际矩阵均通过，开发与实验依据见[集成记录](record.md#39-共享与三个反馈修复合入-dev)。实际dev集成提交的CI单独核对，不用PR通过替代。

B-055 的原生 launch 重试关闭入口已通过 [PR #42](https://github.com/DreamSoul-AI/RPipe/pull/42) 合入 dev `8c15cd84f8a47d5bfeb80edfa69bc5c370998e74`。最终工作提交 `8269bff` 的三项必需检查及实际 dev 的 [Unit Tests](https://github.com/DreamSoul-AI/RPipe/actions/runs/38557561151) / [Package Check](https://github.com/DreamSoul-AI/RPipe/actions/runs/38557561212) 均通过，包含 Windows / Python 3.13；开发证据见 [记录 §40](record.md#40-原生-launch-显式关闭失败重试41验证通过)。默认重试仍保留，#41 不自动关闭。

GitHub #36—#38及#41仍保持open，原v0.3.1/main尚未发布这些修复；当前dev修复与已发布版本必须区分。没有开放软件条目不代表所有后端、设备或研究条件均已验收。

此前撤回的范围外反例保留历史事实，未自动恢复为本目标的实施队列；接续以新的实际反馈和已确认设计为准。旧收缩依据见[当时记录](record.md#32-按用户要求收缩到本机适配与三个下游-issue)。

B-002、B-021 / B-024及B-025至B-029的既有修复证据仍在 [record.md](record.md)。B-022 / B-023 的代表性容量、精度合同与探索边界保留各自报告；本轮新的设备 / 并发结果见 [本轮执行记录](../../studies/main_exp/docs/v030_execution.md)。未实测支持范围不是已通过，缺环境依赖的导入失败也不自动列为库缺陷。

B-031 至 B-056 已使用，编号不复用。下次新增问题从 **B-057** 起编号。

新增条目一条一事，记录编号、open / investigating状态、复现条件、期望 / 实际、影响落点、方案和实际验证证据。
