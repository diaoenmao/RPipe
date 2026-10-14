# Bugs

> 这里只保留尚未解决的缺陷。编号只增不复用；修复并验证后移除条目，开发证据留在 [record.md](record.md)，实验依据留在对应 Study 报告。正常差异或决定不修的事项不列为开放缺陷。权威仍是 [concept.md](../code/concept.md) → [layout.md](../code/layout.md) → 代码。

更新日期：2026-10-14。

## 开放

### B-060：本地测试 manifest 遗漏 packed refs / worktree 的 Git 提交

- 状态：open，候选修复在草案 PR #46 / 394a068；最终代码四环境 CI 与本机66项通过，dev 9324014 尚未集成。原 f162eb6 红色回归保留。
- 来源：本轮采栈要求完整来源时审查 tests/conftest.py::_git_sha，当前只读 .git/HEAD 和松散 ref。不是原 480 秒性能根因，也不等同于包含 dirty 字节的完整来源冻结。
- 期望：本地 manifest 可识别 packed refs / worktree 的实际 HEAD，同时保留普通松散 ref / detached HEAD；无 .git 的复制快照不能搜索并误记父仓库。缺 Git、命令失败或查询超时仍允许 null，不阻断测试。
- 实际：git pack-refs --all 后分支 ref 不再是单文件，返回 null；worktree 的 .git 为指针文件，.git/HEAD 不存在，也返回 null。真实提交存在、Git可读，记录却丢失身份。
- 证据：[测试先行 Unit Tests](https://github.com/DreamSoul-AI/RPipe/actions/runs/38586121972)，f162eb60e1cd3f3faba04ea848e32aaa872bd6c6；四个环境各仅上述两项 failed，Windows core 2 failed / 496 passed / 2 skipped，Ubuntu core 2 failed / 490 passed / 8 skipped；CPU Flow 因 core 失败未执行。原失败日志 .tmp/trace480/b060-baseline-ci.log 保留，不计通过。
- 方案与验证：保留现有松散 ref / detached HEAD 文件读取；仅 .git 存在而直接读取未解决时，采用现有项目 subprocess/Git 模式，5秒有界查询，失败返回 null。8项真实布局/错误边界回归和本机相关验证后重新核对最终提交 CI；GITHUB_SHA 的原优先级不变，不改 Run、训练、调度、旧 manifest 或下游源码。见 record §47；草案仍不合并。

### B-059：共享输入文件合同按原始字符串比较，可绕过保留名与唯一性

- 状态：open，Windows/Python 3.13.9 本机已复现并在草案 PR #46 修复；dev 9324014 尚未集成，原 b20a034 反例保留。
- 来源：继续审查 prepared_resource 文件清单入口，已核对 main_exp / main_probe 两个正式调用方，均使用正常相对文件名。
- 期望：创建缓存根或调用 builder 前拒绝控制 manifest 本身及按本机 Path 语义等价的重复路径；空文件名、目录本身或带锚点的路径也不能作为相对文件输入。
- 实际：files=['./manifest.json'] 进入 builder，随后覆盖作为输入的 manifest 并发布 inputs，校验才失败；下次仍复用这个不一致目录而拒绝。Windows 上 MANIFEST.JSON 同样触发。files=['data','./data'] 被当作不同文件接受。合成缓存复现 .tmp/a51d92a07/repro.json 保留，未触碰真实数据。
- 方案：只在入口建立 Path 对象并按 Path 等价比较保留名/重复项、拒绝 anchor 与无文件名；仍保留原清单字符串作为 manifest 键，不改变既有有效缓存、定义 hash、build/锁或发布格式。Windows case / 分隔符规则按实际平台处理，不在 POSIX 假装大小写不敏感。
- 验证：先增加 portable 与 Windows 专属非法清单断言，并检查 builder 不被调用、根不创建；有效单个 ./weights 和 nested/manifest.json 必须继续构建、hash 校验和复用。该修复不是通用 NTFS 路径/符号链接防护，也不是原 480 秒事件根因。见 record §46；草案不合并。

### B-058：测试事件 started_at 记录的是完成后时间

- 状态：open，候选修复在草案 PR #46；dev 9324014 尚未集成。f3eb08b 是原复现基线，不能当作当前草案状态。
- 来源：本轮追查原量化 480 秒批次，核对上游 tests/conftest.py 的 pytest_runtest_makereport；不是导致原超时的根因结论。
- 期望：events.jsonl 的 started_at 对应其实际记录阶段的起点（正常 call 或失败 setup），duration_ms 保持 pytest 报告的实际耗时。
- 实际：写事件时调用 datetime.now，发生在报告完成以后；长用例的起点因此被推迟约该阶段耗时，误导逐项等待时间线。原日志保留，不回填或重新签收。
- 复现：使用真实记录 hook、固定 TestReport.start 与 1.25 s duration，分别输入成功 call / 失败 setup；断言落盘 started_at 等于已知 start。修复前必须失败，随后仅改记录字段来源，补相关回归。
- 影响与边界：测试工程记录，不改变 Run、训练、评测、锁或调度；不把这个元数据修复当作原 480 秒性能修复。正式验证见 record §45；草案保持未合并。

### B-057：同进程原子写入临时名依赖时钟，可碰撞并丢失 staging

- 状态：open，草案 PR #46 中修复；dev 9324014 尚未集成。
- 来源与条件：本轮新增两个线程同时向同一目标发布完整文本；草案 ebcdb8b 的 Windows Server 2025 / Python 3.10.11 core 实测失败。该提交与 dev 的 _atomic.py 相同，B-056 的预算校验不改变此路径。
- 期望：两个写者拥有不同临时路径，均能完成一次原子发布；最终目标为其中一份完整文本，各自只清理自己的 staging，不碰其他写者文件。
- 实际：[Unit Tests](https://github.com/DreamSoul-AI/RPipe/actions/runs/38571738491) 的 Windows/Python 3.10 报 FileNotFoundError / WinError 2，位置为临时文件 replace；core 为 1 failed / 471 passed / 2 skipped，流程门因 core 失败未执行。其他三个环境通过；早前 Windows/Python 3.13 的并发用例也通过，不能据此忽略反例。
- 原因与方案：临时名只含 PID 和 time_ns；同进程同 tick 的两线程可使用同一临时文件，一方发布后另一方找不到该源。纳秒返回单位不承诺读数唯一；[Python 时钟说明](https://docs.python.org/3.13/library/time.html#time.time_ns)也区分 Windows 不同版本的底层时钟。改用项目已有的随机 UUID nonce 模式，不依赖时钟精度，不改变 replace / 清理 / Windows 有界重试。
- 回归：保留自然时钟的并发用例，增加固定时钟读数的同构边界，检查两个独立 staging、完整最终文本、peer 文件与清理；原始部分写失败、永久拒绝、真实 Windows 句柄拒绝仍须通过。不宣称多写入者 checkpoint / Run 可恢复，也没有证据表明此反例损坏过正式数值结果。
- 原日志：.tmp/cache-lock-budget/draft-first-unit.log（不随 clone），SHA256 5fb955c04db72537de8fda8b72b5b09281782123b15b84315b164babbb5bb437。修复前失败保留，最终工作提交必须重新通过全部必需 CI，不能用单次重跑通过覆盖失败。

### B-056：共享准备锁接受非法超时预算，NaN 可绕过竞争截止

- 状态：open，已复现并在草案 PR #46 修复；当前 dev 9324014 尚未集成，不自动合并。
- 来源：本轮审查 shared.cache_lock 的等待边界，不是量化 480 秒事件或下游 KeyError 的根因结论。
- 条件：Windows Server 2025 / Python 3.13.15，生产 src 与 dev 9324014 相同。直接调用 cache_lock(root, timeout=NaN / ±inf / -1)；NaN 忙锁通过固定 EACCES 和三次循环 watchdog 注入，避免真实无限挂起。
- 期望：timeout 有限且非负，非法值在创建目录、锁文件或等待前抛 ValueError；合法 0 不等待竞争，默认 120 秒不变。
- 实际：四种非法值在无竞争时均可进入临界区；NaN 忙锁不能满足 monotonic >= deadline，被 watchdog 截住。+inf 同样没有有限截止，负预算也未校验；没有声称默认有限预算会失效。
- 证据：[修复前有界回归](https://github.com/DreamSoul-AI/RPipe/actions/runs/38570617049)，提交 7fba2b0f51642b2441ffbe760169e340d03e35dc，5 failed / 26 passed，1.51 s。原日志 SHA256 9241e3708dcb4040314eb3fbfe63820928db1306a9957449d4fe332ddaa38bab；本机 .tmp/cache-lock-budget/boundary-baseline-verified.log 不随 clone 提供。
- 方案：标准库 math.isfinite 提前校验，不新增调度、重试、锁所有者账本或缓存格式。已核对各 Factory、prepare、迁移使用默认 / 0 / 20 等有限非负值。
- 验证：非法值与 watchdog 回归、既有零预算和退出释放锁均须通过。并发写、失败 staging 重建、Windows Unicode 路径的真实句柄拒绝在修复前已通过。最终工作提交 CI 另行核对，不把预期红色任务或未执行项算通过。见 [记录 §43](record.md#43-最小启动--缓存成本诊断与共享锁预算边界分步记录)。

当前 dev 已确认 B-056 / B-057 / B-058 / B-059 / B-060，修复候选在草案工作分支，尚未合入 dev；见对应条目。B-053（跨Study输入共享）和B-054（关闭扩展时的初始冻结拒绝保全）已通过[PR #39](https://github.com/DreamSoul-AI/RPipe/pull/39)合入dev `614d0a6`；工作提交07bcf2e的Unit tests / Build package / Branch flow及实际矩阵均通过，开发与实验依据见[集成记录](record.md#39-共享与三个反馈修复合入-dev)。实际dev集成提交的CI单独核对，不用PR通过替代。

B-055 的原生 launch 重试关闭入口已通过 [PR #42](https://github.com/DreamSoul-AI/RPipe/pull/42) 合入 dev `8c15cd84f8a47d5bfeb80edfa69bc5c370998e74`。最终工作提交 `8269bff` 的三项必需检查及实际 dev 的 [Unit Tests](https://github.com/DreamSoul-AI/RPipe/actions/runs/38557561151) / [Package Check](https://github.com/DreamSoul-AI/RPipe/actions/runs/38557561212) 均通过，包含 Windows / Python 3.13；开发证据见 [记录 §40](record.md#40-原生-launch-显式关闭失败重试41验证通过)。默认重试仍保留，#41 不自动关闭。

GitHub #36—#38及#41仍保持open，原v0.3.1/main尚未发布这些修复；当前dev修复与已发布版本必须区分。没有开放软件条目不代表所有后端、设备或研究条件均已验收。

此前撤回的范围外反例保留历史事实，未自动恢复为本目标的实施队列；接续以新的实际反馈和已确认设计为准。旧收缩依据见[当时记录](record.md#32-按用户要求收缩到本机适配与三个下游-issue)。

B-002、B-021 / B-024及B-025至B-029的既有修复证据仍在 [record.md](record.md)。B-022 / B-023 的代表性容量、精度合同与探索边界保留各自报告；本轮新的设备 / 并发结果见 [本轮执行记录](../../studies/main_exp/docs/v030_execution.md)。未实测支持范围不是已通过，缺环境依赖的导入失败也不自动列为库缺陷。

B-031 至 B-060 已使用，编号不复用。下次新增问题从 **B-061** 起编号。

新增条目一条一事，记录编号、open / investigating状态、复现条件、期望 / 实际、影响落点、方案和实际验证证据。
