# Bugs

> 这里只保留尚未解决的缺陷。编号只增不复用；修复并验证后移除条目，开发证据留在 [record.md](record.md)，实验依据留在对应 Study 报告。正常差异或决定不修的事项不列为开放缺陷。权威仍是 [concept.md](../code/concept.md) → [layout.md](../code/layout.md) → 代码。

更新日期：2026-10-15。

## 开放

### B-068：可选活动提示的编码或存在性读取错误使 status 崩溃

- 状态：open；85d6070真实隔离复现，修复前13项中6 failed /7 passed，候选与既有status回归34 passed；最终提交CI另核对。修复候选继续放PR #46草案，不合dev/main。
- 来源：主动审查read_activity及唯一生产调用方CLI _execute_status；已有实现对JSON/读取失败返回None，但漏掉UTF-8解码失败和try外is_file的权限错误。
- 期望：activity只是可选进度提示，不可用时仍读取权威index/result；没有index则照常退出2并报missing index。保留活动文件、来源和已有成功字节，不掩盖未知编程错误。
- 实际：真实写入b'\xff'触发UnicodeDecodeError，存在性检查PermissionError可在异常边界外外溢；CLI有无index均崩溃，旧结果虽未被改写但无法查看。
- 证据：.tmp/phase068/before.zip /BEFORE_INDEX.json保存生产/新增测试原字节与hash；red.xml /red.log保存6个反例。非法编码用真实合成文件，权限边界用精确方法替身；未修改系统权限或正式Study。临时材料不随clone提供。
- 最小方案：仅把存在性检查纳入既有可选IO异常边界，并显式捕获UnicodeError；保持有效提示、无index、写入/删除及非预期RuntimeError合同。

### B-067：冻结采集错误未归为来源拒绝，旧成功结果可被覆盖

- 状态：open；在80fdce1隔离输入上真实复现，修复前19项为15 failed /4 passed；最终候选34项定向、214 passed /5 deselected相邻正确性检查通过，最终提交CI另核对。PR #46继续草案。
- 来源：主动审查check_frozen→provenance_changes及CLI冻结门、Runner初始错误处理。
- 期望：freeze开启时无法读取/解析来源证据也必须作为冻结拒绝；保留旧成功结果与原始异常cause，不启动计算。make仍直接报告原输入错误，freeze关闭不采集。
- 实际：include唯一文件消失抛FileNotFoundError，读文件拒绝抛PermissionError，损坏provenance.json抛JSONDecodeError；CLI未转拒绝返回，Runner因异常不属于ProvenanceChangedError将已有成功写成failed。
- 证据：.tmp/phase067/before.zip、source-index.json为修改前源码/测试字节与hash；red.xml /red.log /red-results.json保留真实失败与结果。仅合成库/Study，无模型或GPU，临时证据不随clone提供。
- 追加边界：合法JSON但根/files/plan为非mapping时，首候选仍有7 failed /21 passed（shape-red.xml），AttributeError仍可绕过保全。须在load_provenance验证磁盘mapping形状后交由同一冻结拒绝路径；不以捕获AttributeError掩盖编程错误。
- 最小方案：仅在check_frozen读取/比较证据时将OSError或ValueError转现有ProvenanceChangedError并保留cause；不改Runner保全规则、不捕获任意编程异常、不重签旧来源。

- 并发补验（2026-10-15）：新增失败交错与独立Run同时拒绝回归，修复前函数8项全红、当前相关53项通过；生产修复已在PR #46，仍未合入dev/main。详见[本轮汇总](record.md#55-b-058-至-b-067-修复汇总与并发补验)。

### B-066：Study 来源键可覆盖库来源哈希，掩盖真实库修改

- 状态：open；296fac5上已用隔离库/Study真实复现，最终候选14项门和202 passed /5 deselected相邻core通过，最终提交CI另核对。PR #46保持草案，不合dev/main。
- 来源：主动复核source_files的统一files mapping，库键为rpipe/<relative>，Study输入键直接相对Study根；write_provenance及provenance_changes/check_frozen共用该路径。
- 期望：不同物理输入不得静默共用同一个来源键；即使初始字节相同也必须拒绝，防止后续独立修改被掩盖。rpipe/为库来源保留键域；对应库路径存在且可验证为同一物理文件（Study内直接导入库或硬链接）时可复用键，其他Study输入须移到不同键域。格式不变。
- 实际：Study include的rpipe/__init__.py覆盖实际库__init__.py，修改库后check_frozen仍成功。legacy-bypass.json保留库修改前后不同hash、记录值等于Study副本hash以及错误放行事实；未修改真实库或下游vendor。
- 证据：.tmp/phase066/red.xml初始7单元为5 failed/2 passed；加入原生launch冻结门和Runner后red-gates.xml为9 failed/2 passed，0.31s。原实现确实进入了禁止执行的合成阶段；不是只缺元数据。首轮red-final.xml因两个新增测试同basename收集失败1项，重命名后另起basetemp保留旧失败。pre-fix-acceptance.zip和acceptance-source-index.json在实施前保存完整源码/最终测试字节与hash，命名无大小写别名。
- 追加边界：首候选11 passed后，真实删除库路径仍被Study硬链接掩盖；扩展为保留键域的两项控制为2 failed/11 passed，身份PermissionError另1 failed。均保留，未签为最终通过。
- 最小方案：source_files对rpipe/保留键核对对应库路径的samefile，不同文件或身份不可验证均抛现有ProvenanceChangedError，身份OSError保留cause；不新建异常/映射格式、不改CLI/Runner规则。沿用原生拒绝和初始冻结失败保全；旧歧义清单不重签，须移动Study输入到无歧义相对路径后按计划fresh make。


### B-064：完整目标名拼入 staging，合法长文件名无法原子写入或迁移

- 状态：open；870da1b 上真实文件系统复现，短名候选修复及相关 187 passed / 5 deselected，最终提交 CI 另核对；PR #46 保持草案。
- 来源：主动审查 _atomic.atomic_write_text 的所有调用方及 CLI shared --migrate → migrate_shared；与 B-062 已修的 link_input 是不同调用点。
- 期望：系统支持的 245 字符 ASCII 文件名可创建、替换及迁移同字节副本；临时文件拥有独立短名，失败保留旧字节与其他文件。
- 实际：文本临时名附加 PID/UUID、迁移附加 PID/time_ns，单个路径组件超过文件系统上限。Windows 扩展路径中原目标实际创建成功，写入报 Errno 22，迁移报 WinError 123，清理还可遮蔽原错误；未改 LongPathsEnabled。
- 证据：.tmp/phase064/red2.log、red2.xml 为 4 failed / 1 passed，0.22 s；两项文本 create/replace、data/model 两项迁移失败，发布拒绝保全通过。首轮 red.log 因未建立 basetemp 父目录为 5 setup errors，保留且不算缺陷复现。before.zip/BEFORE.json 保存修复前源码与测试原字节，不随 clone 提供。
- 最小方案：两处 staging 沿用已有短 UUID-only basename；保留同目录 replace、锁顺序、冲突检查、失败清理和原文件路径，不迁移正式数据、不引入依赖。补长名与拒绝发布回归；最终 CI 绑定新提交核对。

### B-063：可选测试 Git 元数据的文件读取未防错，损坏输入可中断配置或误记 SHA

- 状态：open，原草案 c4d651d 的实现已复现；候选 870da1b 本机相关108项与四环境必需CI通过，尚未集成。测试工程问题，不是原 480 s 性能慢因，也不改库训练/Run 状态。
- 来源：主动审查 tests/conftest.py::_git_sha 与唯一调用 pytest_configure；B-060 只覆盖布局和 Git 子进程错误，直接 HEAD / loose ref 的读取仍在 try 外。
- 期望：可选本地身份不可读/损坏时记录 null，不阻断实际测试；文件快速路径只记录合法 40/64 位十六进制形状，正常 loose/detached 与 packed/worktree、5 s Git 截止不变。显式 GITHUB_SHA 仍优先，不假装包含全部 dirty 字节。
- 实际：HEAD 或 loose ref 的非 UTF-8、权限拒绝、读取前移除异常外溢；裸 ref: 导致 IndexError，任意文本又可直接被当作 git_sha。真实复制的原生 tests/run.py 会在 pytest_configure 退出 3，测试尚未执行；不是单纯结果缺可选字段。
- 证据：.tmp/e063-red/ 为 10 failed /10 passed，1.10 s；复制入口 .tmp/e063-session-red/ 为 2 failed /1 passed，1.11 s，两个损坏输入子进程退出 3，显式 CI SHA 分支成功。均只创建临时 .git fixture，未触碰真实仓库 HEAD/索引。修复前源码、当时新增测试原字节、逐文件 hash 及 ZIP 在 .tmp/phase063/，不随 clone 提供。
- 最小方案：把现有直接读取纳入原有可选错误边界，安全解析 ref 前缀，校验文件快速返回的 SHA 形状；不取消快速路径、引入依赖、增大 Git 超时或改 CI 显式来源优先级。验证实际入口执行、manifest、事件和原错误保全；草案不合并。


### B-062：同字节共享引用发布的临时名仍依赖时钟，可被并发写者共用

- 状态：open，原草案 c3b8cef 的 link_input 已本机复现；一行候选修复和最终相关 93 项通过，最终提交 CI 待核对；与 B-057 的 atomic_write_text 是不同发布入口，同属临时名唯一性边界，不重复登记旧文本写入反例。
- 来源：主动审查 shared.link_input 的递归临时引用发布及 reference_tree 调用方。标准 DataFactory / 迁移有外层共享锁；本反例针对直接并发调用引用 helper，不声称现有锁保护流程或正式数据已经损坏。
- 条件与期望：两个线程同时把同一已存在、内容相同的独立副本转为同一不可变源的引用。每个写者必须只发布/清理自己的 staging；两次调用均成功、最终 samefile、源字节和 peer 文件不变。固定 time_ns 读数是有效时钟语义，不能作为唯一性保证。
- 实际：.link 临时名只含目标、PID、time_ns。冻结时钟后两个写者共用 staging，Windows 分别出现 PermissionError / FileExistsError，另一写者在用于同步发布的测试 barrier 退出；两种 backend 都失败。没有把 barrier 当生产锁死或原 480 s 根因。
- 证据：.tmp/e062-red/ 的真实文件/线程回归为 2 failed / 3 passed，4.25 s；default 与强制 hardlink 的 frozen-clock 两项失败，正常时钟与发布拒绝保全通过。JUnit 和原日志保留，不随 clone 提供；只使用合成小文件。
- 最小方案：临时名改用 shared.py 已导入的短 UUID-only nonce + .link，不携带完整目标名，复用 B-057 的成熟模式，不新增锁/依赖或改变 symlink→hardlink fallback、原子 replace、冲突拒绝及 finally 清理。验证发布拒绝时源/旧副本/peer 完整；不扩展为可变 Run 多写者协议，也不宣称任意缺失目标的并发创建合同。


- 候选复审：首版完整目标名+UUID 对合法 220 字符文件名产生新回归；旧 c3b8cef 实测成功，候选 .tmp/e062-name-red/ 1 failed 保留。短名修正后新增第六项实际文件系统回归，最终 .tmp/e062-final/ 93 passed；Windows 仅显式扩展路径，未改系统策略。

- 并发补验（2026-10-15）：新增失败交错与独立Run同时拒绝回归，修复前函数8项全红、当前相关53项通过；生产修复已在PR #46，仍未合入dev/main。详见[本轮汇总](record.md#55-b-058-至-b-067-修复汇总与并发补验)。

### B-061：Git status 查询失败被记为可靠的 dirty 布尔值

- 状态：open，原草案 1ba0877 的实现已在本机复现；两行候选修复和本机 87 项回归通过，最终提交 CI 待核对；PR #46 继续草案，不合入 dev/main。
- 来源：主动审查 provenance.capture_git 的两个 subprocess.run 调用及唯一生产调用方 write_provenance。不是原量化 480 秒慢因结论，也不是 B-060 的测试 manifest 布局问题。
- 期望：只有 HEAD 与 status 均查询成功才记录 head/dirty；状态不可取得时沿用现有可选元数据失败语义，返回空 mapping。非零退出码的空或部分 stdout 不能证明干净或脏；保留正常、packed refs、worktree、非 Git、缺命令和既有 10/30 秒截止。
- 实际：临时仓库提交正常后只把合成 .git/index 写坏，rev-parse HEAD 仍成功，git status 返回非零且无 stdout；capture_git 却返回真实 head 和 dirty=false。注入非零 status 的部分 stdout 又得到 dirty=true，均把失败当成可靠状态。
- 证据：新增镜像回归 test_provenance_git.py；修复前 .tmp/e061-red/ 为 3 failed / 8 passed，1.53 s，包括真实损坏索引及两种失败 stdout。只破坏自建临时 fixture，不修改仓库索引、真实数据、旧 provenance 或下游代码。临时日志与 JUnit 不随 Git clone 提供。
- 最小方案：在 status 结果后检查 returncode，失败返回空 mapping；不更改命令、超时、进程生命周期或持久化结构。补齐 Git 查询超时/不可用、正常 clean/dirty 与 packed/worktree 回归；验证后保留红色证据并更新开发记录。


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

## 暂缓（不进入正确性修复队列）

### B-065：单次来源清单采集对重叠输入反复解析、读取和哈希

- 状态：deferred；按最终仅做正确性的要求已用新普通提交撤回ad67f74次数去重优化，原复现与性能记录仅作历史保留，不进入本轮修复队列。见[范围收缩记录](record.md#57-按最终范围暂缓-b-065-性能去重)。
- 来源：source_files 经 study_phase_files、flow/structure 自动扫描、recipe 与 provenance.include 收集同一路径，最终 mapping 覆盖重复键，但此前每次均执行 _inside/读取/哈希。生产调用方为 write_provenance 和 provenance_changes，后者供 launch/prepare/Runner/Study process 冻结检查。
- 期望：同一次清单采集每个解析后 Study 文件只读取/哈希一次；重复原始字符串不反复解析；新一次调用仍完整发现文件并重读，保持变化检测、路径边界和读取错误。
- 实际：两项真实读取计数用例失败，flow 开关关闭时两个 flow 文件各读2次、helper4次，开启时 flow各3次、helper4次。32个合成源文件、2个声明及1个隔离库文件共35个唯一输入，被读取/哈希100次；三次小场景结果相同。
- 证据：.tmp/phase065/red.log/red.xml 为 2 failed /1 passed，0.20 s；before.json/日志逐层记录 total/resolve/glob/open/read_bytes/hash，源码原字节在before.zip；BEFORE.json/before.json大小写别名覆盖的原hash索引缺口及事后重建SNAPSHOT_FILES.json已在record §52如实记录，不随clone。当前量化冻结 helper 的162文件单调用仅诊断其内部阶段，不等于原480s现场，也不是本问题的同一函数。
- 最小方案：仅在一次 source_files 调用内按首次出现去重原始声明，再按已验证的解析 Path 去重 Study 文件；不缓存跨调用内容/清单，不减少任何独立 freeze 检查，不改库文件集合、格式、Run或下游业务。补修改后下一次调用、读取拒绝、空匹配和越界回归，保留 add/edit/delete 旧门。

## 集成与历史说明

归档前已确认 B-056 / B-057 / B-058 / B-059 / B-060 / B-061 / B-062 / B-063 / B-064 / B-066 / B-067 / B-068，修复与回归已在工作分支；维护者已授权PR #46通过最终CI后合dev。条目中的open/草案只记录归档前阶段，实际集成身份见[归档状态](maintenance_archive_2026-10-15.md)和PR当前mergeCommit；合入后按实际证据更新，不把旧文字当新缺陷。B-053（跨Study输入共享）和B-054（关闭扩展时的初始冻结拒绝保全）已通过[PR #39](https://github.com/DreamSoul-AI/RPipe/pull/39)合入dev `614d0a6`；工作提交07bcf2e的Unit tests / Build package / Branch flow及实际矩阵均通过，开发与实验依据见[集成记录](record.md#39-共享与三个反馈修复合入-dev)。实际dev集成提交的CI单独核对，不用PR通过替代。

B-055 的原生 launch 重试关闭入口已通过 [PR #42](https://github.com/DreamSoul-AI/RPipe/pull/42) 合入 dev `8c15cd84f8a47d5bfeb80edfa69bc5c370998e74`。最终工作提交 `8269bff` 的三项必需检查及实际 dev 的 [Unit Tests](https://github.com/DreamSoul-AI/RPipe/actions/runs/38557561151) / [Package Check](https://github.com/DreamSoul-AI/RPipe/actions/runs/38557561212) 均通过，包含 Windows / Python 3.13；开发证据见 [记录 §40](record.md#40-原生-launch-显式关闭失败重试41验证通过)。默认重试仍保留，#41 不自动关闭。

GitHub #36—#38及#41仍保持open，原v0.3.1/main尚未发布这些修复；当前dev修复与已发布版本必须区分。没有开放软件条目不代表所有后端、设备或研究条件均已验收。

此前撤回的范围外反例保留历史事实，未自动恢复为本目标的实施队列；接续以新的实际反馈和已确认设计为准。旧收缩依据见[当时记录](record.md#32-按用户要求收缩到本机适配与三个下游-issue)。

B-002、B-021 / B-024及B-025至B-029的既有修复证据仍在 [record.md](record.md)。B-022 / B-023 的代表性容量、精度合同与探索边界保留各自报告；本轮新的设备 / 并发结果见 [本轮执行记录](../../studies/main_exp/docs/v030_execution.md)。未实测支持范围不是已通过，缺环境依赖的导入失败也不自动列为库缺陷。

B-031 至 B-068 已使用，编号不复用。下次新增问题从 **B-069** 起编号。

新增条目一条一事，记录编号、open / investigating状态、复现条件、期望 / 实际、影响落点、方案和实际验证证据。
