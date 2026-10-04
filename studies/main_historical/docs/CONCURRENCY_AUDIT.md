# 历史长训练并发补充审查

前五节保留 2026-10-04 约 01:09（Asia/Shanghai）的初次观察与当时待验证条件。后续真实暂停探针及两条 eager 的实际协调证据见第六节；正式 hold 点已提前到 CIFAR10 / mlp，以预留后续 CNN 的显存峰值。

## 一、结论

2026-10-04，按当前未修改的 [run.py](../run.py)、[recipe.py](../recipe.py) 和生产训练链进行只读审查。**可以提前启动一个尚未开始的 CIFAR10 / resnet18 / seed 0 Run，以填补当前 GPU 空闲；条件是它在主调度生成该数据集与模型的候选列表前已经成功结束并退出。** 先实测一条的显存峰值、吞吐与对主任务的影响，再决定是否追加 seed 1。

当前控制器没有跨调度器的 Run 锁，不会识别并等待外部启动的同一 Run。无法保证上述完成窗口时，不能把第二个控制器或任意提前启动当作安全的动态排班。完成时间和整轮提速尚未实测，本审查没有启动或暂停进程。

## 二、当前观察与计算影响

审查时主调度仍在 MNIST / linear 的四个 seed 训练组；只读状态显示 4 条 started、60 条 pending，最新训练记录约 step 31800。主调度的实际顺序为 linear、mlp、cnn、resnet18；每个模型内先 MNIST，再 CIFAR10；全部 train 结束后才进入 eval。因此 CIFAR10 / resnet18 是最后一个训练模型与数据集组合，但不能仅凭这一顺序承诺外部重模型一定先完成。

| **项目** | **判断与依据** |
|---|---|
| 随机数与采样 | 每条 `one` 是独立 Python 进程。Flow prepare 分别设置 Python、NumPy、Torch CPU/CUDA seed；历史采样器使用每 Run 的独立 `torch.Generator`。另一进程不会推进本 Run 的 RNG 或重新绑定其迭代器 |
| 训练连续性 | `one` 进入同一个原生 TrainAlgorithm，保留连续 80000-step 迭代器。提前改变 Run 的开始时间没有改变步数、评测周期、scheduler 或数据转换 |
| 确定性控制 | 使用原配置 deterministic=true、cudnn deterministic=true、benchmark=false 及既有 CUBLAS 设置。并发本身不修改这些配置；当前 runtime 的确定性调用使用 `warn_only=True`，已有同环境探针也不能替代任意负载下的逐位保证 |
| 文件与缓存 | 不同 Run ID 使用不同日志、tracker、checkpoint、result。两套数据的 legacy processed 缓存已在探针中生成，训练读取数据进入各自进程内存；归档源文件由 Git blob 校验 |
| 吞吐与显存 | GPU/CPU、显存、磁盘是共享资源，重模型可能提高 GPU 利用率，也可能拖慢现有轻模型。检查训练与完整评测的峰值，而非只看启动后的低占用；显存余量不足时不能追加第二条 |
| source gate | 原 `one` 会核对已冻结的源文件、index 和每 Run config。新增本审查文档不在数学源文件清单中，不需要修改或重写源码门 |

本次未取得完整进程枚举：当前沙箱拒绝 CIM 查询。因此执行者仍应通过已有调度会话、已知子进程 PID 和 Run 日志交叉确认候选确实未启动；仅凭 `pending` 字样不足以作为跨调度器的排他锁。

## 三、重复启动的具体风险

`launch` 在进入每个 model/data 组合时，以 `state(rid) != 'succeeded'` 一次性构造候选列表；`state` 将存在日志而无结果的 Run 标成 `started`，所以仍在运行的外部 Run 也会被选中。后续 `one` 只跳过已成功的 Run，没有针对活跃 PID 的等待或原子排他操作。

1. 如果外部 Run 已成功，主调度会排除它；即使它在候选列表生成之后、子进程检查之前刚好成功，`one` 也会直接跳过。
2. 如果外部 Run 仍运行且已有 checkpoint，主调度启动的第二个 `one` 通常会被 checkpoint guard 拒绝。该错误会写入主调度事件，不能当作正常等待或充分防撞措施。
3. 如果两个 `one` 都在首个 checkpoint 前通过检查，可能同时训练同一 Run 并写同一 tracker、日志与 checkpoint。系统 checkpoint 明确采用每 Run 单 writer 假设，此时原子发布也不能保证两个写者互不干扰。
4. 外部训练失败后保留 checkpoint，主调度稍后会再次尝试该 Run 并被 guard 拒绝。保留失败证据，按既有连续训练要求使用新 version 从头复测；不能把失败的部分 best 当作完整长训结果。

因此，正式冲突条件不是“两个进程是否同时占 GPU”，而是“是否有两个活跃写者使用同一个 Run ID”。

## 四、执行条件与记录

1. 只选择当前 index 中的目标 train Run；核对同一 source/version、seed、80000-step 预算，确认没有 result、run.log、checkpoint 或已有外部 PID。不重新 make，不改变 index、配置和 SOURCE_MANIFEST。
2. 先启动一条 CIFAR10 / resnet18 / seed 0，使用现有 `run.py one <run_id>`；单独保存 stdout、PID、开始时间、命令、退出码及最终状态。外部调度事件写到独立证据文件，不并发修改主控制器拥有的 `EXECUTION.json`。
3. 观察轻任务和重任务的实际吞吐及训练/评测显存峰值，再判断第二条是否有余量和完成窗口。后续主控制器还会启动四条 cnn，需要为该阶段留余量，不能按现在 linear 的占用推算所有后续阶段。
4. 成功窗口以原始 `result.json: succeeded`、该 Run 的 flow succeeded 日志和外部进程已退出共同确认。最迟必须在主控制器进入 CIFAR10 / resnet18 候选生成前确认；不要仅用 ETA 或参数 checkpoint 推断完成。
5. 如果窗口变得不足，需要执行者具备经过验证的“只暂停父调度、保留所有训练子进程连续运行”的进程控制能力，等外部 Run 完成后再恢复父调度。当前审查未验证这种能力；不能以关闭终端、结束整个进程树或中断训练代替父调度暂停，也不能临时修改正在运行控制器的磁盘源码来改变它已加载的循环。
6. 所有父/外部子进程均终止后再由正式入口 process。最终报告合并两份调度证据，逐 Run 确认唯一训练开始、恰好 80000 步和 400 个完整 test 点，再验收 32 train + 32 eval。提前完成的 Run 不会因缺少主调度 start 事件而被解释为未执行。

## 五、审查边界

本次只读检查代码、配置、调度证据和状态，并新增本文件。没有修改计算链、源清单、index、Run config 或控制器；没有启动 GPU 训练。提前排班在不同 Run ID 之间的计算隔离可由实现确认，整轮提速、峰值余量及外部 Run 的真实完成窗口需要执行中的实际观测。

## 六、2026-10-04 的真实协调证据

### （一）12 秒父调度暂停探针

[PARENT_COORDINATION_PROBE.json](PARENT_COORDINATION_PROBE.json) 记录的暂停区间为 01:37:50.208 至 01:38:02.215（Asia/Shanghai），约 12 秒。探针固定核对父 PID 35852、create_time 1791046826.0321603 及完整 `python -B studies/main_historical/run.py launch` 命令，只暂停此父进程。

四条已开始的 CIFAR10 / linear 训练均保持同一进程身份且 live，各自日志由 step 15800 推进至 16000：

| **seed** | **Run ID** | **PID** | **create_time** | **探针结果** |
|---|---|---|---|---|
| 0 | 24db7966c8b8469a | 66288 | 1791048277.3498528 | live，15800 → 16000 |
| 1 | f91f89c4afdcb655 | 36972 | 1791048277.3698573 | live，15800 → 16000 |
| 2 | d8583c714101e7fb | 54072 | 1791048277.3870273 | live，15800 → 16000 |
| 3 | 4b42a10dd5424d93 | 48016 | 1791048277.4101412 | live，15800 → 16000 |

正式证据为 `status=succeeded`、`passed=true`、`parent_suspension_owned=false`，并包含 `parent_suspended`、`parent_resumed`、`probe_verified` 三个事件。探针 JSON 的 SHA256 为 `f59d73a2309eea53e9fd570df921e152e401160b8e36716e3bec279251d8a489`。这验证了本机当前进程控制可单独暂停父调度，并保留其子训练连续推进；它没有验证整轮调度已经完成。

### （二）两条 eager 与 live guard

01:42–01:43 的独立只读检查，通过指定 PID 的非沙箱 psutil 查询取得下列四个 live 身份。前面三个 create_time 与预先记录完全一致；guard 的 create_time 是此次查询补充记录的实际值。默认沙箱中的进程查询曾返回 `NoSuchProcess`，非沙箱指定 PID 查询确认它们均在运行；该隔离视图的报错没有被解释为实际训练退出。

| **职责** | **PID / create_time** | **Run / 进程命令核对** | **执行 handle** |
|---|---|---|---|
| 原父调度 | 35852 / 1791046826.0321603 | 精确 argv 为 `D:\anaconda3\python.exe -B studies/main_historical/run.py launch` | 37125 |
| eager seed 0 | 64656 / 1791047530.421903 | 4b6781cffa2c2b8c；完整 `-u -B -c` wrapper 与既定模板一致 | 11464 |
| eager seed 1 | 59936 / 1791049116.9261825 | 9003fd56d3612f4d；相同 wrapper 仅 Run ID 不同，完整 argv 一致 | 17242 |
| guard watch | 62868 / 1791049184.3098533 | `python -B .tmp/historical_controller_guard.py watch --hold-at cifar-mlp --expect-run 4b6781cffa2c2b8c 9003fd56d3612f4d` | 89871 |

完整 worker 命令保存在 [EAGER_EXECUTION.json](EAGER_EXECUTION.json) 和 [GUARD.json](GUARD.json)；两条 wrapper 都设置唯一的 `sys.argv=['run.py','one',<Run ID>]`，然后执行同一个未修改的 `run.py`。guard 对 eager 使用完整命令比较，对父训练子进程仍只接受标准 `one` 入口。协调 helper 的 SHA256 为 `eeb3c6ff56b175d8143a4c8e074a9cfcd145ec0535ac996e111d2c1ab29820a1`；helper 位于 `.tmp/`，不会随 Git clone 提供，报告保留其哈希与执行证据。

本次 [EXECUTION.json](EXECUTION.json) 显示 MNIST / linear 四条已 succeeded、exit_code 均为 0；父当前在未结束的 CIFAR10 / linear 四并发组，尚无 CNN 或 resnet18 父调度组。01:43 左右当前四条 linear 已约 step 21800，eager seed 0 约 step 24800；这些训练进度只是实测快照，没有用于解锁条件。

guard 当前 `status=running`、`passed=false`、`hold_at=cifar-mlp`，最后观察阶段为 `waiting`。两个 `worker_states` 都是 `identity_alive=true`、`result=null`、`flow_succeeded=false`，与仍在训练的 Run 一致；它们尚未被记为成功。watch 尚无 `parent_suspended` 事件，实际父状态仍 running，所以还未触发正式 hold。

正式 guard 在 CIFAR10 / mlp 的四个 Run 都写入 start 事件后才暂停父，随后再次验证该组尚未结束、未出现 CNN/ResNet 父 group，精确子进程均属于 held group；提前完成的孩子需要正式 succeeded 证据。暂停期间观察已启动孩子的身份、日志与结果；只有两条 eager 均取得 `result.status=succeeded`、`[flow] succeeded` 且各自精确身份已退出，才正常恢复父并放行下一模型。若 eager 提前完成，则无需暂停。异常与 KeyboardInterrupt 会记录失败、尝试恢复 helper 自己暂停的精确父并非零退出；系统拒绝恢复时报告 `resume_error` 与 owned 状态。执行器必须保持 guard 会话存活，不能强杀 helper 或依赖关闭终端触发 Python 的 `finally`。

### （三）重复启动、源码门和显存快照

01:43 的全 Study 日志扫描发现 10 个已开始的 Run，每个恰好一个 `[flow] start phases=`；其中 4 个 MNIST / linear 已完成、4 个 CIFAR10 / linear 正在运行、2 个 eager 正在运行。父调度 start 事件是 8 个不同 Run，未包含两条 eager；没有重复 Flow start 的磁盘证据。后续仍需继续核对，不能将这个快照解释为最终 64 Run 已验收。

独立执行原 `check_unchanged()` 通过：99 个数学源文件与 65 个 expanded plan 文件（index 加 64 个 config）的哈希仍与 SOURCE_MANIFEST 一致。helper、probe、guard 和本审查文档均不在数学源清单中；此次只追加本审查文档，没有修改 helper、计算链、index、Run config 或 SOURCE_MANIFEST，也没有启动、暂停或终止 GPU 训练进程。

GPU 的独立只读快照为 01:41:53 利用率 58%、显存 9889 / 24455 MiB；执行者稍早记录的 64% / 9890 MiB 与这一时段占用相近。两条 eager 已增加设备利用率，但后续 CNN 的完整训练/评测峰值尚未实测，故实际采用较早的 CIFAR10 / mlp hold 点。完成时间、整轮吞吐收益与最终曲线复现仍待完整运行验收。
