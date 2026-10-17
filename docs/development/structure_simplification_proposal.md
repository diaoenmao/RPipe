# Structure 精简与粒度调整 proposal

状态：Enmao 2026-10-17 18:30 批准的硬件职责分离已按先行扫描实施，经 PR219 合 dev 并双侧实际 CI 验收，证据见 record§293；19:25新授权已恢复每15分钟续跑，派生改为新可用里程碑版本后周期同步三个副本，本批未同步，见 record§295。

当前审批入口（2026-10-17 23:31）：剩余项一次性盘点、逐文件结论和合并proposal见§七；PR221/222均已双侧验收。本次只推工作分支、不开PR，完成后停止等批；B2恢复合同未批准不实施。历史基线与已完成硬件方案保留供追溯。

## 一、目标、基线与当前可执行范围

以 dev `1bded360b3a4547cfbc85647320ac95af89e81c9` 为基线，structure 为 78 个 Python 文件、8989 物理行；起点 80 文件、9132 行，净减 143 行（1.57%）。这不足以证明总体精简目标完成。搬移代码、文档缩行和测试数量不算生产代码净删减。

当前继续在原文件内删除可证明重复的检查和旁路，不改变公开签名、返回值、异常、磁盘布局、八包和跨层 API 约束。每项保留入口验证、数据保全和资源退出语义；按同一职责积累里程碑，本地主题 commit，不再一两项就开 PR。

首个候选是 `make/schedule.py::load_launch_plan`：JSON 容器检查和 `wait_groups` 已在 readiness 过滤之前验证所有 job 为 object、安全 Run ID、唯一 Run 及显式 batches 的完整多重集；之后的 job 类型检查和静默过滤重复。只删除成功校验之后的重复防御，保留坏缓存返回 None、配置/设备失配检查和原异常边界。先复用既有坏缓存/重复 Run/多重集回归，补充合法缓存输出顺序与独立列表合同。该项属于已授权的同文件精简，不是架构实施。

## 二、要动哪部分架构

已批准的范围仅限 `make/capacity.py` 的硬件发现职责。建议把 `GpuInfo`、`probe_gpus`、`_probe_torch`、`_probe_nvidia_smi` 放入一个具名 `make/hardware.py`；capacity 保留装箱、容量计算与报告，通过直接导入使用这些对象。保持 make 包与 make_api 的现有导出名称，不新增门面、包装、回退模块或通用框架；更新 structure.md 逐文件登记和对应测试位置。

不把硬件探测塞进只负责配置估算的 `make/estimate.py`，也不把实现放进 API 文件。暂不提出 train/HF/schedule/transfers 的拆分：文件较长本身不是违规，必须先找到具体职责冲突及可删实现。派生副本按19:25确认的新可用里程碑版本后周期同步口径执行；本批不读取或覆盖库外 vendor。

## 三、为什么当前组织需要评审

提案基线 capacity 为 385 行（本轮同文件包装删除后为 372 行），同时包含平台探测（subprocess、可选 torch）、纯装箱和文本报告。硬件依赖和纯计算共处一模块，使依赖边界不清；这是 §2.1.4 的人工粒度评审项，不是已复现的运行缺陷。当前局部导入已避免导入时触发 CUDA，因此没有证据表明仅拆分会修复正确性或产生性能收益。

本次不把分责冒充净瘦身。若确认后仍只能搬移、没有减少重复逻辑或依赖复杂度的可核验收益，则不单独为拆分开 PR，继续同文件净删减。现有布局可继续使用，架构评审不阻塞独立 CPU 精简。

## 四、改完的收益、接口与副本影响

预期收益是把硬件依赖集中到一个模块，capacity 纯计算与探测边界可单独核验。公开 `make_api.GpuInfo/probe_gpus/...` 的名称、签名、返回字段、设备选择与探测失败回退均保持；flow/cli.py 继续经 make_api。库内还包括 make/__init__.py 的导出、capacity 内部调用和 test_capacity.py 的直接导入/patch 位置。

已确认的身份影响：移动 GpuInfo 定义会改变新对象的 `__module__`（涉及 repr/新 pickle 路径），即使 capacity 继续重导出、旧 pickle 的原名称仍可解析，也不能声称完全无公开行为影响。不自行添加兼容别名或修改 __module__ 掩盖变化。18:30 批准要求先扫实际对象，已完成的范围、数量和限制见 record§293.1；若发现实际对象读不回则停下报告，不增加兼容旁路。

库外副本目前未给出确认过的物理根、上游提交和允许同步范围，不能给出受影响副本数量或保证其兼容。待确认时一并确定目标清单、当前 vendor 版本、仅比对或可覆盖范围；先只读 diff，保全下游改动，另行验安装/核心流程，不跑其训练，也不以本机通过替代异机验收。

## 五、预计改动量、风险与验收

| **范围** | **预计改动量（非实测）** | **风险与验收** |
| --- | --- | --- |
| 硬件职责分离 | 新增 1 个源码文件，移动约 80–100 行；总生产行数可能持平或略增，不能计为删减 | 类模块身份、monkeypatch 位置和导入循环；先确认身份/API影响，再做导出/序列化/离线探测合同回归 |
| 文件登记与测试位置 | 约 3–5 个源码/测试/文档文件，依最终批准方案核定 | 不遗漏直接调用方；init 只导出、八包、登记与跨层检查全部通过 |
| 同文件重复防御删除 | 按事实累计，不承诺虚构行数配额 | 坏输入仍在入口拒绝；正常输出、顺序、对象身份、异常和磁盘字节不变 |
| 派生同步 | 三个副本在新可用里程碑版本后周期同步，本批不实施 | 实际同步前明确根/版本/范围与冲突保全办法，绝不覆盖未知下游改动 |

每个里程碑列出相对实际 dev 的生产净行数、删除的重复分支/过滤层、增加的 helper 与文件数，区分搬移量与净删除。完整验证前冻结提交和源码；经 tests/run.py 显式排除 gpu/external/slow，执行 core 与 CPU Flow，新增节点和结构节点实际执行。完整批次再推送/开 PR，最终 head 与实际 dev 分别验收所有真实 CI 矩阵/聚合。不以单行压缩、去空行、删必要边界或放宽测试达成数字。

当前决定边界：按18:30批准迁四个硬件声明，并继续同文件净删减；不改变 make_api 形状/调用方式，不增加兼容别名，19:25已批准并恢复每15分钟续跑，副本待新可用里程碑版本就绪后周期同步，本批不实施同步。

## 六、2026-10-17 22:19 系统盘点与后续批次

当前基线为PR220实际dev2c6bf0c8：structure79个Python文件/8927物理行，15个init只导出、64个非init逐项登记，未登记/登记失效均0。以下全文件清单基于Git跟踪、AST和structure登记表；职责混杂与重复仍属人工判断，不把机械通过写成“无冗余”。原始机器清单在.tmp/structure-systematic/INVENTORY.json，不随clone，正式文件清单如下。

### （一）按净收益与权限划分批次

| **排序/批次** | **具体对象与重复** | **预计生产净删减（非实测）** | **决定与验证** |
| --- | --- | --- | --- |
| 1 / B2待批 | system/factory.py同时写bundle和分片诊断镜像，并兼容payload/meta/分片读取；存在多套序列化/恢复分支 | 若删除一种完整旧路径，约50–90行；须先统计实际旧格式使用，不能承诺全部可删 | 涉及恢复行为/磁盘合同，先提案，不实施；当前镜像与staging保全合同照旧 |
| 2 / B1立即做 | metric.py的metric名称映射与full结果收集；system/factory.py重复bundle选择赋值/返回分支；report.py重复四统计字段输出；logger.py重复elapsed/eta输出 | 合计20–35行，按完成后的真实diff验收，不凑配额 | 仅原文件/函数内归纳，无新helper/API/目录；先原新对照、便携回归，再冻结core+CPU Flow；累计成完整里程碑再PR |
| 3 / B3职责评审 | train/loop.py、transformers_trainer.py、transfers.py、schedule.py | 当前只能确认职责热点，净删减尚无可核验估值 | 不按长度拆文件；有具体共享行为与净收益后另提proposal，不能用搬移量充当产出 |
| 保留 | 各层Config/Registry、API重导出、init、恢复/URL/路径/资源退出门、模型初始化顺序 | 0已确认可删 | 有公开身份、参数、数值或安全合同，不能批量删除；未处理不等于已证实冗余 |

B1是第一可执行批，包含多个同主题改动，不逐项开PR。B2虽潜在净收益更高，但未获实施批准；等待该方案不阻塞B1。完成一批报告生产净行数和真正执行的行为覆盖，不按轮次、扫描量或PR数验收。没有具体新候选的空轮不追加record。

### （二）B2架构/恢复合同候选proposal：仅供审议，未批准

1. 要动的部分：system/factory.py的checkpoint持久化与读取路径。可选方向是未来只发布一个原子bundle，并在明确迁移后移除目录分片旧读路径；当前不改任何文件或目录布局。
2. 为什么评审：现实现为保全原诊断与旧对象恢复，维护bundle、model/optimizer/scheduler分片、tracker/logger JSON、meta.json/meta.pt以及incomplete/staging多个路径。重复读写使维护面扩大，但其必要性有真实恢复合同，不能当无用防御直接删。
3. 预期收益与影响：可能少一套序列化/恢复分支，估计50–90行；会影响System.save_checkpoint/load_checkpoint行为、Algorithm.resume、compare/readout、旧checkpoint与派生副本。公共函数名称可不变，但接受格式、诊断产物与恢复能力会变，必须先批准迁移合同；不新增旧别名或隐式兜底掩盖变化。
4. 风险与前置：批准方向后先只读统计实际bundle-only/legacy-only/混合对象及调用方，列明受影响文件和副本；任何旧对象不得删除或覆盖。必须保留崩溃、取消、最后成功bundle、未发布staging及原异常证据，给出回退/离线转换方案后再定是否实施。若不能同时保全诊断与恢复能力，则维持现状，撤回净删减预期。

### （三）过大/职责混合文件的剩余项

- algorithm/train/loop.py（483行）：配置/精确恢复、训练迭代、eval/checkpoint调度和最终flush；恢复时序与训练循环耦合，不能先搬文件再找收益。
- algorithm/transformers_trainer.py（370行）：HF adapter、loader/model包装、callback与native合同桥接；与native checkpoint选择共享部分已在progress，不强并不同payload路径。
- system/transfers.py（361行）：传输安装/恢复、线程/进程上下文、惰性dataset/media消费者。其清理与所有权共同约束生命周期，归并前必须证明安装/取消/最后退出顺序。
- make/schedule.py（365行）：readiness、缓存计划校验和subprocess生命周期；scripts已分责，下一步只删已被前置门证明重复的处理。
- artifact/readout/compare.py（292行）：标量/张量精确比较与磁盘读取，精度分支不是任意防御，不按长度去掉科学门。
- make/capacity.py（293行）：硬件探测已分离，剩装箱/估算报告同一职责域；大于200行不自动判违规。

### （四）全文件逐项清单

| **文件** | **物理行** | **职责/剩余分类** |
| --- | ---: | --- |
| `__init__.py` | 19 | 导出；保持；exports only |
| `algorithm/__init__.py` | 14 | 导出；保持；exports only |
| `algorithm/base.py` | 83 | 已登记；尚无已证实大项重复；Algorithm base class (no factory import). |
| `algorithm/batch.py` | 22 | 已登记；尚无已证实大项重复；Native batch prep: flatten only when the module is a linear-style head. |
| `algorithm/config.py` | 29 | 已登记；尚无已证实大项重复；AlgorithmConfig dataclass. |
| `algorithm/eval/__init__.py` | 1 | 导出；保持；exports only |
| `algorithm/eval/loop.py` | 72 | 已登记；尚无已证实大项重复；Eval mode: independent algorithm; resume then score. |
| `algorithm/eval_hook.py` | 107 | 已登记；尚无已证实大项重复；Test-split body for train ``AlgorithmHook.on_eval_period`` (not a Flow mode). |
| `algorithm/factory.py` | 56 | 已登记；尚无已证实大项重复；Algorithm factory: mode + source → implementation. |
| `algorithm/hf_map.py` | 101 | 已登记；尚无已证实大项重复；Map algorithm extras onto Transformers TrainingArguments (same keys as native). |
| `algorithm/hook.py` | 72 | 已登记；尚无已证实大项重复；AlgorithmHook: named loop insertion points (not a Flow phase). |
| `algorithm/kwargs.py` | 54 | 已登记；尚无已证实大项重复；Pass config extras through a constructor, keeping only valid parameters. |
| `algorithm/metric.py` | 237 | B1：复核同文件重复；Batch / full metrics compatible with git main names (Loss, Accuracy, MSE, RMSE, GLUE). |
| `algorithm/optim.py` | 96 | 已登记；尚无已证实大项重复；Native optimizer / scheduler (algorithm-layer interface). |
| `algorithm/progress.py` | 218 | 已登记；尚无已证实大项重复；Train budget and checkpoint cadence (epoch or step). |
| `algorithm/resume.py` | 209 | 已登记；尚无已证实大项重复；Resume target resolution (algorithm-layer; system only loads files). |
| `algorithm/tracker.py` | 221 | 已登记；尚无已证实大项重复；AlgorithmTracker: numeric ledger (curves in assets/tracker). |
| `algorithm/train/__init__.py` | 4 | 导出；保持；exports only |
| `algorithm/train/loop.py` | 483 | B3：职责热点，非已认定违规；Train mode. |
| `algorithm/transformers_trainer.py` | 370 | B3：职责热点，非已认定违规；algorithm.source: transformers_trainer — §6.11 keys, real HuggingFace Trainer loop. |
| `api/__init__.py` | 1 | 导出；保持；exports only |
| `api/algorithm_api.py` | 32 | 已登记；尚无已证实大项重复；algorithm_api: Algorithm, AlgorithmTracker, AlgorithmFactory.build, AlgorithmConfig. |
| `api/artifact_api.py` | 40 | 已登记；尚无已证实大项重复；磁盘产物层的纯重导出门面。 |
| `api/data_api.py` | 22 | 已登记；尚无已证实大项重复；data_api: Data, DataFactory.build, DataConfig. |
| `api/make_api.py` | 24 | 已登记；尚无已证实大项重复；脚本生成层的纯重导出门面。 |
| `api/model_api.py` | 9 | 已登记；尚无已证实大项重复；model_api: Model, ModelFactory.build, ModelConfig. |
| `api/system_api.py` | 45 | 已登记；尚无已证实大项重复；system_api: System, Logger, SystemFactory.build, SystemConfig. |
| `artifact/__init__.py` | 49 | 导出；保持；exports only |
| `artifact/_atomic.py` | 80 | 已登记；尚无已证实大项重复；Atomic text write (unique temp file then replace). |
| `artifact/activity.py` | 53 | 已登记；尚无已证实大项重复；Current Study task, so a long make can be seen without waiting for the final print. |
| `artifact/asset/__init__.py` | 12 | 导出；保持；exports only |
| `artifact/asset/kinds.py` | 31 | 已登记；尚无已证实大项重复；Run asset names, directory creation and recursive file listing. |
| `artifact/config/__init__.py` | 3 | 导出；保持；exports only |
| `artifact/config/io.py` | 34 | 已登记；尚无已证实大项重复；Config YAML mapping codec and atomic entity IO. |
| `artifact/index.py` | 117 | 已登记；尚无已证实大项重复；Study ``index.json`` — orchestration plan written before Flow. |
| `artifact/layout.py` | 111 | 已登记；尚无已证实大项重复；Study/Run layout, canonical leaf names and artifact IO errors. |
| `artifact/provenance.py` | 207 | 已登记；尚无已证实大项重复；Study provenance: source / plan hashes, environment, and the freeze gate. |
| `artifact/readout/__init__.py` | 5 | 导出；保持；exports only |
| `artifact/readout/compare.py` | 292 | B3：职责热点，非已认定违规；Compare two Run directories: result metrics, tracker history, checkpoint state. |
| `artifact/readout/logs.py` | 54 | 已登记；尚无已证实大项重复；Print Study run.log event lines in timestamp order. Does not write a file. |
| `artifact/readout/report.py` | 170 | B1：复核同文件重复；Write Experiment and Run number tables from process.json. Does not write a conclusion. |
| `artifact/readout/status.py` | 295 | 已登记；尚无已证实大项重复；Read-only Study listing: index plan joined with each Run result. |
| `artifact/result/__init__.py` | 14 | 导出；保持；exports only |
| `artifact/result/io.py` | 65 | 已登记；尚无已证实大项重复；Result JSON mapping codec, validation and atomic entity IO. |
| `artifact/shared.py` | 250 | 已登记；尚无已证实大项重复；Project input roots, interprocess preparation lock, and explicit cache migration. |
| `control/__init__.py` | 48 | 导出；保持；exports only |
| `control/contract.py` | 44 | 已登记；尚无已证实大项重复；Light contract checks for RunConfig / Control. |
| `control/control.py` | 66 | 已登记；尚无已证实大项重复；Control object: holds RunConfig for one Run. |
| `control/mapping.py` | 76 | 已登记；尚无已证实大项重复；Config mapping and shared layer conversion, deep merge, and stable content identifiers. |
| `control/run_config.py` | 154 | 已登记；尚无已证实大项重复；ExperimentConfig / RunConfig and merge + id hashing. |
| `data/__init__.py` | 4 | 导出；保持；exports only |
| `data/config.py` | 34 | 已登记；尚无已证实大项重复；DataConfig dataclass. |
| `data/factory.py` | 213 | 已登记；尚无已证实大项重复；Dataset registry, factory and vision builders. |
| `data/prepare.py` | 83 | 已登记；尚无已证实大项重复；Materialize project-shared datasets once (not per Run). |
| `data/profile.py` | 173 | 已登记；尚无已证实大项重复；Dataset profile written next to shared data. |
| `data/resume.py` | 67 | 已登记；尚无已证实大项重复；Replay sampler indices, never consumed dataset transforms, on CPU-loader resume. |
| `data/runtime.py` | 193 | 已登记；尚无已证实大项重复；Runtime Data consumption, loader options and step-budget rebinding. |
| `make/__init__.py` | 63 | 导出；保持；exports only |
| `make/capacity.py` | 293 | B3：职责热点，非已认定违规；Batch packing, capacity calculation and reporting; existing hardware exports remain direct imports. |
| `make/estimate.py` | 175 | 已登记；尚无已证实大项重复；Config-only job memory/time estimates and annotation; no hardware probing. |
| `make/expand.py` | 135 | 已登记；尚无已证实大项重复；Expand a Study declaration into Run patches. |
| `make/hardware.py` | 86 | 已登记；尚无已证实大项重复；GpuInfo and GPU discovery through nvidia-smi or optional torch; no packing or reporting. |
| `make/recipe.py` | 234 | 已登记；尚无已证实大项重复；Load a Study recipe and call its ``register(ctx)`` before Data / Model are built. |
| `make/schedule.py` | 365 | B3：职责热点，非已认定违规；Run readiness, launch-plan reuse and subprocess lifecycle. |
| `make/scripts.py` | 208 | 已登记；尚无已证实大项重复；Wait-group validation, shell generation and atomic script publication. |
| `make/write.py` | 76 | 已登记；尚无已证实大项重复；Expand a Study declaration and write Run configs, index, and provenance. |
| `model/__init__.py` | 4 | 导出；保持；exports only |
| `model/config.py` | 24 | 已登记；尚无已证实大项重复；ModelConfig dataclass. |
| `model/custom_torch.py` | 232 | 已登记；尚无已证实大项重复；custom_torch models from git main: linear, mlp, cnn, resnet10/18, wresnet. |
| `model/factory.py` | 273 | 已登记；尚无已证实大项重复；Runtime Model、registry 与模型构建，包含形状解析和参数初始化。 |
| `model/input_norm.py` | 55 | 已登记；尚无已证实大项重复；Input normalize and train augmentation in front of the network. |
| `system/__init__.py` | 14 | 导出；保持；exports only |
| `system/config.py` | 28 | 已登记；尚无已证实大项重复；SystemConfig dataclass. |
| `system/factory.py` | 185 | B1：复核同文件重复；Runtime System object, factory, and Logger. |
| `system/logger.py` | 140 | B1：复核同文件重复；System Logger: stdout + assets/logs/run.log (flush on every emit). |
| `system/origin.py` | 219 | 已登记；尚无已证实大项重复；来源与端点选择、模型/数据下载入口及输入预检；公开名称经 system_api 对外提供。 |
| `system/runtime.py` | 84 | 已登记；尚无已证实大项重复；Apply Run seed and torch runtime flags. Call at the start of prepare. |
| `system/torchvision_load.py` | 35 | 已登记；尚无已证实大项重复；Load torchvision after its native ops are actually registered. |
| `system/transfers.py` | 361 | B3：职责热点，非已认定违规；SDK 传输钩子、并发策略作用域及延迟 reader/media/generator 生命周期；私有状态由本模块独占。 |


## 七、2026-10-17 23:31 剩余项一次性盘点与合并审批入口

本节覆盖§六的“B1待CI”等旧状态：PR221/222均已双侧验收。基线为当前工作分支8f06b63（源码与dev f55a873一致），全structure79文件/8905行；未登记0、失效登记0。本次只推工作分支，不开PR；同文件能安全做的只收录确实无用的返回中间态，Model.to_result_snapshot原预计净少2行、实际净少1行，先核验字典表达式AST与其他声明不变，再跑模型与结构CPU回归。其他可见相似项不为数字强并，详见下方完整表与合并proposal；完成后停下等批。


### （一）清单口径与已执行项

所有79个当前磁盘/跟踪源码均做AST及登记对照，不只看相邻文件；对已登记大文件和重复候选再读具体调用上下文。机械扫描发现的相同import、同名dataclass字段、短导出和不同职责的相似语句，不自动算可删实现。扫描不能证明动态调用和未知库外副本无影响，本节也不声称已数学证明“任何一行都不能再删”。

- 当前结构：79文件；盘点前8905行。本次`model/factory.py::Model.to_result_snapshot`去掉仅用于立即返回的局部`out`，实际净少1行，变为8904行；字典表达式AST、元数据过滤/覆盖/浅引用语义及其他声明保持。不是重排格式、不是一次新架构里程碑；没有拿它凑PR。
- 验证：`tests/run.py --core -- tests/rpipe/structure/model tests/rpipe/structure/test_structure.py`，174 passed/0skip/fail/error，显式排除GPU/external/slow，独立basetemp。未为1行等价返回跳转重复全仓训练/测试；下一获批源代码里程碑冻结后再完整core+CPU Flow。本次不借PR222的完整测试当新源码验收。
- `__init__`15个、非init登记64个，未登记0、登记失效0；八包与跨层API机械检查通过。不存在“先补登记再才能动”的隐藏源码清单。
- 以下预计净删均为评审估算，非已完成产出；同一区域的不同方案不叠加求和。没有可支持的正净收益就写0，不把搬移量当净删除。

### （二）真重复与相似实现逐项裁定

| **项** | **对象/事实依据** | **预计净删行（非实测）** | **现在为何不能继续/决定** |
| --- | --- | ---: | --- |
| R0 已执行 | model/factory.py `Model.to_result_snapshot`仅将字典绑定out后立即返回 | 实测1 | 同文件安全项已做；174节点与原表达式AST通过；无新helper |
| R1 | algorithm/train/loop.py:325与transformers_trainer.py:228各17行checkpoint选择前缀完全相同，后续payload/恢复不同 | 0；未设计的更大归一不作正收益承诺 | 需跨文件/职责设计；必须保留运行前缓存的cadence选项，不能改成每次重新读config。参数透传/新helper会吃掉17行毛删除，当前不推荐单独抽取 |
| R2 | 上述两处resume后optimizer/scheduler装载约8行相同，best_accuracy回退条件不同 | 0 | 跨文件且有行为风险：native只在Accuracy回退，HF旧行为不同；不统一其判定、不新添通用恢复包装 |
| R3 | control/run_config.py两类共享7行字段声明，from_mapping各有相同的9个字段转换 | 严格保持公开行为约0；允许构造/MRO改变可能4–10 | 同文件可见重复，但继承会改变RunConfig构造id位置/字段顺序；提取kwargs helper加入定义/调用后收益有限。当前同文件安全归纳已做to_mapping共享，不擅改class形状 |
| R4 | algorithm/config.py与system/config.py的setting方法体完全相同3行；三个Config已共享from/to mapping | 0 | 跨文件；引入mixin或新helper的定义/导入成本不小于3行，MRO/签名额外风险。DataConfig origin拒绝门不等同普通codec，保留 |
| R5 | data/factory.py与model/factory.py各3个Registry方法同构；AlgorithmRegistry另外有惰性初始化 | 严格保留签名/方法归属约0；放宽后0–8 | 需跨层公共归属或新基类/模块，返回注解和类独立_items不能丢；不把Algorithm的惰性初始化当重复检查删掉。收益不足，不推荐本批实施 |
| R6 | model/factory.py多个builder重复shape解析/元数据，custom_torch.py多个网络有相似forward | 0已确认 | 新通用builder需要更多参数/回调；CNN/ResNet/MLP初始化顺序、元数据字段和模型state键不同。WideBlock分支涉及原地ReLU/shortcut输入，不能为几行改变数值或hook轨迹 |
| R7 | system/factory.py同时bundle、诊断分片及legacy payload/meta恢复 | 旧路径毛删除50–90；包含迁移代码后净收益尚未定，按0–70预算 | 需磁盘/恢复合同批准；见下方P1。不删除旧对象、不取消诊断/保全门。迁移成本若超过毛删减则撤回方案 |
| R8 | data/runtime.py与algorithm/progress.py均从loader/meta推断步数 | 0–6 | 跨层职责与回退不同：algorithm支持任意duck-typed Data，Data自己的元数据兼容独立存在；未经新能力合同不能只信一种方法 |
| R9 | artifact/config/io.py与result/io.py都有编码/读写包装 | 0 | YAML空值、JSON对象/状态校验、错误类型不同；合并成通用codec参数层不会减少真实职责。保留信任边界 |
| R10 | readout的有限数值/路径/错误处理与algorithm.metric等有相似门 | 0 | 单元格占位、严格科学比较、训练指标的布尔/整数/复杂/非有限值政策不同；未证实语义相同，不合成“安全版”公共包装 |
| R11 | make/schedule.py外层与scripts.wait_groups/readiness有看似重复检查 | 0剩余已证实 | 前批已删成功容器校验后的重复过滤；剩余坏缓存、配置设备、依赖train/eval、进程wait/异常保全处于不同时间和信任边界，不再无依据删门 |
| R12 | system/transfers.py的sync/async、reader/generator上下文退出路径相似 | 0已确认 | 返回/await/yield/close时机不同，线程ContextVar、fork、取消与primary error不能合并成另一层兜底。分责可评审，纯搬移不算净产出 |

### （三）较大文件与杂烩候选：19项逐一结论

“≥200行”只作为本轮完整阅读索引，不是§2.1.4行数门。只有明确独立职责才列为可分责；所有行数是盘点前物理行。

| **文件/行** | **可独立命名的职责与判断** | **预计净删/阻塞** |
| --- | --- | --- |
| algorithm/train/loop.py 483 | TrainAlgorithm hook与native精确恢复/迭代/调度/最终flush；确实有多职责 | 分责净0（搬移）；R1/R2未证明正净收益，需跨文件设计，不作删减承诺；P2待批 |
| algorithm/transformers_trainer.py 370 | batch/dataset/model适配器、Trainer/callback桥接、scheduler适配 | 分责净0；R1/R2跨文件，不能删除native/HF恢复差异；P2待批 |
| system/transfers.py 361 | SDK传输安装/还原、线程/fork作用域、惰性reader/media消费 | 分责净0；约束依赖共享私有状态，需架构提案与资源轨迹对照；P3待批 |
| make/schedule.py 365 | readiness/cache计划验证与进程波次/retry生命周期 | 分责净0；scripts已拆出，剩余生命周期门有行为风险，P3仅备选不优先 |
| model/factory.py 273→272 | Model快照、Registry/Factory、shape/init与各builder | R0实删1；移动shape/init会逆转已批准碎片归并。新factory框架不推荐 |
| make/capacity.py 293 | 装箱、容量与报告；硬件发现已分离 | 0已确认；同一容量职责域，没证据必须再拆 |
| artifact/readout/status.py 295 | index/result/config合并、log摘要、展示单元格 | 0已确认；多输入信任边界不是重复门，不能为长度移走readout |
| artifact/readout/compare.py 292 | 精确标量/张量树比较及文件载入/格式化 | 0已确认；Fraction/非有限/混合整数保护是科学门，不删；若分责也净0 |
| artifact/shared.py 250 | 共享根、OS锁、不可变资产引用/迁移 | 0已确认；锁与发布/清理拥有权合同不同，拆分仅搬移、风险高 |
| make/recipe.py 234 | Study recipe加载、隔离namespace与延迟导入 | 0已确认；作用域与sys.modules身份保全不可当防御冗余 |
| model/custom_torch.py 232 | 不同网络构造与forward | 0已确认；算法集合不是杂烩工具，不能合并有不同state键/原地语义的模块 |
| algorithm/metric.py 228 | metric名称/打包、batch/full计算/收集 | 0剩余已证实；前批已归纳映射/收集/MSE公共尾部，不再动数值公式 |
| algorithm/tracker.py 221 | 加权统计、分段history/JSONL、恢复状态 | 0剩余已证实；这些是同一数值主源的生命周期，不删恢复指针或另起数值源 |
| system/origin.py 219 | 国内源选择、模型/数据下载与预检 | 0已确认；transport已在transfers分责，源策略与入口约束不同，保留国内源门 |
| algorithm/progress.py 218 | 预算、周期/百分位、ETA | 0剩余已证实；百分位已简化，跨文件候选R1/R8另列，当前不新增抽象 |
| data/factory.py 213 | 注册表、stub/vision构建、vision规格 | 0剩余已证实；九字段规格37行归一已实施，Registry候选R5不宜单抽 |
| algorithm/resume.py 209 | checkpoint选择、权重载入与RNG状态 | 0已确认；legacy/新格式与partial/exact是实际兼容合同，随P1审批再定 |
| make/scripts.py 208 | wait-group校验、shell生成与发布 | 0剩余已证实；shell平台不同，cache/readiness外仍须独立校验 |
| artifact/provenance.py 207 | 源码/计划清单、hash、freeze与环境来源 | 0已确认；同源键碰撞与路径越界门保证追溯，不删或改哈希口径 |

### （四）合并proposal：请求批准范围与实施门

以下合写在本文件，不另开多个proposal/PR；本次只推分支后停止。**批准本文入库不等于批准实施**。建议优先只评审P1是否允许变更恢复合同；P2/P3若只有搬移收益则不做。当前没有可无风险承诺的“再净删几十行”的同文件批次。

| **方案** | **要改什么/为什么** | **预计净删量（含新增代码）** | **接口/副本影响与风险** | **验证与停止门** |
| --- | --- | --- | --- | --- |
| P1 checkpoint单一发布/显式legacy导入（承接B2） | 评估未来只写原子bundle，诊断与旧piece/payload/meta恢复不再与正常保存并行维护；须先决定哪些诊断职责仍保留，不能直接砍 | 旧路径毛删50–90；迁移/显式读取/诊断新增暂按20–50预算，净约0–70，最终需dry-run diff；两端极值不作承诺 | save/load接受格式、旧持久化对象、algorithm.resume及compare、下游副本可能受影响。函数名相同也属于行为/布局变化；旧数据与staging都不删不覆盖，不加隐式别名掩盖影响 | 先批准方向，再只读统计实际bundle-only/legacy-only/mixed及真实调用；保全旧对象和原hash，设计离线只读导入/新目录转换及回退；取消/断写/最后bundle/诊断/异常身份回归+core/CPU Flow。不能保持要求的诊断或净收益≤0则撤回 |
| P2 algorithm执行职责归一 | 仅在R1/R2的真实共同合同明确后，评估将checkpoint选择和恢复装载放现有progress/resume，train与HF各保留执行hook；如确需拆train loop，单列具体落点后批 | 净0已确认；更大归一未测算，不作正收益承诺；纯分责0 | 运行前一次性读取选项、native exact RNG/data/state与HF partial恢复不得混同；新函数/职责迁移、可能的新模块须先批准，不能直接直连其他层 | 合成native/HF callback轨迹对照，cached选项访问次数、payload/percent提交时序、异常和finalizer保持；不跑真实Trainer/训练/GPU。严格公开合同下无净收益则不做 |
| P3 生命周期职责拆分（备选） | transfers的transport安装与惰性consumer、schedule的计划/readiness与subprocess分开具名；只为阅读边界，非删功能 | 净0，新增导入可能略增；不算瘦身验收项 | 私有全局状态归属、scope/fork/pickle/deferred close及patch位置会变；未提供收益前不建议优先。新增文件/目录需另明确树，不在本次实施 | 完整线程/fork/取消/reader/generator关闭轨迹、patch恢复、缓存/依赖/retry CPU合成测试；若只是搬运无复杂度收益不实施 |

R3–R6的小型mixin/registry/通用builder暂列“有重复但不推荐抽象”，不请求一揽子批准新框架。八子包、init只导出、公开签名与跨层门面继续保持；readout保留artifact。三个派生副本须等指定可用里程碑版本，先确认物理根/上游版本/允许范围，本次不读取/覆盖vendor。未知库外对象数量不能从本仓库扫描推断。

### （五）_staging_from_DEM来源、字节核对与登记决定

该未跟踪根目录是监督助手按人工指示从DEM复制的中文研究计划规范/模板暂存交接材料，中文名各一份、ASCII名各一份，共4文件；不是Run、模型/数据、源码或新的正式docs目录。本次不执行其中内容，不删除、不移动、不忽略隐藏、不git add。

| **暂存文件** | **字节数** | **SHA256** | **原字节一致的正式文件** |
| --- | ---: | --- | --- |
| `Research_Plan_Standard_zh.md` | 25618 | `61fc9fbe1130810b41ef25d3655eef2f83289b60306458ad5e4c17b9d4658bd3` | [研究计划规范 Research Plan Standard.md](<../research/standards/研究计划规范 Research Plan Standard.md>) |
| `Research_Plan_Template_zh.md` | 9309 | `b53a7250fd57f754a57a2ac3ddb918eefe7a0ac7a4c28096f39c858486370f44` | [研究计划模板 Research Plan Template.md](<../research/standards/研究计划模板 Research Plan Template.md>) |
| `研究计划模板 Research Plan Template.md` | 9309 | `b53a7250fd57f754a57a2ac3ddb918eefe7a0ac7a4c28096f39c858486370f44` | [研究计划模板 Research Plan Template.md](<../research/standards/研究计划模板 Research Plan Template.md>) |
| `研究计划规范 Research Plan Standard.md` | 25618 | `61fc9fbe1130810b41ef25d3655eef2f83289b60306458ad5e4c17b9d4658bd3` | [研究计划规范 Research Plan Standard.md](<../research/standards/研究计划规范 Research Plan Standard.md>) |

登记决定：不加进structure逐文件登记（§2.1.3只针对structure/*.py），也不把相同中文正本再次提交成第三份正式文档。正式标准/模板已经由docs/research/standards登记与引用；暂存来源和内容在本节有明确清单即可。保留未跟踪状态与全部原字节；是否清掉暂存留待Enmao另行授权，不因本次盘点自动清理。对照没有读取DEM原盘，因此只证明暂存与仓库正式文件一致，不外推原盘当前状态。

### （六）79文件完整剩余分类（当前源码）

下表的“0”指本轮未证实另有安全且有净收益的删除，不是普遍无冗余的保证；专项R/P项按上表评审。行数在R0实施后更新。保留项也是逐项审议结果，不能当待办逐个机械搬移。

| **文件** | **行数** | **登记** | **预计进一步净删/原因** |
| --- | ---: | --- | --- |
| `__init__.py` | 19 | init豁免、导出通过 | 0；只导出，不以删公开导出名充当净瘦身 |
| `algorithm/__init__.py` | 14 | init豁免、导出通过 | 0；只导出，不以删公开导出名充当净瘦身 |
| `algorithm/base.py` | 83 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `algorithm/batch.py` | 22 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `algorithm/config.py` | 29 | 已登记 | R4：0；mixin成本抵消，保留setting优先级 |
| `algorithm/eval/__init__.py` | 1 | init豁免、导出通过 | 0；只导出，不以删公开导出名充当净瘦身 |
| `algorithm/eval/loop.py` | 72 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `algorithm/eval_hook.py` | 107 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `algorithm/factory.py` | 56 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `algorithm/hf_map.py` | 101 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `algorithm/hook.py` | 72 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `algorithm/kwargs.py` | 54 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `algorithm/metric.py` | 228 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `algorithm/optim.py` | 96 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `algorithm/progress.py` | 218 | 已登记 | R1/R8：共享估算见前表，不叠加；行为风险 |
| `algorithm/resume.py` | 209 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `algorithm/tracker.py` | 221 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `algorithm/train/__init__.py` | 4 | init豁免、导出通过 | 0；只导出，不以删公开导出名充当净瘦身 |
| `algorithm/train/loop.py` | 483 | 已登记 | R1/R2、P2：跨文件/行为风险；0已确认正净收益 |
| `algorithm/transformers_trainer.py` | 370 | 已登记 | R1/R2、P2：跨文件/行为风险；同一0收益候选，不叠加 |
| `api/__init__.py` | 1 | init豁免、导出通过 | 0；只导出，不以删公开导出名充当净瘦身 |
| `api/algorithm_api.py` | 32 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `api/artifact_api.py` | 40 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `api/data_api.py` | 22 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `api/make_api.py` | 24 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `api/model_api.py` | 9 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `api/system_api.py` | 45 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `artifact/__init__.py` | 49 | init豁免、导出通过 | 0；只导出，不以删公开导出名充当净瘦身 |
| `artifact/_atomic.py` | 80 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `artifact/activity.py` | 53 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `artifact/asset/__init__.py` | 12 | init豁免、导出通过 | 0；只导出，不以删公开导出名充当净瘦身 |
| `artifact/asset/kinds.py` | 31 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `artifact/config/__init__.py` | 3 | init豁免、导出通过 | 0；只导出，不以删公开导出名充当净瘦身 |
| `artifact/config/io.py` | 34 | 已登记 | R9：0；YAML codec错误/空值语义独立 |
| `artifact/index.py` | 117 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `artifact/layout.py` | 111 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `artifact/provenance.py` | 207 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `artifact/readout/__init__.py` | 5 | init豁免、导出通过 | 0；只导出，不以删公开导出名充当净瘦身 |
| `artifact/readout/compare.py` | 292 | 已登记 | R10：0；科学比较与有限/精度门保留 |
| `artifact/readout/logs.py` | 54 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `artifact/readout/report.py` | 167 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `artifact/readout/status.py` | 295 | 已登记 | R10：0；多输入/路径/错误语义保留 |
| `artifact/result/__init__.py` | 14 | init豁免、导出通过 | 0；只导出，不以删公开导出名充当净瘦身 |
| `artifact/result/io.py` | 65 | 已登记 | R9：0；JSON/status合同独立 |
| `artifact/shared.py` | 250 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `control/__init__.py` | 48 | init豁免、导出通过 | 0；只导出，不以删公开导出名充当净瘦身 |
| `control/contract.py` | 44 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `control/control.py` | 66 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `control/mapping.py` | 76 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `control/run_config.py` | 154 | 已登记 | R3：0安全；MRO/构造顺序风险 |
| `data/__init__.py` | 4 | init豁免、导出通过 | 0；只导出，不以删公开导出名充当净瘦身 |
| `data/config.py` | 34 | 已登记 | R4：0；独立origin信任边界保留 |
| `data/factory.py` | 213 | 已登记 | R5：0安全；Registry跨层/签名问题；规格归一已做 |
| `data/prepare.py` | 83 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `data/profile.py` | 173 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `data/resume.py` | 67 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `data/runtime.py` | 193 | 已登记 | R8：0–6待跨层合同；sampler/RNG恢复保留 |
| `make/__init__.py` | 63 | init豁免、导出通过 | 0；只导出，不以删公开导出名充当净瘦身 |
| `make/capacity.py` | 293 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `make/estimate.py` | 175 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `make/expand.py` | 135 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `make/hardware.py` | 86 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `make/recipe.py` | 234 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `make/schedule.py` | 365 | 已登记 | R11/P3：0；生命周期门不可删，跨文件分责净0 |
| `make/scripts.py` | 208 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `make/write.py` | 76 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `model/__init__.py` | 4 | init豁免、导出通过 | 0；只导出，不以删公开导出名充当净瘦身 |
| `model/config.py` | 24 | 已登记 | R4：0；共享codec已实施，独立类字段保留 |
| `model/custom_torch.py` | 232 | 已登记 | R6：0；模块注册/原地计算/state键顺序风险 |
| `model/factory.py` | 272 | 已登记 | R0实删1；R5/R6进一步0安全，构造/state身份风险 |
| `model/input_norm.py` | 55 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `system/__init__.py` | 14 | init豁免、导出通过 | 0；只导出，不以删公开导出名充当净瘦身 |
| `system/config.py` | 28 | 已登记 | R4：0；同上，独立类身份保留 |
| `system/factory.py` | 176 | 已登记 | R7/P1：需恢复/磁盘合同批准；净0–70条件预算 |
| `system/logger.py` | 139 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `system/origin.py` | 219 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `system/runtime.py` | 84 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `system/torchvision_load.py` | 35 | 已登记 | 0；导出入口/独立职责已登记，未证实可安全删除的重复实现 |
| `system/transfers.py` | 361 | 已登记 | R12/P3：0；资源/线程/fork/延迟退出风险，分责待批 |

### （七）本次交付与等批状态

本次生产实删1行，structure79文件/8904行，相对起点9132净少228行约2.50%；不把1行说成“大项已完成”，主要交付是可审议的剩余清单和合同取舍。原始AST/清单/字节核对/174节点XML在.tmp/structure-remaining，不随clone；正式依据完整保留在本文。没有新PR、没有重新合并或验收PR222、没有主分支/标签/发布/数据删除。

工作分支推送后停止等本proposal批准；在明确批准之前不实施P1/P2/P3，不继续空轮扫描或空轮record。2026-10-17 23:33 Enmao已取消15分钟自跑，automation rpipe已删除，不自动恢复；后续由监督在有具体工作时派发。当前保持行为、API、文件架构和磁盘布局不变的安全净瘦身已到头，结束主动瘦身；不把P1条件性恢复合同变更或P2/P3纯分责当已确认可删代码。待监督决定是否批准具体提案，或安排回正常修bug，不为净删数字硬删。


### （八）2026-10-18 01:11 Wrap-up决定：P1待批，不开PR

本次只将当前主题提交和本proposal推至 `codex/structure-artifacts`。P1当前选择为保留现有bundle/诊断分片/legacy读取及恢复合同；条件净收益估算不构成实施批准。所有旧数据、Run、结果与staging保持；不启动对象转换、架构/目录迁移或新实验。安全同文件瘦身已收口，下一步建议由监督安排正常正确性bug修复，或明确批准P1方向与前置调查范围；本次不自行启动任一路线。clone接手入口及可复验命令见[handover最新章节](handover.md)。
