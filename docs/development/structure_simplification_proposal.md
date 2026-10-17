# Structure 精简与粒度调整 proposal

状态：Enmao 2026-10-17 18:30 批准的硬件职责分离已按先行扫描实施，经 PR219 合 dev 并双侧实际 CI 验收，证据见 record§293；19:25新授权已恢复每15分钟续跑，派生改为新可用里程碑版本后周期同步三个副本，本批未同步，见 record§295。

当前计划更新（2026-10-17 22:19）：全包盘点与批次排序见§六；B1同文件归并已本机验收、净少22行（core2800+1既有skip/CPU Flow525），待合批PR实际CI，B2 checkpoint合同变更仅提案待批，B3职责热点待具体方案。历史基线与已完成硬件方案保留供追溯。

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
