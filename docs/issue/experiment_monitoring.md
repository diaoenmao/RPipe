# 实验监控与 logger 选型调研

独立探索 Issue：[90](https://github.com/DreamSoul-AI/RPipe/issues/90)。

2026-10-16 按最新周期性清理指示，已核对 PR92 的调研文档交付及实际 dev CI，关闭本交付 Issue。本文仍是初步调研与条件性建议；后续选型、迁移和实现未获自动授权，需要新的具体合同与任务。

## 一、结论与状态

记录日期2026-10-16。**保留当前 logger/tracker，不立即迁回 legacy-dayan，不引入新依赖或服务。** TensorBoard 仍可作为本地曲线/媒体的候选展示工具；是否“够用”取决于是否需要跨Run查询、团队协作、制品目录或自动硬件遥测，不能只按流行度判断。迁回旧logger不是恢复一个无状态适配器：旧实现还持有指标聚合、计数和恢复状态，直接替换会影响已有合同。

本次仅只读源码和官方文档；没有安装/运行任何logger、起服务、联网同步指标、开通付费资源、采样GPU或做速度比较。下述是能力比较和建议，不是部署验收或最终选型。

## 二、实际读取的本地实现

| **实现** | **已核对职责** | **迁移影响** |
| --- | --- | --- |
| RPipe [system.Logger](../../src/rpipe/structure/system/logger.py) | stdout与assets/logs/run.log逐行追加，含时间/level/Run/event；`report`从tracker读均值/last；state只记录path | logs读取、异常原文、追加保全属于现行磁盘合同，不能用dashboard替代权威状态 |
| RPipe [AlgorithmTracker](../../src/rpipe/structure/algorithm/tracker.py) | 带权统计、分段展示、JSONL曲线、原子状态文件；记录optimizer_step/epoch与JSONL offset用于恢复 | 改成另一套均值/横轴/恢复计数可能改变科学含义；展示输出必须基于同一记录 |
| legacy-dayan 库 `_logging/logger.py` | Python logging单例与硬件/config/summary格式化，StreamHandler | 它不是TensorBoard曲线logger；硬件观察与指标聚合分开评估 |
| legacy-dayan `examples/metric/logger.py`（benchmarks同类） | SummaryWriter可选；mean/counter/history/iterator、metric状态及合并；add_scalar/add_text；带可选profiler配置 | iterator按每指标write计数，与optimizer_step不等价；不能整体替换当前tracker/恢复或开启profiler |

只读 checkout `DreamSoul-AI/dayan` 的固定快照 `a7a1b4072df22c5d1e2ce5cf9aa646709b4c60d6`，具体范围为[库logger](https://github.com/DreamSoul-AI/dayan/blob/a7a1b4072df22c5d1e2ce5cf9aa646709b4c60d6/src/legacy-dayan/src/dayan/_logging/logger.py)、[example logger](https://github.com/DreamSoul-AI/dayan/blob/a7a1b4072df22c5d1e2ce5cf9aa646709b4c60d6/src/legacy-dayan/examples/metric/logger.py)及benchmarks同名文件。工作区原字节与Git blob经换行归一核对一致，SHA存本机.tmp/research-docs/SOURCES.json（不随clone）；未改下游源码、状态或实验。较早RPipe归档metric/logger.py也使用SummaryWriter，但不是此次legacy-dayan文件的代称。

## 三、候选按需求比较

| **候选** | **适合解决的问题** | **新增负担与未验证项** |
| --- | --- | --- |
| 当前文本日志+JSONL/状态 | 无外部服务的故障解释、状态读出和正式结果追溯 | 没有自动TensorBoard展示，不意味着没有监控证据 |
| TensorBoard | 本地曲线、图像等交互展示；可从event数据读取DataFrame作分析 | 需要显式writer生命周期与恢复横轴合同；不取代Run/result/provenance |
| MLflow Tracking | 多Run参数/指标查询、实验组织、制品追踪与可选自托管 | SDK、后端/制品存储、备份与服务运维；布局和写入边界必须另设计，不自动生成根目录产物 |
| W&B | 指标/媒体记录、workspace比较与团队协作；有offline模式 | 客户端/账号/同步/隐私策略，offline不等于本机完整协作服务；费用与部署未评估，未启用 |
| 迁回legacy-dayan logger | 重用熟悉的SummaryWriter与格式化习惯 | 绑定旧metric和状态/iterator，必须剥离展示职责才可评估；不回退当前恢复/统计修复 |

PyTorch官方教程展示SummaryWriter写标量/图像/图，并要求按需flush、完成后close；TensorBoard官方提供event到DataFrame的读取方式。因此对单机曲线与事后分析，它仍有对应能力。**足够支撑本项目当前需求**是待验证判断，不由这两篇教程证明。[PyTorch教程](https://docs.pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html)、[TensorBoard DataFrame API](https://www.tensorflow.org/tensorboard/dataframe_api)

MLflow将运行元数据与制品存储分开，并支持本地或自托管。当前官方不同入口对默认本地存储的描述存在版本/调用方式差异；若将来采用，必须固定版本和显式存储URI，不能依赖默认值，尤其不能违反本仓库layout。[MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)、[Backend Stores](https://mlflow.org/docs/latest/tracking/backend-stores/)

W&B官方文档说明指标/媒体可通过log记录，SDK支持offline本地记录后另行同步。本文未核验价格、托管条件、可用配额或数据驻留，不推荐自动上传现有实验。[W&B Run API](https://docs.wandb.ai/ref/python/experiments/run/)、[官方SDK的mode说明](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_init.py)

## 四、建议的下一步与准入合同

先列实际缺失的监控问题：当前哪个Run/指标/错误不能由status/logs/JSONL回答，是否必须跨机器/多人协作，再决定工具。若只缺交互曲线，可先设计从现有JSONL导出到独立展示的最小候选；**这里只建议，不新增文件、写入位置或writer**。若需要团队/制品追踪，再单独评估MLflow/W&B，不能把模板落库或跨项目shared作为其前置。

任何后续实施先明确layout内合法位置、依赖是否可选、关闭展示时零额外产物/外部请求、记录失败是否影响任务状态。正确性回归应覆盖：sample-weighted均值不变；同一optimizer_step/epoch横轴；恢复/回滚不把旧失败尾段计入有效曲线；writer缺依赖/拒写/flush/close失败不覆盖正式result和peer；每Run隔离、单写者、无泄漏凭据。采用模拟writer和合成记录，不跑训练/GPU或性能比较。工程缺陷另开Issue与红测试，不能将本次选型探索当已复现bug。

## 五、阅读边界

检索/阅读日期2026-10-16。实际阅读上列官方教程/能力文档及明确源码文件；未开展产品全面审计、运行集成、网络服务/权限测试或历史logger科学等价复验。最初PyTorch旧URL跳转及W&B部分guides读取失败，改用可读官方教程/API/SDK入口；不把读取失败当产品不支持。文中推荐为条件性推断，实施仍需具体需求与合同测试。
