# main → dev 迁移与合并准备

> 本文保存2026-10-04发布前的迁移审查快照，引用的 main/dev SHA 和“尚未提交/合并”适用于该时点。随后阶段版本的交付范围与分支约定见 [v0.3.0](releases/v0.3.0.md)；实际 Release、CI 和 refs 以远端为准。原数值报告继续绑定实测源码版本。

## 一、摘要

2026-10-04。**可以着手迁移和合并准备。** 当前 dev 已实现 Study / Experiment / Run、可安装的 `rpipe` 包、原生 train/eval、调度、结果聚合和故障恢复；本机现代 main 数值对照与历史 README 曲线验收分别完成。此次替换涉及命令、Python 接口、配置和磁盘格式的大范围变化，需要作为一次明确的使用方式迁移发布。

本文依据当前 Git refs、源码、配置、CI 工作流及实际报告重新审查。没有实际 merge、push 或删除旧产物，也没有修改冻结的科学证据。本轮统一 CPU 执行门为 288 passed / 3 deselected；wheel 与 sdist 分别安装后通过 9 项公开 CLI 命令验收。新增 CI 尚未在远端执行，详见第六、七节。

| **事项** | **当前结论** |
| --- | --- |
| Git 分支关系 | main 是 dev 的祖先；已知 refs 可以快进，无 main 独有提交 |
| 新执行链与数值依据 | 有明确范围的本机实测和独立审计，见第五节 |
| 旧命令与 Python import | 新入口不兼容旧调用，需要迁移 |
| 旧 checkpoint / Stats / 结果 | 保留原值与原读取环境，不能直接当新 Run 恢复 |
| 安装与 CI | 本机 wheel / sdist 功能通过；复用现有依赖。新增干净安装和跨平台 CI 待远端执行 |
| 实际合并 | 尚未执行；发布前完成变更整理、迁移说明和最终门核对 |

## 二、固定分支关系与交付范围

根任务重新执行 `git fetch origin` 后，本次独立读取的 refs 为：

| **引用** | **提交** |
| --- | --- |
| main / origin/main | `98648f3a5c7db7dccf3ca806410d5b6fdee9484c` |
| dev / origin/dev / HEAD | `8bccbac321d4c3ac1ea9892a5e774c114e0298c6` |
| merge-base(origin/main, dev) | `98648f3a5c7db7dccf3ca806410d5b6fdee9484c` |
| origin/main…dev 左右提交数 | `0 / 61` |

本次对比有 314 个文件变化，包含删除旧 `src/train_model.py`、`test_model.py`、`make.py`、`process.py`、`config.py` 和 `dataset/model/module/metric` 包，新增 `src/rpipe/`、Study、测试与打包配置。没有分支分叉冲突，不等于这些公共使用方式自动兼容。

合并前仍需检查最新 refs，避免用本次快照替代实际合并时的远端状态。旧 main 提交仍是迁移与回退的固定依据；可在发布记录中固定旧提交，或按仓库发布流程保留可识别的旧版本引用。

本机新增的 `studies/main_historical/`、`studies/main_reproduction/current_device.py` 及本机正式报告在本次审查开始时尚未提交。只合并已推送的 dev 不会自动携带这些文件。应将可复跑入口、声明、报告、图和来源清单纳入待审变更，继续按 [LAYOUT.md](LAYOUT.md) 排除数据、Run 资产、生成计划、归档导出和 `.tmp/`。冻结清单对应原实测版本，不能为了清理目录重写清单或旧报告里的来源 SHA。

## 三、旧入口与新公共接口

### （一）命令映射

旧 main 从源码目录运行脚本，并依赖工作目录中的 `config.yml`。新版本先在仓库根安装包，以 Study 路径声明任务；安装后可使用 `rpipe` 或 `python -m rpipe`。

```bash
python -m pip install -e ".[dev]"
```

| **旧 main 入口** | **新入口** | **迁移说明** |
| --- | --- | --- |
| `bash make.sh` / `python make.py --mode base` | `python -m rpipe make studies/<name>` | 先定义 `study.yaml`、`experiment_config.yaml`，展开因素和 seed；调度落 `scripts/jobs.json` |
| `python train_model.py --control_name MNIST_linear` | `python -m rpipe run studies/<name>`，或 `launch --mode train` | `data.name`、`model.name` 和其他字段由 YAML 声明，不解析旧 `control_name` |
| `python test_model.py --control_name CIFAR10_resnet18` | `python -m rpipe launch studies/<name> --mode eval` | 声明独立 eval Run，默认加载同因素同 seed 的 train best |
| `python make_dataset.py` | `python -m rpipe data studies/<name>` | 写入新的 `stats.yaml` profile；不是旧 Stats 对象的直接转换，精度边界见第四节 |
| `python process.py` | `python -m rpipe process studies/<name>`，随后 `report` | 生成 `process.json`、曲线和 `docs/NUMBERS.md`；没有旧 Excel 的直接兼容出口 |
| 按 tag 查 `output/` | `python -m rpipe status studies/<name>` / `logs` | 通过当前 index 定位 Run 和日志 |

`run --skip-launch` 只展开配置，适合迁移后先核对预算和因素；`run-one` 执行一个明确 Run ID。`launch` 可复用既有 jobs，声明改变后应核对是否需要 `--remake`。`--include-done` 不清 checkpoint，也不保证从头训练；独立新实测应使用新的 `fixed.version` 和 Run ID。详见 [STUDY_GUIDE.md](STUDY_GUIDE.md)。

历史复现 Study 的自定义数据/模型注册器使用自己的 [run.py](../studies/main_historical/run.py) 入口。Registry 注册只存在于当前 Python 进程，普通 CLI 的 `run-one` 子进程不会继承父进程临时注册。迁移自定义旧模型时，应在实际 worker 启动入口执行注册；不能只在启动调度的父进程调用 `register`。

### （二）Python 调用与数据格式

| **旧接口** | **新接口与契约** |
| --- | --- |
| `from config import cfg`，修改全局字典 | `RunConfig` / `Control` 与每次 Flow 的配置；四层分别为 data、model、algorithm、system |
| `from dataset import make_dataset, make_data_loader` | `DataFactory` / `DataRegistry` / `DataConfig`；`Data.iter_batches` 的原生通路给出 `(images, targets)`，不直接接收旧字典 batch |
| `model.make_model`、`model(**input)` | `ModelFactory` / `ModelRegistry` 返回运行对象；`model.module(images)` 返回 logits，Loss 与指标由 algorithm 计算 |
| 旧 `Base` 返回 `{'pred', 'loss'}` | 新模块不会保持这份返回字典，外部推理代码需要更新 |
| 旧 `metric.Logger` 同时处理指标和 TensorBoard | `AlgorithmTracker` 维护数字，system `Logger` 输出文本；原生入口写 JSONL/JSON 和 `run.log`，不保持旧 TensorBoard 事件目录契约 |
| 向旧 `model/` 包加实现并 wildcard import | 显式 Registry 注册 `(name, source)`；自定义 batch、归一化和 loss 也要按新算法接口适配 |

新包只安装 `rpipe*`。旧 `config`、`dataset`、`model`、`metric`、`module` 不是保留的兼容模块。新 API 与扩展边界见 [CODE_STRUCTURE.md](CODE_STRUCTURE.md)、[structure.md](code_structure/structure.md)。

## 四、配置与产物迁移

### （一）配置字段映射

旧 `config.yml`、`process_control()` 和 `output/config/<control_name>.yml` 共同组成实际配置。迁移应读取执行后的有效 cfg，而不是只复制旧 YAML 中的少量基础字段。新的配置布局示例见 [main_reproduction/experiment_config.yaml](../studies/main_reproduction/experiment_config.yaml) 与 [study.yaml](../studies/main_reproduction/study.yaml)；该声明本身不代替本机条件对照脚本。

| **旧有效字段** | **新字段** | **需要保持的语义** |
| --- | --- | --- |
| `control.data_name` / `data_name` | `data.name`，配合 `data.source: torch` | 明确真实数据源，避免误用 stub |
| `control.model_name` / `model_name` | `model.name`，配合 `model.source: custom_torch` | 主体结构与超参数在 `model.config`，不是旧完整 model cfg 的直接粘贴 |
| `init_seed` + `num_experiments` | Study `seeds: [...]` | 明确列出每次 Run 的 seed；seed 不放 axes |
| `batch_size` / optimizer 的 train/test batch 字典 | `data.config.batch_size` + `test_batch_ratio` | 例如 train250 / test1000 对应 250 / 4；完整 test 仍另设 eval 限制 |
| `pin_memory`、`num_workers` | `data.config.pin_memory`、`num_workers` | worker 数变化也会改变随机性与性能条件 |
| `num_steps` / `step_period` | `algorithm.num_steps` / `step_period` | `num_steps` 是 optimizer 更新数，step_period 为梯度累积批数 |
| `num_epochs` | `algorithm.num_epochs` | 同时设置时会由 epoch cardinality 推导 num_steps；迁移 step 配方应清除继承的 num_epochs |
| `eval_period`、`save_period` | `algorithm.eval_period`、`checkpoint_period` | 两者按 `progress_unit: step` 或 `epoch` 计数；旧 step 配方明确写 step |
| `eval.num_steps` | `algorithm.eval_num_steps` | `-1` 或缺省为完整 test；正数是 test batch 数，不是样本数 |
| optimizer/scheduler 参数 | `algorithm.optimizer`、`scheduler`、`lr`、`momentum`、`weight_decay`、`nesterov` 等 | 原生支持相关名称 alias；不要继承 template 的额外默认值 |
| `save_checkpoint`、save period | `algorithm.checkpoint`、`checkpoint_period`、`save_best` | latest 整包与分件镜像；percent 额外保留进度快照，含义不同 |
| `metric.best_split`、`best_metric_name` | `algorithm.best_split`、`best_metric`、可选 `best_mode` | 新默认 Accuracy/max；当前 main 是 Loss/min，需要显式写 Loss |
| `resume_mode` | `algorithm.resume` / `resume_from` | 缺省 train=latest、eval=best；只给训练设置 false，避免固定 false 同时覆盖 eval |
| `device`、脚本 cuDNN 设置 | `system.device`、`deterministic`、`cudnn_benchmark` 等 | 数值对照还需明确进程启动环境与 CUBLAS 设置 |
| `tag` / `control_name` | 因素 + seed + 可选 `fixed.version` 的内容 Run ID | 新 ID 不是旧 tag 的重命名；description 不进入 ID，version 进入 ID |

复制 `_template` 后尤其注意其 epoch 预算。将 fixed 的 `num_steps` 改为 60 但保留基底 `num_epochs: 20`，不会得到明确的旧 60-step 配方；需清除或设为 null，再检查展开配置。旧 `profile` 的 Torch profiler 开关也没有同名的直接迁移承诺。

### （二）checkpoint 与模型权重

旧 main 的 `output/exp/<tag>/checkpoint/` 和 `best/` 保存无扩展名的 `cfg`、`model`、`optimizer`（Torch）与 `scheduler`、`logger`（pickle）。新 [System.load_checkpoint](../src/rpipe/structure/system/factory.py) 读取 `.pt` 整包、`payload.pt` 或 `meta.json/.pt`、`model/optimizer/scheduler.pt`、`tracker/logger.json` 分件；不会读取那组旧无扩展名文件。

旧 main `Base` 注册网络为 `model`，权重前缀为 `model.*`；新 `InputNorm` 注册网络为 `net`，前缀为 `net.*`。这是固定 main `98648f3` 的情况；不要与历史 `4ccb28d` 的模型包装命名混用。旧字典 batch、包装层、归一化配置、模型结构及 optimizer 参数顺序也要核对，单纯改文件名或替换 key 前缀不足以宣布完整恢复兼容。

**没有已验收的通用旧 checkpoint → 新 Run 转换器。** 如需延续旧任务，应先在旧代码环境读取其存档，显式转换和验证模型、统计 profile、optimizer、scheduler、进度与指标状态，并使用新的 version 保存。仅做推理权重导入，也应单独核对 logits / Loss / 正确样本数。原目录保持不变，不能把新 bundle 覆盖进旧目录来掩盖格式差异。

当前 native resume 不保存完整 RNG / sampler 位置，[Data.rebind_train_steps](../src/rpipe/structure/data/factory.py) 会重新从 seed 取剩余预算对应的采样前缀。因此现有 checkpoint 恢复与故障保护不承诺和无中断训练逐位相同；本次历史 80k 验收全部连续，无训练 resume。

### （三）数据、Stats 与结果保留

| **旧产物** | **新位置 / 保留方法** |
| --- | --- |
| `data/<dataset>/raw`、旧 processed 缓存 | 新数据归属 Study `shared/data/<vision_root>/`；torchvision 的布局与旧自定义 dataset 不同，按目标 reader 复制原始文件后核对 SHA，不直接搬旧 processed |
| `output/stats/<dataset>` 的 `module.stats.Stats` Torch 对象 | 新 reader 接收 `shared/data/<vision_root>/stats.yaml` 的 mean/std；需在旧类可导入的环境读取并显式转成普通数值，保留来源与精度 |
| `output/exp/<tag>/...` | 保留旧环境与原目录；新 Run 落 `studies/<name>/runs/<id>/` |
| `output/result/<tag>`、processed_result、Excel | 保留原文件和旧 `process.py` 的读取方式；新 `process.json` / `NUMBERS.md` 不会自动导入或重写旧 result |
| 旧学习曲线与论文引用 | 保留原图片、原 source ref 和统计口径，另列新的结果，避免覆盖后失去依据 |

普通 `rpipe data` 的 profile 采用 batch256、sum/sumsq population variance，并把数值舍入到六位小数。原 main `Stats(dim=1)` 使用逐批 mean/std 更新；本机现代严格对照特意在完整 train、顺序 batch250 下重新执行该原 Stats，并保存全精度 profile。两种生成方式不能声称逐位相同。新 Factory 读取已有 profile；没有 profile 时使用常量 fallback。要复现已声明的数值条件，应复用对应准备入口和精度，而不是只运行普通 `rpipe data`。

旧结果的保留方式是固定源提交与有效 cfg、保留数据/存档/统计对象的 SHA 和可读取环境、提供独立导出结果。需要继续旧入口时，可使用独立的旧提交 checkout 或导出的旧源码目录，不把旧全局包重新混入新 `src/rpipe/`。清理 `.tmp`、`data` 或 `output` 前先区分可重建缓存与唯一原始证据；本迁移审查没有授权或执行这些目录的删除。

### （四）统计口径

旧 `process.py` 使用 `np.mean`、`np.std`，std 为 population std，ddof=0。新通用 [aggregate.py](../src/rpipe/flow/process/aggregate.py) 的 `summarize_numbers` 与 `summarize_histories` 使用 `statistics.stdev`，是 sample std；同一四 seed 数据的 std 也会不同。通用 `process.json` / `NUMBERS.md` 的 std 不能直接替代旧图口径。

历史 [compare.py](../studies/main_historical/compare.py) 则明确按 optimizer step 对齐四 seed，用 population std；其末 50 点时间 std 是均值曲线随时间的波动，与每个 step 的跨 seed std 是不同统计量。原图阴影 std 没有数字化验收。本轮的通用聚合表与专用科学验收分别保存，没有为了接近图片改变通用 std 定义。

## 五、已有真实验证及适用边界

| **证据** | **实测结论** | **不覆盖的内容** |
| --- | --- | --- |
| [现代 main 本机结果](../studies/main_reproduction/docs/CURRENT_DEVICE_RESULT.md) / [CPU 重载审计](../studies/main_reproduction/docs/CURRENT_DEVICE_CPU_RELOAD_AUDIT.json) | seed0、8组合、60步/eval30、step30/60共16段参数最大差0，optimizer/RNG/scheduler一致，独立全 test 相同 | 原 Stats B250 全精度 profile + deterministic 条件；不覆盖无 profile 默认常量、普通 profile 重算、benchmark=true 或旧 checkpoint 文件兼容 |
| [历史同机前缀](../studies/main_historical/docs/PREFLIGHT.json) / [CPU 核验](../studies/main_historical/docs/PREFLIGHT_VERIFICATION.json) | 8组合、seed0、200/400/600三段原码/native数值对照通过 | 只证明前缀计算桥，不证明全80k原码/native权重逐位一致 |
| [历史完整结果](../studies/main_historical/docs/STUDY_REPORT.md) / [FINAL_RESULT.json](../studies/main_historical/docs/FINAL_RESULT.json) | 32连续80000-step train + 32 own-best eval，四seed完整；8组固定五门和两模型排序门通过 | 原图估读合同；缺原始历史日志、实际seed/环境及阴影数据；不证明所有原图点逐位一致 |
| [原图独立终审](../studies/main_historical/docs/FINAL_REFERENCE_AUDIT.md) / [执行审计](../studies/main_historical/docs/FINAL_EXECUTION_AUDIT.json) | 原图main/历史SHA一致，参考/比较绑定，冻结99 source+65计划一致、完整Run和来源核对 | 在线正式eval终态模型没有单独指纹；来源证明限定于实际resume日志、canonical best和冻结加载链 |
| 2026-10-03核心unit c1/c2执行记录 | 266 passed / 23 deselected，对应同一dev生产源；原记录在本机 `.tmp/test-results/20261003T164247Z_38a72e/` | 非全量integration/e2e，临时文件不随clone提供；本轮新CPU门另记 |

上述科学报告属于本机 Python3.13.9 / Torch2.11.0+cu130 / Windows / RTX5090 D v2 的新运行证据。旧 cu128 报告保留自身边界，不代替本机验收；混合负载时长不是算法 benchmark。数值结果可用于评估迁移的计算行为，不能据此扩称旧 CLI、磁盘文件、外部 import 或所有依赖版本都兼容。

## 六、依赖与 CI 的兼容范围

旧 requirements 采用固定版本，例如 NumPy1.23.5、Torch2.0.1、torchvision0.15.2；旧源码还导入未全部列在该文件中的 Kornia、TensorBoard、evaluate、transformers 等模块。新依赖事实源为 [pyproject.toml](../pyproject.toml)，Python≥3.10，项目版本0.3.0；[requirements.txt](../requirements.txt) 只是 `-e .[dev]` 入口。base 列出 Torch/torchvision/Kornia 等，`dev` 为 pytest，HF 的 accelerate/evaluate/transformers 放在 `hf` extra。历史归档 runner 的补充依赖另由 [runtime-requirements.txt](../studies/main_historical/runtime-requirements.txt) 管理。

因此 `pip install -r requirements.txt` 已从安装旧固定环境变成安装当前本地项目。它不是旧环境重建命令，也不是与旧 checkpoint 序列化类兼容的承诺。新声明没有把全部科学环境锁为固定版本；复跑科学结果仍须保留对应环境记录。HF extra 不表示所有下游模型、数据源、框架或版本已验收。

已推送 dev `8bccbac` 的原工作流使用 Ubuntu / Python3.10：Unit Tests 只执行 core unit c1/c2；Package Check 构建 wheel/sdist，`--no-deps` 安装 wheel 后只验轻量 import。对应 [Unit Tests 原提交运行](https://github.com/diaoenmao/RPipe/actions/runs/37125686675) 与 [Package Check 原提交运行](https://github.com/diaoenmao/RPipe/actions/runs/37125686622) 均为 success。这些远端记录不覆盖本轮尚未提交的 CI 改动。

本轮已更新两个工作流：

1. [Unit Tests](../.github/workflows/unit-tests.yml) 使用 Ubuntu / Windows × Python3.10 / 3.13，保留 core 门，另执行 c1/c2、非 external/gpu/slow 的 integration/e2e 选择。当前两个 e2e 都带 external 标签，实际新增门为 20 项本地 integration，不声称执行了外部 e2e。
2. [Package Check](../.github/workflows/package-check.yml) 使用 Ubuntu / Windows、Python3.10，隔离构建 wheel/sdist，各自创建普通新 venv、完整安装依赖，再运行 [package_smoke.py](../tests/package_smoke.py)。验收 module 和 console 入口、make/launch/process/status/logs/report，以及成功 Run 的重复 launch 跳过行为。
3. 本机对应执行通过，但新的远端矩阵尚未运行。collect-only 的退出0只表示收集完成，不计入执行通过；本机共享已有依赖的包环境也不代替远端新 venv 的依赖解析。

无需为了文档和打包整理重复已完成的 GPU 长矩阵；生产计算源如发生改动，应重新判断原数值证据的适用性。

## 七、本轮新验收记录

完整汇总见 [仓库整理报告](REPOSITORY_CLEANUP.md) 与 [机器可读结果](REPOSITORY_CLEANUP_RESULT.json)。原始本机日志在 `.tmp/repo-cleanup/` 和 `.tmp/test-results/`，不会随 Git clone 提供。

| **验收门** | **实测结果与边界** |
| --- | --- |
| CPU unit / integration 执行 | `python tests/run.py --all --cost-class c1 --cost-class c2 -- -m "not external and not gpu and not slow"`：288 passed / 3 deselected，退出0，20.14秒；排除 GPU、external、slow 及 c3/c4 |
| 统一测试子进程 | 显式传递环境与 stdout/stderr；两个真实子进程回归覆盖参数、诊断输出、环境和退出码，已包含在288项中 |
| wheel / sdist 构建 | 本机 setuptools80.9.0 / wheel0.45.1 backend，无构建隔离；wheel 的91个 Python 源文件与生产源逐字节一致，console entry 正确，无 Study/Run/临时资产 |
| 安装后公开 CLI | wheel、sdist 各安装到自己的包环境；各9命令退出0、4个 Toy/Stub Run 成功、2个 Experiment 完整，配置不变且第二次 launch 跳过成功 Run；证明安装和产物合同，不证明真实 ML 准确率 |
| 本机依赖范围 | 新包环境共享已实测的基础科学依赖，补入本地 Kornia；使用 `--no-index --no-deps --no-build-isolation`，没有全新网络依赖解析验收 |
| 新 CI 与仓库边界 | 新远端 CI 待执行；正式导航与本地证据边界统一核对，科学报告、冻结来源和旧进度快照保留 |

## 八、迁移交付顺序与合并决策

1. 整理待入库的新运行入口、声明、报告和来源清单，更新导航；保留所有原数值与执行故障证据，排除本地大产物。
2. 将本文的命令、字段、磁盘格式、Stats精度、std与resume边界写入发布说明，明确旧调用需要迁移。需要旧存档转换时，作为独立任务验收，不隐含在分支合并中。
3. 完成本轮打包、CPU执行、文档检查与新增CI对应门，记录真实退出码；必要时修复被实际暴露的问题，不为未声称支持的旧入口添加虚假兼容层。
4. 合并前再次检查远端祖先关系、工作区变更、CI和待审提交。将dev到main的变更做成可审阅的迁移发布，再执行获授权的合并；本审查未实际merge/push。

已有证据支持继续这项迁移准备。最终合并的技术条件是交付新入口及报告、明确不兼容项、补齐本轮运行和打包门，并在实际合并时核对refs；不是扩大历史图门限或重做全部GPU实验。
