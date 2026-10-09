# 历史 main 曲线复现

## 一、目标与完成状态

2026-10-04，本 Study 完成 MNIST / CIFAR10 × linear / mlp / cnn / resnet18 × seeds0–3 的 32 条连续 80000-step 训练和 32 条自身 best 独立评估。完整四 seed 曲线通过事先固定的 PNG 估读门，[整体比较](docs/COMPARISON.json)为 `complete=true / final_gates_applied=true / passed=true`。

先读 [完整报告](docs/STUDY_REPORT.md)、[计划](docs/PLAN.md) 与 [固定合同](docs/TARGET.md)。历史来源固定为 `4ccb28d0496110253e9f8e3f3df658853f07996b`，本次生产代码基线为 dev `8bccbac321d4c3ac1ea9892a5e774c114e0298c6`。参考值是 PNG 像素估读，未恢复历史原始指标、实际 seed 集合或原依赖环境；完整结果保留这层限制。

现代 main `98648f3` 的 60-step 配方另见 [本机现代对照](../main_reproduction/docs/CURRENT_DEVICE_RESULT.md)。历史长曲线通过与现代短预算数值一致分别限定范围，不把 600-step 参数桥扩称为全部 80000-step 权重逐位一致。

## 二、专用入口与依赖

本 Study 使用 [recipe.py](recipe.py) 的 `historical_4ccb28d` 数据和模型 Registry。`study.yaml` 写了 `recipe: recipe.py`，通用 `rpipe launch` 的每个子进程在 Flow prepare 中调用 `register(ctx)`，见 [flow.md](../../docs/code/flow.md) §14。`register` 还负责本 Study 的三条约束：设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 与2线程；train 已有 checkpoint 时拒绝续跑；`docs/PREFLIGHT.json` 未通过或 Torch 版本变化时拒绝运行。`freeze: true` 让 make 之后源码、声明或计划有任何变化时拒绝 launch。

运行需要可安装的 RPipe、本机可用的 Torch / torchvision / NumPy / PyYAML，以及 [runtime-requirements.txt](runtime-requirements.txt) 列出的隔离依赖。完整本次版本见 [ENVIRONMENT.json](docs/ENVIRONMENT.json)。归档源码按 Git blob 原字节导出，需要本地 Git 对象 `4ccb28d`；原始数据由 [prepare_data.py](prepare_data.py) 下载或复用缓存，逐文件核对 [历史数据清单](../main_reproduction/docs/HISTORICAL_BRIDGE_DATA_MANIFEST.json)，写入本 Study 的 `shared/data/`。

独立新环境应先让 pip 正常解析基础包及 TensorBoard / Evaluate 的传递依赖，再用隔离 runtime 覆盖本次固定版本。`--no-deps` 只适合依赖已经齐全时补入固定包，不是全新环境的依赖安装方案。应使用完整 clone；浅 clone 需要先补齐固定提交的 Git 历史。

在尚未保存既有运行证据的独立 clone 或工作树中，准备与执行顺序为：

```powershell
python -m pip install -e .
python -m pip install tensorboard==2.21.0 evaluate==0.4.6
python -m pip install --target .tmp/runtime --no-deps -r studies/main_historical/runtime-requirements.txt
python -B studies/main_historical/prepare_data.py
python -m rpipe make studies/main_historical
python -B studies/main_historical/recipe.py preflight
python -B studies/main_historical/verify_preflight.py
python -m rpipe launch studies/main_historical
python -m rpipe report studies/main_historical
python -B studies/main_historical/compare.py
```

`make` 展开 64 个 Run，并在 Study 根写 `provenance.json`（源码、声明、recipe、计划哈希与环境）；`recipe.py preflight` 在新设备执行原代码 / 当前链 600-step 八格 CUDA 对照，采样和 scheduler 总预算仍为 80000。`verify_preflight` 只用 CPU 重载探针。`launch` 先 train 后 eval，失败格子按通用规则重试，eval 只在 sibling train 成功后运行；同组并行数默认按显存装箱，原入口固定为每组4条、resnet18 每组2条，需要时用 `--round N` 指定。`compare.py` 是本 Study 对原图估读的终验门，不是通用 Run 对比。准备、调度与比较会写本轮清单和报告；应使用独立目录保留此前实测快照。

本机既有 64 个 Run 已成功，Run ID 在改接后不变。2026-10-10 起专用 `run.py` 已删除，`docs/` 中的 `SOURCE_MANIFEST.json`、`PREFLIGHT.json` 等记录的是验收时点的源码哈希，与当前文件不再逐字节一致。需要按原清单严格核验既有 Run 时，在保留的实测源码快照 `b95873f` 中执行当时的命令：

```powershell
python -B studies/main_historical/run.py status
python -B studies/main_historical/verify_group.py --data CIFAR10 --model resnet18 --with-eval --output .tmp/independent-historical-audit
```

当前树中查看状态用 `python -m rpipe status studies/main_historical`。

`verify_group.py` 不启动训练或 GPU，只读取已完成 Run；额外 `--replay` 仅支持 linear。完整比较默认要求全部 32 train + 32 eval 和同点四 seed；`compare.py --partial` 只供未完成阶段查看，不应用终验。已有连续长训中断后保留证据，另用新 version 从头执行；专用入口不会把缺少完整 RNG / 采样位置的恢复当成连续训练。

## 三、正式证据导航

| **证据范围** | **入口** |
|---|---|
| 完整结果、图和最终判定 | [STUDY_REPORT](docs/STUDY_REPORT.md)、[FINAL_RESULT](docs/FINAL_RESULT.json)、[COMPARISON](docs/COMPARISON.json) |
| 执行前配方、原图估读与门限 | [PLAN](docs/PLAN.md)、[TARGET](docs/TARGET.md)、[REFERENCE_CURVES](docs/REFERENCE_CURVES.json) |
| 源码、数据和设备版本 | [SOURCE_MANIFEST](docs/SOURCE_MANIFEST.json)、[DATA_MANIFEST](docs/DATA_MANIFEST.json)、[ENVIRONMENT](docs/ENVIRONMENT.json) |
| 同机原代码 / 当前链 600-step 桥与 CPU 重载 | [PREFLIGHT](docs/PREFLIGHT.json)、[PREFLIGHT_VERIFICATION](docs/PREFLIGHT_VERIFICATION.json)、[verify_preflight.py](verify_preflight.py) |
| 八组 saved-state / history / own-best 独立审计 | [GROUP_INTEGRITY_AUDIT](docs/GROUP_INTEGRITY_AUDIT.md)、[GROUP_AUDIT_METHOD](docs/GROUP_AUDIT_METHOD.md)、[verify_group.py](verify_group.py) |
| 全部 64 条正式运行与真实执行来源 | [FINAL_EXECUTION_AUDIT](docs/FINAL_EXECUTION_AUDIT.md)、[EXECUTION](docs/EXECUTION.json)、[COMPLETION_AUDIT](docs/COMPLETION_AUDIT.json) |
| 原图合同与比较输出独立终审 | [FINAL_REFERENCE_AUDIT](docs/FINAL_REFERENCE_AUDIT.md) |
| 提前排班、guard 与容量的历史观察 | [CONCURRENCY_AUDIT](docs/CONCURRENCY_AUDIT.md)、[GPU_CAPACITY](docs/GPU_CAPACITY.md)、[WAVE1_COMPLETION](docs/WAVE1_COMPLETION.json) |
| 原报告日期快照和早期曲线 | [PROGRESS_REPORT_20261004](docs/PROGRESS_REPORT_20261004.md)、[EARLY_COMPARISON](docs/EARLY_COMPARISON.md)、[ACTIVE_SESSION_PROGRESS_SNAPSHOT](docs/ACTIVE_SESSION_PROGRESS_SNAPSHOT.json) |
| CIFAR linear eval 控制器写记录失败与恢复 | [组报告](docs/CIFAR_LINEAR_RESULT.md)、[原失败](docs/EARLY_EVAL_CIFAR10_LINEAR_FAILED_CONTROLLER.json)、[原未提交记录](docs/EARLY_EVAL_CIFAR10_LINEAR_UNCOMMITTED_RECORD.json)、[恢复记录](docs/EARLY_EVAL_CIFAR10_LINEAR_EXECUTION.json)、[实际 helper 源码](docs/CIFAR_LINEAR_EVAL_EXECUTION_HELPER.py) |

## 四、本机原始产物与验证边界

配置、入口、正式 JSON / Markdown、来源 SHA 和图可随 Git 提供。`runs/` 的 result、checkpoint、tracker、日志，`shared/data/`，`index.json`、`process.json`，以及 `.tmp/` 中的 Git 导出、preflight、容量诊断、运行 helper 和隔离依赖属于本机忽略产物，不随 clone 提供。CPU 审计需要这些实际文件，正式汇总不能替代缺失的原始证据。

原失败、恢复、进度快照和前三组正式 JSON 保留各自实际版本；历史快照中的 running/pending 描述只适用于记录时点。CIFAR linear 的记录控制器实际退出1，四个 eval worker 均退出0/succeeded，恢复仅补写执行记录，没有重跑 worker。

README 首页和根 `asset/` 发布本次实测的新图；旧参考 PNG 逐字节保存在 [历史图归档](docs/reference/README.md)。冻结报告中的旧 `asset/` 路径描述的是验收时点，重读旧参考时使用归档路径和固定 Git blob；训练与 `compare.py` 的门计算仍读取原固定合同。

两组 linear 有额外 CPU 完整 test 推理；六组非线性仅核对保存状态、history、正式 eval 实际加载来源与结果。正式 80000-step worker 未逐样本记录输入字节，也未保存完整 RNG / 迭代位置；同机前缀桥和日志连续性分别作为已取得证据。更多范围说明见完整报告第五节。返回 [Study 使用指南](../README.md)。
