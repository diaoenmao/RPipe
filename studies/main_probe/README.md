# main_probe

固定 main `98648f3` 原代码与当前 RPipe 的 60-step 同设备数值探针。目标、配方、门限见 [计划](docs/PLAN.md)。2026-10-13 version06八组合全部通过完整Study终验，实际来源和数值见[实测报告](docs/STUDY_REPORT.md)。此前version05两组子集与version03历史完整证据均保留在[版本快照](docs/versions/)，进展见[复跑记录](docs/RERUN.md)。

## 一、流程

本轮仅重构 Study 内部代码：成对 Algorithm 与计算实现放在 `structure/algorithm/`，`flow/` 保留准备、比较、判定、投影和整轮终验，`recipe.py` 只校验并注册专用 source。原计算顺序、精度和数值门限不变。

**当前验证边界：** 库加载 `flow/<phase>/`，包括构造前的 prepare.before，两柱Python源码自动冻结。version06八组真实60-step Run覆盖prepare / 成对训练 / collect / 数值判定 / 库compare / Study终验；来源在计算期间不变。它不是长曲线验收；再次实测须新version / make，不能覆盖既有Run。

[study.yaml](study.yaml) 声明八个成对 Run（两种数据集 × 四种模型，seed0）。设计上由 [flow/prepare/](flow/prepare/) 在对象构造前准备该组合，[recipe.py](recipe.py) 调用 [structure/algorithm/](structure/algorithm/) 注册成对 Algorithm。Algorithm 在同一进程先计算原版，再计算当前版并执行各自 best 独立评测，[flow/execute/](flow/execute/) 只确认一对计算已完成。当前侧直接使用库 prepare 的 Data / Model / System / Tracker，训练 checkpoint、history 和 result 使用普通 Run 合同。

`flow/prepare/`、`flow/collect/`、`flow/summarize/`、`flow/write/` 与 `flow/process/` 按现有阶段合同执行。collect 比较输入、RNG、样本计数、逐段参数与自身 best，summarize 执行数值门；write 投影两侧已有观测并调用库 compare，在库成功结果定稿前验收；没有独立探针脚本入口。

六个阶段的执行实现均在各自的 `run.py`，helper 保留在 `data.py` / `comparison.py` 等独立模块。`__init__.py` 只导出 `run`（prepare 另导出 `before`），不承载业务实现；导入接口和执行时序不变。

## 二、命令

以下命令用于后续完整复跑，本轮未执行完整 launch；先使用新 version，不能复用旧 Run 或来源清单。在仓库根运行；原始数据复用 `main_exp/shared/data/`（先按 [main_exp](../main_exp/README.md) 准备 raw）。make 只展开配置；launch 才执行准备和计算。

原归档即使跑视觉组合也会导入 `evaluate` / `datasets`；此前缺依赖的prepare失败保留。隔离目录.tmp/runtime使用evaluate0.4.6 / datasets5.1.0；2026-10-13完整MNIST / CIFAR10缓存已按历史hash核对，见[数据清单](../main_exp/docs/capacity_data_20261012.json)。异机仍须重新准备依赖和核对raw，不把本机就绪外推到目标电脑。

```powershell
python -m rpipe make studies/main_probe
python -m rpipe launch studies/main_probe
python -m rpipe report studies/main_probe
```

每个 Run 的证据在 `runs/<id>/assets/probe/`：CPU 准备、原代码归档、环境、来源清单与 `COMPARISON.json`；`matrix/<data>_<model>/observed/runs/` 保存两侧已有观测的 artifact 投影，`RUN_COMPARISON.json` 保存库 compare 结果。数值门失败写 failed Run 并保留证据。重新执行必须使用新 Run 目录，不能覆盖既有工作区。

Study process 只读当前 index，八个组合齐全、同设备与同来源且全部通过后，`PROBE_COMPARISON.json` 才报告 complete/passed；子集不是完整验收。正式数字整理进 `docs/STUDY_REPORT.md`。后续正式复跑按计划先准备数据，再运行八组合。CPU 伪计算不能替代正式数值验收。

## 三、2026-10-12 重构验证

- 搬迁后的成对计算函数与 PairedAlgorithm 的 AST 与重构前一致（来源清单的 Study 根路径随搬迁修正）；阶段包导入检查通过。
- 隔离副本展开八个配置；两柱源码全部进入 provenance，修改副本源码会被 freeze 检查发现。未改写正式 Run 或来源记录。
- 使用现有 AlgorithmFactory 验证注册、准备证据缺失时拒绝，以及 Data / Model / System / Tracker / RNG 的参数交接；成对计算被替身替换，没有执行训练。
- 接通库前的 31 项回归只证明旧加载合同。接通后针对性回归 44 passed / 2 skipped，覆盖新路径、两柱相对导入、准备时序、失败定稿与 freeze；两个跳过项因本机不能创建目录符号链接。真实 Study loader 通过 raw 缺失时拒绝、成对观测存在 / 缺失的 execute 检查及 Run process scope 检查，计算与数值验收未执行。
- 随后将六个阶段实现从 `__init__.py` 移到 `run.py`，包入口仅导出；全库与 Study 的 13 个搬迁模块 AST 不变，15 个 Flow 包入口无业务实现。新的 Flow 针对性回归 103 passed / 2 skipped，CPU core 321 passed / 2 skipped，CPU integration 40 passed；隔离 Study 构造与真实 loader 自检再次通过，没有训练。

专用自检与 pytest 临时输出在本机 `.tmp/`，不会随 Git clone 提供。随后仅执行下面两个真实 Run；历史 version03 数字不作为重构后的验收结果。

```powershell
python -B -m rpipe make studies/main_probe --round 1
python -B -m rpipe run-one studies/main_probe 075534e416d6cc8d
python -B -m rpipe run-one studies/main_probe aa502aa512c16e4b
python -B -m rpipe process studies/main_probe
python -B -m rpipe report studies/main_probe
```

这是已完成的 version05 执行记录，不要重复运行这些 ID。process exit 1 是缺少六组时的预期拒绝，两个成功 result 未丢失；复跑使用新 version，从新 index 选择对应 ID。详细证据见 [flow_smoke_20261012.json](docs/flow_smoke_20261012.json)；根 `docs/PAIR_RESULTS.json` 属于历史 version03，不是当前子集结果。
