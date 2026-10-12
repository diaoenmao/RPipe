# main_exp

历史 main `4ccb28d` 配方的四 seed、80000-step 曲线复现。目标、配方和门限见 [计划](docs/PLAN.md)，原图估读与固定门见 [TARGET.md](docs/TARGET.md)。

本 Study 使用库调度、六阶段链与来源冻结，recipe 提供固定来源适配，prepare 和 process 提供研究检查与终验。当前 version03 显式关闭 cuDNN / matmul TF32，须通过同设备 600-step 八格前置门后执行完整矩阵；实测进展、设备数值边界与恢复方案见 [复跑记录](docs/RERUN.md) 和 [eval_recovery.md](docs/eval_recovery.md)。

2026-10-11 version03 完整矩阵与固定曲线终验已通过，正式数字见 [NUMBERS.md](docs/NUMBERS.md)。2026-10-12 重构后只补跑 MNIST / linear 的 200-step 前置对照，参数及 Loss 差值为 0；它不替代八格 600-step 门或长矩阵。最新结果与边界见 [STUDY_REPORT.md](docs/STUDY_REPORT.md)；历史串行速度校准已记录在 [速度报告](docs/speed_calibration.md)。

## 一、文件

本轮仅重构 Study 内部代码：历史建构实现放在 `structure/`，研究检查放在 `flow/`，根 `recipe.py` 保留注册、安全门和显式命令。训练循环、精度配置与数值门限不变。

**当前验证边界：** 库已加载 `flow/<phase>/`，两柱 Python 源码自动进入 provenance；手动速度校准和空阶段目录不会自动执行。本轮完成合同回归及手动单组 200-step 前置对照，未执行新完整矩阵或正式 Flow Run。实测后清除了库 collect 入口重复实现，针对性回归通过；后续完整验收需使用新 version、重新 make 和计划规定的前置门。

| **文件** | **作用** |
|---|---|
| `study.yaml`、`experiment_config.yaml` | 64 个 Run 的声明；`recipe`、`freeze`、`provenance.include` 见 [Study 指南](../README.md) |
| `recipe.py` | `register(ctx)`：调用 structure 注册 `historical_4ccb28d` 数据与模型，设置确定性环境，拒绝续跑中断的 train，检查 preflight。`python -B recipe.py preflight` 保留显式 600-step 原代码对照入口，不在导入或注册时训练 |
| `flow/execute/speed.py` | 由 `recipe.py calibrate` 调用的历史 / 当前链同条件阶段计时，独立测量 ID 与数值校验；手动校准，不自动进入阶段链，见 [校准方案](docs/speed_calibration.md) |
| `flow/prepare_data.py` | 下载或复用 MNIST / CIFAR10 原始文件，逐文件核对 [EXPECTED_DATA.json](docs/EXPECTED_DATA.json)，写入 `shared/data/` |
| `flow/process/curves.py` | 终验：四 seed 均值曲线对原图估读的固定门，输出 `docs/COMPARISON.json` 与图；同一实现支持手动完整 / partial 读取，不再保留根目录转调脚本 |
| `flow/prepare/run.py` | 库 prepare 后核对实际 Data/Model 的配方来源与实际运行条件；建构前注册和安全门仍在 recipe |
| `flow/process/run.py` | 库 Study 聚合后调用 `curves.py` 执行曲线门；每条 Run 的 process 不重复执行终验。手动读取直接调用 `curves.py` |
| `flow/<phase>/__init__.py` | 仅导出阶段函数，不写实现；没有额外检查的阶段复用库默认实现，不填写空 hook |
| `structure/` | 注册编排调用现有 Registry；`api/` 保留固定 Git 归档与观测支持，`data/` 和 `model/` 分别提供历史建构适配。其余层暂留空，无新训练循环 |
| `runtime-requirements.txt` | 归档代码需要的隔离依赖，装到 `.tmp/runtime` |
| `docs/reference/` | 两张原图，逐字节保留 |

## 二、命令

2026-10-13候选声明为main-exp-flow-20261013-04，按本机峰值采用system.config.vram_bytes=4.5GiB；库仍只用50%空闲显存自动装箱。该预算仅针对此处已测配方 / 设备，不是通用推荐。旧version03长结果保留在版本快照，新的声明尚未长训。容量与精度诊断入口及证据见[速度报告](docs/speed_calibration.md#九2026-10-13-cifar10-容量与总体吞吐)；变更源码 / version后重新前置验收，旧PREFLIGHT不能直接launch。

以下是正式执行顺序，本轮未执行这些实验；先使用新 version，再完成准备、make 和前置门，不能沿用旧的前置报告直接 launch。命令在仓库根运行；归档源码按 Git blob 导出，需要完整 clone 中的提交 `4ccb28d`。

```powershell
python -m pip install -e .
python -m pip install tensorboard==2.21.0 evaluate==0.4.6
python -m pip install --target .tmp/runtime --no-deps -r studies/main_exp/runtime-requirements.txt
python -B studies/main_exp/flow/prepare_data.py
python -m rpipe make studies/main_exp
python -B studies/main_exp/recipe.py preflight
python -m rpipe launch studies/main_exp
python -m rpipe status studies/main_exp
python -m rpipe report studies/main_exp
python -B studies/main_exp/flow/process/curves.py
```

1. `flow.prepare_shared: false` 关闭通用 Data 预构造，raw 由 prepare_data.py 准备、专用 source 在 Run prepare 内注册。`make` 展开 64 个 Run，写 `provenance.json`。`freeze: true`，之后改任何源码、声明或计划，`launch` 都会拒绝，重新 make 才接受。
2. `preflight` 写 `docs/PREFLIGHT.json`；未通过时每条 Run 在 prepare 阶段失败。
3. `launch` 先 train 后 eval，eval 只在同 seed 的 train 成功后运行。
4. 阶段目录为 `flow/<phase>/`，launch / process 自动调用终验。两柱源码及手动工具自动纳入 provenance；`provenance.include` 只列额外参考输入。`flow/process/curves.py --partial` 可手动查看，不应用终验。完整终验失败会报错，并保留已写出的通用聚合、Run result 和终验报告。

## 三、2026-10-12 重构验证

- 非代码清理：重复的归档运行条件回归 JSON 合并为 `docs/ARCHIVE_RUNTIME_REGRESSION.json`，两次验证记录保留；删除旧 Python 缓存及已有实现目录的多余 `.gitkeep`。固定输入、正式结果、失败证据和版本快照保留；剩余 36 个 JSON 均可解析，两个版本快照的 manifest hash 校验通过。
- 根目录仅保留 `recipe.py`：删除冗余转调入口 `compare.py`，速度校准实现收进 `flow/execute/speed.py`。数值实现和历史证据不删除，工具与建构源码由库自动覆盖 provenance。
- 历史 Data / Model 建构、原代码归档与 preflight 数值门的函数 AST 与重构前一致；修正搬迁后的 Study / 仓库根路径，阶段包导入检查通过。
- 隔离副本展开 64 个配置；两柱源码全部进入 provenance，修改副本源码会被 freeze 检查发现。未改写正式 Run 或来源记录。
- 微型 CPU 输入验证现有 Factory 注册、来源与 seed 元数据，并确认错误 batch、变更 sampler horizon 和中断续训仍被拒绝。
- 接通库前的 31 项回归只证明旧加载合同。接通后针对性回归 44 passed / 2 skipped，覆盖新路径、两柱相对导入、准备时序、失败定稿与 freeze；两个跳过项因本机不能创建目录符号链接。真实 Study loader 也通过历史来源 / 实际 runtime 成功门、错误来源拒绝和 Run process scope 检查，均未训练。
- 随后将两个阶段实现从 `__init__.py` 移到 `run.py`，包入口仅导出；全库与 Study 的 13 个搬迁模块 AST 不变，15 个 Flow 包入口无业务实现。新的 Flow 针对性回归 103 passed / 2 skipped，CPU core 321 passed / 2 skipped，CPU integration 40 passed；隔离 Study 构造与真实 loader 自检再次通过，没有训练。

专用自检与 pytest 临时输出在本机 `.tmp/`，不会随 Git clone 提供。随后通过现有 `recipe.preflight` 完成单组 200-step 真实对照，见 [短实验证据](docs/flow_smoke_20261012.json)；未执行八格 600-step 门或正式矩阵，历史 version03 数字不作为重构后的验收结果。

相同预算的手动复跑可从仓库根调用已有 API，输出目录每次换新，不覆盖正式 `docs/PREFLIGHT.json`：

```python
from pathlib import Path
from runpy import run_path
study = Path('studies/main_exp').resolve()
recipe = run_path(str(study / 'recipe.py'))
recipe['bootstrap']()
recipe['preflight'](
    study / 'shared/data', study / 'shared/validation/<new-validation-id>',
    steps=200, seed=0, device='cuda', combinations=[('MNIST', 'linear')])
```
