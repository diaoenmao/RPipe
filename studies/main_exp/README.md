# main_exp

历史 main `4ccb28d` 配方的四 seed、80000-step 曲线复现。目标、配方和门限见 [计划](docs/PLAN.md)，原图估读与固定门见 [TARGET.md](docs/TARGET.md)。

本 Study 使用库调度、六阶段链与来源冻结，recipe 提供固定来源适配，prepare 和 process 提供研究检查与终验。本轮固定发布 `v0.3.0` / `804b3823`，version 为 `main-exp-v030-20261013-final`；完整八格 600-step 前置门通过，32 条连续 80000-step train、32 条自身 best eval、完整性及固定曲线终验均通过。当前状态、资源与完成边界见 [本轮执行记录](docs/v030_execution.md) 和 [实测报告](docs/STUDY_REPORT.md)。

当前轮的进程中断与同源码接续已完成，真实Run身份和验证见[接续方案](docs/v030_recovery.md)及[64条审计](docs/v030_long_audit.json)。该接续index保留原30条来源及两条新训练；直接remake原声明会重新选择中断的旧ID。后续独立新轮按下方执行顺序使用新version。

## 一、文件

历史建构实现放在 `structure/`，研究检查放在 `flow/`，根 `recipe.py` 保留注册、安全门和显式命令。库加载 `flow/<phase>/`，两柱 Python 源码自动进入 provenance；手动速度校准和空阶段目录不会自动执行。换设备或重新实测须使用新 version、fresh make，并通过同设备八格前置门。

| **文件** | **作用** |
|---|---|
| `study.yaml`、`experiment_config.yaml` | 64 个 Run 的声明；`recipe`、`freeze`、`provenance.include` 见 [Study 指南](../README.md) |
| `recipe.py` | `register(ctx)`：调用 structure 注册 `historical_4ccb28d` 数据与模型，设置确定性环境，拒绝续跑中断的 train，检查 preflight。`python -B recipe.py preflight` 保留显式 600-step 原代码对照入口，不在导入或注册时训练 |
| `flow/execute/speed.py` | 由 `recipe.py calibrate` 调用的历史 / 当前链同条件阶段计时，独立测量 ID 与数值校验；手动校准，不自动进入阶段链，见 [校准方案](docs/speed_calibration.md) |
| `flow/prepare_data.py` | 下载或复用 MNIST / CIFAR10 原始文件，逐文件核对 [EXPECTED_DATA.json](docs/EXPECTED_DATA.json)，写入项目级 `shared/data/` |
| `flow/process/curves.py` | 终验：四 seed 均值曲线对原图估读的固定门，输出 `docs/COMPARISON.json` 与图；同一实现支持手动完整 / partial 读取，不再保留根目录转调脚本 |
| `flow/prepare/run.py` | 库 prepare 后核对实际 Data/Model 的配方来源与实际运行条件；建构前注册和安全门仍在 recipe |
| `flow/process/run.py` | 库 Study 聚合后调用 `curves.py` 执行曲线门；每条 Run 的 process 不重复执行终验。手动读取直接调用 `curves.py` |
| `flow/<phase>/__init__.py` | 仅导出阶段函数，不写实现；没有额外检查的阶段复用库默认实现，不填写空 hook |
| `structure/` | 注册编排调用现有 Registry；`api/` 保留固定 Git 归档与观测支持，`data/` 和 `model/` 分别提供历史建构适配。其余层暂留空，无新训练循环 |
| `runtime-requirements.txt` | 归档代码需要的固定依赖；在隔离环境准备，另核对归档导入所需的 HF 依赖 |
| `docs/reference/` | 两张原图，逐字节保留 |

## 二、命令

原始输入默认放在项目根 `shared/data/`，与其他 Study 共用。已有本机缓存先运行 `python -m rpipe shared studies/main_exp --migrate`，按 hash 核对并保留旧路径引用，原正式 Run / 来源清单不改写。prepare_data、preflight 和容量 / 精度工具解析同一共享根；独立复验仍需新 version 与 fresh make。

processed按固定4ccb28d格式、raw hash与依赖版本缓存在`shared/data/derived/`，与其他归档格式分开；对象、transform和sampler每次重新构造，不共享训练状态。缓存完整发布后才读取，失败或损坏保留诊断。

本轮实测后采用 `system.config.vram_bytes=4.5 GiB`、原生自动两并发及 safety=0.5，推导和资源保护见 [资源选择](docs/resource_choice_v030.json)。预算只针对本轮设备和配方，不能直接套用到 main_probe 或其他电脑。模型、数据增强、CPU 2 线程、workers0、batch250、完整 test10000、评测 / 保存频率和总预算不变。

以下为后续独立复跑的执行顺序。使用隔离环境安装仓库及归档依赖（本轮实际版本见 [环境清单](docs/environment_v030_20261013.json)），先验证 CUDA，核对已有 `.tmp/runtime` 不会遮蔽所选环境；设置新 version 后执行命令。CUBLAS 环境必须在 CUDA 初始化前设置。命令在仓库根运行，需要完整 clone 中的历史提交 `4ccb28d`；旧前置报告不能直接用于新来源 launch。

```powershell
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
python -B studies/main_exp/flow/prepare_data.py
python -m rpipe make studies/main_exp
python -B studies/main_exp/recipe.py preflight
python -m rpipe launch studies/main_exp --round auto --num-gpus 1
python -m rpipe status studies/main_exp
python -m rpipe report studies/main_exp
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
from rpipe.structure.artifact.shared import shared_paths
from runpy import run_path
study = Path('studies/main_exp').resolve()
recipe = run_path(str(study / 'recipe.py'))
recipe['bootstrap']()
recipe['preflight'](
    shared_paths(study)['data'], study / 'shared/validation/<new-validation-id>',
    steps=200, seed=0, device='cuda', combinations=[('MNIST', 'linear')])
```
