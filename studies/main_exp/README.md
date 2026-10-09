# main_exp

历史 main `4ccb28d` 配方的四 seed、80000-step 曲线复现。目标、配方和门限见 [计划](docs/PLAN.md)，原图估读与固定门见 [TARGET.md](docs/TARGET.md)。

本 Study 使用库调度、六阶段链与来源冻结，recipe 提供固定来源适配，prepare 和 process 提供研究检查与终验。当前重构版本尚未运行。先前版本的实测证据见 [`18cd76c` 的报告](https://github.com/diaoenmao/RPipe/tree/18cd76c/studies/main_historical)。

## 一、文件

| **文件** | **作用** |
|---|---|
| `study.yaml`、`experiment_config.yaml` | 64 个 Run 的声明；`recipe`、`freeze`、`provenance.include` 见 [Study 指南](../README.md) |
| `recipe.py` | `register(ctx)`：注册 `historical_4ccb28d` 数据与模型，设置确定性环境，拒绝续跑中断的 train，检查 preflight。`python -B recipe.py preflight` 跑同设备 600-step 原代码对照 |
| `prepare_data.py` | 下载或复用 MNIST / CIFAR10 原始文件，逐文件核对 [EXPECTED_DATA.json](docs/EXPECTED_DATA.json)，写入 `shared/data/` |
| `compare.py` | 终验：四 seed 均值曲线对原图估读的固定门，输出 `docs/COMPARISON.json` 与图 |
| `prepare/` | 库 prepare 后核对实际 Data/Model 的配方来源；建构前注册和安全门仍在 recipe |
| `process/` | 库 Study 聚合后执行曲线门；每条 Run 的 process 不重复执行终验。`compare.py` 是手动完整/partial 读取入口 |
| `runtime-requirements.txt` | 归档代码需要的隔离依赖，装到 `.tmp/runtime` |
| `docs/reference/` | 两张原图，逐字节保留 |

## 二、命令

在仓库根运行。归档源码按 Git blob 导出，需要完整 clone 中的提交 `4ccb28d`。

```powershell
python -m pip install -e .
python -m pip install tensorboard==2.21.0 evaluate==0.4.6
python -m pip install --target .tmp/runtime --no-deps -r studies/main_exp/runtime-requirements.txt
python -B studies/main_exp/prepare_data.py
python -m rpipe make studies/main_exp
python -B studies/main_exp/recipe.py preflight
python -m rpipe launch studies/main_exp
python -m rpipe status studies/main_exp
python -m rpipe report studies/main_exp
python -B studies/main_exp/compare.py
```

1. `flow.prepare_shared: false` 关闭通用 Data 预构造，raw 由 prepare_data.py 准备、专用 source 在 Run prepare 内注册。`make` 展开 64 个 Run，写 `provenance.json`。`freeze: true`，之后改任何源码、声明或计划，`launch` 都会拒绝，重新 make 才接受。
2. `preflight` 写 `docs/PREFLIGHT.json`；未通过时每条 Run 在 prepare 阶段失败。
3. `launch` 先 train 后 eval，eval 只在同 seed 的 train 成功后运行。
4. `flow: {study_phases: true}` 启用阶段包，源码自动进 provenance。launch / process 完成 Study 聚合后自动调用终验；不完整时只写 partial，不宣告通过。`compare.py --partial` 可手动中途查看，不应用终验。完整终验失败会报错，并保留已写出的通用聚合、Run result 和终验报告。
