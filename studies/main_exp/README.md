# main_exp

历史 main `4ccb28d` 配方的四 seed、80000-step 曲线复现。目标、配方和门限见 [计划](docs/PLAN.md)，原图估读与固定门见 [TARGET.md](docs/TARGET.md)。

2026-10-10 起本 Study 按 [flow.md](../../docs/code/flow.md) §14 改为只写声明：调度、阶段链、来源清单都用库。上一轮（2026-10-04，原名 `main_historical`）的结果与证据已从当前树删除，需要时查看提交 [`18cd76c`](https://github.com/diaoenmao/RPipe/tree/18cd76c/studies/main_historical)。本轮尚未运行。

## 一、文件

| **文件** | **作用** |
|---|---|
| `study.yaml`、`experiment_config.yaml` | 64 个 Run 的声明；`recipe`、`freeze`、`provenance.include` 见 [Study 指南](../README.md) |
| `recipe.py` | `register(ctx)`：注册 `historical_4ccb28d` 数据与模型，设置确定性环境，拒绝续跑中断的 train，检查 preflight。`python -B recipe.py preflight` 跑同设备 600-step 原代码对照 |
| `prepare_data.py` | 下载或复用 MNIST / CIFAR10 原始文件，逐文件核对 [EXPECTED_DATA.json](docs/EXPECTED_DATA.json)，写入 `shared/data/` |
| `compare.py` | 终验：四 seed 均值曲线对原图估读的固定门，输出 `docs/COMPARISON.json` 与图 |
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

1. `make` 展开 64 个 Run，写 `provenance.json`。`freeze: true`，之后改任何源码、声明或计划，`launch` 都会拒绝，重新 make 才接受。
2. `preflight` 写 `docs/PREFLIGHT.json`；未通过时每条 Run 在 prepare 阶段失败。
3. `launch` 先 train 后 eval，eval 只在同 seed 的 train 成功后运行。
4. `compare.py --partial` 只用于中途查看，不应用终验。
