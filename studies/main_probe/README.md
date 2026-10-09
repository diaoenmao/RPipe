# main_probe

固定 main `98648f3` 原代码与当前 RPipe 的 60-step 同设备数值探针。目标、配方、门限见 [计划](docs/PLAN.md)。本版本尚未运行正式矩阵。

## 一、流程

[study.yaml](study.yaml) 声明八个成对 Run（两种数据集 × 四种模型，seed0）。[recipe.py](recipe.py) 在库 prepare 内准备该组合并注册成对 Algorithm；[execute/](execute/) 在同一进程先计算原版，再计算当前版并执行各自 best 独立评测。当前侧直接使用库 prepare 的 Data / Model / System / Tracker，训练 checkpoint、history 和 result 使用普通 Run 合同。

[prepare/](prepare/)、[collect/](collect/)、[summarize/](summarize/)、[write/](write/) 与 [process/](process/) 接入库阶段链。write 模块投影两侧已有观测并调用库 compare，保留输入、RNG、样本计数、逐段参数与自身 best 的严格门；没有独立探针脚本入口。

## 二、命令

在仓库根运行。原始数据复用 `main_exp/shared/data/`（先按 [main_exp](../main_exp/README.md) 准备 raw）。make 只展开配置；launch 才执行准备和计算。

```powershell
python -m rpipe make studies/main_probe
python -m rpipe launch studies/main_probe
python -m rpipe report studies/main_probe
```

每个 Run 的证据在 `runs/<id>/assets/probe/`：CPU 准备、原代码归档、环境、来源清单与 `COMPARISON.json`；`matrix/<data>_<model>/observed/runs/` 保存两侧已有观测的 artifact 投影，`RUN_COMPARISON.json` 保存库 compare 结果。数值门失败写 failed Run 并保留证据。重新执行必须使用新 Run 目录，不能覆盖既有工作区。

Study process 只读当前 index，八个组合齐全、同设备与同来源且全部通过后，`PROBE_COMPARISON.json` 才报告 complete/passed；子集不是完整验收。正式数字整理进 `docs/STUDY_REPORT.md`。本轮仅做 CPU 合同验证，不重跑正式探针。
