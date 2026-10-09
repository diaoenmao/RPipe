# main_probe

固定 main `98648f3` 原代码与当前 RPipe 的 60-step 同设备数值探针。目标、配方、门限见 [计划](docs/PLAN.md)。

2026-10-10 由 `main_reproduction` 改名并清理：旧的 24 个复跑脚本和全部旧证据已从当前树删除，需要时查看提交 [`18cd76c`](https://github.com/diaoenmao/RPipe/tree/18cd76c/studies/main_reproduction)。本轮尚未运行。

## 一、文件

| **文件** | **作用** |
|---|---|
| `study.yaml`、`experiment_config.yaml` | 当前 RPipe 一侧的 8 train + 8 eval 声明 |
| `probe.py` | 原 main 与当前 RPipe 同进程对照：`prepare` 只做 CPU 核对，`run --device cuda` 跑八格 |
| `write/` | 将已经观测到的两段快照投影成两侧的 Run 目录，供库 `rpipe compare` 比较；不会执行模型 |

## 二、命令

在仓库根运行。原始数据复用 `main_exp` 的 `shared/data/`（先运行 `studies/main_exp/prepare_data.py`）。

```powershell
python -m rpipe make studies/main_probe
python -m rpipe launch studies/main_probe
python -m rpipe report studies/main_probe
python -B studies/main_probe/probe.py prepare
python -B studies/main_probe/probe.py run --workspace .tmp/main-probe-<date>-<id> --device cuda
```

`probe.py` 每次在 `.tmp/main-probe-*` 新建工作区，比较结果写在工作区的 `COMPARISON.json`；正式数字整理进 `docs/STUDY_REPORT.md`。

每个组合另写 `matrix/<data>_<model>/observed/runs/original` 与 `current`，包括 config、result、tracker history、latest/best checkpoint，并调用库对比后写 `RUN_COMPARISON.json`。只规范化已知的 `model.` / `net.` 包装前缀，保存投影来源，保留原始快照。库对比作为附加门，不能代替原有输入、RNG、正确样本计数、逐段参数和自身 best 评测的严格门。

这两个目录是既有观测的 artifact 投影，不是通过 make/launch 新运行的实验；不额外生成训练、seed 或重放结果。可以显式复核：

```powershell
python -m rpipe compare <cell>/observed/runs/original <cell>/observed/runs/current --atol 0.000001 --rtol 0.00001 --checkpoint latest --checkpoint best
```

本 Study 没有启用 `flow.study_phases`，`write/` 由 probe 的显式命令调用；加载目录或运行普通 CLI 不会自动启动原代码对照矩阵。本轮仅改代码和检查 CPU artifact 合同，不重跑探针。
