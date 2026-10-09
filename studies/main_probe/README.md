# main_probe

固定 main `98648f3` 原代码与当前 RPipe 的 60-step 同设备数值探针。目标、配方、门限见 [计划](docs/PLAN.md)。

2026-10-10 由 `main_reproduction` 改名并清理：旧的 24 个复跑脚本和全部旧证据已从当前树删除，需要时查看提交 [`18cd76c`](https://github.com/diaoenmao/RPipe/tree/18cd76c/studies/main_reproduction)。本轮尚未运行。

## 一、文件

| **文件** | **作用** |
|---|---|
| `study.yaml`、`experiment_config.yaml` | 当前 RPipe 一侧的 8 train + 8 eval 声明 |
| `probe.py` | 原 main 与当前 RPipe 同进程对照：`prepare` 只做 CPU 核对，`run --device cuda` 跑八格 |

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
