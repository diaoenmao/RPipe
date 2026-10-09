# main_probe 复跑记录

## 一、摘要

本次按 [PLAN.md](PLAN.md) 复跑 Flow 重构后的八组合 60-step 数值探针。代码基线为 PR #25 合入 dev 的 `d73b217`；本工作分支追加独立实测 version 与记录，不调整训练配方或门限。当前完成环境和数据准备，尚无本次正式数值结论。

## 二、执行条件

| **项目** | **本次值** |
| --- | --- |
| version | main-probe-flow-20261010-01 |
| 原代码 | 98648f3a5c7db7dccf3ca806410d5b6fdee9484c |
| 设备 | NVIDIA GeForce RTX 5090 D v2，CUDA |
| Python / Torch / torchvision | 3.13.9 / 2.11.0+cu130 / 0.26.0+cu130 |
| seed / 预算 | seed0；每组合两侧 60 steps，step30 / 60 全 test，各自 best 独立 eval |
| 数据 | 复用 main_exp 的 raw；[数据清单](../../main_exp/docs/DATA_MANIFEST.json) 的 16/16 文件匹配固定 SHA-256 |
| 顺序 | 先真实单组合验收，再统一 CLI 完成八组合；不覆盖此前 Run |

## 三、2026-10-10

1. Flow 重构的 CPU 核心 286 项、集成 36 项通过；隔离伪计算验证成功、数值失败和 artifact compare 失败，不属于正式数值结果。
2. GPU 与归档依赖已核对。当前尚未运行正式组合，Run ID、日志路径和门限结果随实际执行追加。

运行产物保存在本 Study 的 runs，按仓库规则不随 Git clone 提供。完整结果、失败与结论最终写入 STUDY_REPORT.md；未完成八组合前不报告完整通过。
