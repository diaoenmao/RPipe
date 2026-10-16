# main_probe 复跑记录

## 一、摘要

本次按 [PLAN.md](<../../../../../docs/research/研究计划 RPipe.md#study-main-probe>) 复跑 Flow 重构后的八组合 60-step 数值探针。Flow 基线为 PR #25 合入 dev 的 `d73b217`，真实复验含修复的实际提交为 `e682f25`。version02 的八组合及 Study 终验完整通过，16段参数最大差0、train/test Loss 最大差6.66e-16、独立 eval Loss 差0。失败与修复保留在本记录，正式结论见 [STUDY_REPORT.md](STUDY_REPORT.md)。

## 二、执行条件

| **项目** | **本次值** |
| --- | --- |
| version | 当前 main-probe-flow-20261010-02；保留失败的 main-probe-flow-20261010-01 |
| 原代码 | 98648f3a5c7db7dccf3ca806410d5b6fdee9484c |
| 设备 | NVIDIA GeForce RTX 5090 D v2，CUDA |
| Python / Torch / torchvision | 3.13.9 / 2.11.0+cu130 / 0.26.0+cu130 |
| seed / 预算 | seed0；每组合两侧 60 steps，step30 / 60 全 test，各自 best 独立 eval |
| 数据 | 复用 main_exp 的 raw；[数据清单](../../../../main_exp/docs/DATA_MANIFEST.json) 的 16/16 文件匹配固定 SHA-256 |
| 顺序 | 先真实单组合验收，再统一 CLI 完成八组合；不覆盖此前 Run |

## 三、2026-10-10

1. Flow 重构的 CPU 核心 286 项、集成 36 项通过；隔离伪计算验证成功、数值失败和 artifact compare 失败，不属于正式数值结果。
2. version01 的 MNIST/linear，Run f6db09ee2f38fec9，在 prepare 阶段 CPU parity 通过，写环境文件时因缺少 os 导入而 failed，未进入正式训练。源码基线 5f1ff5c；[原始日志](../../../runs/f6db09ee2f38fec9/assets/logs/run.log)（本机产物，不随 Git clone 提供）及 PREPARED.json 保留。
3. 进一步核对准备证据发现，来源清单递归扫描 Study 时将本 Run 生成的32个归档 Python 文件计为源码变化（before105 / after137），因此总体 prepared.passed=false。数据与模型 CPU 数值检查本身通过，不能将它表述为完整准备通过。
4. 补充 os 导入，并改用库来源清单，仍保留原归档的独立逐字节完整性检查。使用 version02 的新 Run 目录复验；未改训练配方或数值门限，尚待真实 GPU 全阶段结果。缺陷跟踪见 [bugs.md](../../../../../docs/development/bugs.md)。

5. version02 / MNIST-linear，Run 785f86e74f36f408，实际提交 e682f25，完整 GPU 阶段 succeeded。step30 / 60 参数最大差均为0，独立 eval Loss 差0；来源、原归档、数值门与库 compare 全通过。[日志](../../../runs/785f86e74f36f408/assets/logs/run.log)、[比较报告](../../../runs/785f86e74f36f408/assets/probe/COMPARISON.json) 为本机原始产物，不随 Git clone 提供。
6. 统一 launch 顺序执行当前 index 的余下七组合，首组合不重复训练。2026-10-10 06:30 完成，8 succeeded / 0 failed / 0 pending，PROBE_COMPARISON.json 的 complete / passed 均为 true。库 compare 全八组通过，所有提交、来源清单、Torch 与设备一致；详见正式报告和 PAIR_RESULTS.json。

运行产物保存在本 Study 的 runs，按仓库规则不随 Git clone 提供。完整结果、失败与结论最终写入 STUDY_REPORT.md；未完成八组合前不报告完整通过。
