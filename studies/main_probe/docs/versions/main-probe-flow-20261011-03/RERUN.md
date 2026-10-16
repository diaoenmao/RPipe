# main_probe 复跑记录

## 一、摘要

本次按 [PLAN.md](<../../../../../docs/research/研究计划 RPipe.md#study-main-probe>) 复跑 Flow 重构后的八组合 60-step 数值探针。2026-10-11 的 main-probe-flow-20261011-03 在实际提交 c9c782b5d5bb62da4e5e2292e96e842740c24f53 上全部通过，两侧显式 FP32。8 succeeded / 0 failed / 0 pending，Study complete / passed；16 段参数最大差 0、train / test Loss 最大差 4.44e-16、独立 eval Loss 差 0。运行实际 System 结果确认两个 TF32 开关均为 false，同源清单 hash 为 65bbec5ee797a557ae419df90c351723764ad9e0a54f3d15821a2b8121693658。正式结论见 [STUDY_REPORT.md](STUDY_REPORT.md) 与 [PAIR_RESULTS.json](PAIR_RESULTS.json)。旧 version02 报告已归档，不能替代新长实验；main_exp FP32 的 600-step 前置门在执行。

## 二、执行条件

| **项目** | **本次值** |
| --- | --- |
| version | 当前 main-probe-flow-20261011-03；version02 完整通过的证据已归档，version01 失败保留 |
| 原代码 | 98648f3a5c7db7dccf3ca806410d5b6fdee9484c |
| 设备 | NVIDIA GeForce RTX 5090 D v2，CUDA |
| Python / Torch / torchvision | 3.13.9 / 2.11.0+cu130 / 0.26.0+cu130 |
| seed / 预算 | seed0；每组合两侧 60 steps，step30 / 60 全 test，各自 best 独立 eval |
| 数据 | 复用 main_exp 的 raw；[数据清单](../../main_exp/docs/DATA_MANIFEST.json) 的 16/16 文件匹配固定 SHA-256 |
| 顺序 | 先真实单组合验收，再统一 CLI 完成八组合；不覆盖此前 Run |

## 2026-10-11

1. main_exp 默认精度的 best 独立评测不稳定，串行恢复停止条件触发；两 Study 统一声明 FP32，并通过新 Run ID 做复验，完整诊断与方案见 main_exp 的 EVAL_DIAGNOSTIC.json / eval_recovery.md。没有覆盖旧结果或放宽数值门。
2. 在 c9c782b 上使用 rpipe make --round 3 展开 8 个唯一 Run ID，与 version02 无重叠；统一 launch 会话 75657 于 04:37 完成，exit0。全部六阶段链成功，Study 终验 complete / passed，rpipe report 写正式数字。
3. 核对当前八个正式 result：control.version=main-probe-flow-20261011-03，System.cudnn_allow_tf32=false、matmul_allow_tf32=false；八组来源、设备、提交一致。输入 / RNG、参数 / buffer、优化器 / 调度器、正确样本数、best 与独立 eval、原归档完整性及库 compare 均通过。原始日志和 OBSERVATIONS.pt 在本机 runs 下，正式逐段数字保存在 PAIR_RESULTS.json。


## 三、2026-10-10

1. Flow 重构的 CPU 核心 286 项、集成 36 项通过；隔离伪计算验证成功、数值失败和 artifact compare 失败，不属于正式数值结果。
2. version01 的 MNIST/linear，Run f6db09ee2f38fec9，在 prepare 阶段 CPU parity 通过，写环境文件时因缺少 os 导入而 failed，未进入正式训练。源码基线 5f1ff5c；[原始日志](../runs/f6db09ee2f38fec9/assets/logs/run.log)（本机产物，不随 Git clone 提供）及 PREPARED.json 保留。
3. 进一步核对准备证据发现，来源清单递归扫描 Study 时将本 Run 生成的32个归档 Python 文件计为源码变化（before105 / after137），因此总体 prepared.passed=false。数据与模型 CPU 数值检查本身通过，不能将它表述为完整准备通过。
4. 补充 os 导入，并改用库来源清单，仍保留原归档的独立逐字节完整性检查。使用 version02 的新 Run 目录复验；未改训练配方或数值门限，尚待真实 GPU 全阶段结果。缺陷跟踪见 [bugs.md](../../../docs/development/bugs.md)。

5. version02 / MNIST-linear，Run 785f86e74f36f408，实际提交 e682f25，完整 GPU 阶段 succeeded。step30 / 60 参数最大差均为0，独立 eval Loss 差0；来源、原归档、数值门与库 compare 全通过。[日志](../runs/785f86e74f36f408/assets/logs/run.log)、[比较报告](../runs/785f86e74f36f408/assets/probe/COMPARISON.json) 为本机原始产物，不随 Git clone 提供。
6. 统一 launch 顺序执行当前 index 的余下七组合，首组合不重复训练。2026-10-10 06:30 完成，8 succeeded / 0 failed / 0 pending，PROBE_COMPARISON.json 的 complete / passed 均为 true。库 compare 全八组通过，所有提交、来源清单、Torch 与设备一致；详见正式报告和 PAIR_RESULTS.json。

运行产物保存在本 Study 的 runs，按仓库规则不随 Git clone 提供。完整结果、失败与结论最终写入 STUDY_REPORT.md；未完成八组合前不报告完整通过。
