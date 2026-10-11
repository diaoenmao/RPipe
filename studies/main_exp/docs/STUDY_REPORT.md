# main_exp version03 实测报告

## 一、摘要

main-exp-flow-20261011-03 的新源码八格 600-step 原代码 / 当前链前置对照全部通过，参数最大差均为 0。完整矩阵于 2026-10-11 05:03 启动，按 round=3 调度 32 条连续 80000-step train 与 32 条自身 best 独立 eval。已有 32 条训练正式成功，均完成 80000-step 连续训练及 400 个完整评测点，单体 checkpoint / tracker 核查通过；32 条自身 best 独立评估也全部成功，64 个 Run 的 checkpoint / tracker 合同核查通过。2026-10-11 19:48 原生 Flow 正常退出（exit 0），Study process 应用固定四 seed 曲线门后通过终验。

实际源码提交 09e495b2d006caa197bb41e2773d47b817a4bd46；Torch 2.11.0+cu130、RTX 5090 D v2、CPU 2 线程。新基线显式关闭 cuDNN / matmul TF32，并在 prepare 完成后检查实际 deterministic、cuDNN 和精度开关；数据、seed、预算和固定估读门沿用 [PLAN.md](PLAN.md)。旧版训练及评估全部执行成功但终验 invalid，完整证据见 [version02 报告](versions/main-exp-flow-20261010-02/STUDY_REPORT.md)，不能混入当前精度基线。

## 二、改前与改后

| **范围** | **改前** | **当前状态** |
| --- | --- | --- |
| 设备条件 | 归档导入把声明的 benchmark=false 覆盖为 true；TF32 未显式固定 | 导入前开关保存并恢复，TF32 显式关闭，实际值写入 runtime_validation.json；全部 64 个 Run 均通过实际值检查 |
| 前置来源 | 新 make 后可能沿用旧源码的成功前置报告 | register 核对前置源码 hash；旧源码报告拒绝回归通过，新源码八格对照通过 |
| 调度 | 重型四 seed 被逐条串行安排 | 新完整矩阵显式 round=3，首批三条训练同时推进；B-022 容量估算器仍开放 |
| 验收 | 旧版 best 独立评估存在一个样本差异 | 新版本全部 32 组自身 best 独立评估一致，八组固定曲线门与两组模型次序门通过；B-023 的本次 FP32 基线复验通过，具体 kernel 根因仍未完全确定 |

## 三、证据与未完成项

1. 前置证据：[PREFLIGHT.json](PREFLIGHT.json)。导入副作用与失败边界回归：[ARCHIVE_RUNTIME_REGRESSION.json](ARCHIVE_RUNTIME_REGRESSION.json)。正式 Flow 前缀辅助核查：[FLOW_PREFIX_AUDIT.json](FLOW_PREFIX_AUDIT.json)，当前已核对 8/8 组 seed0，step200 / 400 / 600 共 96 个 train/test Loss / Accuracy 值与通过的前置链完全一致；只覆盖这些前缀的指标，不代替参数或长预算终验。精度和恢复条件：[eval_recovery.md](eval_recovery.md)。
2. 当前曲线证据：[COMPARISON.json](COMPARISON.json)，状态 passed，complete=true、final_gates_applied=true。八组均包含四个 seed；参考叉号为原图估读，不是恢复的原始指标。正式更新日期和执行细节见 [RERUN.md](RERUN.md)。
3. 32 条训练、32 条各自 best 独立评估、完整性核查和固定曲线门均已完成；正式数字见 [NUMBERS.md](NUMBERS.md)。同设备 CNN / ResNet18 各三次串行历史速度校准已完成，见 [速度报告](speed_calibration.md)；B-022 的容量与并发吞吐调查及 B-021 的 checkpoint 精确续跑修复仍待完成。原图的实际 seed、环境及原始指标没有恢复，终验结论限于预先固定的估读门。
4. 原始 Run 日志、数据和 checkpoint 保留在本机 runs / shared 下，不随 Git clone 提供。临时 launch 日志 .tmp/main-exp-flow-20261011-03-launch.log 和会话 13127 也仅为本机执行证据。

## 四、已完成训练

截至 2026-10-11 19:48，以下 32 条训练正式成功，完整性证据见 [CHECKPOINT_AUDIT.json](CHECKPOINT_AUDIT.json)。数字为各 seed 的训练末点与全程 best，全部矩阵与独立 best 评估已完成；各 seed 数字与完整四 seed 终验分开呈现。

| **数据** | **模型** | **seed** | **Run ID** | **末点 Accuracy** | **全程 best Accuracy** | **训练耗时** |
| --- | --- | --- | --- | --- | --- | --- |
| MNIST | linear | 0 | 7cd22ce8f92311e3 | 92.6500% | 92.8000% | 0:31:13 |
| MNIST | linear | 1 | cbd61fd5c36c10ad | 92.6900% | 92.7700% | 0:31:31 |
| MNIST | linear | 2 | ccc82976e799fc4b | 92.6800% | 92.7700% | 0:31:05 |
| MNIST | linear | 3 | c9715d794ad3049b | 92.6800% | 92.8000% | 0:31:24 |
| MNIST | mlp | 0 | 614c69d396e317b9 | 98.0800% | 98.1300% | 0:32:28 |
| MNIST | mlp | 1 | 3fa6abbc9027faad | 98.2600% | 98.3100% | 0:32:33 |
| MNIST | mlp | 2 | 9d53348ea5c1165c | 98.1700% | 98.2400% | 0:32:50 |
| MNIST | mlp | 3 | 787e4985f98a2e63 | 98.0900% | 98.1500% | 0:32:59 |
| MNIST | cnn | 0 | f4133f0d7a248646 | 99.4100% | 99.4900% | 0:37:20 |
| MNIST | cnn | 1 | 2921357483f031cd | 99.4600% | 99.5000% | 0:39:05 |
| MNIST | cnn | 2 | 96c74c6b1e44e9cf | 99.4800% | 99.5400% | 0:39:07 |
| MNIST | cnn | 3 | fd2727682e116c02 | 99.3500% | 99.4300% | 0:38:59 |
| MNIST | resnet18 | 0 | cf087b07765f4be4 | 99.6000% | 99.6700% | 2:09:19 |
| MNIST | resnet18 | 1 | 977b2e85d87dcc31 | 99.5500% | 99.5900% | 2:08:46 |
| MNIST | resnet18 | 2 | 9d380eea53541fa8 | 99.5700% | 99.6200% | 2:09:25 |
| MNIST | resnet18 | 3 | 069305e87da05b12 | 99.6000% | 99.6300% | 1:13:49 |
| CIFAR10 | linear | 0 | fc085db588d295b4 | 40.3900% | 40.6700% | 1:26:00 |
| CIFAR10 | linear | 1 | 385bf2b1307b622f | 40.5300% | 40.6600% | 1:25:48 |
| CIFAR10 | linear | 2 | dae98504b3fa274d | 40.3200% | 40.6200% | 1:25:48 |
| CIFAR10 | linear | 3 | a9148272760d5180 | 40.2400% | 40.4400% | 1:26:02 |
| CIFAR10 | mlp | 0 | d2b0e7f33699ecd4 | 63.6500% | 63.8500% | 1:26:11 |
| CIFAR10 | mlp | 1 | 6c2cda2cf02c2a9b | 63.2400% | 63.4200% | 1:26:31 |
| CIFAR10 | mlp | 2 | 28f2368d1634417c | 63.6900% | 63.8800% | 1:26:30 |
| CIFAR10 | mlp | 3 | 8e5f67f6b365822f | 63.2900% | 63.5000% | 1:26:19 |
| CIFAR10 | cnn | 0 | b99775641e1800c7 | 88.8500% | 89.0700% | 1:32:19 |
| CIFAR10 | cnn | 1 | 48ee1a36f59ac74d | 88.9000% | 89.1300% | 1:32:13 |
| CIFAR10 | cnn | 2 | 30cc6a335b81cf34 | 88.4200% | 88.6600% | 1:32:13 |
| CIFAR10 | cnn | 3 | 6ce695c58b4b1f98 | 88.8500% | 89.0700% | 1:36:43 |
| CIFAR10 | resnet18 | 0 | 5bd65c61c30c12d2 | 92.8100% | 93.0000% | 2:17:35 |
| CIFAR10 | resnet18 | 1 | caf78c8d80ad9f10 | 93.0200% | 93.2300% | 2:17:34 |
| CIFAR10 | resnet18 | 2 | 8c413f14e3d8f742 | 93.1000% | 93.2600% | 2:03:14 |
| CIFAR10 | resnet18 | 3 | 6b9fced9385ffbad | 93.2100% | 93.3200% | 2:03:14 |

![完整四 seed 训练曲线与原图估读](figures/historical_comparison.png)
