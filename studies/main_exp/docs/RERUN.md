# main_exp 复跑记录

## 一、摘要

本次按 [PLAN.md](PLAN.md) 在 Flow 重构后的代码基线上复跑 32 条连续 80000-step 训练与 32 条自身 best 独立评测。固定图像估读门见 [TARGET.md](TARGET.md)。当前完成环境及原始数据准备，600-step 八格前置门与完整训练尚未运行，无本版本数值结论。

## 二、执行条件

| **项目** | **本次值** |
| --- | --- |
| Flow 基线 | PR #25 合入 dev，d73b217 |
| version | main-exp-flow-20261010-01 |
| 原代码 | 4ccb28d0496110253e9f8e3f3df658853f07996b |
| 设备与环境 | NVIDIA GeForce RTX 5090 D v2；Python 3.13.9，Torch 2.11.0+cu130，torchvision 0.26.0+cu130 |
| 数据 | [DATA_MANIFEST.json](DATA_MANIFEST.json)，16/16 原始文件匹配固定 SHA-256 |
| seed / 预算 | 0–3；连续 80000 optimizer steps，每 200 steps 完整 test，共 400 点 |
| 前置条件 | 同设备 600-step / eval200 八格对照通过后才能 launch；采样与 scheduler 总预算仍为 80000 |

## 三、2026-10-10

1. MNIST 从官方文件准备。CIFAR10 官方下载速度较慢，主动停止该进程后，通过 prepare_data.py --cache-root 复用本机此前探针缓存；所有文件逐一核对固定 SHA-256，未使用旧 processed 结果作为新训练成果。
2. 已核对 split：MNIST train60000 / test10000，CIFAR10 train50000 / test10000。下载的未完成 part 保留在忽略的 raw 目录，不作为有效数据。
3. 当前尚未执行前置门与正式矩阵。其 Run ID、日志、失败和数字随执行追加；最终完整结果写入 STUDY_REPORT.md。

数据、checkpoint、日志及临时前置门输出按仓库规则保留在本机，不随 Git clone 提供。旧数值结果不替代本次验收；失败不通过修改固定门限消除。
