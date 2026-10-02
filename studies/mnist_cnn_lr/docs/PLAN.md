# Study Plan: mnist_cnn_lr

> 2026-10-02 已完成。首次中断轮保留作故障证据；干净轮 `cnn-lr-clean-20261002` 18/18 succeeded，无 retry，flow 墙钟 86.049s。结果与局限见 [STUDY_REPORT.md](STUDY_REPORT.md)。下文保留预先设计、估时与复跑前修订。

## 1. 问题与假设

2026-10-02，承接 [main_base 报告](../../main_base/docs/STUDY_REPORT.md)：MNIST CNN 在 step 15 的 test Accuracy 为 31.00%，step 20 为 Loss-best（23.07%），step 25 后降至 11.35%。CNN 结构、初始化及 MNIST Normalize 与旧 main 配方一致；完整评测后会恢复 train 模式，独立 eval 也复现 best。尚无证据证明低精度是模型迁移错误。

本轮只改变初始学习率，检查 `lr=0.1` 是否导致该短预算训练不稳定。三档预先固定为 0.1 / 0.03 / 0.01，每档 seed 0 / 1 / 2，共 9 train + 9 eval。降低学习率不保证在相同短预算下得到更高分；不在看结果后删 seed 或挑最好的 Run 代表整组。

已确认旧日志的 lr 错用了 epoch 初始缓存，但 checkpoint 证明 cosine 实际衰减。本轮先修正日志读数并用测试确认；不改变训练数学、初始化或调度顺序，不改写旧日志。

## 2. 配方与隔离

- Study：`mnist_cnn_lr`；首次 version：`cnn-lr-20261002`；干净复跑 version：`cnn-lr-clean-20261002`（原因见下）。与 main_base 分开，旧 Run 保留。
- MNIST 全量 train 60000 / test 10000；复用 main_base 已有 MNIST 缓存的文件副本，不重新下载，复制后核对哈希和 stats。train mean 0.130660、std 0.308108。
- `custom_torch` CNN，hidden=[64,128,256,512]；MNIST 仅 Normalize，沿用 `augment=true` 配置（MNIST 无随机增强）。
- 60 optimizer step，batch 250；test batch 1000；每 5 step 训练记录及 full test；13 train / 12 test 观测，不增加预算或训练 epoch。
- SGD、momentum 0.9、Nesterov、weight decay 5e-4、无 clipping；cosine T_max=60、eta_min=0；只有初始 lr 作为配方变量。
- latest 每 30 step；best 按最低 test Loss；独立 eval 读取同 lr / seed / version 的 sibling best。
- 同一 RTX 5090 D v2，GPU 0；`--round 2 --console shared`，最多两条同模型 Run 并发，train 全部结束后再 eval。保持相同预算和并发上限，不把跨轮耗时当作算法加速。

## 3. 执行与预估

```powershell
python -B -m rpipe make studies/mnist_cnn_lr --num-gpus 1 --init-gpu 0 --round 2 --console shared
python -B -m rpipe launch studies/mnist_cnn_lr --num-gpus 1 --init-gpu 0 --round 2 --console shared
python -B -m rpipe report studies/mnist_cnn_lr
```

launch 前检查 18 条 fresh config、seed / lr / mode、数据统计、没有旧 checkpoint、依赖分组与可用显存。make 的初始每条 / 分组估时须在启动前追加在这里。上一轮 MNIST CNN 的 flow train 11.959s、eval 4.185s，只作参考；不同配对和调度负载影响实际时间，不作完成承诺。

2026-10-02 启动前核对通过：18 条 fresh Run，10 个 wait（5 train + 5 eval，每组至多 2 条）；每条初始估计 train 25s / eval 5s，整轮按组最慢者求和 **150s**。GPU 总显存 24455 MiB、已用 3742 MiB、利用率 1%；MNIST 缓存 10 个文件复制后哈希一致。原 main_base 的 370 个 Run / index / process 文件已记录启动前哈希，结束后复核。

### 首轮故障后的执行修订（在复跑前记录）

首轮 `7cf84c35085a4225`（lr 0.03、seed 0）在发布 best 分件目录时遇到 Windows `PermissionError`；launch 在 eval 波结束后才从 latest step 30 重试 train。虽然最终显示 18/18 succeeded，这一格已经发生恢复采样、调度错位和 eval 陈旧，不能混入 fresh 单因素比较。记录 BUGS，保留初轮 Run / 日志及 `.tmp/cnn-lr-20261002/interrupted/` 中的清单与聚合，不删除 checkpoint、不抹掉失败。

为维持三档 × 三 seed 的完整 fresh 矩阵，不手改 index 拼接历史 Run，也不为一次补测新增选择框架：换 `cnn-lr-clean-20261002` 重跑相同 18 条，配方 / 预算 / 并发不变，初始估时仍为 150s。由于首次在受限 Windows 环境写 checkpoint 失败，复跑申请标准提权执行，不修改 ACL 或安全配置；这不代表已经证明或修复 WinError 5 根因。复跑仍须检查没有 retry、曲线点数和 best / eval 配对；若再失败，保留证据，不能放宽验收冒充干净结果。

## 4. 验收与分析

1. 18/18 succeeded，无未解释的失败或 retry；结果、日志、checkpoint、process 完整，旧 main_base 运行产物未变。
2. 每条 train 完成 60 step，12 次 full test；13/12 曲线点，实际 lr 按 cosine 衰减；每条独立 eval 与所加载 best 一致。
3. 报告逐 lr / seed 列 last / best（Loss、Accuracy、best step）及 eval；跨 seed 使用现有 process mean / std，并说明 n=3 的局限。
4. 检查旧 seed 0、lr 0.1 的退化能否复现。若更低 lr 在多个 seed 下消除回落，只能说明该预算下的学习率敏感性，不能自动认定内部机制或推广到其他模型 / 数据集。
5. 根据结果区分确认的软件缺陷、训练现象及待验证假设，更新报告与 brainstorm；不为了改善数字修改 main_base 默认配方。

必要时只读检查 best / latest 在固定 MNIST test 样本上的预测类别分布，判断低 Accuracy 是否来自退化到单类别；不做新的训练或另造推理框架。

## 5. 本轮不做

不增加训练预算，不扫架构 / 初始化 / optimizer，不自动运行旧 main，不新增依赖，不提交或推送。后续扩大实验先根据本轮证据决定。
