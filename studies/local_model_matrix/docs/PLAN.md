# Study Plan: 本地小型多模型矩阵

2026-10-03 用户授权“修 bug，继续执行后面的目标”。承接 [brainstorm](../../../docs/BRAINSTORM.md) 当时的矩阵建议；先通过 B-013 单元 / 原生 Windows 验证和 seed 2 无中断补测，再启动本 Study。遵循 [STUDY_GUIDE](../../../docs/STUDY_GUIDE.md)。

## 1. 研究问题

在统一 600-step 的现有训练配方下，CNN / ResNet18 在 MNIST / CIFAR10 上能否持续学习？各 seed 差异、best / last 分离和本机成本如何？这是一轮固定配方的本地研究，不是调优后模型排名或旧 main 的严格复现。

## 2. 规模与固定条件

2 数据集 × 2 模型 × train / eval = 8 Experiment，每个 seed 0 / 1 / 2，共 24 Run（12 train + 12 eval）。新 version `local-matrix-600step-20261003`，全部从头训练。使用当前已有原生模型 / 数据通路。

MNIST train 60000 / test 10000；CIFAR10 train 50000 / test 10000；不截子集。train batch 250，test batch 1000，pin_memory=true、workers=0。沿用各训练集 stats；MNIST 仅 Normalize，CIFAR10 训练态 flip / crop 与 Normalize、test 仅 Normalize。源缓存从已有 Study 复制并核对哈希。

所有点采用同一个起始 lr 0.03、SGD momentum 0.9 / Nesterov / weight decay 0.0005、无梯度裁剪、cosine T_max=600、eta_min=0；不因中途分数调整 lr 或延长预算。该 lr 是明确的固定配方选择，不声称适合所有数据集 / 模型。每条 train 150000 次采样，MNIST 约 2.5、CIFAR10 约 3 个数据集规模；同 step 不代表相同算力成本。

train log 每 10 step；全量 test、latest 每 30 step；按 test Loss 最低选 best，独立 eval 加载对应同 seed best。deterministic=false、cudnn_benchmark=true，不承诺逐位复现。独立 eval 是同 test split 复算，不是额外泛化验证。

## 3. 高效率排班

单 GPU 0，按 mode / model / dataset 同类分组，最多 2 条并发；每个 3-seed 条件拆成 [0,1]、[2] 两个 wait，共 8 train wait + 8 eval wait。全部 train 波及必要重试结束后才进入 eval；持续失败的 train 阻断对应 eval，无关 seed 继续。组末 wait 保证资源释放。

CLI make 展开并估计后，通过已有 `write_launch_scripts(..., batches=groups, round_size=2)` 为本 Study 写同类分组，launch 复用该 jobs 清单。不添加调度器或库选项，不手写绕过恢复保护的 run-one 脚本。执行前核对每个双并发组显存估计可容纳于当前空闲显存的一半；不满足时将对应组降为 1 并记录。

```powershell
python -m rpipe make studies/local_model_matrix --num-gpus 1 --init-gpu 0 --round 2 --console shared
# 用现有 schedule API 生成本轮同类 batches；见临时执行证据中的 helper
python -m rpipe launch studies/local_model_matrix --num-gpus 1 --init-gpu 0 --round 2 --console shared
python -m rpipe report studies/local_model_matrix
```

## 时长预估

make 后 launch 前填写逐 Run 内置估时与 wait 序号。人工保守参考：CNN train 120s / eval 10s，ResNet18 train 240s / eval 15s；四个数据 / 模型条件各两 train / eval 组，总墙钟参考 1540s（约 26 分钟），资源排班参考上限 30 分钟，不含下载、验证和报告。初始参考来自已有 MNIST CNN 测量与较重 ResNet 的保守裕量，CIFAR10 / ResNet18 当前实际成本仍需测量；不作为性能质量阈值。

## 4. 验收与交付

2026-10-03 已完成 24/24 succeeded，12 组 best / eval 对齐；12 条 train 均无 checkpoint resume / execute 错误。初始启动会话在 3 条 CNN 完成、2 条 ResNet prepare 后结束，确认无存活进程后继续清单，成功项跳过；中断前没有训练观测或 checkpoint。第二次 launcher 503.028s，整轮 flow 时间包络 860.246s（含启动间隔），完整成本与边界见 [报告](STUDY_REPORT.md)。

实际 preflight：CIFAR10 ResNet18 train 单条估计显存 6.1 GiB，两条超出当前空闲显存一半（9.8 GiB），因此改为单并发；其余条件最多 2 条。最终 9 train wait + 8 eval wait，共 17 组，内置墙钟预估 786s。24 个 Run 身份 / seed / 配方均核对，全部无旧 result / latest。

| wait | data | model | mode | seed | Run ID | est 内置（s） |
|---:|---|---|---|---:|---|---:|
| 1 | MNIST | cnn | train | 0 | 150d215770eb2efe | 58 |
| 1 | MNIST | cnn | train | 1 | 9a20a9f71cae60b6 | 58 |
| 2 | MNIST | cnn | train | 2 | 38d35e935ec4490a | 58 |
| 3 | MNIST | resnet18 | train | 0 | 1650bcd5c847c5ea | 102 |
| 3 | MNIST | resnet18 | train | 1 | 6b2e49d29850f9dd | 102 |
| 4 | MNIST | resnet18 | train | 2 | 4b00a4e8da8230c2 | 102 |
| 5 | CIFAR10 | cnn | train | 0 | 1bff05cdc5dee8f5 | 58 |
| 5 | CIFAR10 | cnn | train | 1 | 359e6573a29ed364 | 58 |
| 6 | CIFAR10 | cnn | train | 2 | efebcf9e6ea88420 | 58 |
| 7 | CIFAR10 | resnet18 | train | 0 | ef103fc5d04a21a1 | 102 |
| 8 | CIFAR10 | resnet18 | train | 1 | 43fadbe44304b2c5 | 102 |
| 9 | CIFAR10 | resnet18 | train | 2 | 86a73491c9c1d7fa | 102 |
| 10 | MNIST | cnn | eval | 0 | 69290f0c093f4590 | 5 |
| 10 | MNIST | cnn | eval | 1 | 479dc0ca7a2ef6c9 | 5 |
| 11 | MNIST | cnn | eval | 2 | 592f9c51da9b38d2 | 5 |
| 12 | MNIST | resnet18 | eval | 0 | 215ba19b83e3d9aa | 6 |
| 12 | MNIST | resnet18 | eval | 1 | 94c25ad41eff1a76 | 6 |
| 13 | MNIST | resnet18 | eval | 2 | a5c17619978e1d20 | 6 |
| 14 | CIFAR10 | cnn | eval | 0 | 453459a9d02e1761 | 5 |
| 14 | CIFAR10 | cnn | eval | 1 | 1ad960571d1d8d54 | 5 |
| 15 | CIFAR10 | cnn | eval | 2 | d60cbcdc28e263fc | 5 |
| 16 | CIFAR10 | resnet18 | eval | 0 | 051d90effb2ecdab | 6 |
| 16 | CIFAR10 | resnet18 | eval | 1 | 1c89f5a8572b9ffd | 6 |
| 17 | CIFAR10 | resnet18 | eval | 2 | 727df651da5b491b | 6 |

四组 CNN train 各 58s、MNIST ResNet18 两组各 102s、CIFAR10 ResNet18 三组各 102s；八组 eval 共 44s，合计 786s。人工参考按增加一组 ResNet train 调整为 1780s（约 30 分钟）；实际墙钟另测。


24/24 succeeded；12 train 均到 step 600，20 次完整 test，核对实际 lr、latest / best scheduler 进度与 Loss best；12 组 eval 权重来源和指标对齐。报告每个数据 / 模型条件的三 seed mean / 样本 std、last / best / best step、曲线与真实成本。发生恢复的 Run 单独标记，不能与无中断统计混称；保留所有失败和重试记录。

过程不根据短期结果改变配方。若出现低分、早期回落、seed 差异或最佳点早于末尾，先用已有观测解释，不直接认定代码缺陷。正式 PLAN / YAML / 报告 / 图沿用 Study 目录；临时证据留 `.tmp/`。旧 Run 与已有结果保护检查通过后更新 BUGS / BRAINSTORM。
