# 历史超参前缀探针报告

2026-10-03。用户选择 main README 历史曲线对应的超参，随后明确本轮只做到探针。按 [PLAN](PLAN.md#用户确认的历史超参前缀探针2026-10-03执行前) 执行，**8/8 通过，本轮结束，不启动长实验**。逐格原值与哈希见 [HISTORICAL_PREFIX_PROBE.json](HISTORICAL_PREFIX_PROBE.json)。

## 配方与实际执行

固定图最后更新提交 `4ccb28d0496110253e9f8e3f3df658853f07996b` 的候选配方：MNIST / CIFAR10 × linear / mlp / cnn / resnet18，seed0；train/test batch250；SGD lr0.01、momentum0.9、Nesterov、weight_decay0.0005；每次更新前clip_grad_norm1；cosine T_max80000。旧CNN四层BN、CPU torchvision增强和常量归一化全部沿用归档，当前生产模型不改。

每边每组合仅200次optimizer更新（50000个实际训练样本），在step200完整评测一次（10000张test）。原代码仍按80000步创建采样器和scheduler，当前短预算取同一采样前缀，不将scheduler缩到200。原代码直接调用train/test函数；当前链通过临时DataRegistry / ModelRegistry适配器进入原生TrainAlgorithm。本轮共8对训练链、16次内嵌test，没有独立eval任务或正式Flow调度矩阵。

单进程逐组合串行，对每组合先原代码、再当前链；CUDA确定性控制统一为deterministic=true、cudnn.deterministic=true、benchmark=false、CUBLAS_WORKSPACE_CONFIG=:4096:8。本机RTX 5090 D v2、Torch 2.11.0+cu128、CPU threads=2；先导入NumPy再Torch沿用Windows环境适配。没有重建旧Torch2.0.1环境。

命令：

```powershell
python -B .tmp/main-reproduction-20261003/historical_prefix_probe.py
python -B .tmp/main-reproduction-20261003/verify_historical_prefix.py
```

两条均退出0，训练无失败或重试。原始日志、复制数据、checkpoint和tracker目录为
`.tmp/main-reproduction-20261003/historical-prefix-9f51acfe2a55/`，不入Git。

## 数值与前缀核对

8格全部初始化参数相同，初始及训练/test后的CPU/CUDA RNG相同；50000个实际训练索引、CPU增强后的输入/标签字节哈希相同，完整test输入哈希相同。step200浮点参数及整数buffer均逐位一致，scheduler状态相同，实际SGD组与下一步lr核对通过；下一步lr为 `0.009999845788223966`。

训练Loss最大差 6.661e-16，test Loss最大差 5.551e-17；这些只是均值累计的浮点顺序差异。train/test正确样本数均严格相同。原门限保持：Loss绝对差≤1e-6、参数atol1e-6/rtol1e-5、整数buffer与正确样本数相同；没有放宽门限。

| 数据 | 模型 | step200 test Accuracy (%) | test Loss | 参数最大差 | 判定 |
|---|---|---:|---:|---:|---|
| MNIST | linear | 91.31 | 0.307476 | 0 | 通过 |
| MNIST | mlp | 92.27 | 0.268805 | 0 | 通过 |
| MNIST | cnn | 97.74 | 0.092014 | 0 | 通过 |
| MNIST | resnet18 | 97.82 | 0.084479 | 0 | 通过 |
| CIFAR10 | linear | 36.22 | 1.837898 | 0 | 通过 |
| CIFAR10 | mlp | 42.24 | 1.644279 | 0 | 通过 |
| CIFAR10 | cnn | 52.31 | 1.330396 | 0 | 通过 |
| CIFAR10 | resnet18 | 44.01 | 1.519303 | 0 | 通过 |

重新加载原step200与当前latest整包，通过torch.equal独立复核全部参数/buffer，核对scheduler T_max/last_epoch、SGD momentum/Nesterov/wd/lr、tracker的train/test段指标及实际JSONL坐标。Loss和Accuracy各一条test记录，optimizer_step均200；tracker内部batch计数240不作为训练步数。8/8存档复核通过，不仅依赖探针里的passed标记。

独立复核脚本首次错误地把每指标一行的JSONL当成每评测一行，断言失败；按既有tracker契约修正后通过。这是验证脚本错误，不是生产bug，没有重训。

## 分项时间（秒）

| 数据 | 模型 | 原train | 原test | 当前run | 当前eval hook | 当前checkpoint |
|---|---|---:|---:|---:|---:|---:|
| MNIST | linear | 3.514 | 0.886 | 4.750 | 0.652 | 0.027 |
| MNIST | mlp | 3.702 | 0.639 | 4.194 | 0.730 | 0.032 |
| MNIST | cnn | 4.143 | 0.664 | 4.514 | 0.627 | 0.068 |
| MNIST | resnet18 | 7.134 | 0.854 | 8.396 | 0.865 | 0.322 |
| CIFAR10 | linear | 10.283 | 0.776 | 10.891 | 0.790 | 0.027 |
| CIFAR10 | mlp | 11.198 | 0.841 | 11.411 | 0.792 | 0.036 |
| CIFAR10 | cnn | 10.888 | 0.872 | 11.748 | 0.827 | 0.070 |
| CIFAR10 | resnet18 | 15.785 | 1.096 | 16.594 | 1.020 | 0.301 |

内部总墙钟 148.740 秒，含复制数据、准备、两边执行、哈希和存档；数据复制/哈希 0.410 秒。计时从依赖导入完成后开始，不含Python启动/导入。每个训练/test计时边界前后CUDA同步。

原train/test含逐样本转换/哈希、指标及TensorBoard IO；原test不含随后另存参数快照。当前run含训练、完整test、tracker与checkpoint IO，eval hook含tracker IO，checkpoint含观测器CPU快照与磁盘IO。两边边界不同，单次串行数值对照不是加速基准，也不据此外推80k时长。准备、剩余训练/杂项分项原值保存在JSON。

## Bug与证据保护

本探针没有发现新的当前生产缺陷，没有修改生产代码。当前91个源文件执行前后及存档复核时均与B-018快照一致；归档历史提交36个文件逐一git blob核对，0差异。16个复制原始数据文件SHA-256一致，旧Study与已有证据不改写。

[BUGS](../../../docs/BUGS.md)此前“开放”标题下是已关闭说明，不是真正open条目；已清除这些重复记录，现在明确“目前没有开放缺陷”，下个编号B-019。B-014～B-018修复证据及286项本地回归见[summary](../../../docs/SUMMARY.md)；源码未变，本轮不重复执行同一回归。B-007属于维护关闭，不称根因修复；B-013原外部占用者仍未识别。归档旧best比较缺陷保留在[历史审计](HISTORICAL_AUDIT.md)，不移植到当前实现。

## 结论与边界

真实历史候选eval200的首个200-step前缀，在当前机器的统一确定性控制下，两套计算8/8对齐。这补齐此前4-step/eval2不能证明真实eval200前缀的证据缺口，已有Registry足够承载旧BN、CPU增强、常量归一化及裁剪，暂不需要生产兼容开关。

**未执行**80000-step、4-seed完整矩阵；没有历史曲线、多点评测、长期收敛、历史默认非确定性环境或独立eval/best全程选择的验收。旧原始结果/权重仍未找回，图最后更新提交只是候选运行配方依据。当前main默认60-step严格门4/8、确定性8/8的旧结论保留。本轮按用户要求到探针结束，不以8/8前缀宣称README历史图已复现。
