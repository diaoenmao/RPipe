# Study Plan: MNIST CNN seed 2 无中断补测

2026-10-03 用户授权修复 B-013 并继续后续目标。背景与旧恢复证据见 [原报告](../../mnist_cnn_budget/docs/STUDY_REPORT.md)，执行约定见 [STUDY_GUIDE](../../../docs/STUDY_GUIDE.md)。

## 1. 问题与固定条件

验证文件替换有界重试修复后，原 lr 0.03 / 600-step 配方的 seed 2 可以从头完成训练及配对 eval。两份配置沿用原 Study 的 batch 250、全量 MNIST、Normalize、CNN、SGD momentum/Nesterov/weight decay、cosine T_max=600；train log=10、全 test=30、latest=30、best=test Loss。新 version `cnn-600step-seed2-fixed-20261003`，旧恢复 Run 不覆盖。

## 2. 规模与验收

seed 2 × train / eval，共 2 Run。train 到 step 600，61 train / 20 test 观测，无 execute 失败、无 resume 或训练重启；latest / best scheduler 与 step 一致，eval 加载同 seed 最终 best，Loss / Accuracy 对齐。若发生训练恢复，不称无中断补测，应保留证据并另用新 version 完成。保存内的短暂 IO 重试不改变训练更新或采样位置。

补测与原 seed 0 / 1 的结果汇总时记录不同 version / 启动轮及 IO 修复；恢复 seed 2 不重复计入干净三 seed。仍不承诺逐位重现，也不把一次未再发生错误当作外部文件占用者已经消失的证明。

## 3. 高效率排班与时长预估

GPU 0，每组 1 条：train 成功后 eval，组末 wait；train 波内失败重试后才允许对应 eval。执行前 GPU 总 24455 MiB、已用约 4003 MiB。复用 MNIST 缓存并核对哈希。

make 后 launch 前写每条 Run ID / 内置 est。原无中断 train flow 约 26s（双并发），本次人工参考 60s train / 8s eval，共 68s；内置估时单列。实际计完整 launcher 与每条 flow，不能只用恢复段 algorithm elapsed。

| wait | mode | seed | Run ID | est 内置 / 人工（s） |
|---:|---|---:|---|---:|
| 1 | train | 2 | 488fd0974795ae18 | 58 / 60 |
| 2 | eval | 2 | 0825131829b320a8 | 5 / 8 |
| | 整轮 | | | 63 / 68 |

preflight 已确认 2 个 fresh Run 无 result / latest；正式训练前 core 225、integration 14 通过，真实 Windows 分件 / 整包短暂占用与持续占用四个探针均符合提交 / 旧档保护契约。

```powershell
python -m rpipe make studies/mnist_cnn_budget_repeat --num-gpus 1 --init-gpu 0 --round 1 --console shared
python -m rpipe launch studies/mnist_cnn_budget_repeat --num-gpus 1 --init-gpu 0 --round 1 --console shared
python -m rpipe report studies/mnist_cnn_budget_repeat
```

## 4. 交付

2026-10-03 已完成：2/2 succeeded、train 无中断到 600、best / eval 对齐；launcher 28.486s，详见 [报告](STUDY_REPORT.md)。旧产物保护复核通过。

报告、可点击曲线、日志、NUMBERS；旧 Run / index / process 文件保护检查，临时验证文件在 `.tmp/b013-followup-20261003/`。补测完成后继续已授权的小型矩阵。
