# main_exp 复跑记录

## 一、摘要

本次按 [PLAN.md](<../../../../../docs/research/研究计划 RPipe.md#study-main-exp>) 在 Flow 重构后的代码基线上复跑 32 条连续 80000-step 训练与 32 条自身 best 独立评测，固定图像估读门见 [TARGET.md](../../TARGET.md)。中断版本 main-exp-flow-20261010-01 的原始结果保留，未完成终验。当前版本 main-exp-flow-20261010-02 已通过同设备 600-step 八格前置门；64 个新 Run ID 与中断版本无重叠。截至 2026-10-11 04:16，32 train 与 32 eval 均正式 succeeded，训练各有 400 个完整 test 点，64 个 Run 的单体 checkpoint / tracker 审计通过。剩余 CIFAR10 ResNet18 seed 1–3 已并发跑完，配方和预算未变。

Study 终验仍未通过：CIFAR10 CNN seed 0 的独立评测为 89.1400%，训练 best 为 89.1500%，相差 1 个正确样本；实际加载路径和 step76800 均匹配自身 best。其余 31 条 eval 的 best 对照通过。当前 [COMPARISON.json](COMPARISON.json) 为 invalid，固定四 seed 曲线门未正式应用；执行成功不等于质量通过。已记录 B-023 并开展只读复查，保留原评测和失败证据，不放宽门限或用临时重评覆盖正式结果。

## 二、执行条件

| **项目** | **本次值** |
| --- | --- |
| Flow 基线 | PR #25 合入 dev，d73b217 |
| 当前 version | main-exp-flow-20261010-02；中断版本 01 保留 |
| 原代码 | 4ccb28d0496110253e9f8e3f3df658853f07996b |
| 设备与环境 | NVIDIA GeForce RTX 5090 D v2；Python 3.13.9，Torch 2.11.0+cu130，torchvision 0.26.0+cu130 |
| 数据 | [DATA_MANIFEST.json](../../DATA_MANIFEST.json)，16/16 原始文件匹配固定 SHA-256 |
| seed / 预算 | 0–3；连续 80000 optimizer steps，每 200 steps 完整 test，共 400 点 |
| 前置条件 | 同设备 600-step / eval200 八格对照通过后才能 launch；采样与 scheduler 总预算仍为 80000 |

## 三、2026-10-11

62. 2026-10-11 00:16，新版本 main-exp-flow-20261010-02 的 MNIST ResNet18 seed 0–3 分别达到 60000、61200、61200、60800 steps，完成 300、306、306、304 个 test 点，均达到 80000-step 预算的四分之三。同一 launch 会话 23271 仍在运行，已有 Loss / Accuracy mean 均有限，评测坐标连续，暂无 error。当前新版本 24 succeeded / 40 pending，无失败结果；此项仅为活动进度，完整 ResNet 训练预算、32 条自身 best 独立评测及固定曲线门尚未完成。

63. 2026-10-11 00:43，新版本 main-exp-flow-20261010-02 的 MNIST ResNet18 seed 0–3 正式 result 均为 succeeded；逐 Run 核对 latest checkpoint step=80000，test Loss / Accuracy 各 400 点，坐标按 200 连续至 80000，数值均有限，best/model.pt 均存在。末点 Accuracy 按 seed 为 99.6000%、99.6300%、99.6300%、99.6300%；检查明细为 .tmp/main-exp-flow-20261010-02-mnist-resnet-completed.json（仅本机临时证据）。至此新版本 28 条训练完成，当前 28 succeeded / 36 pending，无失败结果。同一 launch 会话 23271 仍在运行，CIFAR10 ResNet18 seed 0 已自动启动并完成至 step400 的两个 test 点，暂无 error。其余 ResNet 训练、32 条自身 best 独立评测及固定曲线门尚未完成，训练末点不代替整套验收。

64. 2026-10-11，对当前版本已正式成功的 28 条训练调用最终曲线验收使用的 inspect_run 做只读 CPU checkpoint bundle 审计，28/28 通过。核对实际 latest.pt / best.pt，而非仅检查拆分镜像文件：完整 test10000 / batch250、400 个连续评测点、最终 checkpoint 的 80000-step 进度及轨迹、best 的全程最高 Accuracy 和对应 step、正式 result 与 tracker 一致，均无错误。明细为 .tmp/main-exp-flow-20261010-02-completed-bundle-audit.json（仅本机临时证据）。此审计不代替剩余训练、32 条自身 best 独立评测或四 seed 固定曲线门。

65. 2026-10-11 01:27，新版本 main-exp-flow-20261010-02 的 CIFAR10 ResNet18 seed 0 达到 step40200，完成 201 个 test 点，超过 80000-step 预算的一半；seed 1–3 尚未进入活动训练。同一 launch 会话 23271 仍在运行，已有 Loss / Accuracy mean 均有限，评测坐标连续，暂无 error。当前 28 succeeded / 36 pending，无失败结果；此项仅为活动进度，其余训练、32 条自身 best 独立评测及固定曲线门尚未完成。

66. 2026-10-11 01:49，新版本 main-exp-flow-20261010-02 的 CIFAR10 ResNet18 seed 0 达到 step60200，完成 301 个 test 点，超过 80000-step 预算的四分之三；seed 1–3 尚未进入活动训练。同一 launch 会话 23271 仍在运行，已有 Loss / Accuracy mean 均有限，评测坐标连续，暂无 error。当前 28 succeeded / 36 pending，无失败结果；此项仅为活动进度，其余训练、32 条自身 best 独立评测及固定曲线门尚未完成。

67. 2026-10-11 02:13，新版本 main-exp-flow-20261010-02 的 CIFAR10 ResNet18 seed 0（20b54c915faeb9b9）正式 result 为 succeeded，训练耗时 1:29:57，末点 Accuracy 92.8300%，全程 best 93.0800%。使用最终曲线验收的 inspect_run 核对实际 latest.pt / best.pt、80000-step 进度、400 个连续评测点、完整 test10000 / batch250、全程最高值与对应 step、正式 result 与 tracker 一致，29/29 条完成训练审计通过，无错误。明细为 .tmp/main-exp-flow-20261010-02-completed-bundle-audit.json（仅本机临时证据，更新至 29 条）。当前 29 succeeded / 35 pending，无失败结果；同一 launch 会话 23271 仍在运行，seed 1（ecf45d08ca5c09b3）已自动启动。剩余三条训练、32 条自身 best 独立评测及固定曲线门尚未完成。

68. 2026-10-11 02:18–02:19，按用户要求直接调整剩余训练的调度。原 launch 父进程 74680 在启动时已读入串行队列，不能热更新 jobs.json；核对身份后仅停止该调度父进程，保留 seed 1 原训练子进程 69116。通过现有 rpipe run-one 全流程分别启动尚未执行的 seed 2（f3d208f04c982393，进程 84836）和 seed 3（ada9dd7042544f81，进程 66100），保留原 Run ID、version、源码、数据、seed 和训练预算。确认每个 seed 只有一个进程，seed 1 继续推进至 step5600，seed 2 完成首个 step200 test，seed 3 已 prepare。设备单次快照 GPU 利用率 97%，显存 9903MiB、剩余 14133MiB；尚无失败，此快照不证明长期吞吐提升或完整稳定性。并发进程日志为 .tmp/main-exp-flow-20261010-02-seed2-concurrent.log 和 seed3-concurrent.log（仅本机临时证据）。原串行调度被主动撤下，后续须在三条 train 正式成功后统一启动 32 条 eval，再完成 Study process / report；不得依赖原父进程自动进入评估。B-022 仍开放，此次调度调整不等于容量估算器已经修复。

69. 2026-10-11 03:15，CIFAR10 ResNet18 seed 1–3 分别达到 45400、40200、40000 steps，完成 227、201、200 个完整 test 点，均达到连续 80000-step 预算的一半。只读核验 seed 1 原进程 69116 与 seed 2 / 3 会话 29007 / 73498 仍存活；已有 Loss / Accuracy mean 均有限，评测坐标连续，暂无 error。当前仍为 29 succeeded / 35 pending，无失败结果。此项确认三条并发训练持续推进，不代替完整预算、32 条自身 best 独立评测或四 seed 固定曲线门；后续评测按第 68 项接续。

70. 2026-10-11 03:45，CIFAR10 ResNet18 seed 1–3 分别达到 65400、60200、60000 steps，完成 327、301、300 个完整 test 点，均达到连续 80000-step 预算的四分之三。只读核验 seed 1 原进程 69116 与 seed 2 / 3 会话 29007 / 73498 仍存活；已有 Loss / Accuracy mean 均有限，评测坐标连续，暂无 error。当前仍为 29 succeeded / 35 pending，无失败结果。完整训练预算、32 条自身 best 独立评测及四 seed 固定曲线门尚未完成，后续评测按第 68 项接续。

71. 2026-10-11 04:08，CIFAR10 ResNet18 seed 1（ecf45d08ca5c09b3）正式 result 为 succeeded，连续完成 80000 steps 与 400 个 test 点，训练耗时 1:54:24，末点 Accuracy 93.1100%，全程 best 93.3200%。调用最终曲线验收使用的 inspect_run，只读 CPU 审计实际 latest.pt / best.pt、完整 test10000 / batch250、评测轨迹、全程 best 与正式 result 一致性，30/30 条完成训练通过，无错误。明细为 .tmp/main-exp-flow-20261010-02-completed-bundle-audit.json（仅本机临时证据，更新至 30 条）。当前 30 succeeded / 34 pending，无失败结果；seed 2 / 3 分别到 75000 / 74800 steps，尚未正式完成，32 条自身 best 独立评测和固定曲线门仍待执行。

72. 2026-10-11 04:15，CIFAR10 ResNet18 seed 2（f3d208f04c982393）与 seed 3（ada9dd7042544f81）均正式 succeeded，连续完成 80000 steps 和 400 个完整 test 点。末点 / best 分别为 92.9800% / 93.0900%、92.7800% / 92.9100%，实测训练耗时分别为 1:55:56、1:55:59。两条 run-one 会话均 exit0；32/32 条训练的实际 checkpoint bundle 审计通过。04:16 使用现有计划通过 rpipe launch --mode eval 统一启动 32 条自身 best 独立评测，未重跑训练。

73. 2026-10-11 04:16，32 条 eval 全部正式 succeeded，64 个 Run 的单体 inspect_run 审计通过，无单体错误。但 Study process 因 CIFAR10 CNN seed 0（00597cc7bfc1fb1d）的独立评测与训练 best 不一致退出：8914 / 10000 对 8915 / 10000，实际 best 路径和 step76800 均正确；其余 31 个 best 对照通过。COMPARISON.json 状态为 invalid、complete=false、final_gates_applied=false，曲线门尚未正式应用。原评测日志、result 和 tracker 保留；B-023 记录在 bugs.md。只读同 checkpoint 重评（正式 GPU topk 指标）得到 8915，重复评测及训练前向 / 反向后恢复权重均得到相同输出；根因尚未确认，不据此替换正式结果。临时诊断为 .tmp/cnn-best-eval-replay.json / .pt，完整单体审计为 .tmp/main-exp-flow-20261010-02-completed-bundle-audit.json，均只在本机提供。

74. 2026-10-11 04:21–04:24，使用原生 Flow prepare / execute、同一 best checkpoint 和隔离临时输出做数值诊断。首次默认精度八进程并发有 5 次为 8914、3 次为 8915；逐样本 logits 对照定位到 test index4970（零起始），预测在类别 2 / 3 间变化，最大 logit 差 0.005742。补齐输入及模型 SHA-256 后，第二批默认 cuDNN TF32 并发为 4 次 8914、4 次 8915；禁用 cuDNN / matmul TF32 的 FP32 对照八次均为 8915。16 次输入和模型 hash 各自一致，deterministic / cudnn_deterministic=true、benchmark=false，matmul TF32 默认已关闭。证据收敛到允许 cuDNN TF32 时的设备数值执行差异，具体 kernel 选择仍待跟踪；FP32 诊断不等于原矩阵已经通过或新精度训练已验收。正式摘要证据在 [EVAL_DIAGNOSTIC.json](EVAL_DIAGNOSTIC.json)，原始 logits 和日志只在本机 .tmp/native-eval-diagnostics 下；不覆盖原失败，不修改固定门。

## 四、2026-10-10

1. MNIST 从官方文件准备。CIFAR10 官方下载速度较慢，主动停止该进程后，通过 prepare_data.py --cache-root 复用本机此前探针缓存；所有文件逐一核对固定 SHA-256，未使用旧 processed 结果作为新训练成果。
2. 已核对 split：MNIST train60000 / test10000，CIFAR10 train50000 / test10000。下载的未完成 part 保留在忽略的 raw 目录，不作为有效数据。
3. main_probe 八组合先通过后，在实际提交 e682f25 启动 recipe.py preflight。前置门输出为 .tmp/historical-preflight-20261009-80e740a9（UTC 命名日期与本机执行日期不同；本机执行为2026-10-10）。该进程仍在运行，完整八格判定待验证；输出为本机临时证据，不随 Git clone 提供。
4. 2026-10-10 06:39 前置门正常结束，8/8 passed，PREFLIGHT.json 写入本 Study docs。启动前逐项核对 source_hashes 与当前 source_files 相等，未复用旧前置门。
5. 64 个 Run 已展开，自动调度14个 wait组，先32 train，再32 eval；首组8个 linear。make 的12h43m为估计，不是实测完成时间。2026-10-10 06:41 统一 launch 已启动，实际提交保持 e682f25。首组 MNIST seed0–3 已写 step200 完整 test 和 step400 train，CIFAR10 seed0–3 已完成 prepare；当前0 succeeded / 0 failed，仅表示预算尚未跑满，不将活动 Run 计为完成。
6. launch 父日志位于 .tmp/main-exp-flow-20261010-01-launch.log（本机临时文件）；每 Run 的正式日志在 runs/<id>/assets/logs/run.log。首组八个 Run 为 MNIST：9aeace4808c332c3、07b8220973f650e4、e91a005e9858dd8f、2e1963e326caecbb；CIFAR10：3d616b38a1653b30、5847eb6a49e9d9b2、81372c17eb918c62、d9e4a3f801c9b76b。原始训练日志与 checkpoint 保留，不因失败放宽固定门。

7. 2026-10-10 06:45 核对首组实际 scalars.jsonl：MNIST 四 seed 已完成46–47个 test点（step9200–9400），CIFAR10 四 seed 已完成18个 test点（step3600）。已完成坐标均从200起按200连续，无缺点；已有 mean 均为有限数，未见重复 start 或 error。launch 进程仍存活，0 succeeded / 0 failed；这些是活动进度，不能代表80k或曲线门通过。临时只读检查快照为 .tmp/main-exp-progress.json，不随 Git clone 提供。

8. 2026-10-10 06:51，首组 MNIST 四 seed 超过20000 steps，已完成111–112个 test点（step22200–22400）；CIFAR10 四 seed 已完成43个 test点（step8600）。同一 launch 进程仍存活，未见 error、非有限 mean、缺失评测坐标或重复 start；尚无完成 Run，完整预算与终验继续等待。

9. 2026-10-10 06:57，首组 MNIST 四 seed 完成209–210个 test点（step41800–42000），均已超过80k预算的一半；CIFAR10 四 seed 完成79个 test点（step15800）。同一 launch 进程仍存活，八条活动训练没有 error、非有限 mean、缺点或重复 start；尚无完成 Run，不作完整质量结论。

10. 2026-10-10 07:01，首组 CIFAR10 四 seed 达到20000–20200 steps，完成100–101个 test点；MNIST 达到53600–54000 steps，完成268–270个 test点。同一 launch 进程仍存活，八条活动训练均没有 error、非有限 mean、缺点或重复 start。当前仍无完成 Run，后续按完整预算与固定曲线门验收。

11. 2026-10-10 07:09，首批 MNIST linear seed0–3 的正式 result 均为 succeeded；逐 Run 核对 latest checkpoint step=80000，test Loss / Accuracy 各有400个点，坐标按200连续至80000，数值均有限，best/model.pt 均存在。末点评测 Accuracy 按 seed 为92.6500%、92.6900%、92.6800%、92.6800%；这些是训练末点，独立 own-best eval 与固定曲线对照门仍待后续执行。当前总计4 succeeded / 60 pending / 0 failed，CIFAR10 linear 四 seed 继续训练，未将整个 Study 判为完成。检查明细位于 .tmp/main-exp-mnist-linear-completed.json，仅为本机临时证据。

12. 2026-10-10 07:19，首批 CIFAR10 linear 四 seed 均已超过40000 steps，完成203–204个 test点（step40600–40800）。同一 launch 进程仍存活，已有 mean 均有限，评测坐标连续，无 error 或重复 start；MNIST linear 四个已完成 Run 的8万步与400点检查见上一条。当前4 succeeded / 60 pending / 0 failed，仍等待完整矩阵、独立 eval 与固定曲线门。

13. 2026-10-10 07:38，首批 CIFAR10 linear 四 seed 达到60000–60200 steps，完成300–301个 test点，均已达到8万步预算的四分之三。核对同一 launch 进程仍存活，已有 mean 均有限，坐标连续，无 error 或重复 start；当前仍为4 succeeded / 60 pending / 0 failed。此项仅记录活动进度，完整预算、独立 eval 与曲线门尚未完成。

14. 2026-10-10 07:55，首组 MNIST / CIFAR10 linear 共8个 train Run 均正式 succeeded。新增核对 CIFAR10 四 Run 的 latest checkpoint step=80000，test Loss / Accuracy 各400点，坐标按200连续至80000，数值均有限，best/model.pt 均存在；末点 Accuracy 按 seed 为40.3900%、40.5300%、40.3200%、40.2400%。检查明细为 .tmp/main-exp-cifar-linear-completed.json（本机临时证据）。同一 launch 自动进入第二 wait 组，MNIST MLP seed0–3 已到step1200，CIFAR10 MLP seed0–3 已到step400；八条新训练暂无 error、非有限 mean、缺点或重复 start。当前8 succeeded / 56 pending / 0 failed；独立 own-best eval 和完整固定曲线门尚未完成。

15. 2026-10-10 08:11，第二组 MNIST MLP 四 seed 已超过40000 steps，完成209–210个 test点（step41800–42000）；CIFAR10 MLP 四 seed 达到step16000，完成80个 test点。同一 launch 进程仍存活，八条活动训练的已有 mean 均有限，评测坐标连续，无 error 或重复 start。当前8 succeeded / 56 pending / 0 failed；活动进度不替代完整预算、独立 eval 与固定曲线门。

16. 2026-10-10 08:24，MNIST MLP seed0–3 的正式 result 均为 succeeded；逐 Run 核对 latest checkpoint step=80000，test Loss / Accuracy 各400点，坐标按200连续至80000，数值均有限，best/model.pt 均存在。末点 Accuracy 按 seed 为98.0800%、98.2600%、98.1700%、98.0900%；独立 own-best eval 与固定曲线对照门仍待后续执行。检查明细位于 .tmp/main-exp-mnist-mlp-completed.json，仅为本机临时证据。当前12 succeeded / 52 pending / 0 failed，同一 launch 进程仍存活；CIFAR10 MLP 四 seed 达到31200–31400 steps，完成156–157个 test点，暂无 error、非有限 mean、缺点或重复 start。

17. 2026-10-10 08:32，CIFAR10 MLP 四 seed 均已超过40000 steps，完成206–208个 test点（step41200–41600）。同一 launch 进程仍存活，已有 mean 均有限，评测坐标连续，无 error 或重复 start。当前12 succeeded / 52 pending / 0 failed；活动训练尚未达到完整预算，独立 eval 与固定曲线门仍待后续执行。

18. 2026-10-10 08:50，CIFAR10 MLP 四 seed 达到60200–60400 steps，完成301–302个 test点，均超过8万步预算的四分之三。同一 launch 进程仍存活，已有 mean 均有限，评测坐标连续，无 error 或重复 start。当前12 succeeded / 52 pending / 0 failed；此项仅记录活动进度，完整预算、独立 eval 与固定曲线门尚未完成。

19. 2026-10-10 09:09，CIFAR10 MLP seed 0–3 的正式 result 均为 succeeded；逐 Run 核对 latest checkpoint step=80000，test Loss / Accuracy 各 400 点，坐标按 200 连续至 80000，数值均有限，best/model.pt 均存在。末点 Accuracy 按 seed 为 63.6500%、63.2400%、63.6900%、63.2900%；检查明细为 .tmp/main-exp-cifar-mlp-completed.json（仅为本机临时证据）。当前 16 succeeded / 48 pending / 0 failed，同一 launch 进程仍存活，已自动启动第三组 CNN：MNIST seed 0–3、CIFAR10 seed 0–1；首次完整 test 尚未写入，暂无 error。独立 own-best eval 和固定曲线门尚未完成，不以训练末点代替完整验收。

20. 2026-10-10 09:28，第三组 MNIST CNN seed 0–3 已超过 40000 steps，完成 205–206 个 test 点（step41000–41200）；CIFAR10 CNN seed 0–1 达到 step17600，完成 88 个 test 点。同一 launch 进程仍存活，六条活动训练的已有 mean 均有限，评测坐标连续，无 error 或重复 start。当前 16 succeeded / 48 pending / 0 failed；此项为活动进度，完整预算、独立 own-best eval 与固定曲线门尚未完成。

21. 2026-10-10 09:36，MNIST CNN seed 0–3 达到 60200–60400 steps，完成 301–302 个 test 点，均超过 80000-step 预算的四分之三；CIFAR10 CNN seed 0–1 达到 step25800，完成 129 个 test 点。同一 launch 进程仍存活，已有 mean 均有限，评测坐标连续，无 error 或重复 start。当前 16 succeeded / 48 pending / 0 failed；完整训练预算、独立 own-best eval 和固定曲线门尚未完成。

22. 2026-10-10 09:46，MNIST CNN seed 0–3 的正式 result 均为 succeeded；逐 Run 核对 latest checkpoint step=80000，test Loss / Accuracy 各 400 点，坐标按 200 连续至 80000，数值均有限，best/model.pt 均存在。末点 Accuracy 按 seed 为 99.3700%、99.4000%、99.4800%、99.4100%；检查明细为 .tmp/main-exp-mnist-cnn-completed.json（仅为本机临时证据）。当前 20 succeeded / 44 pending / 0 failed，同一 launch 进程仍存活；第三组 CIFAR10 CNN seed 0–1 达到 34600–34800 steps，评测坐标连续，已有 mean 均有限，无 error 或重复 start。剩余训练、独立 own-best eval 和固定曲线门仍待完成，不以训练末点代替完整验收。

23. 2026-10-10 09:52，第三组 CIFAR10 CNN seed 0–1 均达到 step41200，完成 206 个 test 点，超过 80000-step 预算的一半。同一 launch 进程仍存活，已有 mean 均有限，评测坐标连续，无 error 或重复 start。当前 20 succeeded / 44 pending / 0 failed；其余 seed、剩余模型训练、独立 own-best eval 与固定曲线门尚未完成。

24. 2026-10-10 10:11，第三组 CIFAR10 CNN seed 0–1 达到 60600–60800 steps，完成 303–304 个 test 点，均超过 80000-step 预算的四分之三。同一 launch 进程仍存活，已有 mean 均有限，评测坐标连续，无 error 或重复 start。当前 20 succeeded / 44 pending / 0 failed；此项仅为活动进度，完整预算、其余训练、独立 own-best eval 和固定曲线门仍待完成。

25. 2026-10-10 10:30，CIFAR10 CNN seed 0–1 的正式 result 均为 succeeded；逐 Run 核对 latest checkpoint step=80000，test Loss / Accuracy 各 400 点，坐标按 200 连续至 80000，数值均有限，best/model.pt 均存在。末点 Accuracy 按 seed 为 88.8600%、89.2000%；检查明细为 .tmp/main-exp-cifar-cnn-first-completed.json（仅为本机临时证据）。当前 22 succeeded / 42 pending / 0 failed，同一 launch 进程仍存活，已自动进入第四组 CIFAR10 CNN seed 2–3（Run 3f83046826175282、05818c56ff0d46db），均已完成 step600 的第三个 test 点，暂无 error、非有限 mean、缺点或重复 start。剩余训练、独立 own-best eval 和固定曲线门尚未完成。

26. 2026-10-10 11:02，第四组 CIFAR10 CNN seed 2–3 均达到 step40800，完成 204 个 test 点，超过 80000-step 预算的一半。同一 launch 进程仍存活，已有 mean 均有限，评测坐标连续，无 error 或重复 start。当前 22 succeeded / 42 pending / 0 failed；此项仅为活动进度，剩余训练、32 条自身 best 独立评测和固定曲线门尚未完成。

27. 2026-10-10 11:17，第四组 CIFAR10 CNN seed 2–3 达到 60000–60200 steps，完成 300–301 个 test 点，均达到 80000-step 预算的四分之三。同一 launch 进程仍存活，已有 mean 均有限，评测坐标连续，无 error 或重复 start。当前 22 succeeded / 42 pending / 0 failed；此项仅为活动进度，完整预算、剩余训练、32 条自身 best 独立评测和固定曲线门尚未完成。

28. 2026-10-10 11:33，CIFAR10 CNN seed 2–3 的正式 result 均为 succeeded；逐 Run 核对 latest checkpoint step=80000，test Loss / Accuracy 各 400 点，坐标按 200 连续至 80000，数值均有限，best/model.pt 均存在。末点 Accuracy 按 seed 为 88.5700%、88.9800%；检查明细为 .tmp/main-exp-cifar-cnn-second-completed.json（仅为本机临时证据）。至此 linear、MLP 和 CNN 共 24 个 train Run 已完成。当前 24 succeeded / 40 pending / 0 failed，同一 launch 进程仍存活，已自动进入第五组 MNIST ResNet18 seed 0–3，均已完成 step200 的首个 test 点，暂无 error、非有限 mean、缺点或重复 start。剩余 ResNet18 训练、32 条自身 best 独立评测和固定曲线门尚未完成，不以训练末点代替完整验收。

29. 2026-10-10 12:28，第五组 MNIST ResNet18 seed 0–3 达到 40400–40800 steps，完成 202–204 个 test 点，均超过 80000-step 预算的一半。同一 launch 进程仍存活，已有 mean 均有限，评测坐标连续，无 error 或重复 start。当前 24 succeeded / 40 pending / 0 failed；此项仅为活动进度，完整训练预算、后续 CIFAR10 ResNet18 训练、32 条自身 best 独立评测和固定曲线门尚未完成。

30. 2026-10-10 12:55，第五组 MNIST ResNet18 seed 0–3 达到 60400–61000 steps，完成 302–305 个 test 点，均超过 80000-step 预算的四分之三。同一 launch 进程仍存活，已有 mean 均有限，评测坐标连续，无 error 或重复 start。当前 24 succeeded / 40 pending / 0 failed；此项仅为活动进度，完整训练预算、后续 CIFAR10 ResNet18 训练、32 条自身 best 独立评测和固定曲线门尚未完成。

31. 2026-10-10 13:22，MNIST ResNet18 seed 0–3 的正式 result 均为 succeeded；逐 Run 核对 latest checkpoint step=80000，test Loss / Accuracy 各 400 点，坐标按 200 连续至 80000，数值均有限，best/model.pt 均存在。末点 Accuracy 按 seed 为 99.5800%、99.6000%、99.5300%、99.5500%；检查明细为 .tmp/main-exp-mnist-resnet-completed.json（仅为本机临时证据）。当前 28 succeeded / 36 pending / 0 failed，同一 launch 进程仍存活，已自动进入 CIFAR10 ResNet18 seed 0（Run 837ba1f4d0d49513），已完成 step200 的首个 test 点，暂无 error。剩余 CIFAR10 ResNet18 训练、32 条自身 best 独立评测及固定曲线门仍待完成，不以训练末点代替完整验收。

32. 2026-10-10 14:04，CIFAR10 ResNet18 seed 0（Run 837ba1f4d0d49513）达到 step40600，完成 203 个 test 点，超过 80000-step 预算的一半。同一 launch 进程仍存活，已有 mean 均有限，评测坐标连续，无 error 或重复 start。当前 28 succeeded / 36 pending / 0 failed；此项仅为活动进度，完整预算、后续 seed 1–3、32 条自身 best 独立评测和固定曲线门尚未完成。

33. 2026-10-10 14:26，CIFAR10 ResNet18 seed 0（Run 837ba1f4d0d49513）达到 step60400，完成 302 个 test 点，超过 80000-step 预算的四分之三。同一 launch 进程仍存活，已有 mean 均有限，评测坐标连续，无 error 或重复 start。当前 28 succeeded / 36 pending / 0 failed；此项仅为活动进度，完整预算、后续 seed 1–3、32 条自身 best 独立评测和固定曲线门尚未完成。

34. 2026-10-10 14:46，CIFAR10 ResNet18 seed 0（Run 837ba1f4d0d49513）的正式 result 为 succeeded；核对 latest checkpoint step=80000，test Loss / Accuracy 各 400 点，坐标按 200 连续至 80000，数值均有限，best/model.pt 存在。末点 Accuracy 为 92.7600%，训练期间 best Accuracy 为 93.0000%；检查明细为 .tmp/main-exp-cifar-resnet-seed0-completed.json（仅为本机临时证据）。当前 29 succeeded / 35 pending / 0 failed，同一 launch 进程仍存活，已自动启动 seed 1（Run bd80210a90d99c46），首次完整 test 尚未写入，暂无 error。剩余 seed 1–3、32 条自身 best 独立评测及固定曲线门仍待完成，单 seed 末点不代替四 seed 聚合验收。

35. 2026-10-10 15:30，CIFAR10 ResNet18 seed 1（Run bd80210a90d99c46）达到 step40600，完成 203 个 test 点，超过 80000-step 预算的一半。同一 launch 进程仍存活，已有 mean 均有限，评测坐标连续，无 error 或重复 start。当前 29 succeeded / 35 pending / 0 failed；此项仅为活动进度，完整预算、后续 seed 2–3、32 条自身 best 独立评测和固定曲线门尚未完成。

36. 2026-10-10 15:51，CIFAR10 ResNet18 seed 1（Run bd80210a90d99c46）达到 step60000，完成 300 个 test 点，达到 80000-step 预算的四分之三。同一 launch 进程仍存活，已有 mean 均有限，评测坐标连续，无 error 或重复 start。当前 29 succeeded / 35 pending / 0 failed；此项仅为活动进度，完整预算、后续 seed 2–3、32 条自身 best 独立评测和固定曲线门尚未完成。

37. 2026-10-10 16:15，CIFAR10 ResNet18 seed 1（Run bd80210a90d99c46）的正式 result 为 succeeded；核对 latest checkpoint step=80000，test Loss / Accuracy 各 400 点，坐标按 200 连续至 80000，数值均有限，best/model.pt 存在。末点 Accuracy 为 92.9000%，训练期间 best Accuracy 为 93.0300%；检查明细为 .tmp/main-exp-cifar-resnet-seed1-completed.json（仅为本机临时证据）。当前 30 succeeded / 34 pending / 0 failed，同一 launch 进程仍存活，已自动进入 seed 2（Run d377dbb7570e248b），完成 step200 的首个 test 点，暂无 error。剩余 seed 2–3、32 条自身 best 独立评测及固定曲线门仍待完成，不以两条 seed 的末点代替四 seed 聚合验收。

38. 2026-10-10 17:22，原 launch 会话句柄 51952 已不存在，系统进程与 GPU 列表未见对应训练进程。seed 2（Run d377dbb7570e248b）的日志最后为 16:37:46、step17400，共 87 个完整 test 点；无正式 result，不能记为 succeeded，也不能从无 error 日志推断正常结束。中断原因尚未确定，未发现足以归因代码 Bug 的证据。30 个已完成训练保留，seed 2 未完成，seed 3 与 32 个 eval 未执行。按 PLAN 的连续训练要求，保留原 Run、index、provenance 和前置门副本（.tmp/main-exp-flow-20261010-01-interrupted，仅本机证据），改用新 version main-exp-flow-20261010-02 从头执行 64 Run，重新运行同设备 600-step 八格前置门；不改变配方、预算、seed 或固定门限。

39. 2026-10-10 17:33，新版本 main-exp-flow-20261010-02 的 preflight 进程正常退出 0；600-step / eval200 八组全部 passed，参数最大差均为 0，报告 source_hashes 与当前源码逐项一致。make 正常退出 0，核对 64 个新 Run ID 均属于版本 02，与版本 01 无重叠且无已有 result；按原计划完整展开 32 train 与 32 eval、14 个 wait group。launch 已启动，原始输出为 .tmp/main-exp-flow-20261010-02-launch.log（仅本机证据）。实际代码提交仍为 e682f25，experiment_config.yaml 的 version 为本次未提交声明改动，依赖、设备、配方、seed、预算及固定门限均沿用计划。此前 30 条成功训练仅作保留证据，不计入新版本终验。

40. 2026-10-10 17:51，新版本 main-exp-flow-20261010-02 的 MNIST linear seed 0–3 达到 40600–41000 steps，完成 203–205 个 test 点，均超过 80000-step 预算的一半；CIFAR10 linear seed 0–3 均达到 step16000，完成 80 个 test 点。同一 launch 会话 23271 仍在运行，已有 Loss / Accuracy mean 均有限，评测坐标连续，暂无 error。当前版本 64 个 Run 尚无正式成功结果；此项仅记录活动进度，完整预算、32 条自身 best 独立评测及固定曲线门尚未完成。进度明细为 .tmp/main-exp-progress.json（仅本机临时证据，后续观察会覆盖）。

41. 2026-10-10 18:00，新版本 main-exp-flow-20261010-02 的 MNIST linear seed 0–3 达到 60400–60600 steps，完成 302–303 个 test 点，均超过 80000-step 预算的四分之三；CIFAR10 linear seed 0–3 均达到 step23800，完成 119 个 test 点。同一 launch 会话 23271 仍在运行，已有 Loss / Accuracy mean 均有限，评测坐标连续，暂无 error。当前版本仍为 64 pending，无正式成功结果；此项仅为活动进度，完整预算、32 条自身 best 独立评测及固定曲线门尚未完成。

42. 2026-10-10 18:10，新版本 main-exp-flow-20261010-02 的 MNIST linear seed 0–3 正式 result 均为 succeeded；逐 Run 核对 latest checkpoint step=80000，test Loss / Accuracy 各 400 点，坐标按 200 连续至 80000，数值均有限，best/model.pt 均存在。末点 Accuracy 按 seed 为 92.6500%、92.6900%、92.6800%、92.6800%；检查明细为 .tmp/main-exp-flow-20261010-02-mnist-linear-completed.json（仅本机临时证据）。当前新版本 4 succeeded / 60 pending，无失败结果，同一 launch 会话 23271 仍在运行；CIFAR10 linear 四 seed 均到 step31800，已有 159 个完整 test 点，暂无 error。其余训练、32 条自身 best 独立评测及固定曲线门尚未完成，四条训练完成不代替整套验收。

43. 2026-10-10 18:19，新版本 main-exp-flow-20261010-02 的 CIFAR10 linear seed 0–3 达到 40200–40400 steps，完成 201–202 个 test 点，均超过 80000-step 预算的一半。同一 launch 会话 23271 仍在运行，已有 Loss / Accuracy mean 均有限，评测坐标连续，暂无 error。当前新版本 4 succeeded / 60 pending，无失败结果；此项仅为活动进度，完整训练预算、后续模型训练、32 条自身 best 独立评测及固定曲线门尚未完成。

44. 2026-10-10 18:39，新版本 main-exp-flow-20261010-02 的 CIFAR10 linear seed 0–3 均达到 step60600，完成 303 个 test 点，超过 80000-step 预算的四分之三。同一 launch 会话 23271 仍在运行，已有 Loss / Accuracy mean 均有限，评测坐标连续，暂无 error。当前新版本 4 succeeded / 60 pending，无失败结果；此项仅为活动进度，完整训练预算、后续模型训练、32 条自身 best 独立评测及固定曲线门尚未完成。

45. 2026-10-10 18:56，新版本 main-exp-flow-20261010-02 的 CIFAR10 linear seed 0–3 正式 result 均为 succeeded；逐 Run 核对 latest checkpoint step=80000，test Loss / Accuracy 各 400 点，坐标按 200 连续至 80000，数值均有限，best/model.pt 均存在。末点 Accuracy 按 seed 为 40.3900%、40.5300%、40.3200%、40.2400%；检查明细为 .tmp/main-exp-flow-20261010-02-cifar-linear-completed.json（仅本机临时证据）。至此新版本 8 条 linear 训练完成，当前 8 succeeded / 56 pending，无失败结果。同一 launch 会话 23271 仍在运行，已自动进入 MNIST / CIFAR10 MLP 四 seed 训练，首次完整 test 尚未写入，暂无 error。后续模型训练、32 条自身 best 独立评测及固定曲线门尚未完成，训练末点不代替整套验收。

46. 2026-10-10 19:12，新版本 main-exp-flow-20261010-02 的 MNIST MLP seed 0–3 达到 40600–40800 steps，完成 203–204 个 test 点，均超过 80000-step 预算的一半；CIFAR10 MLP 四 seed 均到 step15400，完成 77 个 test 点。同一 launch 会话 23271 仍在运行，已有 Loss / Accuracy mean 均有限，评测坐标连续，暂无 error。当前新版本 8 succeeded / 56 pending，无失败结果；此项仅为活动进度，完整训练预算、后续模型训练、32 条自身 best 独立评测及固定曲线门尚未完成。

47. 2026-10-10 19:20，新版本 main-exp-flow-20261010-02 的 MNIST MLP seed 0–3 达到 60800–61000 steps，完成 304–305 个 test 点，均超过 80000-step 预算的四分之三；CIFAR10 MLP 四 seed 达到 23200–23400 steps，完成 116–117 个 test 点。同一 launch 会话 23271 仍在运行，已有 Loss / Accuracy mean 均有限，评测坐标连续，暂无 error。当前新版本 8 succeeded / 56 pending，无失败结果；此项仅为活动进度，完整训练预算、后续模型训练、32 条自身 best 独立评测及固定曲线门尚未完成。

48. 2026-10-10 19:29，新版本 main-exp-flow-20261010-02 的 MNIST MLP seed 0–3 正式 result 均为 succeeded；逐 Run 核对 latest checkpoint step=80000，test Loss / Accuracy 各 400 点，坐标按 200 连续至 80000，数值均有限，best/model.pt 均存在。末点 Accuracy 按 seed 为 98.0800%、98.2600%、98.1700%、98.0900%；检查明细为 .tmp/main-exp-flow-20261010-02-mnist-mlp-completed.json（仅本机临时证据）。当前新版本 12 succeeded / 52 pending，无失败结果；同一 launch 会话 23271 仍在运行，CIFAR10 MLP 四 seed 达到 31200–31400 steps，完成 156–157 个 test 点，暂无 error。其余训练、32 条自身 best 独立评测及固定曲线门尚未完成，训练末点不代替整套验收。

49. 2026-10-10 19:39，新版本 main-exp-flow-20261010-02 的 CIFAR10 MLP seed 0–3 达到 40000–40200 steps，完成 200–201 个 test 点，均达到 80000-step 预算的一半。同一 launch 会话 23271 仍在运行，已有 Loss / Accuracy mean 均有限，评测坐标连续，暂无 error。当前新版本 12 succeeded / 52 pending，无失败结果；此项仅为活动进度，完整训练预算、后续模型训练、32 条自身 best 独立评测及固定曲线门尚未完成。

50. 2026-10-10 20:01，新版本 main-exp-flow-20261010-02 的 CIFAR10 MLP seed 0–3 达到 60400–60600 steps，完成 302–303 个 test 点，均超过 80000-step 预算的四分之三。同一 launch 会话 23271 仍在运行，已有 Loss / Accuracy mean 均有限，评测坐标连续，暂无 error。当前新版本 12 succeeded / 52 pending，无失败结果；此项仅为活动进度，完整训练预算、后续模型训练、32 条自身 best 独立评测及固定曲线门尚未完成。

51. 2026-10-10 20:19，新版本 main-exp-flow-20261010-02 的 CIFAR10 MLP seed 0–3 正式 result 均为 succeeded；逐 Run 核对 latest checkpoint step=80000，test Loss / Accuracy 各 400 点，坐标按 200 连续至 80000，数值均有限，best/model.pt 均存在。末点 Accuracy 按 seed 为 63.6500%、63.2400%、63.6900%、63.2900%；检查明细为 .tmp/main-exp-flow-20261010-02-cifar-mlp-completed.json（仅本机临时证据）。至此新版本 16 条 linear / MLP 训练完成，当前 16 succeeded / 48 pending，无失败结果。同一 launch 会话 23271 仍在运行，已自动进入 CNN 训练；首次完整 test 尚未写入，暂无 error。后续模型训练、32 条自身 best 独立评测及固定曲线门尚未完成，训练末点不代替整套验收。

52. 2026-10-10 20:38，新版本 main-exp-flow-20261010-02 的 MNIST CNN seed 0–3 达到 41400–41600 steps，完成 207–208 个 test 点，均超过 80000-step 预算的一半；CIFAR10 CNN 当前活动 seed 0–1 均到 step17600，完成 88 个 test 点。同一 launch 会话 23271 仍在运行，已有 Loss / Accuracy mean 均有限，评测坐标连续，暂无 error。当前新版本 16 succeeded / 48 pending，无失败结果；此项仅为活动进度，完整训练预算、后续模型训练、32 条自身 best 独立评测及固定曲线门尚未完成。

53. 2026-10-10 20:46，新版本 main-exp-flow-20261010-02 的 MNIST CNN seed 0–3 达到 61800–62000 steps，完成 309–310 个 test 点，均超过 80000-step 预算的四分之三；CIFAR10 CNN 当前活动 seed 0–1 均到 step26200，完成 131 个 test 点。同一 launch 会话 23271 仍在运行，已有 Loss / Accuracy mean 均有限，评测坐标连续，暂无 error。当前新版本 16 succeeded / 48 pending，无失败结果；此项仅为活动进度，完整训练预算、后续模型训练、32 条自身 best 独立评测及固定曲线门尚未完成。

54. 2026-10-10 20:54，新版本 main-exp-flow-20261010-02 的 MNIST CNN seed 0–3 正式 result 均为 succeeded；逐 Run 核对 latest checkpoint step=80000，test Loss / Accuracy 各 400 点，坐标按 200 连续至 80000，数值均有限，best/model.pt 均存在。末点 Accuracy 按 seed 为 99.3900%、99.4700%、99.3600%、99.4100%；检查明细为 .tmp/main-exp-flow-20261010-02-mnist-cnn-completed.json（仅本机临时证据）。当前新版本 20 succeeded / 44 pending，无失败结果；同一 launch 会话 23271 仍在运行，CIFAR10 CNN 当前活动 seed 0–1 均到 step34800，完成 174 个 test 点，暂无 error。其余训练、32 条自身 best 独立评测及固定曲线门尚未完成，训练末点不代替整套验收。

55. 2026-10-10 20:59，新版本 main-exp-flow-20261010-02 的 CIFAR10 CNN 当前活动 seed 0–1 均达到 step40400，完成 202 个 test 点，超过 80000-step 预算的一半；seed 2–3 尚未进入活动训练。同一 launch 会话 23271 仍在运行，已有 Loss / Accuracy mean 均有限，评测坐标连续，暂无 error。当前新版本 20 succeeded / 44 pending，无失败结果；此项仅为活动进度，完整训练预算、后续模型训练、32 条自身 best 独立评测及固定曲线门尚未完成。

56. 2026-10-10 21:18，新版本 main-exp-flow-20261010-02 的 CIFAR10 CNN 当前活动 seed 0–1 均达到 step60400，完成 302 个 test 点，超过 80000-step 预算的四分之三；seed 2–3 尚未进入活动训练。同一 launch 会话 23271 仍在运行，已有 Loss / Accuracy mean 均有限，评测坐标连续，暂无 error。当前新版本 20 succeeded / 44 pending，无失败结果；此项仅为活动进度，完整训练预算、后续模型训练、32 条自身 best 独立评测及固定曲线门尚未完成。

57. 2026-10-10 21:37，新版本 main-exp-flow-20261010-02 的 CIFAR10 CNN seed 0–1 正式 result 均为 succeeded；逐 Run 核对 latest checkpoint step=80000，test Loss / Accuracy 各 400 点，坐标按 200 连续至 80000，数值均有限，best/model.pt 均存在。末点 Accuracy 按 seed 为 89.0400%、89.0000%；检查明细为 .tmp/main-exp-flow-20261010-02-cifar-cnn-seeds-0-1-completed.json（仅本机临时证据）。当前新版本 22 succeeded / 42 pending，无失败结果；同一 launch 会话 23271 仍在运行，CIFAR10 CNN seed 2–3 已自动启动并各完成首个 step200 test 点，暂无 error。其余训练、32 条自身 best 独立评测及固定曲线门尚未完成，两个 seed 的训练末点不代替四 seed 整套验收。

58. 2026-10-10 22:15，新版本 main-exp-flow-20261010-02 的 CIFAR10 CNN 当前活动 seed 2–3 均达到 step40400，完成 202 个 test 点，超过 80000-step 预算的一半。同一 launch 会话 23271 仍在运行，已有 Loss / Accuracy mean 均有限，评测坐标连续，暂无 error。当前新版本 22 succeeded / 42 pending，无失败结果；此项仅为活动进度，完整训练预算、后续 ResNet 训练、32 条自身 best 独立评测及固定曲线门尚未完成。

59. 2026-10-10 22:33，新版本 main-exp-flow-20261010-02 的 CIFAR10 CNN 当前活动 seed 2–3 达到 60800、60600 steps，完成 304、303 个 test 点，均超过 80000-step 预算的四分之三。同一 launch 会话 23271 仍在运行，已有 Loss / Accuracy mean 均有限，评测坐标连续，暂无 error。当前新版本 22 succeeded / 42 pending，无失败结果；此项仅为活动进度，完整训练预算、后续 ResNet 训练、32 条自身 best 独立评测及固定曲线门尚未完成。

60. 2026-10-10 22:53，新版本 main-exp-flow-20261010-02 的 CIFAR10 CNN seed 0–3 正式 result 均为 succeeded；逐 Run 核对 latest checkpoint step=80000，test Loss / Accuracy 各 400 点，坐标按 200 连续至 80000，数值均有限，best/model.pt 均存在。末点 Accuracy 按 seed 为 89.0400%、89.0000%、88.6400%、88.9200%；检查明细为 .tmp/main-exp-flow-20261010-02-cifar-cnn-completed.json（仅本机临时证据）。至此新版本 24 条 linear / MLP / CNN 训练完成，当前 24 succeeded / 40 pending，无失败结果。同一 launch 会话 23271 仍在运行，MNIST ResNet18 四 seed 已自动启动并各完成至 step400 的两个 test 点，暂无 error。后续 ResNet 训练、32 条自身 best 独立评测及固定曲线门尚未完成，训练末点不代替整套验收。

61. 2026-10-10 23:48，新版本 main-exp-flow-20261010-02 的 MNIST ResNet18 seed 0–3 分别达到 40000、40800、40800、40600 steps，完成 200、204、204、203 个 test 点，均达到 80000-step 预算的一半。同一 launch 会话 23271 仍在运行，已有 Loss / Accuracy mean 均有限，评测坐标连续，暂无 error。当前新版本 24 succeeded / 40 pending，无失败结果；此项仅为活动进度，完整 ResNet 训练预算、32 条自身 best 独立评测及固定曲线门尚未完成。

数据、checkpoint、日志及临时前置门输出按仓库规则保留在本机，不随 Git clone 提供。旧数值结果不替代本次验收；失败不通过修改固定门限消除。
