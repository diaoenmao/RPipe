# main_exp 复跑记录

## 一、摘要

本次按 [PLAN.md](PLAN.md) 复跑 Flow 重构后的 32 条连续 80000-step 训练与 32 条自身 best 独立评测。main-exp-flow-20261010-02 全部执行成功且单体 checkpoint / tracker 审计通过，但 Study 终验 invalid：CIFAR10 CNN seed0 独立评测少一个正确样本。设备诊断证明相同输入 / 模型在默认 cuDNN TF32 条件下存在预测差异，串行五次也有一次失败，因此停止串行恢复，未替换正式结果。完整旧报告和来源证据保留在 [version02](versions/main-exp-flow-20261010-02/STUDY_REPORT.md)。

当前声明 main-exp-flow-20261011-03，显式关闭 cuDNN / matmul TF32，两侧前置对照与正式链使用一致精度。runtime 针对性 13 passed，核心回归 295 passed / 40 deselected；归档导入副作用修复提交 09e495b 的 600-step 八格前置门已通过，完整矩阵于 2026-10-11 05:03 启动，round=3；全部 64 个 Run 的真实设备开关核验通过。19:48 全部 32 train / 32 eval 成功，原生 Flow exit 0，完整性核查与固定四 seed 终验通过。数据、预算、seed 与固定曲线门不变；旧训练不能充当新精度基线。方案与停止条件见 [eval_recovery.md](eval_recovery.md)。

## 二、执行条件

| **项目** | **本次值** |
| --- | --- |
| Flow 基线 | PR #25 合入 dev，d73b217 |
| 当前 version | main-exp-flow-20261011-03；版本 01 中断 / 02 数值终验失败的证据保留 |
| 原代码 | 4ccb28d0496110253e9f8e3f3df658853f07996b |
| 设备与环境 | NVIDIA GeForce RTX 5090 D v2；Python 3.13.9，Torch 2.11.0+cu130，torchvision 0.26.0+cu130 |
| 数据 | [DATA_MANIFEST.json](DATA_MANIFEST.json)，16/16 原始文件匹配固定 SHA-256 |
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

74. 2026-10-11 04:21–04:24，使用原生 Flow prepare / execute、同一 best checkpoint 和隔离临时输出做数值诊断。首次默认精度八进程并发有 5 次为 8914、3 次为 8915；逐样本 logits 对照定位到 test index4970（零起始），预测在类别 2 / 3 间变化，最大 logit 差 0.005742。补齐输入及模型 SHA-256 后，第二批默认 cuDNN TF32 并发为 4 次 8914、4 次 8915；禁用 cuDNN / matmul TF32 的 FP32 对照八次均为 8915。16 次输入和模型 hash 各自一致，deterministic / cudnn_deterministic=true，matmul TF32 默认已关闭。更正：当时配置声明 benchmark=false，但原始诊断实际值为 true，归档导入覆盖配置的原因见第 78 项。证据收敛到允许 cuDNN TF32 时的设备数值执行差异，具体 kernel 选择仍待跟踪；FP32 诊断不等于原矩阵已经通过或新精度训练已验收。正式摘要证据在 [EVAL_DIAGNOSTIC.json](EVAL_DIAGNOSTIC.json)，原始 logits 和日志只在本机 .tmp/native-eval-diagnostics 下；不覆盖原失败，不修改固定门。

75. 2026-10-11，原精度的五次独立新进程串行原生评测依次为 8915、8915、8915、8915、8914，预先声明的串行恢复停止条件触发。没有移开或替换正式 eval；version02 保持 invalid。终验失败、来源、配置、全部数字与图归档在 docs/versions/main-exp-flow-20261010-02/，原 Run / checkpoint 继续留在本机 runs 下；归档文件逐字节核对并记录 SHA-256。没有以再次重试获得通过。

76. 2026-10-11，先明确 [精度方案](eval_recovery.md) 再实现：System runtime 增加可选 cudnn_allow_tf32 / matmul_allow_tf32 布尔配置，构造对象前应用并记录实际值，未配置保留 Torch 当前值，非法类型拒绝。main_exp 与 main_probe 两侧明确 FP32，prepare 检查声明和前置证据。runtime 针对性回归 13 passed、核心回归 295 passed / 40 deselected。声明 main-exp-flow-20261011-03 / main-probe-flow-20261011-03，计划在真实前置数值门通过后执行新矩阵，以 --round 3 使用已验证的三任务并发；不改变原始数据、seed、预算和质量门。B-023 在完整复验前仍开放，B-021 / B-022 未在此次精度改动中修复。

77. 2026-10-11，新精度源码提交 c9c782b5d5bb62da4e5e2292e96e842740c24f53，CPU Flow 集成 / CLI 回归 37 passed / 298 deselected。main_probe FP32 version03 八组合真实六阶段链与 Study 终验全部通过，逐段参数最大差 0、train / test Loss 最大差 4.44e-16、独立 eval Loss 差 0。main_exp 已 make 64 个新 Run ID、与 version02 无重叠，各配置显式关闭两个 TF32 开关，round=3；600-step 八格前置门已通过 MNIST 四模型与 CIFAR10 linear，共 5/8，其他组合仍在执行，正式 80000-step 训练尚未启动。前置进程会话 45574、日志 .tmp/main-exp-flow-20261011-03-preflight.log、阶段结果 .tmp/historical-preflight-20261010-d31b2472/COMPARISON.json 均为本机临时证据；UTC 输出目录日期与本地执行日期不同。未完成八格前不 launch，不复用根目录仍保留的旧 PREFLIGHT.json。

78. 2026-10-11，c9c782 的八格 600-step 前置门全部通过，证据保留在 [PREFLIGHT_C9C782.json](PREFLIGHT_C9C782.json)。启动长训前发现归档 train_model 导入设置 cudnn.benchmark=True，覆盖 Flow 已应用的 false；前置门在导入后应用 runtime，未暴露正式链的顺序问题。修复 Archive 导入副作用，prepare 检查并保存真实设备开关，register 核对前置源码 hash。实际归档导入 false / true 两种输入均保留，实际开关异常被拒绝，归档字节不变，见 [ARCHIVE_RUNTIME_REGRESSION.json](ARCHIVE_RUNTIME_REGRESSION.json)。同旧 checkpoint、TF32 允许且实际 benchmark=false 的八次并发诊断均为 8914，输入 hash 一致，见 [BENCHMARK_ISOLATION_DIAGNOSTIC.json](BENCHMARK_ISOLATION_DIAGNOSTIC.json)；这证明该受控条件下八次一致，不能证明旧训练 best 的 8915 已恢复。第 74 项及旧版快照中 benchmark=false 的文字描述不准确，以原始实际值 true 为准；旧快照保留当时记录，不改写其 hash。version03 尚无正式 result，重新冻结未执行计划并复验新源码后再启动完整矩阵，B-023 仍开放。

79. 2026-10-11，归档导入与实际开关修复提交为 09e495b2d006caa197bb41e2773d47b817a4bd46。未执行的 version03 已重新 make，仍为 64 个新 Run、round=3，与旧版无 ID 重叠；没有旧 Run 输出被移走或重写。旧源码成功前置门的拒绝回归通过，文档 161 个链接与旧版快照 hash 检查无错误。新的八格前置门启动于会话 72825，日志 .tmp/main-exp-flow-20261011-03-runtime-fix-preflight.log，输出 .tmp/historical-preflight-20261010-c63aa887（两者仅本机证据）；MNIST linear 已通过、参数最大差 0，其他组合仍待完成。正式长训尚未启动，须全部通过且与当前冻结来源一致才 launch。

80. 2026-10-11 05:03，09e495b 新源码的八格前置门全部通过，参数最大差均为 0，进程 72825 正常退出 0。启动前核对当前源码 hash 与 PREFLIGHT 相等、冻结来源有效、64 个新 ID 无已有 result、完整 80000-step / eval200 / test10000 配方及 round=3 计划一致。随后启动完整原生 launch，会话 13127，日志 .tmp/main-exp-flow-20261011-03-launch.log（仅本机证据）。首批 MNIST linear seed0 / 1 / 2 同时达到 step200，三个 runtime_validation.json 实际值全部匹配 deterministic=true、cudnn_deterministic=true、benchmark=false、两个 TF32=false，未见 error、缺点或重复 start。当前 COMPARISON 状态 partial，未应用终验，报告和数字入口切换到 version03；旧版完整失败报告与数字仍在 version02 快照中。

81. 2026-10-11，正式长训启动后核对实际 Flow 与已通过前置链的 seed0 前缀。MNIST linear 的 step200 / 400 / 600，共 12 个 train/test Loss / Accuracy 值完全一致，最大指标差 0；实际设备开关通过、当前来源冻结与前置源码 hash 仍匹配，见 [FLOW_PREFIX_AUDIT.json](FLOW_PREFIX_AUDIT.json)。此辅助核查只覆盖已执行的一组早期指标，其他七组尚未执行到该阶段；不替代前置参数门、完整预算或独立评估终验，不作为 B-023 关闭依据。

82. 2026-10-11 05:11，version03 首批 MNIST linear seed0 / 1 / 2 分别达到 step20200 / 20000 / 20200，完成 101 / 100 / 101 个完整 test 点，均达到连续 80000-step 预算的四分之一。launch 会话 13127 确认存活；已有 Loss / Accuracy 均有限，评测坐标连续，无重复 start 或 error，三条实际设备开关核验均通过。当前尚无正式 succeeded result；此项只记录活动训练进度，不代替完整预算、其余训练、独立 best 评估和四 seed 曲线终验。只读临时进度与设备开关摘要分别在 .tmp/main-exp-progress.json 和 .tmp/main-exp-runtime-audit.json，仅本机提供。

83. 2026-10-11 05:19，version03 首批 MNIST linear seed0 / 1 / 2 达到 step41200 / 40800 / 41400，分别完成 206 / 204 / 207 个完整 test 点，均超过 80000-step 预算的一半。同一 launch 会话 13127 仍存活，已有指标有限、评测坐标连续，无重复 start 或 error；三条实际设备条件核验通过，首组前缀指标最大差仍为 0，源码冻结与前置来源一致。按当前 index 更新 partial COMPARISON 与部分曲线，status=partial、complete=false、final_gates_applied=false；当前仅三 seed 的部分轨迹，不应用四 seed 终验。剩余完整预算、其他训练和自身 best 独立评估继续执行。

84. 2026-10-11 05:27，version03 首批 MNIST linear seed0 / 1 / 2 达到 step61800 / 61200 / 62000，完成 309 / 306 / 310 个完整 test 点，均超过 80000-step 预算的四分之三。launch 会话 13127 确认仍在运行，已有指标有限、评测坐标连续，无重复 start 或 error；首组实际 Flow 前缀指标最大差仍为 0，当前源码冻结与前置来源匹配。三条完整预算尚未结束，当前没有正式 succeeded result，其余训练、独立 best 评估及四 seed 固定曲线门继续待执行。

85. 2026-10-11 05:35，MNIST linear seed0 / 1 / 2 正式 result 全部 succeeded，均完成连续 80000 steps；使用终验 inspect_run 对真实 latest.pt / best.pt 与 tracker 核查，3/3 通过：400 个完整 test 点、最终进度、全程最高值与 best step、正式 result 与轨迹一致，无错误。正式摘要见 [CHECKPOINT_AUDIT.json](CHECKPOINT_AUDIT.json)，逐 seed 末点、best 与实际耗时见 [STUDY_REPORT.md](STUDY_REPORT.md)。同一 launch 会话 13127 自动进入下一批 MNIST linear seed3、MLP seed0 / 1；六条已 prepare Run 的实际设备开关均通过。MLP seed0 的 600-step 实际 Flow 前缀也与前置链一致，目前辅助核查 2/8 通过、24 个指标最大差 0。当前 3 succeeded / 61 pending，无失败；32 条独立评估及四 seed 终验尚未执行，单体成功不作为整个 Study 已通过。

86. 2026-10-11 05:43，第二批 MNIST linear seed3 与 MLP seed0 / 1 分别达到 step21000 / 20200 / 20200，完成 105 / 101 / 101 个完整 test 点，均超过连续 80000-step 预算的四分之一。同一 launch 会话 13127 仍在运行，已有指标有限、评测点连续，无重复 start 或 error；已 prepare 的六条 Run 实际设备条件检查通过。MNIST linear / MLP seed0 的实际 Flow 前缀辅助核查仍为 2/8，24 个指标最大差 0，当前源码冻结与前置来源匹配。累计 3 succeeded / 61 pending，无失败；其余完整预算和独立评估继续执行，尚未应用四 seed 终验。

87. 2026-10-11 05:51，第二批 MNIST linear seed3 与 MLP seed0 / 1 分别达到 step42200 / 40800 / 40800，完成 211 / 204 / 204 个完整 test 点，均超过连续 80000-step 预算的一半。launch 会话 13127 仍存活，已有指标有限、评测坐标连续，无重复 start 或 error。已 prepare 的六条 Run 实际设备条件核验通过；linear / MLP seed0 前缀辅助对照仍为 2/8、24 个指标最大差 0，当前来源冻结与通过的前置源码一致。累计 3 succeeded / 61 pending，无失败；完整矩阵及独立评估尚未完成，不应用固定四 seed 终验。

88. 2026-10-11 06:00，第二批 MNIST linear seed3 与 MLP seed0 / 1 分别达到 step63400 / 61200 / 61000，完成 317 / 306 / 305 个完整 test 点，均超过连续 80000-step 预算的四分之三。launch 会话 13127 仍在运行，已有指标有限、评测坐标连续，无重复 start 或 error；六条已 prepare Run 的实际设备条件检查通过。linear / MLP seed0 前缀辅助对照仍为 2/8，24 个指标最大差 0，当前冻结来源与前置源码匹配。累计 3 succeeded / 61 pending，无失败；其余完整训练、独立 best 评估与固定四 seed 终验继续待完成。

89. 2026-10-11 06:08，MNIST linear seed3 与 MLP seed0 / 1 正式 succeeded，累计六条训练均完成连续 80000 steps；真实 latest.pt / best.pt、400 个完整 test 点、全程 best 与正式 result / tracker 的单体核查 6/6 通过，无错误，完整摘要及逐 seed 数字更新在 [CHECKPOINT_AUDIT.json](CHECKPOINT_AUDIT.json) 与 [STUDY_REPORT.md](STUDY_REPORT.md)。同一 launch 会话 13127 已自动启动第三批 MLP seed2 / 3 与 CNN seed0；九条已 prepare Run 的实际设备开关均通过。CNN seed0 的实际 Flow 前缀亦与前置链一致，辅助核查累计 3/8、36 个指标最大差 0。当前 6 succeeded / 58 pending，无失败；完整矩阵和 32 条独立 best 评估尚未完成，不能据已完成的 linear 四 seed 宣布整个 Study 已通过。

90. 2026-10-11 06:17，第三批 MNIST MLP seed2 / 3 与 CNN seed0 达到 step23400 / 23400 / 20400，完成 117 / 117 / 102 个完整 test 点，均超过连续 80000-step 预算的四分之一。launch 会话 13127 确认存活，已有指标有限、评测坐标连续，无重复 start 或 error；九条已 prepare Run 实际设备条件检查通过。linear / MLP / CNN seed0 实际前缀辅助对照仍为 3/8，36 个指标最大差 0，当前源码冻结与前置来源匹配。累计 6 succeeded / 58 pending，无失败；其余完整预算、独立 best 评估与四 seed 终验继续待完成。

91. 2026-10-11 06:26，第三批 MNIST MLP seed2 / 3 与 CNN seed0 达到 step46200 / 46200 / 40400，完成 231 / 231 / 202 个完整 test 点，均超过连续 80000-step 预算的一半。launch 会话 13127 确认存活，已有指标有限、评测坐标连续，无重复 start 或 error；九条已 prepare Run 的实际设备条件核验通过。linear / MLP / CNN seed0 实际前缀辅助对照仍为 3/8，36 个指标最大差 0，当前冻结源码与前置来源一致。累计 6 succeeded / 58 pending，无失败；其余完整训练、32 条自身 best 独立评估与固定四 seed 终验尚未完成。

92. 2026-10-11 06:36，第三批 MNIST MLP seed2 / 3 与 CNN seed0 达到 step69400 / 69200 / 60400，完成 347 / 346 / 302 个完整 test 点，均超过连续 80000-step 预算的四分之三。launch 会话 13127 仍在运行，已有指标有限、评测坐标连续，无重复 start 或 error；九条已 prepare Run 的实际设备条件检查通过。linear / MLP / CNN seed0 实际前缀辅助对照仍为 3/8，36 个指标最大差 0，当前源码冻结与前置来源匹配。累计 6 succeeded / 58 pending，无失败；其余完整训练、32 条自身 best 独立评估与固定四 seed 终验尚未完成。

93. 2026-10-11 06:41，MNIST MLP seed2 / 3 正式 succeeded，累计八条训练完成连续 80000 steps。对八条已完成 Run 的真实 latest.pt / best.pt、400 个完整 test 点、全程 best 与正式 result / tracker 做单体核查，8/8 通过，无错误；逐 seed 数字及证据更新在 [STUDY_REPORT.md](STUDY_REPORT.md) 与 [CHECKPOINT_AUDIT.json](CHECKPOINT_AUDIT.json)，部分曲线更新在 [COMPARISON.json](COMPARISON.json)。launch 会话 13127 仍存活，第三批 CNN seed0 继续训练至 step70200；九条已 prepare Run 的实际设备条件核验通过。当前 8 succeeded / 56 pending，无失败；完整矩阵、32 条独立 best 评估及固定四 seed 终验尚未完成，单体审计和部分曲线不代替 Study 验收。

94. 2026-10-11 06:45，MNIST CNN seed0 正式 succeeded，累计九条训练完成连续 80000 steps。已完成 Run 的真实 latest.pt / best.pt、400 个完整 test 点、全程 best 与正式 result / tracker 单体核查 9/9 通过，无错误；[CHECKPOINT_AUDIT.json](CHECKPOINT_AUDIT.json)、[STUDY_REPORT.md](STUDY_REPORT.md) 与 [COMPARISON.json](COMPARISON.json) 已更新。launch 会话 13127 仍存活，自动启动第四批 CNN seed1 / 2 / 3，三条均完成首个 step200 test；十二条已 prepare Run 的实际设备条件核验通过。当前 9 succeeded / 55 pending，无失败；其余完整预算、32 条自身 best 独立评估及固定四 seed 终验继续待完成。

95. 2026-10-11 06:55，第四批 MNIST CNN seed1 / 2 / 3 均达到 step20200，完成 101 个完整 test 点，超过连续 80000-step 预算的四分之一。launch 会话 13127 确认存活，已有指标有限、评测坐标连续，无重复 start 或 error；十二条已 prepare Run 的实际设备条件检查通过。linear / MLP / CNN seed0 实际前缀辅助对照仍为 3/8，36 个指标最大差 0，当前源码冻结与前置来源匹配。累计 9 succeeded / 55 pending，无失败；剩余完整训练、32 条自身 best 独立评估与固定四 seed 终验尚未完成。

96. 2026-10-11 07:05，第四批 MNIST CNN seed1 / 2 / 3 达到 step40200 / 40200 / 40400，完成 201 / 201 / 202 个完整 test 点，均超过连续 80000-step 预算的一半。launch 会话 13127 确认存活，已有指标有限、评测坐标连续，无重复 start 或 error；十二条已 prepare Run 的实际设备条件核验通过。linear / MLP / CNN seed0 实际前缀辅助对照仍为 3/8，36 个指标最大差 0，当前冻结源码与前置来源一致。累计 9 succeeded / 55 pending，无失败；其余完整训练、32 条自身 best 独立评估与固定四 seed 终验尚未完成。

97. 2026-10-11 07:14，第四批 MNIST CNN seed1 / 2 / 3 达到 step60200 / 60200 / 60400，完成 301 / 301 / 302 个完整 test 点，均超过连续 80000-step 预算的四分之三。launch 会话 13127 仍在运行，已有指标有限、评测坐标连续，无重复 start 或 error；十二条已 prepare Run 的实际设备条件核验通过。linear / MLP / CNN seed0 实际前缀辅助对照仍为 3/8，36 个指标最大差 0，当前源码冻结与前置来源匹配。累计 9 succeeded / 55 pending，无失败；其余完整训练、32 条自身 best 独立评估与固定四 seed 终验尚未完成。

98. 2026-10-11 07:24，MNIST CNN seed1 / 2 / 3 正式 succeeded，累计十二条训练完成连续 80000 steps。对已完成 Run 的真实 latest.pt / best.pt、400 个完整 test 点、全程 best 与正式 result / tracker 做单体核查，12/12 通过，无错误；[CHECKPOINT_AUDIT.json](CHECKPOINT_AUDIT.json)、[STUDY_REPORT.md](STUDY_REPORT.md) 与 [COMPARISON.json](COMPARISON.json) 已更新。launch 会话 13127 仍存活，自动启动第五批 MNIST ResNet18 seed0 / 1 / 2；十五条已 prepare Run 的实际设备条件核验通过。当前 12 succeeded / 52 pending，无失败；其余完整训练、32 条自身 best 独立评估和固定四 seed 终验尚未完成，已完成三个模型的四 seed 训练不代替整个 Study 验收。

99. 2026-10-11 07:26，第五批 MNIST ResNet18 seed0 / 1 / 2 均完成至 step800，无 error 或重复 start，已有指标有限、评测坐标连续，launch 会话 13127 仍存活。ResNet18 seed0 的实际 Flow step200 / 400 / 600 train/test Loss / Accuracy 与通过的前置链完全一致；[FLOW_PREFIX_AUDIT.json](FLOW_PREFIX_AUDIT.json) 累计核查 4/8 组、48 个指标，最大差 0，来源冻结与前置源码一致，报告同步更新此范围。十五条已 prepare Run 的实际设备条件核验通过，当前 12 succeeded / 52 pending，无失败。此辅助证据仅覆盖前缀指标，不代替参数核查、长预算训练、独立 best 评估或固定四 seed 终验。

100. 2026-10-11 07:57，第五批 MNIST ResNet18 seed0 / 1 / 2 达到 step20200 / 20400 / 20200，完成 101 / 102 / 101 个完整 test 点，均超过连续 80000-step 预算的四分之一。launch 会话 13127 确认存活，已有指标有限、评测坐标连续，无重复 start 或 error；十五条已 prepare Run 的实际设备条件核验通过。MNIST 四模型 seed0 实际前缀辅助对照仍为 4/8，48 个指标最大差 0，当前源码冻结与前置来源匹配。累计 12 succeeded / 52 pending，无失败；其余完整训练、32 条自身 best 独立评估与固定四 seed 终验尚未完成。

101. 2026-10-11 08:30，第五批 MNIST ResNet18 seed0 / 1 / 2 达到 step40600 / 40800 / 40400，完成 203 / 204 / 202 个完整 test 点，均超过连续 80000-step 预算的一半。launch 会话 13127 确认存活，已有指标有限、评测坐标连续，无重复 start 或 error；十五条已 prepare Run 的实际设备条件核验通过。累计 12 succeeded / 52 pending，无失败。半程进度不代表完整训练或质量通过，32 条自身 best 独立评估与固定四 seed 终验仍待完成。

102. 2026-10-11 09:02，第五批 MNIST ResNet18 seed0 / 1 / 2 达到 step60600 / 61000 / 60400，完成 303 / 305 / 302 个完整 test 点，均超过连续 80000-step 预算的四分之三。launch 会话 13127 确认存活，已有指标有限、评测坐标连续，无重复 start 或 error；十五条已 prepare Run 的实际设备条件核验通过。累计 12 succeeded / 52 pending，无失败。此进度核查不代替完整训练、32 条自身 best 独立评估或固定四 seed 终验。

103. 2026-10-11 09:34，第五批 MNIST ResNet18 seed0 / 1 / 2 正式 succeeded，累计 15 条训练完成、49 条 Run pending，无失败。逐条读取实际 latest / best checkpoint 与 tracker 的核查通过：均完成连续 80000-step、400 个完整 test 点，best 与全程指标一致，证据已更新至 [CHECKPOINT_AUDIT.json](CHECKPOINT_AUDIT.json) 和 [STUDY_REPORT.md](STUDY_REPORT.md)。partial 比较与图已同步，仍不应用完整四 seed 终验。launch 会话 13127 持续运行，自动启动第六批 MNIST ResNet18 seed3（069305e87da05b12）、CIFAR10 linear seed0（fc085db588d295b4）/ seed1（385bf2b1307b622f）；十八条已 prepare Run 的实际设备条件核验通过。32 条自身 best 独立评估与固定四 seed 质量门仍待完整矩阵完成。

104. 2026-10-11 09:35，第六批 MNIST ResNet18 seed3 达到 step1400，CIFAR10 linear seed0 / 1 均达到 step1200，已有指标有限、评测坐标连续，无重复 start 或 error，launch 会话 13127 仍存活。CIFAR10 linear seed0 的实际 Flow step200 / 400 / 600 train/test Loss / Accuracy 与通过的前置链完全一致；[FLOW_PREFIX_AUDIT.json](FLOW_PREFIX_AUDIT.json) 累计核查 5/8 组、60 个指标，最大差 0，来源冻结与前置源码一致，报告同步更新此范围。十八条已 prepare Run 的实际设备条件核验通过，当前 15 succeeded / 49 pending，无失败。此辅助证据仅覆盖前缀指标，不代替参数核查、长预算训练、独立 best 评估或固定四 seed 终验。

105. 2026-10-11 09:56，第六批 MNIST ResNet18 seed3 达到 step24000，CIFAR10 linear seed0 / 1 分别达到 step20400 / 20600，三条均超过连续 80000-step 预算的四分之一。完整 test 点分别为 120 / 102 / 103，已有 Loss / Accuracy mean 均有限，评测坐标连续，无重复 start 或 error；launch 会话 13127 仍在运行。当前 15 succeeded / 49 pending，无失败；十八条已 prepare Run 的实际设备条件核验通过。此项仅记录活动进度，剩余训练、32 条自身 best 独立评测及固定四 seed 质量门尚未完成。

106. 2026-10-11 10:17，第六批 MNIST ResNet18 seed3 达到 step46800，CIFAR10 linear seed0 / 1 分别达到 step40000 / 40200，三条均达到连续 80000-step 预算的一半。完整 test 点分别为 234 / 200 / 201，已有 Loss / Accuracy mean 均有限，评测坐标连续，无重复 start 或 error；launch 会话 13127 仍在运行。当前 15 succeeded / 49 pending，无失败；十八条已 prepare Run 的实际设备条件核验通过。此项仅记录活动进度，剩余训练、32 条自身 best 独立评测及固定四 seed 质量门尚未完成。

107. 2026-10-11 10:39，第六批 MNIST ResNet18 seed3 达到 step70800，CIFAR10 linear seed0 / 1 均达到 step60400，三条均超过连续 80000-step 预算的四分之三。完整 test 点分别为 354 / 302 / 302，已有 Loss / Accuracy mean 均有限，评测坐标连续，无重复 start 或 error；launch 会话 13127 仍在运行。当前 15 succeeded / 49 pending，无失败；十八条已 prepare Run 的实际设备条件核验通过。此项仅记录活动进度，剩余训练、32 条自身 best 独立评测及固定四 seed 质量门尚未完成。

108. 2026-10-11 10:48，MNIST ResNet18 seed3（069305e87da05b12）正式 succeeded，连续完成 80000 steps 和 400 个完整 test 点，末点 Accuracy 99.6000%，全程 best 99.6300%，训练耗时 1:13:49。至此当前版本 MNIST 的四模型 × 四 seed 共 16 条训练全部完成；只读 CPU 核查实际 latest / best checkpoint、完整 test10000 / batch250、评测轨迹、全程 best 与正式 result 一致性，16/16 条完成训练通过，无错误。[CHECKPOINT_AUDIT.json](CHECKPOINT_AUDIT.json)、[STUDY_REPORT.md](STUDY_REPORT.md) 与 partial 比较图已更新，完整质量门尚未应用。launch 会话 13127 仍在运行，当前 16 succeeded / 48 pending，无失败；同批 CIFAR10 linear seed0 / 1 分别到 step68600 / 68800。剩余 CIFAR10 训练、32 条自身 best 独立评测及固定四 seed 质量门仍待完成。

109. 2026-10-11 11:00，CIFAR10 linear seed0（fc085db588d295b4）/ seed1（385bf2b1307b622f）均正式 succeeded，连续完成 80000 steps 和 400 个完整 test 点。末点 / best Accuracy 分别为 40.3900% / 40.6700%、40.5300% / 40.6600%，训练耗时分别为 1:26:00、1:25:48。累计 18 条训练完成；只读 CPU 审计实际 latest / best checkpoint、完整 test10000 / batch250、评测轨迹、全程 best 与正式 result 一致性，18/18 通过，无错误。[CHECKPOINT_AUDIT.json](CHECKPOINT_AUDIT.json)、[STUDY_REPORT.md](STUDY_REPORT.md) 与 partial 比较图已更新，完整质量门尚未应用。launch 会话 13127 仍在运行，自动启动第七批 CIFAR10 linear seed2（dae98504b3fa274d）/ seed3（a9148272760d5180）及 MLP seed0（d2b0e7f33699ecd4），三条均完成首个 step200 test，暂无 error。当前 18 succeeded / 46 pending，无失败；二十一条已 prepare Run 的实际设备条件核验通过。剩余训练、32 条自身 best 独立评测及固定四 seed 质量门仍待完成。

110. 2026-10-11 11:01，第七批 CIFAR10 linear seed2 / 3 与 MLP seed0 均达到 step1400，已有指标有限、评测坐标连续，无重复 start 或 error，launch 会话 13127 仍在运行。MLP seed0 的实际 Flow step200 / 400 / 600 train/test Loss / Accuracy 与通过的前置链完全一致；[FLOW_PREFIX_AUDIT.json](FLOW_PREFIX_AUDIT.json) 累计核查 6/8 组、72 个指标，最大差 0，来源冻结与前置源码一致，报告同步更新此范围。二十一条已 prepare Run 的实际设备条件核验通过，当前 18 succeeded / 46 pending，无失败。此辅助证据仅覆盖前缀指标，不代替参数核查、长预算训练、独立 best 评测或固定四 seed 终验。

111. 2026-10-11 11:21，第七批 CIFAR10 linear seed2 / seed3 与 MLP seed0 分别达到 step20200、20200、20000，完成 101、101、100 个完整 test 点，均达到 80000-step 预算的四分之一。launch 会话 13127 经直接轮询确认仍在运行；已有指标有限、评测坐标连续，无重复 start 或 error。当前 18 succeeded / 46 pending，无失败；二十一条已 prepare Run 的实际运行条件核验通过。此项为活动进度，剩余训练、32 条自身 best 独立评测与固定四 seed 终验尚未完成。

112. 2026-10-11 11:43，第七批 CIFAR10 linear seed2 / seed3 与 MLP seed0 分别达到 step40200、40000、40000，完成 201、200、200 个完整 test 点，均达到 80000-step 预算的一半。launch 会话 13127 经直接轮询确认仍在运行；已有指标有限、评测坐标连续，无重复 start 或 error。当前 18 succeeded / 46 pending，无失败；二十一条已 prepare Run 的实际运行条件核验通过。此项为活动进度，剩余训练、32 条自身 best 独立评测与固定四 seed 终验尚未完成。

113. 2026-10-11 12:05，第七批 CIFAR10 linear seed2 / seed3 与 MLP seed0 分别达到 step60600、60400、60400，完成 303、302、302 个完整 test 点，均超过 80000-step 预算的四分之三。launch 会话 13127 经直接轮询确认仍在运行；已有指标有限、评测坐标连续，无重复 start 或 error。当前 18 succeeded / 46 pending，无失败；二十一条已 prepare Run 的实际运行条件核验通过。此项为活动进度，剩余训练、32 条自身 best 独立评测与固定四 seed 终验尚未完成。

114. 2026-10-11 12:26，第七批 CIFAR10 linear seed2（dae98504b3fa274d）/ seed3（a9148272760d5180）与 MLP seed0（d2b0e7f33699ecd4）均正式 succeeded，连续完成 80000 steps 和 400 个完整 test 点。末点 / best Accuracy 分别为 40.3200% / 40.6200%、40.2400% / 40.4400%、63.6500% / 63.8500%，训练耗时分别为 1:25:48、1:26:02、1:26:11。累计 21 条训练完成；只读 CPU 审计实际 latest / best checkpoint、完整 test10000 / batch250、评测轨迹、全程 best 与正式 result 一致性，21/21 通过，无错误。[CHECKPOINT_AUDIT.json](CHECKPOINT_AUDIT.json)、[STUDY_REPORT.md](STUDY_REPORT.md) 与 partial 比较图已更新，图中保持 INCOMPLETE / PROVISIONAL 标记，完整质量门尚未应用。launch 会话 13127 仍在运行，自动启动第八批 CIFAR10 MLP seed1（6c2cda2cf02c2a9b）/ seed2（28f2368d1634417c）/ seed3（8e5f67f6b365822f），均完成首个 step200 test，暂无 error。当前 21 succeeded / 43 pending，无失败；二十四条已 prepare Run 的实际运行条件核验通过。剩余训练、32 条自身 best 独立评测及固定四 seed 终验仍待完成。

115. 2026-10-11 12:49，第八批 CIFAR10 MLP seed1 / seed2 / seed3 分别达到 step20600、20800、20800，完成 103、104、104 个完整 test 点，均超过 80000-step 预算的四分之一。launch 会话 13127 经直接轮询确认仍在运行；已有指标有限、评测坐标连续，无重复 start 或 error。当前 21 succeeded / 43 pending，无失败；二十四条已 prepare Run 的实际运行条件核验通过。此项为活动进度，剩余训练、32 条自身 best 独立评测与固定四 seed 终验尚未完成。

116. 2026-10-11 13:10，第八批 CIFAR10 MLP seed1 / seed2 / seed3 分别达到 step40200、40400、40400，完成 201、202、202 个完整 test 点，均超过 80000-step 预算的一半。launch 会话 13127 经直接轮询确认仍在运行；已有指标有限、评测坐标连续，无重复 start 或 error。当前 21 succeeded / 43 pending，无失败；二十四条已 prepare Run 的实际运行条件核验通过。此项为活动进度，剩余训练、32 条自身 best 独立评测与固定四 seed 终验尚未完成。

117. 2026-10-11 13:31，第八批 CIFAR10 MLP seed1 / seed2 / seed3 分别达到 step60000、60000、60200，完成 300、300、301 个完整 test 点，均达到 80000-step 预算的四分之三。launch 会话 13127 经直接轮询确认仍在运行；已有指标有限、评测坐标连续，无重复 start 或 error。当前 21 succeeded / 43 pending，无失败；二十四条已 prepare Run 的实际运行条件核验通过。此项为活动进度，剩余训练、32 条自身 best 独立评测与固定四 seed 终验尚未完成。

118. 2026-10-11 13:53–13:54，第八批 CIFAR10 MLP seed1 / seed2 / seed3 正式 result 均为 succeeded，完成 80000-step 连续训练及 400 个完整 test 点；末点 / 全程 best Accuracy 分别为 63.2400% / 63.4200%、63.6900% / 63.8800%、63.2900% / 63.5000%，耗时分别为 1:26:31、1:26:30、1:26:19。累计二十四条已完成训练的实际 latest / best checkpoint、tracker 与正式结果合同核验全部通过，已更新 CHECKPOINT_AUDIT.json、STUDY_REPORT.md、部分 COMPARISON.json 和阶段对照图，并确认图中保留未完整标记。相同 launch 会话 13127 自动进入第九批 CIFAR10 CNN seed0 / seed1 / seed2（Run ID b99775641e1800c7、48ee1a36f59ac74d、30cc6a335b81cf34），13:54 均到 step800，无 error；二十七条已 prepare Run 的实际运行条件核验通过。CNN seed0 的 step200 / 400 / 600 指标前缀核对通过，累计 7/8 组、84 个 train/test Loss / Accuracy 值与前置链一致，最大差异为 0。当前 24 succeeded / 40 pending，无失败；剩余训练、32 条自身 best 独立评测与固定四 seed 终验尚未完成，训练末点和前缀核对不代替完整验收。

119. 2026-10-11 14:16，第九批 CIFAR10 CNN seed0 / seed1 / seed2 均达到 step20200，完成 101 个完整 test 点，超过 80000-step 预算的四分之一。launch 会话 13127 经直接轮询确认仍在运行；已有指标有限、评测坐标连续，无重复 start 或 error。当前 24 succeeded / 40 pending，无失败；二十七条已 prepare Run 的实际运行条件核验通过。此项为活动进度，剩余训练、32 条自身 best 独立评测与固定四 seed 终验尚未完成。

120. 2026-10-11 14:39，第九批 CIFAR10 CNN seed0 / seed1 / seed2 分别达到 step40000、40200、40200，完成 200、201、201 个完整 test 点，均达到 80000-step 预算的一半。launch 会话 13127 经直接轮询确认仍在运行；已有指标有限、评测坐标连续，无重复 start 或 error。当前 24 succeeded / 40 pending，无失败；二十七条已 prepare Run 的实际运行条件核验通过。此项为活动进度，剩余训练、32 条自身 best 独立评测与固定四 seed 终验尚未完成。

121. 2026-10-11 15:02，第九批 CIFAR10 CNN seed0 / seed1 / seed2 均达到 step60000，完成 300 个完整 test 点，达到 80000-step 预算的四分之三。launch 会话 13127 经直接轮询确认仍在运行；已有指标有限、评测坐标连续，无重复 start 或 error。当前 24 succeeded / 40 pending，无失败；二十七条已 prepare Run 的实际运行条件核验通过。此项为活动进度，剩余训练、32 条自身 best 独立评测与固定四 seed 终验尚未完成。

122. 2026-10-11 15:25–15:26，第九批 CIFAR10 CNN seed0 / seed1 / seed2 的正式 result 均为 succeeded，分别为 Run b99775641e1800c7、48ee1a36f59ac74d、30cc6a335b81cf34。按实际 Study inspect_run 核对全部二十七条已完成训练，latest step=80000，test Loss / Accuracy 各 400 点，latest / best checkpoint 与 tracker 合同均通过。三条 CNN 末点 Accuracy 为 88.8500%、88.9000%、88.4200%，训练期间 best 为 89.0700%、89.1300%、88.6600%，耗时分别为 1:32:19、1:32:13、1:32:13。已更新 CHECKPOINT_AUDIT.json、STUDY_REPORT.md、partial COMPARISON.json 和历史对照图，图仍标记未完成与临时结果。launch 会话 13127 经直接轮询确认存活，自动启动第十批 CNN seed3（6ce695c58b4b1f98）、ResNet18 seed0（5bd65c61c30c12d2）与 seed1（caf78c8d80ad9f10），分别已完成 step400、200、200；三十条已 prepare Run 的实际运行条件核验通过。当前 27 succeeded / 37 pending，无失败；其余训练、32 条自身 best 独立评测与固定四 seed 终验尚未完成。

123. 2026-10-11 15:27，第十批 CNN seed3 已完成 step1400，ResNet18 seed0 / seed1 均完成 step1000，无 error；launch 会话 13127 经直接轮询确认仍在运行。补齐 CIFAR10 ResNet18 seed0 的正式 Flow 前缀核查后，八组 seed0 的 step200 / 400 / 600 共 96 个 train/test Loss / Accuracy 值均与当前通过的前置链完全一致，最大指标差为 0。FLOW_PREFIX_AUDIT.json 与 STUDY_REPORT.md 已同步；此项仅覆盖前缀指标，不代替参数、独立 best 评估或固定四 seed 长预算终验。三十条已 prepare Run 的实际运行条件核验通过，当前 27 succeeded / 37 pending，无失败。

124. 2026-10-11 15:50，第十批 CIFAR10 CNN seed3（Run 6ce695c58b4b1f98）达到 step20200，完成 101 个完整 test 点，超过 80000-step 预算的四分之一；同批 ResNet18 seed0 / seed1 均达到 step13800，完成 69 个完整 test 点。launch 会话 13127 经直接轮询确认仍在运行；已有指标有限、评测坐标连续，无重复 start 或 error。三十条已 prepare Run 的实际运行条件核验通过，当前 27 succeeded / 37 pending，无失败。此项为活动进度，剩余训练、32 条自身 best 独立评测与固定四 seed 终验尚未完成。

125. 2026-10-11 16:01，第十批 CIFAR10 ResNet18 seed0 / seed1（Run 5bd65c61c30c12d2 / caf78c8d80ad9f10）均达到 step20000，完成 100 个完整 test 点，达到 80000-step 预算的四分之一；同批 CNN seed3 达到 step29200，完成 146 个完整 test 点。launch 会话 13127 经直接轮询确认仍在运行；已有指标有限、评测坐标连续，无重复 start 或 error。三十条已 prepare Run 的实际运行条件核验通过，当前 27 succeeded / 37 pending，无失败。此项为活动进度，剩余训练、32 条自身 best 独立评测与固定四 seed 终验尚未完成。

126. 2026-10-11 16:14，第十批 CIFAR10 CNN seed3（Run 6ce695c58b4b1f98）达到 step40000，完成 200 个完整 test 点，达到 80000-step 预算的一半；同批 ResNet18 seed0 / seed1 均达到 step27400，完成 137 个完整 test 点。launch 会话 13127 经直接轮询确认仍在运行；已有指标有限、评测坐标连续，无重复 start 或 error。三十条已 prepare Run 的实际运行条件核验通过，当前 27 succeeded / 37 pending，无失败。此项为活动进度，剩余训练、32 条自身 best 独立评测与固定四 seed 终验尚未完成。

127. 2026-10-11 16:37，第十批 CIFAR10 ResNet18 seed0 / seed1（Run 5bd65c61c30c12d2 / caf78c8d80ad9f10）均达到 step40400，完成 202 个完整 test 点，超过 80000-step 预算的一半；同批 CNN seed3 达到 step59000，完成 295 个完整 test 点。launch 会话 13127 经直接轮询确认仍在运行；已有指标有限、评测坐标连续，无重复 start 或 error。三十条已 prepare Run 的实际运行条件核验通过，当前 27 succeeded / 37 pending，无失败。此项为活动进度，剩余训练、32 条自身 best 独立评测与固定四 seed 终验尚未完成。

128. 2026-10-11 16:39，第十批 CIFAR10 CNN seed3（Run 6ce695c58b4b1f98）达到 step60600，完成 303 个完整 test 点，超过 80000-step 预算的四分之三；同批 ResNet18 seed0 / seed1 均达到 step41400，完成 207 个完整 test 点。launch 会话 13127 经直接轮询确认仍在运行；已有指标有限、评测坐标连续，无重复 start 或 error。三十条已 prepare Run 的实际运行条件核验通过，当前 27 succeeded / 37 pending，无失败。此项为活动进度，剩余训练、32 条自身 best 独立评测与固定四 seed 终验尚未完成。

129. 2026-10-11 17:02–17:03，第十批 CIFAR10 CNN seed3（Run 6ce695c58b4b1f98）正式 succeeded，至此 CIFAR10 CNN 四 seed 全部完成连续 80000 steps；seed3 末点 Accuracy 为 88.8500%，全程 best 为 89.0700%，耗时 1:36:43。对累计 28 条完成训练的真实 latest.pt / best.pt、400 个完整 test 点、最终进度及全程 best 与正式 result / tracker 做单体核查，28/28 通过，无错误；逐 seed 数字与正式证据已同步 [STUDY_REPORT.md](STUDY_REPORT.md) / [CHECKPOINT_AUDIT.json](CHECKPOINT_AUDIT.json)，部分曲线已更新 [COMPARISON.json](COMPARISON.json) 与对照图，保留未完整标记。launch 会话 13127 经直接轮询确认仍在运行，ResNet18 seed0 / seed1 均到 step55000，完成 275 个完整 test 点，暂无 error。当前 28 succeeded / 36 pending，无失败；剩余四条 ResNet18 训练、32 条自身 best 独立评测与固定四 seed 终验尚未完成，单体核查和部分曲线不代替 Study 验收。

130. 2026-10-11 17:10，第十批 CIFAR10 ResNet18 seed0 / seed1（Run 5bd65c61c30c12d2 / caf78c8d80ad9f10）均达到 step60000，完成 300 个完整 test 点，达到 80000-step 预算的四分之三。launch 会话 13127 经直接轮询确认仍在运行；已有指标有限、评测坐标连续，无重复 start 或 error。三十条已 prepare Run 的实际运行条件核验通过，当前 28 succeeded / 36 pending，无失败。此项为活动进度，剩余四条 ResNet18 训练、32 条自身 best 独立评测与固定四 seed 终验尚未完成。

131. 2026-10-11 17:43，第十批 CIFAR10 ResNet18 seed0 / seed1（Run 5bd65c61c30c12d2 / caf78c8d80ad9f10）正式 succeeded，均完成连续 80000 steps；末点 Accuracy 分别为 92.8100% / 93.0200%，全程 best 为 93.0000% / 93.2300%，耗时 2:17:35 / 2:17:34。对累计 30 条完成训练的真实 latest.pt / best.pt、400 个完整 test 点、最终进度及全程 best 与正式 result / tracker 做单体核查，30/30 通过，无错误；逐 seed 数字和正式证据已同步 [STUDY_REPORT.md](STUDY_REPORT.md) / [CHECKPOINT_AUDIT.json](CHECKPOINT_AUDIT.json)，部分曲线已更新 [COMPARISON.json](COMPARISON.json) 与对照图，保留未完整标记。launch 会话 13127 经直接轮询确认仍在运行，自动启动第十一批 CIFAR10 ResNet18 seed2 / seed3（Run 8c413f14e3d8f742 / 6b9fced9385ffbad）；首次完整 test 尚未写入，暂无 error。三十二条已 prepare Run 的实际运行条件核验通过，当前 30 succeeded / 34 pending，无失败。剩余两条训练、32 条自身 best 独立评测与固定四 seed 终验尚未完成，单体核查和部分曲线不代替 Study 验收。

132. 2026-10-11 18:16，第十一批 CIFAR10 ResNet18 seed2 / seed3（Run 8c413f14e3d8f742 / 6b9fced9385ffbad）均达到 step20200，完成 101 个完整 test 点，超过连续 80000-step 预算的四分之一。launch 会话 13127 经直接轮询确认仍在运行，评测坐标连续、已有 Loss / Accuracy mean 均有限，暂无 error；32 条已 prepare Run 的实际运行条件核验通过。当前 30 succeeded / 34 pending，无失败。剩余两条训练、32 条自身 best 独立评测与固定四 seed 终验尚未完成，此进度不代替 Study 验收。

133. 2026-10-11 18:48，第十一批 CIFAR10 ResNet18 seed2 / seed3（Run 8c413f14e3d8f742 / 6b9fced9385ffbad）均达到 step40400，完成 202 个完整 test 点，超过连续 80000-step 预算的一半。launch 会话 13127 经直接轮询确认仍在运行，评测坐标连续、已有 Loss / Accuracy mean 均有限，暂无 error；32 条已 prepare Run 的实际运行条件核验通过。当前 30 succeeded / 34 pending，无失败。剩余两条训练、32 条自身 best 独立评测与固定四 seed 终验尚未完成，此中点检查不代替 Study 验收。

134. 2026-10-11 19:18，第十一批 CIFAR10 ResNet18 seed2 / seed3（Run 8c413f14e3d8f742 / 6b9fced9385ffbad）均达到 step60400，完成 302 个完整 test 点，超过连续 80000-step 预算的四分之三。launch 会话 13127 经直接轮询确认仍在运行，评测坐标连续、已有 Loss / Accuracy mean 均有限，暂无 error；32 条已 prepare Run 的实际运行条件核验通过。当前 30 succeeded / 34 pending，无失败。剩余两条训练、32 条自身 best 独立评测与固定四 seed 终验尚未完成，此进度不代替 Study 验收。

135. 2026-10-11 19:48，version03 原生 launch 正常结束（exit 0）：32 train 与 32 eval 全部 succeeded，无 failed / pending。全部训练连续 80000 steps、400 个完整 test 点；64/64 checkpoint / tracker 合同核查通过，64/64 实际设备开关检查通过。COMPARISON.json 为 complete=true、final_gates_applied=true、passed=true，八组固定曲线门及 MNIST / CIFAR10 模型次序门全部通过，32 组自身 best 独立评估一致。正式数字由原生 rpipe report 从 process.json 生成，见 [NUMBERS.md](NUMBERS.md)；完整结论见 [STUDY_REPORT.md](STUDY_REPORT.md)。旧失败保留；下一步执行维护者要求的历史速度校准，数值终验通过不代表速度问题解决。

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
