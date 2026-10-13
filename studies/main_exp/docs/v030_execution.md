# v0.3.0 本机复验执行记录

## 一、摘要

- 基线：发布标签 `v0.3.0`，提交 `804b3823ab6d2af46cbda1a65688ca8b76c60b22`。本次重新准备隔离环境、核对数据并实测；不使用历史通过报告解锁新实验。
- 授权：2026-10-13 维护者授权小规模验证、受控并发校准及前置门通过后的完整两个 Study。旧 PLAN 中“本次仅短实验”描述的是发布准备阶段，当前授权覆盖完整矩阵；配方、数值门与正式预算不变。
- 当前状态：最终依赖下 core 397 / CPU integration 51 项通过；单格 GPU / Flow smoke、六次串行对照、27 次容量 Flow、main_probe 独立校准和新正式八格终验均通过。main_exp 新来源八格 600-step 前置门通过；原调度完成30条训练后中断，现按同源码 / 同科学配置的 [接续方案](v030_recovery.md) 重跑受影响两条及执行32条评测，32条连续训练、32条own-best评测、完整性与固定曲线终验均已通过；原30条与新两条分别保留来源。旧声明、生成清单与正式报告已逐字节归档，旧 Run 保留。

## 二、顺序与定稿条件

1. 核对标签及两个历史提交，创建 `.tmp/v030-20261013/venv`，独立安装依赖；真实 CUDA 计算后记录设备、依赖和磁盘。
2. 通过现有 `flow/prepare_data.py` 核对完整 MNIST / CIFAR10 的 16 项 raw hash 与 split 数量，main_probe 复用这些 raw。
3. 依测试规范运行核心与 CPU integration，再用新 version / index 执行真实 GPU / Flow smoke，记录子集终验拒绝边界。
4. main_exp 复用 calibrate 与 calibrate-capacity，从 round1 起测 CNN、ResNet18、混合组，相同 800-step 任务量与完整 test。分别校准 main_probe，不借用 main_exp 的显存预算，不改变一对内的先后顺序。
5. 保留 allocator 峰值与整卡样本，补充 CPU、主存、磁盘 I/O 和父墙钟观察。WDDM 无可靠逐进程显存时明确其边界；context / 驱动开销用整卡增量辅助估计，不宣称精确归因。
6. 仅在无资源故障、checkpoint / history 数值门通过且总体吞吐改善时采用并发。保持 safety=0.5；根据本轮实测确定各 Study 的资源配置，然后 fresh make。
7. main_probe 新来源完整八格 60-step 与 Study 终验通过；main_exp 新来源完整八格 600-step 前置门通过后，执行 32 条连续 80000-step train 和 32 条各自 best 独立 eval。每 200-step 完整 test、400 点与固定曲线门不变。
8. 源码 / 配置变化重新冻结与复验，不将中断训练续跑作为连续训练；保留失败证据。最终区分执行、数值、Study 完整终验与未执行范围。

## 三、证据位置

正式环境、校准数字及终验报告在本 Study `docs/`；原始临时日志与安装缓存在 `.tmp/v030-20261013/`，不会随 Git clone 提供。Run、数据及校准工作区沿用 Study 的原生磁盘合同。

## 四、2026-10-13 准备与验证

1. 标签与指定提交一致，两项历史提交均在本 clone 可读取。工作区原先干净，创建 `codex/v030-local-studies`；旧报告和本机可取得清单逐字节保存到两个 Study 的 `docs/versions/v030-before-local-20261013/`。这些文件可能描述不同历史源码，快照没有补造缺失的 Run 证据。
2. 新建独立 venv，不继承系统 site-packages。旧 `.tmp/runtime` 原样移到 `.tmp/runtime-before-v030-20261013`，避免 main_exp 的旧 bootstrap 优先导入旧依赖。首次 ensurepip 受沙箱限制，在授权范围内初始化后成功。
3. [环境](environment_v030_20261013.json)：Windows 11、Ryzen 9 9950X3D、16 核 / 32 线程、约 47 GiB 主存，RTX 5090 D v2 / 24455 MiB、驱动 595.97。Torch 2.11.0+cu130 / torchvision 0.26.0+cu130；实际 FP32 矩阵计算通过。未使用另一环境的 MKL 补丁。
4. [数据](data_v030_20261013.json)：prepare_data verify-only 核对现有 raw，16/16 SHA-256 匹配；MNIST 60000 / 10000，CIFAR10 50000 / 10000，无需新增下载。NumPy 对旧 pickle align 的警告保留，数据 hash 未变。
5. 初始 core 395 passed / 2 skipped / 54 deselected；CPU integration 51 passed / 2 skipped / 398 deselected。两项是缺 transformers 的整模块 collection skip，未算通过；七条旧 checkpoint 部分恢复警告保留。归档首次校准也在 import transformers 时退出，尚未训练，[失败资源记录](speed_measurements/v030-speed-cnn-telemetry.json)保留。补齐归档依赖后使用新 ID 校准，最终依赖环境的相关门待复验。
6. 最终依赖下 core 397 passed / 55 deselected，无 skip；包含补齐依赖后可收集的 HF 轻量合同测试，未运行真实 HF 研究。CPU 流程门和后续实测另行记录，不将核心门扩大为全后端通过。
7. 最终 CPU integration 51 passed / 401 deselected，无 skip；最终两次命令、Run ID 和原生报告见 [测试证据](v030_test_audit.json)。真实 MNIST / linear 成对 Flow smoke 为 `dfad40a92a173601`，60-step 与完整 test 的门通过；Study process exit 1 且 incomplete / passed=false 是缺七格的预期边界，见 [独立 smoke 证据](../../main_probe/docs/v030_smoke.json)。
8. [六次串行汇总](speed_v030_summary.json)：CNN / ResNet18 各三次独立进程，初始和最终参数精确一致，完整 test 门通过、来源未变。剔除第一段预热，old / current train200 均值为 8.6304 / 8.6819 秒及 14.9220 / 15.0610 秒，当前分别高约 0.60% / 0.93%。保存成本另列，不将该微基准当作长矩阵总时间。
9. [容量矩阵](speed_measurements/capacity-20261013-033729-c49faec2.json)：九组共 27 个原生 800-step Flow 全部通过，strict checkpoint / test history 对串行一致，0 OOM / 0 failed，完整与来源门通过。CNN 的 round1/2/3 墙钟 129.63 / 84.38 / 45.37 秒；ResNet18 为 209.41 / 141.80 / 109.15 秒；混合组为 150.25 / 111.05 / 74.09 秒。每组三任务；这是本轮一次受控矩阵，不是多轮速度排名。[外部资源观察](speed_measurements/v030-capacity-telemetry.json)补充 CPU、主存、进程 I/O 与整卡信息，包含背景活动和观察开销。
10. [正式资源选择](resource_choice_v030.json)：main_exp 按本轮 reserved、串行整卡增量、已有 context 开销下限与余量核对后取 4.5 GiB；main_probe 独立自动分组校准取 5 GiB。两项 fresh make 均使用原生自动两并发与 safety=0.5；手动 round3 的较快结果没有绕过正式保护池。版本分别为 `main-exp-v030-20261013-final`、`main-probe-v030-20261013-final`。
11. [main_exp 新来源前置门](PREFLIGHT.json)及[核对](v030_preflight_audit.json)：八格各600 steps / eval200，sampler 与 scheduler 总预算仍80000，参数、输入、RNG、整数正确数、Loss 与 scheduler 门全部通过，来源与新声明一致。main_probe 新正式八格结果见其[本轮记录](../../main_probe/docs/v030_execution.md)。
12. 2026-10-13 约04:26开始完整原生 launch：32条连续80000-step train，全部 train 结束后32条自身best独立eval；batch250、eval200、test10000、400点、latest2000、全部固定曲线门保持不变。当前运行资源与状态写入 [观察摘要](v030_long_resources.json)，原始逐10秒样本和启动日志在 `.tmp/v030-20261013/`，仅本机。启动不等于完整通过；终验与合同核对完成后再定稿。

13. 本轮发现并独立复现 [B-030](../../../docs/development/bugs.md)：fresh make 后原 process 保留旧轮次，公开 report 仍 exit0 导出64个旧ID、0个当前ID，见 [复现证据](v030_stale_report_repro.json)。计算源码仍固定804b3823，长矩阵不受报告读取缺陷影响；完整定稿额外核对 index / process / COMPARISON 的ID集合，旧聚合不作为本轮通过依据。未在运行期间改被冻结的源码，也未宣称已修复。

14. 维护者要求依据最新main后，重新fetch核对origin/main仍为804b3823；本地main已快进到同一提交，当前工作分支HEAD与两Study冻结合同未变，见[来源核对](v030_git_audit.json)。旧根阶段目录仅含忽略的pyc缓存，已隔离到本机.tmp并保留hash；正式计算仍在flow / structure，运行不中断。

15. 原调度于20:59左右中断，30条完整性经原生curves.inspect_run核对通过；两条中断训练不resume，原证据见[v030_interruption.json](v030_interruption.json)。按[原生接续清单](v030_recovery_plan.json)仅替换受影响4个身份槽位，源码映射和其余科学字段一致，原30条结果hash不变；当前index为64条，两份provenance分别保留真实执行来源。
16. 21:14左右以授权隐藏后台进程启动同一public launch，34个待执行任务。临时观察器首次未处理Windows不可用磁盘计数，在训练开始前退出；证据保留，改为null并保留进程I/O，见[v030_observer_repair.json](v030_observer_repair.json)。计算源码未变；观察日志与进程可见性不替代正式Run状态。接续实时观察在[v030_recovery_resources.json](v030_recovery_resources.json)，原资源摘要明确标为interrupted_observed。

17. 2026-10-13 23:11左右接续结束，public launch exit0，source_unchanged=true，32train / 32eval成功、0failed / 0pending。64条实权重 / committed scheduler / history / 自身best / 原结果hash及两份同源码provenance通过，固定八曲线及两模型次序门通过；原中断与观察器启动失败保留。正式数字已由native report生成，详细结论见[报告](STUDY_REPORT.md)和[v030_long_audit.json](v030_long_audit.json)。
