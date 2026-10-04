# 历史训练 GPU 容量诊断

## 一、概要

2026-10-04，本机三个独立历史 native 容量诊断进程顺序完成，主控制器工具句柄 `41748` 真实结束、退出码为 `0`。执行时间为北京时间 03:03:17–03:04:53。独立 CPU 重载审查通过，完整原始控制器 summary、三个 worker 报告、117 条全局采样和逐项审查保存在 [GPU_CAPACITY.json](GPU_CAPACITY.json)。

各格均连续训练 400 个 optimizer step，批量为 train 250 / test 250；在第 200、400 步分别完成 10,000 样本、40 个 batch 的完整 test。累计每格 train 100,000 样本 / 400 batch、test 20,000 样本 / 80 batch。它们是容量诊断，不属于正式 80k Run，也不替代曲线复现验收。

| **历史模型组合** | **本进程 allocated 峰值 MiB** | **本进程 reserved 峰值 MiB** | **同期全局采样最大占用 MiB** |
| --- | ---: | ---: | ---: |
| MNIST / cnn | 362.93 | 502 | 8553 |
| CIFAR10 / cnn | 445.67 | 842 | 8893 |
| MNIST / resnet18 | 1018.50 | 1414 | 9465 |

全局共有 117 个有效样本、0 次采样错误；设备可见总量为 24,455 MiB，全局采样最大占用 9465 MiB，采样时剩余 14,990 MiB。全局包含继续运行的正式训练进程；该最大值是轮询记录中的最大值。

容量证据支持有条件准备第二波两个 CIFAR10 / resnet18，与后续四个 CNN、两个 MNIST / resnet18 协调运行。四个 CNN 阶段的保守规划余量仅比 5 GiB 实际余量门多 871 MiB，因此仍需依据真实全局显存和精确进程状态继续投放。

## 二、计算与仪表范围

### （一）历史配方与隔离

诊断使用冻结的历史 `4ccb28d0496110253e9f8e3f3df658853f07996b` 模型和数据 adapter、当前 native TrainAlgorithm。正式 seed 0 配置仅在内存中把训练终点改为 400 步并禁用 resume；SGD lr 0.01、momentum 0.9、Nesterov、weight decay 0.0005、clip 1、cosine `T_max=80000`、80k sampler、完整数据集、确定性设置和 checkpoint period 2000 均保留。

三个进程拥有独立工作区、processed 数据副本、历史 source 导出、tracker 和 checkpoint。正式 Study 的 shared 数据、归档、Run 资产、index、配置与 SOURCE_MANIFEST 均未被容量诊断改写。

### （二）显存记录

显存计数器从模型放入 CUDA 前启动；每格覆盖初始放置和 0→200 训练、200→400 训练、两次完整评测，以及第 200 步 best、第 400 步 best / latest 保存，共 11 段。仪表只同步 CUDA 和重置峰值计数器，保留 allocator 缓存；eval / checkpoint 委托原生 `super`。两次 eval 返回后 `model.training=true`。

allocated / reserved 是单个进程的 PyTorch CUDA allocator 统计，未覆盖非 Torch context、库分配及其他进程。`nvidia-smi` 按 0.75 秒等待加命令耗时采样；它不提供连续、无遗漏的全局峰值。设备、seed、批量和输入形状与本次证据不同，或者更晚出现新的分配路径时，需要重新观察实际占用。

诊断开始前一条全局样本为 7436 MiB；执行者记录当时已有五个正式 worker。三个 case 的全局采样最大值相对此样本分别多 1117 / 1457 / 2029 MiB，恰比各自 reserved 值多 615 MiB。此差值包含同期背景负载变化，只是一组观测差值，不能当成恒定、精确的 context 大小。

## 三、独立完成与来源核对

1. 控制器保留三个互异 PID：16044、11676、38596，顺序启动且各自 `poll()` 记录退出码 `0`。执行者报告主工具句柄真实终态 `exit 0`。独立 targeted unsandboxed、只读 psutil 查询确认三个 PID 均不存在；没有控制任何训练进程。
2. 三个 worker 报告的实际 SHA256 与控制器保存的 SHA256 完全相同，日志末条成功记录与 case、显存峰值一致。全局 JSONL 逐条重算得到 117 样本、0 错误及完全相同的整体 / 分 case 最大值。
3. CPU 以 `map_location='cpu'` 重载三个 `latest.pt`，全部 optimizer step 400、tracker batch ledger 480。train / test 的 Loss、Accuracy 均有两个 history，scalars 的 optimizer 坐标为 200 / 400；tracker 坐标依次为 train 200、test 240、train 440、test 480。
4. CPU 检查 SGD momentum 张量齐全、所有模型和 momentum 张量有限，调度 `T_max=80000`、`last_epoch=400`、实际 lr 正确。每格 200 / 400 的累计与分段 live batch 计数，以及原生 ledger 相互对应。私有 processed 文件 CPU 重载确认 MNIST train 60,000 / test 10,000、CIFAR10 train 50,000 / test 10,000。
5. GPU 前后冻结快照相同，当前重新哈希的 99 个计算源文件、65 个 expanded plan 文件、16 个官方原始数据文件仍一致。共享归档的 36 个文件与原 Git blob 逐字节哈希对应；三个私有导出共 108 文件、九个私有 processed 文件及其正式来源也全部匹配。

独立审查仅执行 CPU 重载和只读进程查询，并新增本报告。审查的 PID 查询属于独立的已完成观测；CPU 审查脚本不会自行查询或控制进程。

## 四、下一波的规划与实际门

### （一）规划估算

[CONCURRENCY_AUDIT.md](CONCURRENCY_AUDIT.md) 第 92 行保存的 01:41:53 独立快照为两条 CIFAR10 / resnet18 加四条 CIFAR10 / linear，共 9889 / 24455 MiB。执行者另有约 10.3 GB 的同期观察，未保存精确时刻与完整全局峰值，故不把它列为精确峰值。

人为采用 **11,000 MiB** 的保守规划基线，保留原四个 linear 的预算而不扣除。每个新增进程在实测 reserved 之外再留 **1024 MiB** 的非 allocator 预算。以下数字都是规划估算，尚不是这些未来组合的实测联合峰值。

| **未来阶段** | **规划公式 MiB** | **估算占用 MiB** | **估算剩余 MiB** | **超过 5 GiB 门的估算余量 MiB** |
| --- | --- | ---: | ---: | ---: |
| 两条 CResNet 加四条 CNN，按较大的 CIFAR10 CNN 预算 | 11000 + 4 × (842 + 1024) | 18464 | 5991 | 871 |
| 两条 CResNet 加两条 MNIST ResNet | 11000 + 2 × (1414 + 1024) | 15876 | 8579 | 3459 |

### （二）必须可观察的前置条件

1. 第一波 seed 0 / 1 均正式 `succeeded`、flow 成功且精确 PID 身份退出；旧 guard 真实 `exit 0`、`parent_suspension_owned=false` 后，才归档旧 EAGER / GUARD 并更换 membership。容量诊断三个进程也必须全部退出、主控制器真实 `exit 0`。
2. 父控制器 PID / create_time / 完整命令仍匹配原身份，源与计划保持冻结。第二波 `b218a0bde4013fb8`、`3b6bdafdab45a4ef` 仍为 pending，没有已有 checkpoint 或竞争写者。
3. 复用未改的、probe 通过的 helper，新的 current EAGER 只含这两个新 Run，记录精确身份；新 watch 使用 `--hold-at mnist-resnet --expect-run b218a0bde4013fb8 3b6bdafdab45a4ef`。父调度在第一组 MNIST / resnet18 未结束时被协调，暂停后重新确认尚未生成 CIFAR10 / resnet18 组或重复子进程。
4. 投放、完整 train / test 与 checkpoint 窗口持续观察实际全局显存，保留至少 **5120 MiB（5 GiB）** 余量。实际余量不足时停止追加投放；规划求和、预计完成时间和历史快照都不能充当并发锁。
5. 确认原 CNN 子进程已经退出，再进入 MNIST ResNet 阶段；guard 仅在第二波两个 eager 成功、flow 成功、精确身份退出后放行父控制器，避免父调度把仍在运行的同 Run 再次列入 CIFAR10 / resnet18 候选。

若上述门尚未满足，较早的 CIFAR10 / mlp hold 仍是可选的保守协调位置，是否存在真实未结束的四 worker 组必须即时核对。本报告不执行启动、暂停或终止操作。

## 五、证据位置与可复核性

完整原始记录和复核 SHA256 保存在 [GPU_CAPACITY.json](GPU_CAPACITY.json)。原始测量文件位于本机 `.tmp/historical-capacity-20261004-01/`，测量 helper 为 `.tmp/historical_gpu_capacity.py`，SHA256：

`128026214515ea2f5b074d99e768e2bb89d22768b527cb9d473b183a35af9652`

独立 CPU 审查脚本为 `.tmp/historical_capacity_cpu_audit.py`，其 SHA256 保存在正式 JSON 的 `evidence_sha256`。以上 `.tmp` 原始文件和脚本不会随 Git clone 提供；正式 JSON 内已保留完整原始 summary、三个 worker 报告、117 个采样以及冻结快照。
