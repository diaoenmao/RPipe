# 历史版本速度校准

## 一、目标与状态

2026-10-11 维护者要求核对当前速度与早年版本。完整 version03 数值终验已通过；CNN / ResNet18 各三次独立进程的串行历史速度校准也已完成，六组最终参数精确一致。当前平均train200相对旧版，CNN增加约3.34%、ResNet18减少约5.13%，波动与阶段拆分见第六节。这不是完整并发吞吐结论，B-022 的容量与受控并发调查保持 investigating。现有长训包含不同并发数及设备竞争，不能直接用各 Run 耗时给新旧版本排名。

## 二、对照条件

1. 早年对照固定为本 Study 已采用的 `4ccb28d0496110253e9f8e3f3df658853f07996b`，通过 Git blob 导出；当前链记录实际提交及测量脚本 hash。归档代码不得修改。
2. 对齐设备、数据 hash、模型、seed、batch250、完整 test10000、eval200、CPU2线程、连续采样和 optimizer / scheduler 总预算80000。两个实现均 deterministic=true、benchmark=false、cuDNN / matmul TF32=false。
3. 历史默认精度、checkpoint 格式与保存策略的差异单列；默认设置计时不能用于证明 Flow 本身变慢。旧环境缺少可核对的硬件和依赖记录时，以当前设备重跑作为同条件对照。

## 三、测量顺序与边界

1. 先做 CIFAR10 CNN / ResNet18 的串行短程诊断，原代码和当前链交替执行。CUDA 同步后记录训练段、完整 test、checkpoint、初始化、整体墙钟和显存峰值。首次段作为预热，正式比较取后续段；每侧至少三次独立进程复测。
2. 初步可复用现有成对前置链定位阶段耗时，但其输入 hash、参数快照和比较均有额外开销。这种诊断只定位候选瓶颈，不替代正式吞吐基准，也不得覆盖 PREFLIGHT.json 或完整实验结果。
3. 随后去除探针额外观测开销，保留训练、评测及保存的真实职责，核对两侧数值一致性后做串行、round2、round3 对比。记录外部 GPU 进程、持续利用率、显存峰值和总体墙钟；比较总完成量，不能只看单个 seed 延时。
4. 调度计时从父进程展开、创建子进程到各阶段入口分别记录，与子进程内部计算分开。保存策略不同则拆分保存开销，不把额外产物的成本计入纯训练速度。
5. 依据测量决定优化，保留失败、OOM 和波动记录。不得缩短正式预算、减少评测、放宽数值门或修改已完成 Run 来取得加速。

## 四、证据位置

正式测量代码和报告属于 main_exp，输出采用独立测量 ID；大体积临时 checkpoint 放 `.tmp/`，报告注明其不随 clone 提供。当前正式训练结果见 [STUDY_REPORT.md](STUDY_REPORT.md)，问题记录见 [B-022](../../../docs/development/bugs.md#b-022-gpu-容量估算与调度过于保守重型训练被串行安排)。

串行阶段测量接入既有 recipe 命令，不改正式 Study 声明或覆盖 PREFLIGHT.json：

```powershell
python -u -B studies/main_exp/recipe.py calibrate --model cnn --steps 800 --order old-first
python -u -B studies/main_exp/recipe.py calibrate --model resnet18 --steps 800 --order current-first
```

[speed.py](../speed.py) 使用归档原 train / test 和当前 TrainAlgorithm，不启用前置探针的逐输入 hash 或参数快照。每侧真实执行四段 train200 / test10000 / checkpoint，首段预热；保留初始与最终参数数值门、各段 Loss / Accuracy 对照。JSON 放 `docs/speed_measurements/`，有唯一 ID、代码与数据 hash。当前训练计时包含段收口与 tracker bookkeeping，评测包含各自的日志职责；保存格式及 best 策略不同，单列保存成本。此命令暂未测父调度开销与并发吞吐，不代表完整 B-022 验收。

校准使用 `algorithm_config(800)`，两侧 latest 均每200steps保存；正式 version03 声明的 latest 周期为2000steps，且 best 按改善保存。短程测量统一频率用于拆分单次保存成本，不能把其全部保存耗时直接外推成正式矩阵的成本。终点当前链另写一次 latest，完整运行总耗时包含这次保存；表中阶段成本保留实际观测，不把两种保存行为误称为相同磁盘合同。

## 五、首轮实测（2026-10-11）

CIFAR10 CNN，old-first，800-step，后面三个 200-step 段均值。初始与最终参数精确一致，四段 test Loss 差不超过 1e-6，Accuracy 一致。证据为 [speed-20261011-195455-7dec6afb.json](speed_measurements/speed-20261011-195455-7dec6afb.json)。这是单次测量，尚未完成三个独立进程复测及 ResNet18 对照。

| **阶段 / 范围** | **旧版** | **当前链** |
| --- | --- | --- |
| train200，当前包含段收口与 bookkeeping | 10.9942 秒 | 11.2924 秒 |
| test10000，包含各自评测日志 | 0.8008 秒 | 0.8351 秒 |
| checkpoint，保存策略与格式不同 | 0.0202 秒 | 0.0984 秒 |
| 完整800-step运行，包含预热 | 47.8654 秒 | 49.0768 秒 |
| PyTorch peak allocated | 443.97 MiB | 443.65 MiB |

这次串行测量未出现数倍的训练链速度差。保存开销相对增加，但绝对值较小；现有证据不能解释长训的全部耗时，也不能判断并发吞吐与容量估算是否合理。PyTorch allocated 不包含 CUDA context、其他进程和驱动开销，不能直接用来修改显存装箱上限。

20:01，CNN 三次独立进程（old-first / current-first / old-first）全部完成，三组初始与最终参数精确一致，完整 test 数值门通过。剔除各自第一段预热后，进程段均值的总体平均：旧版 train200=10.9545 秒，当前=11.3207 秒（约增加3.34%，当前包含段收口与记账）；完整 test 为0.8119 / 0.8183 秒；checkpoint 为0.0197 / 0.0953 秒。旧 / 当前训练段的进程间总体标准差为0.0435 / 0.1973 秒。ResNet18 复测仍在执行，尚无该模型完整对照结论。

20:06，ResNet18 的 old-first / current-first 两次独立进程对照均通过，最终参数精确一致。第三次仍在原生会话66360顺序执行；不得仅凭首轮当前15.3667秒、旧17.4685秒的训练段均值宣布完整性能结论。先完成剩余测量，再汇总进程间波动。

## 六、三次独立进程串行复测

CNN / ResNet18 各三次独立进程，old-first / current-first 交替。每个进程先剔除首个200-step预热段，再计算其余三个段的均值，下表对三个进程均值取平均与总体标准差。全部六组初始和最终参数精确一致，四段完整 test 的 Loss / Accuracy 数值门均通过。每个训练段200steps，test10000，保存策略与格式不同；不含父调度计时，当前训练段包含段收口与记账。

| **模型** | **实现** | **阶段** | **均值（秒）** | **进程间总体标准差（秒）** | **进程均值范围（秒）** | **独立进程数** |
| --- | --- | --- | --- | --- | --- | --- |
| cnn | old | 训练段（含各自记账） | 10.9545 | 0.0435 | 10.8939–10.9942 | 3 |
| cnn | old | 完整评测 | 0.8119 | 0.0099 | 0.8008–0.8249 | 3 |
| cnn | old | checkpoint | 0.0197 | 0.0004 | 0.0193–0.0202 | 3 |
| cnn | current | 训练段（含各自记账） | 11.3207 | 0.1973 | 11.0946–11.5753 | 3 |
| cnn | current | 完整评测 | 0.8183 | 0.0184 | 0.7927–0.8351 | 3 |
| cnn | current | checkpoint | 0.0953 | 0.0023 | 0.0930–0.0984 | 3 |
| resnet18 | old | 训练段（含各自记账） | 17.5546 | 0.0751 | 17.4685–17.6514 | 3 |
| resnet18 | old | 完整评测 | 1.2695 | 0.0075 | 1.2617–1.2797 | 3 |
| resnet18 | old | checkpoint | 0.1127 | 0.0019 | 0.1104–0.1151 | 3 |
| resnet18 | current | 训练段（含各自记账） | 16.6534 | 0.9433 | 15.3667–17.6016 | 3 |
| resnet18 | current | 完整评测 | 1.2102 | 0.1045 | 1.0625–1.2847 | 3 |
| resnet18 | current | checkpoint | 0.3556 | 0.0137 | 0.3364–0.3675 | 3 |

原始报告：

- [speed-20261011-195455-7dec6afb.json](speed_measurements/speed-20261011-195455-7dec6afb.json)
- [speed-20261011-195658-d36bd8a7.json](speed_measurements/speed-20261011-195658-d36bd8a7.json)
- [speed-20261011-195847-df1c84fc.json](speed_measurements/speed-20261011-195847-df1c84fc.json)
- [speed-20261011-200037-45c800e6.json](speed_measurements/speed-20261011-200037-45c800e6.json)
- [speed-20261011-200314-c5e7464a.json](speed_measurements/speed-20261011-200314-c5e7464a.json)
- [speed-20261011-200554-81ff5851.json](speed_measurements/speed-20261011-200554-81ff5851.json)

## 七、正式 Flow 时间边界与准备优化

[NATIVE_FLOW_TIMING.json](NATIVE_FLOW_TIMING.json) 核对当前 index 的全部64个成功 Run 日志时间戳。训练准备平均6.0231秒、execute平均4605.1953秒、execute之后收尾平均0.0781秒；准备与收尾占训练 Run 累计 Flow 时长约0.1323%。eval 准备平均5.8959秒、execute平均1.2885秒，准备与收尾占其累计 Flow 时长约82.23%。这些边界不包含 Flow start 之前的 Python 导入、父进程启动和整轮 Study process，不能解释为完整调度成本。

32条训练的 Flow 时长之和除以首条开始至最后结束的时间跨度为2.7851，说明此次 round3 的长训实际存在并行重叠；这不是相对串行的实测速比，也不能证明所有模型在 round3 下最优。原始日志只在本机 runs 中保留，JSON 提供已核对的逐 Run 时间边界。

归档导出当前每个 blob 启动一次 `git cat-file`。2026-10-11 同进程三次完整导出固定36个文件，分别耗时3.0772、2.9644、3.0156秒。下一步改为 Git 原生 `cat-file --batch` 一次读取所有 blob，逐条核对 oid、类型、长度和最终分隔，并保持已有文件字节比对、不同内容拒绝及 manifest 不变。该优化只减少准备阶段的进程创建，不改训练、数据、精度或调度容量系数。验收包括所有归档文件与 Git 字节相同、错误内容仍拒绝、真实导入开关回归及新进程计时；正式数值复验另用新版本，不覆盖 version03。

批量读取已实现。同进程三次导出为0.1821、0.1786、0.1732秒，固定36个文件 manifest 完全相同，见 [ARCHIVE_EXPORT_TIMING.json](ARCHIVE_EXPORT_TIMING.json)。新空目录导出、不同内容拒绝且不覆盖、缺失 header / 错误对象类型 / 截断内容 / 多余尾部拒绝均通过，见 [ARCHIVE_EXPORT_REGRESSION.json](ARCHIVE_EXPORT_REGRESSION.json)。真实归档导入在 benchmark=true / false 两种条件下仍保存原开关，运行条件异常仍拒绝，见 [ARCHIVE_BATCH_RUNTIME_REGRESSION.json](ARCHIVE_BATCH_RUNTIME_REGRESSION.json)。这些是准备阶段回归与微基准，不是优化后的完整矩阵或端到端速度结果。

20:23，新提交 `8c17d5dbe64d37cdfc09a154d29bb63d47597da8` 的独立八格600-step原代码 / 当前链回归完成，验证 ID 为 main-exp-batch-prefix-20261011-3e158abd。八组24个分段全部通过，参数最大差0，Loss最大差4.44e-16，输入、RNG、正确样本数与 scheduler 一致；源码清单前后未变。证据见 [BATCH_EXPORT_PREFLIGHT.json](BATCH_EXPORT_PREFLIGHT.json)。原 version03 前置报告与64个正式 Run 未覆盖；这份回归证明准备优化后的前缀数值合同，不宣称新的完整80000-step矩阵。
