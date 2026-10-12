# 历史版本速度校准

## 一、目标与状态

2026-10-13：16项历史 raw 校验通过；固定源码容量测量27个原生 Flow Run 全部成功，round2/3在相同预算与严格 checkpoint / history 门下提高总体吞吐。16次 FP32逐样本一致；TF32对照观察到不同 kernel 和预测。见第九、十节。后续 fresh make 使用显式4.5GiB预算，库默认模型系数与 safety=0.5不变，不外推到未知设备或配方。

### 2026-10-13 受控并发验证计划

用户授权持续完成缺陷验证及 main 合并准备。本轮先从既有缓存恢复16个原始文件并逐一核对固定 hash，清单为 [capacity_data_20261012.json](capacity_data_20261012.json)，不覆盖旧 DATA_MANIFEST。CIFAR10 / ResNet18 的单进程历史 / 当前链800-step数值门先通过，再执行以下短程容量矩阵；它不启动32条80000-step长训。

1. CIFAR10 / cnn、resnet18，固定 seeds0–2；各采用相同800-step、eval200完整 test10000、batch250、CPU2线程、FP32、采样 / scheduler 总预算80000。每个模型分别以 round1、2、3 完成相同任务量，使用新的 version / Run ID；初始串行结果作为同 seed 的并发数值基线。
2. 增补 cnn / resnet18 / linear、各seed0的混合任务 round1 / 2 / 3，覆盖不同资源开销。显式 round 是受控实验，不代表默认装箱已采用混合策略；不修改正式矩阵声明。
3. 复用库的展开、FlowRunner、launch_jobs、wait 分组、result / tracker / checkpoint 与 compare，不新建训练循环。仅增加 Study 手动测量入口和来源明确的隔离测量 Study；记录父调度墙钟、子进程阶段时间、全过程GPU显存 / 利用率采样及每个进程的allocated / reserved峰值。输出到本 Study 的新 shared/validation 目录，正式 JSON 和报告留在 docs/。
4. 对照同 seed 的完整 test history 和 latest / best 参数、optimizer、scheduler，严格精确相等；elapsed等运行耗时不参加科学数值门，但差异报告保留。每个 Run 必须真实 succeeded、达到800step且拥有4个完整test点，不能只以退出码通过判定。
5. 每一组开始前核对设备空闲预算；保留原始失败 / OOM / 不稳定结果，不覆盖旧 Run，不在计时期间并行跑 CPU / CUDA 回归。测量结束后再更新启发式或采用显式预算，不直接取消显存保护。结果只限定于本设备 / 精度 / batch / 已测 source，未知配方仍使用保守估计或显式校准预算。

2026-10-11 维护者要求核对当前速度与早年版本。完整 version03 数值终验已通过；CNN / ResNet18 各三次独立进程的串行历史速度校准也已完成，六组最终参数精确一致。当前平均train200相对旧版，CNN增加约3.34%、ResNet18减少约5.13%，波动与阶段拆分见第六节。这不是完整并发吞吐结论，B-022 的容量与受控并发调查保持 investigating。现有长训包含不同并发数及设备竞争，不能直接用各 Run 耗时给新旧版本排名。

2026-10-12 接续：MNIST / ResNet18 的单进程800-step短校准已通过，初始 / 最终参数精确一致，四段完整 test 的整数正确数一致；新增 raw hash、实际 runtime、源码不变检查与 reserved 峰值。当前 reserved 峰值为2472 MiB，尚不含完整进程开销，不据此直接修改正式预算。失败、数字与边界见第八节；这不是新完整矩阵或 CIFAR10 并发验收。

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

正式测量代码和报告属于 main_exp，输出采用独立测量 ID；串行工具的临时 checkpoint 放 `.tmp/`，Flow容量及精度诊断在本Study的shared/validation，均不随clone提供。当前正式训练结果见 [STUDY_REPORT.md](STUDY_REPORT.md)，B-022修复事实见[开发记录](../../../docs/development/record.md#2-b-022--b-023-实测收口)。

串行阶段测量接入既有 recipe 命令，不改正式 Study 声明或覆盖 PREFLIGHT.json：

```powershell
python -u -B studies/main_exp/recipe.py calibrate --data CIFAR10 --model cnn --steps 800 --order old-first
python -u -B studies/main_exp/recipe.py calibrate --data CIFAR10 --model resnet18 --steps 800 --order current-first
```

[flow/execute/speed.py](../flow/execute/speed.py) 使用归档原 train / test 和当前 TrainAlgorithm，不启用前置探针的逐输入 hash 或参数快照。2026-10-12 先搬迁实现，随后增加选定数据与显存证据，`recipe.py calibrate` 入口不变，校准不会自动进入阶段链；第六、七节的既有实测来自当时基线。每侧真实执行四段 train200 / test10000 / checkpoint，首段预热；保留初始与最终参数数值门、各段 Loss / Accuracy 对照。JSON 放 `docs/speed_measurements/`，有唯一 ID、代码与数据 hash。当前训练计时包含段收口与 tracker bookkeeping，评测包含各自的日志职责；保存格式及 best 策略不同，单列保存成本。此命令暂未测父调度开销与并发吞吐，不代表完整 B-022 验收。

校准使用 `algorithm_config(800)`，两侧 latest 均每200steps保存；正式 version03 声明的 latest 周期为2000steps，且 best 按改善保存。短程测量统一频率用于拆分单次保存成本，不能把其全部保存耗时直接外推成正式矩阵的成本。终点当前链另写一次 latest，完整运行总耗时包含这次保存；表中阶段成本保留实际观测，不把两种保存行为误称为相同磁盘合同。

### 2026-10-12 本机接续计划

仅做 MNIST / ResNet18、seed0、800-step 的单进程历史 / 当前链短校准，两侧实际完整 test10000、CPU2线程、batch250、FP32 与 sampler / scheduler 总预算80000不变。新增 `--data`，缺省仍为 CIFAR10；不下载缺失 CIFAR10、不改正式声明或 PREFLIGHT。分别记录 train / test / checkpoint 阶段结束的 allocated / reserved 峰值及实际 runtime、源码清单、原始数据 hash；峰值采集覆盖建构与执行，不只覆盖训练。此单进程结果不证明并发安全，不作为 CIFAR10 或默认精度的修复验收，也不自动写入正式显存预算。

数值门沿用计划的正确样本数一致，而非要求 Accuracy 百分数字符串逐位相同：由完整 test 的百分数和样本数还原整数计数，先验证还原误差不超过 float32 batch Accuracy 的舍入边界 `test_size × 2^-23`（10000样本时约0.0012个样本），且该边界必须小于0.5，再严格比较整数。该检查只排除指标计算及汇总舍入，不能容许一个样本的差异；Loss 与参数门不变。初始与最终参数、原始 Accuracy 差值和整数计数均保留。

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

批量读取已实现。同进程三次导出为0.1821、0.1786、0.1732秒，固定36个文件 manifest 完全相同，见 [ARCHIVE_EXPORT_TIMING.json](ARCHIVE_EXPORT_TIMING.json)。新空目录导出、不同内容拒绝且不覆盖、缺失 header / 错误对象类型 / 截断内容 / 多余尾部拒绝均通过，见 [ARCHIVE_EXPORT_REGRESSION.json](ARCHIVE_EXPORT_REGRESSION.json)。真实归档导入在 benchmark=true / false 两种条件下仍保存原开关，运行条件异常仍拒绝，结果字段见 [ARCHIVE_RUNTIME_REGRESSION.json](ARCHIVE_RUNTIME_REGRESSION.json)。2026-10-12 清理时确认批量导出后的回归 JSON 与此前回归逐字节一致，均未包含独立执行 ID、时间或源码 hash，因此合并为同一份结果文件；两次验证的发生顺序仍分别保留在本节和 [复跑记录第 78 项](RERUN.md)。这些是准备阶段回归与微基准，不是优化后的完整矩阵或端到端速度结果。

20:23，新提交 `8c17d5dbe64d37cdfc09a154d29bb63d47597da8` 的独立八格600-step原代码 / 当前链回归完成，验证 ID 为 main-exp-batch-prefix-20261011-3e158abd。八组24个分段全部通过，参数最大差0，Loss最大差4.44e-16，输入、RNG、正确样本数与 scheduler 一致；源码清单前后未变。证据见 [BATCH_EXPORT_PREFLIGHT.json](BATCH_EXPORT_PREFLIGHT.json)。原 version03 前置报告与64个正式 Run 未覆盖；这份回归证明准备优化后的前缀数值合同，不宣称新的完整80000-step矩阵。

## 八、2026-10-12 本机 MNIST 短校准

### （一）有效复验

使用 `calibrate --data MNIST --model resnet18 --steps 800 --order current-first`，提交为 dev `ce8b3e6` 加本机未提交修复，实际源码 hash 在 [speed-20261012-234436-a294bed5.json](speed_measurements/speed-20261012-234436-a294bed5.json) 中。计算期间源码未变化；8个 MNIST raw 文件 hash 与固定清单一致。设备为 RTX 5090 D v2、Torch2.11.0+cu130，实际 deterministic / cudnn_deterministic=true、benchmark / 两个 TF32=false。

初始和最终参数精确一致，四段 Loss 差为0，step200 / 400 / 600 / 800 的正确样本数两侧分别为9774 / 9882 / 9890 / 9930。原始 Accuracy 最大差约1.42e-14，保留不覆盖；还原整数时的约1e-4样本偏差来自 float32 batch 指标计算，不代表预测分歧。计数门仍拒绝一个正确样本的差异，doctest 覆盖整数还原、一个样本差异与非整数拒绝。

下表训练 / test / checkpoint 时间为剔除第一段预热后的三个段均值；本次只有一次有效独立进程测量，不给速度排名。峰值为包含建构到执行结束的累计 PyTorch 数值，不含 context / 驱动或其他进程开销，也不是完整 GPU 进程峰值。

| **指标** | **历史版** | **当前链** |
| --- | --- | --- |
| train200（各自记账职责） | 8.9764 s | 9.0709 s |
| test10000 | 0.8995 s | 0.9020 s |
| checkpoint（保存格式及次数不同） | 0.1169 s | 0.3565 s |
| 完整800-step执行，含预热 | 39.6712 s | 41.3964 s |
| peak allocated | 2358.78 MiB | 2315.23 MiB |
| peak reserved | 2476 MiB | 2472 MiB |

### （二）失败与测量边界

1. 首轮 [speed-20261012-233820-1a85de53.json](speed_measurements/speed-20261012-233820-1a85de53.json) 为 passed=false：两侧参数精确一致，但旧门直接比较 Accuracy 汇总浮点数，step200 相差约1.42e-14。同时新增 raw 校验未规范清单中的反斜杠，raw_sha256 为空；运行中修正工具，不能把此轮作为固定源码有效测量。
2. 第二轮 [speed-20261012-234002-079ae7b2.json](speed_measurements/speed-20261012-234002-079ae7b2.json) 仍为 passed=false，原指标舍入问题保留；该轮开始时已校验8个raw文件，但修正门的源码变更使 source_unchanged=false。它不作为验收通过或重复稳定性证据。
3. 第三轮 `speed-20261012-234145-70367b20` 在计算后因整数还原检查过严报 ValueError：旧百分数还原为9774.000091552733等值，初版1e-7样本边界没有计入 float32 指标计算。该轮未生成最终 JSON；本机 `.tmp/` 下的 checkpoint 与当前侧日志保留，不随 clone 提供。随后改为第四节说明的可验证 float32 舍入边界，重新用新 ID 完整复验，未重标旧失败。
4. 本机 CIFAR10 raw 只有未完成下载，且没有旧 B-023 的 Run checkpoint；未做 CIFAR10 并发吞吐、kernel 重放或新完整矩阵。make 已支持显式 `system.config.vram_bytes` 预算，仍保留 safety=0.5，但本次结果没有自动写入正式声明，B-022 仍需整体进程峰值与并发校准。

## 九、2026-10-13 CIFAR10 容量与总体吞吐

### （一）来源与验收

使用恢复并逐一校验的 raw 缓存，不下载替代数据：[数据清单](capacity_data_20261012.json)。CIFAR10 / ResNet18 原版与当前链的800-step单进程数值门先通过：[原始对照](speed_measurements/speed-20261013-000633-801ee70a.json)。

实际27次六阶段 Flow 运行，cnn / resnet18各seeds0–2，以及三个模型各seed0的混合组；每个组以round1/2/3完成相同任务量。每次800 optimizer steps、4次完整 test10000、batch250、CPU2线程、FP32，采样和cosine总预算80000不变。latest / best checkpoint及test history与相应串行 seed严格零容差一致，steps和曲线点数完整，无 OOM / 重试 / 任务遗漏。计时字段按设计不同，不用于科学相等判定。

最终测量[capacity-20261013-005151-ea033fed](speed_measurements/capacity-20261013-005151-ea033fed.json)完整、通过、来源未变。基于 dev ce8b3e6 加当时未提交修复；库由本机独立只读字节快照导入，JSON记录实际 runtime_package和逐文件hash。它不是修复后所有后续源码都已跑过27次；后续边界补丁另做回归和八格前置复验。

### （二）实测数字

| **任务组（均3个任务）** | **round1 总墙钟 s** | **round2 总墙钟 s** | **round3 总墙钟 s** | **round3 / 串行总体加速** |
| --- | ---: | ---: | ---: | ---: |
| CNN | 147.81 | 102.74 | 45.18 | 3.27× |
| ResNet18 | 236.15 | 166.31 | 110.99 | 2.13× |
| 混合 CNN / ResNet18 / linear | 188.83 | 139.78 | 88.28 | 2.14× |

这是一次受控矩阵、不是多轮统计速度排名。同时运行的单个ResNet18延时增加，但总完成量更快；不把单 seed 延时当总体退化。父墙钟包含启动、wait及2秒GPU采样的尾部观察粒度；串行最大子进程Flow约76.54s、execute约74.77s，三任务总父墙钟236.15s，未观察到足以解释原串行安排的额外Flow开销。精细父阶段逐事件计时未另建框架。

ResNet18每进程reserved峰值2998MiB；串行整卡从首样4016MiB至峰值7903MiB，增量3887MiB包含context / 驱动及背景变化，不等同独占进程精确值。三进程整卡峰值14918MiB（总24455MiB）；平均采样利用率round1约40.84%、round3约88.91%，包括启动 / 等待。CNN reserved650MiB、round3整卡峰值7937MiB；混合round3峰值9529MiB。WDDM未提供可靠逐进程显存，本机还有低水平背景GPU活动，不能声称隔离无竞争。

后续 main_exp统一显式预算4.5GiB，覆盖已测最重峰值并保留余量，仍仅用50%可用显存装箱；约20GiB空闲时重型组自动由单任务改为两个任务。不改库未知模型的默认系数、不让allocator allocated代替进程预算。轻型组也受统一预算限制，可能少于旧自动并行数；这是当前保守配置取舍，不是普遍最优装箱。

### （三）失败与保全

首轮[原始部分报告](speed_measurements/capacity-20261013-001913-81199383.json)只记录4个成功组，不能把其中临时passed=true读成全矩阵通过。ResNet18 round2有两条成功、一条被freeze拒绝，原因是两份库文件的行尾字节变化；函数逻辑未变，未关闭freeze绕过。随后工具遇缺失capacity.json退出；增加失败记录及complete门后换新ID与独立快照重跑。[中止说明](speed_measurements/capacity-20261013-001913-81199383-aborted.json)记录完整状态。原Run、失败日志与源码快照保留在本机shared/validation / .tmp，不随clone提供。

## 十、2026-10-13 固定权重精度与 kernel 复验

### （一）控制与实际结果

使用本次容量测量CNN seed0的800-step best，不是旧 B-023 的76800-step checkpoint。旧权重SHA256已知但未找回；不能声称重放了旧失败的具体kernel。[32次原始评测](speed_measurements/precision-20261013-013123-e9bd81bc.json)和[跨精度分析](speed_measurements/precision-20261013-013123-e9bd81bc-analysis.json)记录checkpoint、输入、模型、逐样本预测及logits哈希。全部10000张test、batch250，benchmark实际false、matmul TF32false；仅cuDNN TF32开关变化。

| **精度** | **串行新进程** | **三并发新进程** | **正确数** | **组内逐样本 / logits** |
| --- | ---: | ---: | ---: | --- |
| FP32 | 8 | 8 | 6212 / 10000，与自身best一致 | 16次精确一致 |
| TF32允许 | 8 | 8 | 6212 / 10000 | 16次精确一致，但与FP32不同 |

两种精度输入和模型哈希相同；改变开关使test index687、2821预测改变，logits最大绝对差0.0034226179。总正确数相同不代表逐样本预测一致；没有topk / argmax分歧。四组首轮取得真实CUDA profiler trace：FP32使用SIMT SGEMM / Winograd，TF32出现cutlass tf32 tensor-op与sm80_xmma tf32卷积kernel。名称只证明本次执行路径，不能还原旧权重上的某个kernel选择。

### （二）结论与边界

支持的严格对照合同是两侧显式同精度、benchmark不被归档导入覆盖，并核对实际runtime；不是只设置seed / deterministic后依赖Torch默认值。PyTorch说明TF32卷积的内部尾数精度低于FP32，且浮点结果受算法 / 环境影响；确定性开关不等于相同精度。[数值精度说明](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html)、[可复现性说明](https://docs.pytorch.org/docs/2.11/notes/randomness.html)。此处机制解释是结合官方说明与本次实测的推断，不宣称找到旧反例的唯一literal kernel。

B-023按严格FP32合同的源码修复与实测复验收口：旧version03完整矩阵已验收，本次修复后固定best复验通过，原benchmark覆盖和TF32默认精度不明确的缺陷不再沿用。默认TF32的跨路径严格等同不是受支持合同；旧失效权重的kernel考古作为非发布阻断的研究边界保留。完整长矩阵本轮没有重跑，也不以32次短评测替代新长曲线验收。

复跑入口为 `python studies/main_exp/recipe.py diagnose-precision --checkpoint <本机固定CNN best.pt>`；它会创建新的测量ID，不改变传入权重或旧正式结果。profiler trace、逐样本logits、worker日志在本Study的shared/validation，不随clone提供。
