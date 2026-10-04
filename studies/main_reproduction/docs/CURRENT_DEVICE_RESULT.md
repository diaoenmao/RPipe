# 当前设备的 main 60-step 完整复现结果

## 一、摘要

2026-10-04，本机当前 main 配方的完整 GPU 矩阵 **8/8 通过**，原 main 与最新 dev 的 step30/60 共16段参数及整数 buffer、SGD optimizer 状态全部严格一致，scheduler 与 Torch CPU/CUDA RNG 一致。完整 test 的 Loss 差为0；train/test 段均值最大 Loss 差为6.6613381477509392e-16，正确样本数全部相同。八格均按 test Loss 选择 step60，两边独立 eval 与对方及各自 best 的指标一致。[完整原值与来源](CURRENT_DEVICE_RESULT.json)和[独立 CPU 重载审计](CURRENT_DEVICE_CPU_RELOAD_AUDIT.json)均保留。

本结论限定为 seed0、60步、相同确定性控制和准备好的统计精度 profile。无 profile 默认常量、原默认非确定性条件和历史 README PNG 的80000步目标不在本次验收范围。

## 二、来源、环境与执行

固定原 main 为`98648f3a5c7db7dccf3ca806410d5b6fdee9484c`，最新 dev 为`8bccbac321d4c3ac1ea9892a5e774c114e0298c6`。使用[冻结脚本](../current_device.py)，SHA-256为`b2059c369e2067d9f9f5369f4e1162fc5b8f546f4e7b930424c55e8ebca1f461`；原 main 的40个文件逐 Git blob 导出并在结束时重新核对，92个生产 Python 与本脚本 SHA 在 prepare 前后、每格和终验均保持不变。原 main blob、生产 src、历史实验注册器与配置均未修改。

本次环境为 Python3.13.9、Torch2.11.0+cu130、NumPy2.4.4、CUDA build13.0、RTX5090 D v2，CPU threads2。执行使用`deterministic=true`、`cudnn.deterministic=true`、`benchmark=false`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`。原 main train/test与当前原生 Factory / TrainAlgorithm / EvalAlgorithm均在这组条件下重新运行；没有使用此前cu128结果替代本机计算。

根线程显式启动单进程串行8格矩阵，正式工具句柄49399返回`exit_code=0`；没有`--combination`子集选择。原始结果记录`scope=full_matrix`、`complete/passed=true`、`selected_complete/selected_passed=true`、`full_matrix_complete/full_matrix_passed=true`。矩阵本体于北京时间2026-10-04 01:56:36至01:59:11完成，原始 UTC 时戳与完整数字保留于JSON。

## 三、数据与统计条件

MNIST train60000/test10000、CIFAR10 train50000/test10000，原/当前完整数据集像素和标签已在CPU准备阶段逐数组核对；17份raw文件（含CIFAR压缩归档）复制后SHA一致。训练从完整 train 中按原采样规则连续取60batch，每批250，共15000个样本观测；这里的60步并不表示遍历全部 train。step30、step60各完整评测test10000样本/10batch，test batch1000。每段train7500样本/30batch，连续两段之间不中断采样或重建训练状态；各自独立eval也实际完成10000样本/10batch。

归档原`Stats(dim=1)`在完整train上按顺序batch250重新计算，完整精度保存到`native-data/mnist/stats.yaml`与`native-data/cifar10/stats.yaml`；当前DataFactory读取这些profile，而非自行重新计算Stats。实际值为：

| **数据集** | **mean** | **std** |
| --- | --- | --- |
| MNIST | 0.13066047430038452 | 0.30810844898223877 |
| CIFAR10 | 0.4913996756076813, 0.4821585416793823, 0.4465310871601105 | 0.24703294038772583, 0.2434857338666916, 0.26158812642097473 |

本次没有验证缺少profile时的默认归一化常量，也没有覆盖DataFactory自动重算Stats的行为。

## 四、配方与数值门

MNIST/CIFAR10 × linear/mlp/cnn/resnet18，seed0；CNN使用当前main的无BN结构，ResNet18保留原main结构；Kornia增强与归一化在模型前执行。连续60个optimizer steps，eval30，SGD lr0.1/momentum0.9/Nesterov/wd0.0005，无梯度裁剪，cosine T_max60，训练不resume。Loss-best采用test Loss最小值；原`test_model.test`和当前`EvalAlgorithm`分别重新构造模型及DataLoader，加载各自best完整评测。

执行前门为浮点参数及momentum atol1e-6/rtol1e-5，整数buffer、optimizer配置、正确样本数、scheduler和Torch RNG相同，train/test Loss绝对差≤1e-6；实际sample/batch计数也单独作为硬门。实测16段参数最大差为0，optimizer逐字段exact；train样本索引、实际图像/标签及增强归一化后的核心输入hash相同，test实际像素/标签/核心输入hash相同。当前test元组没有id字段，未伪造id比较。

完整test的Loss和Accuracy两段均相同。train段摘要最大Loss差为6.6613381477509392e-16、最大Accuracy差为1.4210854715202004e-14个百分点，符合浮点均值累加顺序差异，且远小于门限；JSON保留原/当前未经显示舍入的数值。全部best权重严格相同，八格独立eval的跨实现Loss差、各自eval相对own-best的Loss差均为0，正确样本数一致。

## 五、step60完整 test 结果

下表为本次GPU运行的真实step60 test结果；原main、当前dev及各自独立eval在这些test值上相同。Accuracy显示为百分比并仅在表格中保留两位小数，JSON保留完整精度。

| **数据集** | **模型** | **Loss** | **Accuracy %** | **正确/总样本** | **两边best step** |
| --- | --- | ---: | ---: | ---: | ---: |
| MNIST | linear | 0.402580925822258 | 90.73 | 9073/10000 | 60 |
| MNIST | mlp | 0.271013684570789 | 91.82 | 9182/10000 | 60 |
| MNIST | cnn | 2.28220155239105 | 11.35 | 1135/10000 | 60 |
| MNIST | resnet18 | 0.0753846146166325 | 97.69 | 9769/10000 | 60 |
| CIFAR10 | linear | 2.96587171554565 | 29.52 | 2952/10000 | 60 |
| CIFAR10 | mlp | 1.68994369506836 | 40.30 | 4030/10000 | 60 |
| CIFAR10 | cnn | 1.83429354429245 | 32.94 | 3294/10000 | 60 |
| CIFAR10 | resnet18 | 1.44572395086288 | 46.42 | 4642/10000 | 60 |

## 六、独立核验、资源与边界

[CPU重载审计](CURRENT_DEVICE_CPU_RELOAD_AUDIT.json)在本次GPU结束后独立读取step30/60、latest与best存档、TensorBoard原始事件、当前tracker/scalars及resume日志：8个不同组合全通过，16段模型/optimizer/scheduler/RNG核验一致，来源92文件、归档40 Git blobs及17份raw文件SHA均通过，`errors=[]`。审计没有另起训练或推理；当前独立eval末态没有单独`.pt`，其在线权重比较由自己的未变best、明确resume路径及日志/tracker证据支持。

矩阵本体墙钟155.268802秒，不含解释器前置导入及CPU准备；同时有历史长训练共享CPU/GPU资源，因此不把此时间用作算法性能benchmark。本进程Torch allocator的最高allocated为2013.302246MiB，最高reserved为3630MiB，均出现在CIFAR10/resnet18；这不是设备总峰值，也不含其他进程或Torch allocator之外的占用。

本轮补齐了最新dev在当前设备上、相同统计profile与确定性条件下的当前main 60-step验收。原默认benchmark=true/非确定性条件没有在本轮重测；无profile默认常量与其他seed/配方仍未覆盖；历史README PNG的80000步矩阵另行验收，本结果不代表该目标通过。

## 七、复跑与证据保存

执行方式和预先门限见[本机验证计划](CURRENT_DEVICE_PLAN.md)。已完成工作区不能直接再次运行，复跑需选择新工作区并重新prepare：

```powershell
python -B studies/main_reproduction/current_device.py prepare --workspace .tmp/main-current-device-new-run
python -B studies/main_reproduction/current_device.py run --workspace .tmp/main-current-device-new-run --device cuda
```

原始工作区为[.tmp/main-current-device-gpu-20261004-ready](../../../.tmp/main-current-device-gpu-20261004-ready/)，包含COMPARISON、PREPARED、ENVIRONMENT、SOURCE_MANIFEST、ARCHIVE_MANIFEST以及逐格原/当前step存档、best、TensorBoard、tracker和日志；这些临时文件不会随Git clone提供。本目录[正式JSON](CURRENT_DEVICE_RESULT.json)保存本次全部比较、准备、环境、来源SHA和完整数字，独立审计保留存档hash与核验边界。
