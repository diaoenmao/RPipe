# 当前设备的 main 60-step 独立验证计划

## 一、目标与来源

2026-10-04。补齐最新 `dev` 在当前 cu130 设备上的当前 main 配方验证。固定 `main` 为 `98648f3a5c7db7dccf3ca806410d5b6fdee9484c`、`dev` 为 `8bccbac321d4c3ac1ea9892a5e774c114e0298c6`。此前 cu128 的 4/8 默认门及 8/8 确定性门保留，不能当作本次实测。

入口为 [current_device.py](../current_device.py)。每次 CPU 准备创建独立 `.tmp/main-current-device-*` 工作区，原始数据只读复制，原 main 源码逐 Git blob 导出核对。旧 Study、历史长训练的注册器、源码、配置、index 和 manifest 均不修改。

## 二、配方与阶段

MNIST / CIFAR10 × linear / mlp / cnn / resnet18，seed 0；全量 train/test，batch250/test1000，连续 60 optimizer steps，step30/60 完整 test；SGD lr0.1、momentum0.9、Nesterov、wd0.0005，无梯度裁剪，cosine T_max60。使用当前无 BN CNN、原生 Factory 和 Kornia 模型前增强。全 train 的统计由归档原 Stats(dim1)、顺序 batch250 重新计算，保留完整精度，不读取旧 JSON 数字作为新结果。

1. `prepare` 只进行 CPU 数据、统计、初始化/RNG 和固定输入 forward 核对，不启动 GPU 训练。原 dataset 的现代 torch 缓存与历史 pickle 缓存分开。
2. 根线程安排空隙后显式执行 `run --device cuda`。每格先原 main 的未修改 train/test，再当前 TrainAlgorithm；两边统一 deterministic=true、benchmark=false、CUBLAS_WORKSPACE_CONFIG=:4096:8。
3. 原 main 按 test Loss 选 step30/60 的 best；当前按相同口径保存 best。各自重新构造模型和 DataLoader，原 test_model.test 与当前 EvalAlgorithm 加载各自 best 进行独立完整评测。

## 三、执行前门限

step30/60 浮点参数及 SGD momentum 状态 atol1e-6/rtol1e-5、整数 buffer 与 optimizer 配置严格一致；train/test Loss 绝对差≤1e-6，正确样本数严格相同；初始化参数/Torch CPU及CUDA RNG、训练样本索引、实际像素/标签、Kornia 后核心模型输入、各段 RNG 与 scheduler 一致。每段 train 实际7500样本/30batch、test实际10000样本/10batch；独立 eval 两边也各10000样本/10batch。两边 best 选择与权重一致，独立 eval 正确样本数相同且 Loss 差≤1e-6，各自 eval 与 own-best 同权重评测也满足此门。

prepare 前后、run 每格及终验严格核对生产 Python 与本脚本 SHA；原 main 逐 Git blob 导出，终验重新核对所有归档文件。任何来源变化均使门失败。诊断子集单独记录 `selected_complete/selected_passed`；只有准确8个不同组合、全部通过且来源不变时，`full_matrix_complete/full_matrix_passed` 才表示完整验收。

这验证统一确定性条件下的当前 main 配方，不代替原默认 benchmark=true 的重复性结论，不代替正在执行的 80000-step 历史图片验收。参数字段只映射包装器的 `model.` / `net.` 前缀，不删除未知 buffer。独立 eval 的 torchvision 元组没有原 id 字段，因此 test 对比直接使用实际像素、标签和核心输入，不伪造 test id 观测。

## 四、命令与证据

```powershell
python -B studies/main_reproduction/current_device.py prepare --workspace .tmp/main-current-device-gpu-20261004-ready
# 根线程统一安排，CPU prepare 完成不自动启动下条命令。
python -B studies/main_reproduction/current_device.py run --workspace .tmp/main-current-device-gpu-20261004-ready --device cuda
```

独立工作区保留 PREPARED、ARCHIVE_MANIFEST、SOURCE_MANIFEST、ENVIRONMENT、COMPARISON JSON；逐格原 step30/60/best 存档、当前 step30/60 存档、TensorBoard 日志、当前 checkpoint/tracker/log。GPU 执行时间由根线程安排，尚不记录任何新 GPU 数值通过。

预计采用单进程串行；原 train 模型在当前 train 前释放，当前 train 模型在独立 eval 前释放，逐格完成后清理对象和 CUDA 缓存。显存主要来自单条 batch250 ResNet18 激活和 cuDNN 工作区；本轮尚无 GPU 实测。为排班保守预留5GiB余量，运行时每格记录本进程 Torch allocated/reserved 峰值；该估计和计数均不覆盖其他进程占用。旧设备确定性矩阵量级约数分钟，可先预留2–5分钟；该范围仅是待校准的排班参考，当前共享 GPU/CPU 负载可能增加耗时，不承诺时长或据此判断算法速度。

## 五、当前 CPU 验证与边界

冻结版 CPU 准备已在本设备确认8/8初始化、Torch CPU RNG与固定输入 forward 门通过；两个数据集全部 train/test 像素、标签一致，Stats由完整 train 重新计算。`.tmp/main-current-device-cpu-20261004-v7/` 的 MNIST/linear 原/当前连续60step及独立eval通过全部硬门：两段参数最大差0、optimizer逐字段exact、scheduler/RNG一致，实际样本/batch全量，独立eval两边Loss均0.40258089601993563、Accuracy均90.73000106811523%，best均step60；每边独立eval也与自己的best完全相同。单格数值核验用时6.119秒，不含解释器导入和CPU准备。

该结果显式记录`scope=diagnostic_subset`、`selected_complete/selected_passed=true`，`full_matrix_complete=false/full_matrix_passed=null`；原归档40个文件不变，生产src与脚本在准备、训练前后不变。这是一格CPU诊断，不能当作8格GPU结果。独立GPU-ready工作区为`.tmp/main-current-device-gpu-20261004-ready/`，按相同SHA完成CPU准备后交根线程安排CUDA矩阵。

早期失败输出保留在`.tmp/main-current-device-cpu-20261004`、`-v2`与`-v4`：分别为单进程导入两个旧CLI的重复argparse参数、重置原cfg后遗漏dataset形状、观测generator缺少`__len__`。均修复在外部适配器，未修改归档main或生产src。`-v3`至`-v6`为准备/诊断开发版本，`-v7`为冻结版CPU证据；各自manifest保留，脚本修改后不可直接重用旧workspace运行。临时证据不会随Git clone提供。
