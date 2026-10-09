# main_probe 计划

## 一、目标

在同一设备、同一确定性条件下，比较固定 main `98648f3a5c7db7dccf3ca806410d5b6fdee9484c` 的原代码与当前 RPipe 的 60-step 计算是否逐段一致，确认 RPipe 改写没有改变数值行为。这是短预算探针，不代替 [main_exp](../../main_exp/README.md) 的长曲线复现。

## 二、配方

MNIST / CIFAR10 × linear / mlp / cnn / resnet18，seed0；全量 train / test，batch250 / test1000，连续 60 optimizer steps，step30 / 60 完整 test；SGD lr0.1、momentum0.9、Nesterov、wd0.0005，无梯度裁剪，cosine T_max60；当前无 BN CNN、原生 Factory 和 Kornia 模型前增强。全 train 统计由归档原 Stats(dim1) 按顺序 batch250 重算。两边都用 deterministic=true、benchmark=false、`CUBLAS_WORKSPACE_CONFIG=:4096:8`。

## 三、两个入口

1. `rpipe make / launch studies/main_probe`：按 [study.yaml](../study.yaml) 跑当前 RPipe 的 8 train + 8 eval，得到 RPipe 自己的结果、曲线和 best eval。
2. [probe.py](../probe.py)：在一个进程里先跑原 main 的未修改 train / test，再跑当前 TrainAlgorithm 与 EvalAlgorithm，逐段比较。需要同进程才能核对初始化 RNG、采样索引、实际像素与标签、Kornia 后的模型输入。`prepare` 只做 CPU 核对，`run --device cuda` 跑八格。

## 四、门限

step30 / 60 浮点参数与 SGD momentum 状态 atol1e-6 / rtol1e-5，整数 buffer 与 optimizer 配置严格一致；train / test Loss 绝对差≤1e-6，正确样本数相同；初始化参数、CPU 与 CUDA RNG、训练样本索引、实际输入、各段 scheduler 一致。两边按 test Loss 选 best 且选择与权重一致；独立 eval 正确样本数相同、Loss 差≤1e-6。只有八个组合全部通过且来源不变才算完整通过。

## 五、待办

`probe.py` 目前自己比较两边状态。计划把原 main 一侧的输出写成 RPipe Run 目录格式，再用 `python -m rpipe compare` 做比较，删掉 `probe.py` 里的比较代码。这一步在本轮运行前完成。
