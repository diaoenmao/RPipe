# main_probe 计划

## 一、目标

在同一设备、同一确定性条件下，比较固定 main `98648f3a5c7db7dccf3ca806410d5b6fdee9484c` 的原代码与当前 RPipe 的 60-step 计算是否逐段一致，确认 RPipe 改写没有改变数值行为。这是短预算探针，不代替 [main_exp](../../main_exp/README.md) 的长曲线复现。

## 二、配方

MNIST / CIFAR10 × linear / mlp / cnn / resnet18，seed0；全量 train / test，batch250 / test1000，连续 60 optimizer steps，step30 / 60 完整 test；SGD lr0.1、momentum0.9、Nesterov、wd0.0005，无梯度裁剪，cosine T_max60；当前无 BN CNN、原生 Factory 和 Kornia 模型前增强。全 train 统计由归档原 Stats(dim1) 按顺序 batch250 重算。两边都用 deterministic=true、benchmark=false、`CUBLAS_WORKSPACE_CONFIG=:4096:8`。

## 三、统一 Flow 入口

`rpipe make / launch studies/main_probe` 展开八个成对 Run，每个 Run 只计算自己的 data/model 组合，包含两侧训练和各自 best 独立评测。Algorithm source 为 `main_probe`，recipe 仅注册本 Study 的 Data 和成对 Algorithm；CPU 准备在库 prepare 的 recipe 注册时完成，以便原版全 train Stats 在 Data / Model 构造之前落盘。工作区位于该 Run 的 `assets/probe/`，不得共享可变原版 config。

- prepare：归档固定原代码、复制该数据集 raw、重算 Stats，检查该模型 CPU 初始化与前向一致性；之后恢复本 Run 的运行时 seed。普通 make 仅展开声明，不执行探针准备。
- execute：成对 Algorithm 先执行原版，再用库 prepare 构造的当前 Data / Model / System / Tracker 执行当前版；恢复准备时记录的初始化 RNG，保持原版先、当前版后及 seed0 语义。每个 Run 只处理一个组合，不调用独立脚本或启动全矩阵。
- collect / summarize：普通库阶段产出当前训练指标、checkpoint 与结果，Study 附加门限状态与证据路径。
- write：两侧已有观测写成 artifact 投影，由库 compare 补充 latest / best 对比；不增加训练次数。
- process：Run 阶段检查本组合；Study 阶段只汇总当前 index 中的八组证据。缺组、重复、失败或不同来源不能算完整通过。

失败保留原始证据并交由 Flow 写 failed 状态。不接受 checkpoint resume；重试须使用新的 Run 目录，不能覆盖此前探针工作区。

## 四、门限

step30 / 60 浮点参数与 SGD momentum 状态 atol1e-6 / rtol1e-5，整数 buffer 与 optimizer 配置严格一致；train / test Loss 绝对差≤1e-6，正确样本数相同；初始化参数、CPU 与 CUDA RNG、训练样本索引、实际输入、各段 scheduler 一致。两边按 test Loss 选 best 且选择与权重一致；独立 eval 正确样本数相同、Loss 差≤1e-6。只有八个组合全部通过且来源不变才算完整通过。

## 五、验证边界

本轮只重构并验证 CPU 流程和 artifact 合同，不执行正式八组探针。此前数值证据不能替代本版本重跑验收。完整数值通过仍需一次获授权的统一 CLI 运行。
