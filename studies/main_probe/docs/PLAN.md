# main_probe 计划

## 一、目标

在同一设备、同一确定性条件下，比较固定 main `98648f3a5c7db7dccf3ca806410d5b6fdee9484c` 的原代码与当前 RPipe 的 60-step 计算是否逐段一致，确认 RPipe 改写没有改变数值行为。这是短预算探针，不代替 [main_exp](../../main_exp/README.md) 的长曲线复现。

## 二、配方

MNIST / CIFAR10 × linear / mlp / cnn / resnet18，seed0；全量 train / test，batch250 / test1000，连续 60 optimizer steps，step30 / 60 完整 test；SGD lr0.1、momentum0.9、Nesterov、wd0.0005，无梯度裁剪，cosine T_max60；当前无 BN CNN、原生 Factory 和 Kornia 模型前增强。全 train 统计由归档原 Stats(dim1) 按顺序 batch250 重算。两边都用 deterministic=true、benchmark=false、`CUBLAS_WORKSPACE_CONFIG=:4096:8`。

2026-10-11 的 version03 另显式设置 cudnn_allow_tf32=false、matmul_allow_tf32=false。两侧共享精度配置，recipe 拒绝未声明的精度；门限不变。version02 的通过证据保留在 [版本报告](versions/main-probe-flow-20261010-02/STUDY_REPORT.md)，不能代替新源码 / 精度基线的八格实测。变更依据为 main_exp 的 [设备数值诊断](../../main_exp/docs/EVAL_DIAGNOSTIC.json)。

## 三、统一 Flow 入口

专用计算与注册在 `structure/algorithm/`，六阶段实现在 `flow/<phase>/run.py`；各 `__init__.py` 仅导出，不写阶段实现。collect 的数值比较 helper 在 `flow/collect/comparison.py`，prepare 的数据准备在 `flow/prepare/data.py`；recipe 只校验并注册专用 source。两柱 Python 源码自动进入 provenance，不再手列源码 include。

`rpipe make / launch studies/main_probe` 展开八个成对 Run，每个 Run 只计算自己的 data/model 组合，包含两侧训练和各自 best 独立评测。Algorithm source 为 `main_probe`，recipe 注册本 Study 的 Data 和成对 Algorithm；CPU 准备由 Study `prepare.before(ctx)` 在库初始化后、recipe 注册前完成，以便原版全 train Stats 在 Data / Model 构造之前落盘。工作区位于该 Run 的 `assets/probe/`，不得共享可变原版 config。

- prepare：归档固定原代码、复制该数据集 raw、重算 Stats，检查该模型 CPU 初始化与前向一致性；之后恢复本 Run 的运行时 seed。普通 make 仅展开声明，不执行探针准备。
- execute：成对 Algorithm 先执行原版，再用库 prepare 构造的当前 Data / Model / System / Tracker 执行当前版；恢复准备时记录的初始化 RNG，保持原版先、当前版后及 seed0 语义。每个 Run 只处理一个组合，不调用独立脚本或启动全矩阵。
- collect：从 execute 的 OBSERVATIONS.pt 收集输入、RNG、参数、optimizer、scheduler、完整样本计数及独立 eval 对比。summarize 拼接报告与证据路径，拒绝未通过的数值门，保留失败证据。
- write：在库 result 定稿前投影两侧已有观测，调用库 compare，写完整比较报告；附加门失败写 failed，成功后库登记完整资产清单并定稿。不增加训练次数。
- process：Run 阶段检查本组合；Study 阶段只汇总当前 index 中的八组证据。缺组、重复、失败或不同来源不能算完整通过。

失败保留原始证据并交由 Flow 写 failed 状态。不接受 checkpoint resume；重试须使用新的 Run 目录，不能覆盖此前探针工作区。

## 四、门限

step30 / 60 浮点参数与 SGD momentum 状态 atol1e-6 / rtol1e-5，整数 buffer 与 optimizer 配置严格一致；train / test Loss 绝对差≤1e-6，正确样本数相同；初始化参数、CPU 与 CUDA RNG、训练样本索引、实际输入、各段 scheduler 一致。两边按 test Loss 选 best 且选择与权重一致；独立 eval 正确样本数相同、Loss 差≤1e-6。只有八个组合全部通过且来源不变才算完整通过。

## 五、验证边界

### 2026-10-13 发布前完整短探针

用户授权持续缺陷修复和发布准备，本次执行八组合60-step完整短探针，不启动main_exp长训。先逐字节保存可取得的version05声明、index / provenance / process及报告，旧Run不删除；使用main-probe-flow-20261013-06重新make并串行launch。完整raw缓存现已校验，来源与实际FP32条件不变、原数值门不放宽；八组齐全才通过Study终验。

CPU 合同门使用伪计算与微小张量验证阶段职责、正常定稿、数值失败和库 compare 失败，不能证明正式数值一致性。正式复跑使用新的 version / Run ID，保留当前基线、数据、设备与失败证据。此前数值证据不能替代本版本重跑验收；历史授权与当前限定范围分开，不能自动重启完整矩阵。

### 2026-10-12 本轮短实验

本轮仅运行 main-probe-flow-20261012-05 的 MNIST/linear 与 MNIST/cnn 两个 seed0 成对 Run，沿用上述 60-step 配方、完整 test、FP32 和固定数值门，不改预算或门限。先保存可取得的 version03 报告 / 声明快照，再重新 make；本工作区缺失旧 index / provenance / process，不重建历史记录。首次 version04 因缺 evaluate 在 prepare 失败，原 Run 与快照保留；补齐隔离依赖后换新 version05 复验。原计划第二组为 CIFAR10/cnn，但本机 raw 缺失且下载缓慢；完成 MNIST 校验后改用 MNIST/cnn 控制本轮成本，CIFAR10 不在本次实测覆盖内。使用 `rpipe run-one` 顺序执行这两个新 Run，不 launch 其余六组。随后显式 process 验证子集被判为 incomplete，预期该终验拒绝而非八组通过。原始证据保留在新 Run，逐段数字与本轮结论写入 Study 报告。
