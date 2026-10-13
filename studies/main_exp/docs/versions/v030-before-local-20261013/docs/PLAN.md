# main_exp 计划

## 一、目标

用历史 main 提交 `4ccb28d0496110253e9f8e3f3df658853f07996b` 的配方，在当前 RPipe 训练循环上重跑 MNIST / CIFAR10 × linear / mlp / cnn / resnet18 × seed0–3 的 80000-step 曲线，与 main README 的两张原图比较。原运行记录没有找回，所以分别验收三件事：配方能完整重跑、同环境下原代码与当前链计算一致、四 seed 均值曲线接近原图。

## 二、配方

1. 全部训练数据，完整 10000 张 test。数据、模型、旧 CNN 的四层 BN、CPU torchvision 增强、常量统计和采样顺序直接复用归档代码，由 [recipe.py](../recipe.py) 注册为 `historical_4ccb28d` source。
2. 连续 80000 optimizer steps，每 200 step 完整 test，共 400 个曲线点。train/test batch250，SGD lr0.01、momentum0.9、Nesterov、wd0.0005、clip1、cosine T_max80000，见 [experiment_config.yaml](../experiment_config.yaml)。
3. best 按 RPipe 的全程 test Accuracy 选择；归档代码的旧 best 比较缺陷不移植。曲线取训练内 test history。
4. 确定性条件：deterministic=true、benchmark=false、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、CPU 2 线程。2026-10-11 的新 version03 显式设置 cudnn_allow_tf32=false、matmul_allow_tf32=false，两侧前置对照、训练与独立评测一致；version02 的精度未显式固定且独立评测一致性失败，完整证据保留。新旧精度基线不混用，变更与停止失败的恢复条件见 [eval_recovery.md](eval_recovery.md)。
5. 训练必须连续。通用原生 resume 已按单进程合同本机补齐 RNG / 数据位置，但本 Study 的自定义 sampler / 数据适配仍未验收精确恢复，继续拒绝中断 train 续跑，换新 `version` 从头跑。

## 三、门限

1. preflight：同设备原代码与当前链 600-step / eval200 八格对照全部通过才可 launch。参数 atol1e-6 / rtol1e-5，整数 buffer 一致，Loss 差≤1e-6，正确样本数一致，scheduler、RNG 与实际输入一致；scheduler 与采样总预算仍为 80000。
2. 终验：32 train + 32 eval 全部成功，四 seed 在每个评测点对齐。门限预先固定在 [TARGET.md](TARGET.md) 与 [REFERENCE_CURVES.json](REFERENCE_CURVES.json)：均值终点与末 50 点均值距原图估读 ≤0.25 个百分点（MNIST）/ ≤1.00（CIFAR10）；七个锚点 MAE≤0.25 / 1.00、最大偏差≤0.60 / 2.00；末段模型次序与时间波动门。结果出来后不改门限。
3. 原图估读的误差和原始 seed 集合未知，结论限于这组固定估读门。

## 四、执行

### 2026-10-13 校准与精度复验

先用固定源码快照完成 CIFAR10 的 CNN / ResNet18 / 混合任务各三个 seed、800-step、round1/2/3 容量测量，再用新短程 CNN best 做16次显式 FP32和16次 TF32固定权重全 test，覆盖串行与并发；保留逐样本预测 / 输入 / 模型哈希及可取得的 kernel trace。新权重不冒充旧 B-023 的76800-step权重；旧 checkpoint 未找回，旧反例及其 JSON 不覆盖。

容量采用既有 system.config.vram_bytes，库默认经验系数与 safety=0.5 不变。main_exp 的后续 fresh make 使用4.5GiB单进程预算，保守统一覆盖配方的模型；这限制轻型任务的自动并行数，也不保证其他设备 / 配方可沿用。重跑前仍须新的600-step八格前置门。此次只做前置门与短校准，不启动完整矩阵、不覆盖旧 version03。

命令见 [Study 入口](../README.md)。同组并行数由 `rpipe launch` 按显存装箱，必要时用 `--round N`；train 全部结束后再跑 eval。报告写在 `docs/STUDY_REPORT.md`，数字表由 `rpipe report` 生成。

专用 Data / Model 与归档支持放 `structure/`；`flow/prepare/run.py` 核对来源和实际运行条件，`flow/process/run.py` 只在 Study scope 调用 `curves.py`。`__init__.py` 仅导出。raw 准备使用 `flow/prepare_data.py`，速度校准使用 recipe 调用 `flow/execute/speed.py`；手动工具不自动作为 hook，两柱源码自动冻结。

### 2026-10-12 本轮短实验

仅用现有 `recipe.preflight` 做 MNIST/linear、seed0、200-step / 完整 test 的同设备原版与当前链前缀对照，保持 sampler / scheduler 总预算 80000、batch250、原精度与数值门。独立验证标识为 main-exp-flow-smoke-20261012-01，输出到本 Study `shared/validation/` 的新目录，逐段数字写入 `docs/`。这不是正式 64-Run 实验，不改 version03 的声明、index、provenance、Run 或 `PREFLIGHT.json`，也不触发正式 Flow launch；现有 loader / scope 合同检查另行复验。200-step 单组不满足 600-step 八格前置门，不能解锁长实验。以后正式复跑仍需新 version、重新 make 和完整前置门。
