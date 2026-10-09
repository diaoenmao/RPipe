# main_exp 计划

## 一、目标

用历史 main 提交 `4ccb28d0496110253e9f8e3f3df658853f07996b` 的配方，在当前 RPipe 训练循环上重跑 MNIST / CIFAR10 × linear / mlp / cnn / resnet18 × seed0–3 的 80000-step 曲线，与 main README 的两张原图比较。原运行记录没有找回，所以分别验收三件事：配方能完整重跑、同环境下原代码与当前链计算一致、四 seed 均值曲线接近原图。

## 二、配方

1. 全部训练数据，完整 10000 张 test。数据、模型、旧 CNN 的四层 BN、CPU torchvision 增强、常量统计和采样顺序直接复用归档代码，由 [recipe.py](../recipe.py) 注册为 `historical_4ccb28d` source。
2. 连续 80000 optimizer steps，每 200 step 完整 test，共 400 个曲线点。train/test batch250，SGD lr0.01、momentum0.9、Nesterov、wd0.0005、clip1、cosine T_max80000，见 [experiment_config.yaml](../experiment_config.yaml)。
3. best 按 RPipe 的全程 test Accuracy 选择；归档代码的旧 best 比较缺陷不移植。曲线取训练内 test history。
4. 确定性条件：deterministic=true、benchmark=false、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、CPU 2 线程。
5. 训练必须连续。当前 resume 不保存完整 RNG 与迭代器位置，中断的 train 不续跑，换新 `version` 从头跑。

## 三、门限

1. preflight：同设备原代码与当前链 600-step / eval200 八格对照全部通过才可 launch。参数 atol1e-6 / rtol1e-5，整数 buffer 一致，Loss 差≤1e-6，正确样本数一致，scheduler、RNG 与实际输入一致；scheduler 与采样总预算仍为 80000。
2. 终验：32 train + 32 eval 全部成功，四 seed 在每个评测点对齐。门限预先固定在 [TARGET.md](TARGET.md) 与 [REFERENCE_CURVES.json](REFERENCE_CURVES.json)：均值终点与末 50 点均值距原图估读 ≤0.25 个百分点（MNIST）/ ≤1.00（CIFAR10）；七个锚点 MAE≤0.25 / 1.00、最大偏差≤0.60 / 2.00；末段模型次序与时间波动门。结果出来后不改门限。
3. 原图估读的误差和原始 seed 集合未知，结论限于这组固定估读门。

## 四、执行

命令见 [Study 入口](../README.md)。同组并行数由 `rpipe launch` 按显存装箱，必要时用 `--round N`；train 全部结束后再跑 eval。报告写在 `docs/STUDY_REPORT.md`，数字表由 `rpipe report` 生成。
