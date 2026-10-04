# 历史 main 曲线复现计划

## 一、目标与范围

2026-10-04，基于最新远端 dev `8bccbac`。用户授权持续运行，直至较好复现 main baseline。继续上一轮已经确认的 README 历史曲线路线，从首个 200-step 探针推进长训练；既有 main_base 和 main_reproduction 的结果保持。

参考图片为 main README 的 MNIST / CIFAR10 Accuracy mean，候选配方固定为图最后更新提交 `4ccb28d0496110253e9f8e3f3df658853f07996b`。原运行记录没有找回，因此候选配方重跑、同环境跨实现计算一致与原图近似程度分别验收。原图估读和预定数值门见 [TARGET.md](TARGET.md)。

## 二、固定条件与验收

采用 [TARGET.md](TARGET.md) 及 [REFERENCE_CURVES.json](REFERENCE_CURVES.json) 中的预先门限：4-seed mean 终点与末 50 点均值分别距离原 PNG 估读值 ≤0.25 个百分点（MNIST）/ ≤1.00（CIFAR10）；固定七个槽的 MAE≤0.25/1.00、最大偏差≤0.60/2.00。图像提取的误差和来源未知明确保留，不用新结果调整门限。末段模型相对次序与时间波动门沿用该合同。该图片门与原代码的严格计算对照门分别报告。

1. MNIST / CIFAR10 全部训练数据和完整 10000 张 test；linear / mlp / cnn / resnet18；seed 0–3。
2. 连续 80000 optimizer steps，每 200 step 完整 test，共 400 个曲线点。train/test batch250，SGD lr0.01、momentum0.9、Nesterov、wd0.0005、clip1、cosine T_max80000。旧 CNN 的四层 BN、CPU torchvision 增强、常量统计和采样顺序直接复用归档代码。
3. 保持当前 RPipe 的正确全程 Accuracy-best。归档旧 best 比较缺陷不移植；README 曲线来自训练内 test history，与该缺陷分开。
4. 同环境原代码和当前链 600-step/eval200 三段探针先通过 8/8：参数 atol1e-6/rtol1e-5、整数 buffer 一致、Loss 差≤1e-6、正确样本数一致、scheduler/RNG/实际输入一致。覆盖评测后回到训练的行为；scheduler 与采样总预算始终为 80000。探针前后源码/配置 hash 必须相同，长矩阵不得复用另一源码版本的通过门。
5. 32 个 train Run 各 80000 step、400 次完整 test；32 个独立 eval 加载同因素与 seed 的 train best。最终跨 seed mean / population std 按评测 step 对齐，并和原图比较。单 seed 或短前缀不作为完整验收。

## 三、执行与排班

RTX 5090 D v2 24 GiB、Python3.13.9、Torch2.11.0+cu130；原报告为 cu128，因此本机重新运行归档原代码探针。确定性控制为 deterministic=true / benchmark=false / CUBLAS_WORKSPACE_CONFIG=:4096:8，CPU threads2。缺少的小依赖只放 `.tmp/runtime`，不覆盖系统 Torch。

同模型、同数据集、不同 seed 分组，linear / mlp / cnn 最多 4 条，resnet18 最多 2 条，组末 wait；所有 train 结束后再独立 eval。根据首组实际显存/利用率调整并发，不改变计算配方。每条 Run 独立 Flow 和日志、tracker、checkpoint，source/hash/设备环境落正式 JSON。

当前 native resume 没有完整 RNG 和迭代器位置，历史验收训练必须连续。已成功 Run 跳过；中断的 Run 保留证据，使用新 version 从头复测，不能用恢复采样前缀冒充连续长训。checkpoint 的保存用于证据与灾后诊断，不等于保证逐位续训。失败不中断其他格子；最终报告实际未完成项。

## 四、可重跑入口与证据

### 2026-10-04 运行中的并发调整

第一组 MNIST linear 四 seed 稳态 GPU 利用率约16%、总显存约5.1GiB，主计算源与配置保持冻结。为利用剩余算力，提前通过相同 `run.py one` 入口运行 CIFAR10 / resnet18 seed0；它仍是 index 内原计划的 Run，不增加 seed 或改配方。独立进程隔离 RNG、采样器与 tracker；不把混合负载计时作为算法速度比较。

必须在主调度构造 CIFAR10/resnet18 待执行候选前结束并落 `succeeded`，主调度才会跳过这条已完成 Run。原调度没有外部 Run 锁，不能允许它与仍在执行的提前任务重复启动；定期核对具体进程句柄及 EXECUTION 的当前组，出现接近调度冲突时先协调等待，不通过检查点恢复拼接训练。提前任务的实际 handle/PID/日志与退出结果单独保存在 EAGER_EXECUTION.json，主 EXECUTION 的原事件不改写。先观察一条的显存峰值与两类实际吞吐，再决定是否提前第二条。

并行协调使用独立的本地观察器，不修改冻结的数值计算源。先以 PID、创建时间、命令行核对真实 `run.py launch` 父进程，短暂只暂停父调度并验证训练子进程日志持续推进、父可恢复。观察器最迟在首个 train/MNIST/resnet18 group 出现时拦住父，并暂停后再次确认尚无 CIFAR10/resnet18 group 或重复子进程；已有训练子进程保持连续。若提前两条较重任务，则保守地在 train/CIFAR10/mlp 的四个 worker 都已启动时拦住父，避免未验证的四条 CNN 显存峰值与两条 ResNet 同时出现。仅在提前 Run 正式 succeeded、日志成功且对应精确进程身份退出后正常放行。异常必须落独立协调记录并由当前任务处理，不能静默宣称去重或依赖预计时长。

新 Study 保存可追踪的 prepare_data.py、recipe.py 和 run.py，解决旧适配脚本仅在 `.tmp` 导致异机无法重跑的问题；归档源码由 git blob 导出并校验，不修改生产模型。

2026-10-04，MNIST / linear 四条连续训练均正式成功且完整存档已独立核验后，提前串行执行它们原计划内的四条 sibling-best eval。每条只依赖自身已经完成的 train，不依赖其他模型训练；这不改变 32 train + 32 eval、全量 test 或 best 口径。四个 eval 必须在主控制器进入 eval 阶段前正式成功且进程退出，控制器随后按既有 succeeded 检查跳过。启动与退出证据单独写 EARLY_EVAL_EXECUTION.json，不改主 EXECUTION，也不把独立 best eval 当成 step80000 曲线终点。

同样的提前评测可用于后续已经完整成功的 linear / mlp / cnn 四种子组：必须先从主 EXECUTION 确认四个 train 子进程各自 exit0 且 succeeded，再串行调用相同冻结的 `run.py one`。每组使用独立的 EARLY_EVAL_<DATA>_<MODEL>_EXECUTION.json，不覆盖既有证据；已经有评测日志、结果或执行记录的组不能由该入口重启。主控制器进入 eval 候选构造前必须核实提前 eval 正式成功且进程退出。

```powershell
python -m pip install --target .tmp/runtime --no-deps -r studies/main_historical/runtime-requirements.txt
python studies/main_historical/prepare_data.py
python studies/main_historical/run.py preflight
python studies/main_historical/run.py make
python studies/main_historical/run.py launch
python studies/main_historical/run.py status
python studies/main_historical/run.py process
```

数据、Run、归档和临时依赖不入 Git。正式 docs 保留数据 SHA256、环境、探针数值、源码 hash、全矩阵运行状态、对照数字和曲线。报告区别已有历史结果与本设备新运行，不把待完成训练写成已复现。
