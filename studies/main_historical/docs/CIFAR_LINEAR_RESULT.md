# CIFAR10 / linear 四 seed 独立 CPU 审查

## 一、摘要

审查时间：2026-10-04T02:43:12.521776+08:00。固定 dev `8bccbac321d4c3ac1ea9892a5e774c114e0298c6`，历史来源 `4ccb28d0496110253e9f8e3f3df658853f07996b`。只读取正式 Run 的实际结果、完整 JSONL、tracker 和 CPU 重载的 canonical checkpoint；未启动、恢复或重启任何正式 Run，未使用 GPU。

四 seed 训练与原计划独立 eval 均已正式 succeeded，训练 checkpoint / 完整 history、eval own-best 来源及额外 CPU 完整 test 回放全部通过。四 seed mean 终点 **40.37000083%**、末 50 点 mean **40.36940076%**；五项预定单组合诊断均在原门限内。整项历史 Study 的最终门未应用，`historical_reproduction_passed=null`。

本次全矩阵快照为 train8 succeeded / 6 started / 18 pending、eval8 succeeded / 24 pending；MNIST / linear 与 CIFAR10 / linear 合计 **2/8 个组合完成训练和独立 eval**，其余六组尚未完成。

## 二、训练终态与来源

| **seed / Run** | **状态** | **test 点数** | **latest step** | **终点 Accuracy (%) / 正确样本** | **自身 global Accuracy-best step / (%)** | **完整性检查** |
|---|---|---:|---:|---|---|---|
| 0 / 24db7966c8b8469a | succeeded | 400 | 80000 | 40.39000063 / 4039 | 73000 / 40.67000093 | 41/41，失败项 [] |
| 1 / f91f89c4afdcb655 | succeeded | 400 | 80000 | 40.53000097 / 4053 | 74600 / 40.66000090 | 41/41，失败项 [] |
| 2 / d8583c714101e7fb | succeeded | 400 | 80000 | 40.32000089 / 4032 | 72200 / 40.62000055 | 41/41，失败项 [] |
| 3 / 4b42a10dd5424d93 | succeeded | 400 | 80000 | 40.24000082 / 4024 | 73400 / 40.44000092 | 41/41，失败项 [] |

四条各有一次 Flow start、一次 fresh tracker start，无 resume/error/retry；400 点显式 optimizer_step 覆盖200–80000，train/test 内部 counter 按200/40 batch连续。每次 test10000 来自固定 batch250、完整数据元信息与40 batch 差复算，正式 worker 未记录逐样本输入字节。latest global_step80000、tracker counter96000，所有 train/test Loss/Accuracy 历史与原始 JSONL 一致；best 只取自身完整 Accuracy 曲线最大值，并核对对应历史前缀、scheduler 和 SGD momentum。保存的配置、shape 和有限数检查不等同重新执行全部80000次更新。

来源审查：`{"files": {"files_checked": 99, "differences": []}, "plan_files": {"files_checked": 65, "differences": []}, "raw_data": {"files_checked": 16, "differences": []}, "passed": true}`。机器证据保存各result/config/log/scalars/tracker/latest/best的SHA-256，未读取分件镜像作为checkpoint权威。

## 三、固定曲线门的单组合诊断

同一步四seed mean终点40.37000083%，末50点mean40.36940076%；按population std (ddof=0)聚合全部400点，未用best值替代终点。参考值为PNG估读，不是历史原始指标。

| **诊断** | **实测误差 / 时间 std (百分点)** | **预先门限 (百分点)** | **门内** |
|---|---:|---:|---|
| endpoint | 0.02000083 | ≤1.0 | True |
| late_50_mean | 0.00940076 | ≤1.0 | True |
| anchor_mae | 0.03714316 | ≤1.0 | True |
| anchor_max | 0.16000062 | ≤2.0 | True |
| late_50_temporal_std | 0.05270570 | ≤0.5 | True |

| **槽 / optimizer_step** | **mean Accuracy (%)** | **population std (百分点)** | **PNG (%)** | **有符号差 (百分点)** |
|---|---:|---:|---:|---:|
| 50 / 10200 | 37.70250070 | 0.41390665 | 37.72 | -0.01749930 |
| 150 / 30200 | 38.42750071 | 0.40220474 | 38.42 | +0.00750071 |
| 200 / 40200 | 39.12000062 | 0.27964270 | 38.96 | +0.16000062 |
| 250 / 50200 | 39.39750072 | 0.31283981 | 39.38 | +0.01750072 |
| 300 / 60200 | 39.75500079 | 0.09912120 | 39.76 | -0.00499921 |
| 350 / 70200 | 40.22250071 | 0.09417411 | 40.19 | +0.03250071 |
| 399 / 80000 | 40.37000083 | 0.10653642 | 40.35 | +0.02000083 |

上述五项仅为该四 seed 完整组的诊断，不调整门限。最大锚点偏差在 step40200，为 +0.16000062 个百分点；保留实测前段偏差，不仅凭尾段相近作结论。四模型排序及其余组合完整性尚待，不能声明整体 goal 完成。

## 四、正式独立 eval 与 CPU 回放

执行记录状态为 `workers_succeeded_record_recovered`，四条child.wait退出0及正式succeeded已独立核对；执行来源审查 passed=True。

原eval调度器在第四条child.wait已返回0后，保存执行JSON的 `os.replace` 遭遇WinError5，调度器实际exit1，`controller_status=failed`保留。主线程只恢复执行记录，未重跑worker；原failed-controller记录、原未提交记录字节和实际执行helper源码均独立归档并核对SHA-256。后续修改的临时helper SHA不混入本轮结果。这是记录写入失败，不能称为调度器exit0，也没有证据据此归因为GPU worker失败。

[恢复后的执行记录](EARLY_EVAL_CIFAR10_LINEAR_EXECUTION.json)、[原控制器记录](EARLY_EVAL_CIFAR10_LINEAR_FAILED_CONTROLLER.json)、[原未提交记录](EARLY_EVAL_CIFAR10_LINEAR_UNCOMMITTED_RECORD.json)、[实际执行 helper](CIFAR_LINEAR_EVAL_EXECUTION_HELPER.py)。

| **seed / eval Run** | **own-best step** | **Accuracy (%) / 正确样本** | **完整性检查** | **CPU 回放** |
|---|---:|---|---|---|
| 0 / 7530405c3b1156bf | 73000 | 40.67000093 / 4067 | 20/20，失败项 [] | passed=True，Loss差1.49011611938e-08 |
| 1 / 22b493f4c03737da | 74600 | 40.66000090 / 4066 | 20/20，失败项 [] | passed=True，Loss差1.49011611938e-08 |
| 2 / dbe4dda86857bcbc | 72200 | 40.62000055 / 4062 | 20/20，失败项 [] | passed=True，Loss差2.68220901045e-08 |
| 3 / 28954c019536f1b7 | 73400 | 40.44000092 / 4044 | 20/20，失败项 [] | passed=True，Loss差1.49011611938e-08 |

正式eval没有导出在线模型终态哈希；实际resume_path、未变化的canonical own-best、冻结加载器及独立CPU严格加载/逐tensor检查构成权重来源证据。CPU回放直接使用官方原始test文件、归档Linear类及归档test归一化；不增加正式GPU Run。

## 五、证据与边界

[机器 JSON](CIFAR_LINEAR_RESULT.json) 包含完整400点、七锚点、每项检查、实际终态、审计源和哈希；[预定曲线门](TARGET.md)、[原图估读](REFERENCE_CURVES.json)、[源码/展开配置 manifest](SOURCE_MANIFEST.json)、[数据 manifest](DATA_MANIFEST.json) 提供固定判定和来源。

本次实际审计程序位于本机 `.tmp/historical_group_audit.py`，SHA-256为 `1b00c0eb77ea4ee25bbc7252f77502dc0497c29d810d7ce9dbb9cee2c8ec8c2f`，正式JSON继续绑定该来源。后续新增的 [verify_group.py](../verify_group.py) 适配Study目录定位，报告模板引用当次真实路径/SHA，可随Git提供；[方法与回归记录](GROUP_AUDIT_METHOD.md) 说明源码两处适配及两组合CPU科学字段逐项一致性，不倒写本次审查来源。旧 `.tmp` 程序与原始runs/checkpoint不随Git clone提供。

冻结训练源与配置未改变。正式长训练未保存完整RNG/迭代位置，本审查以无resume连续日志和正式终态为证据，不声称重新证明每一次未记录的RNG状态。
