# CIFAR10 / cnn 四 seed 独立 CPU 审查

## 一、摘要

审查时间：2026-10-04T18:39:13.827088+08:00。固定 dev `8bccbac321d4c3ac1ea9892a5e774c114e0298c6`，历史来源 `4ccb28d0496110253e9f8e3f3df658853f07996b`。只读取正式 Run 的实际结果、完整 JSONL、tracker 和 CPU 重载的 canonical checkpoint；未启动、恢复或重启任何正式 Run，未使用 GPU。

训练四 seed complete=True，完整性审查 passed=True；正式独立 eval complete=True。本报告只审查该完整组合，不应用整体曲线门；自身字段`historical_reproduction_passed=null`。整体判定由COMPARISON记录。

全矩阵快照：`{"counts": {"train": {"succeeded": 32}, "eval": {"succeeded": 32}}, "completed_training_groups": 8, "completed_train_and_eval_groups": 8}`。

## 二、训练终态与来源

| **seed / Run** | **状态** | **test 点数** | **latest step** | **终点 Accuracy (%) / 正确样本** | **自身 global Accuracy-best step / (%)** | **完整性检查** |
|---|---|---:|---:|---|---|---|
| 0 / 67f1f12af61edb4c | succeeded | 400 | 80000 | 88.74000149 / 8874 | 79000 / 89.00000153 | 41/41，失败项 [] |
| 1 / dbc8e2a4f684cf8e | succeeded | 400 | 80000 | 89.11000156 / 8911 | 75600 / 89.21000195 | 41/41，失败项 [] |
| 2 / e9999e28241568aa | succeeded | 400 | 80000 | 88.98000164 / 8898 | 73800 / 89.05000153 | 41/41，失败项 [] |
| 3 / e91adb37db5f9927 | succeeded | 400 | 80000 | 89.27000179 / 8927 | 70000 / 89.45000114 | 41/41，失败项 [] |

四条均已正式 succeeded，各一次 Flow start、一次 fresh tracker start，无 resume/error/retry；400点显式 optimizer_step覆盖200–80000，train/test内部counter按200/40batch连续。每次test10000来自固定batch250、完整数据元信息与40batch差复算，正式worker未记录逐样本输入字节。latest global_step80000、tracker counter96000，所有train/test Loss/Accuracy历史与原始JSONL一致；best只取自身完整Accuracy曲线最大值，并核对对应历史前缀、scheduler和SGD momentum。保存的配置/shape/有限数检查不等同重新执行全部80000次更新。

来源审查：`{"files": {"files_checked": 99, "differences": []}, "plan_files": {"files_checked": 65, "differences": []}, "raw_data": {"files_checked": 16, "differences": []}, "passed": true}`。机器证据保存各result/config/log/scalars/tracker/latest/best的SHA-256，未读取分件镜像作为checkpoint权威。

## 三、固定曲线门的单组合诊断

本非线性组合的 SGD 检查覆盖保存配置、momentum 字段和有限数；未应用 Linear 参数形状与 momentum 的逐一映射检查，也未声称完成额外 CPU 推理。

同一步四seed mean终点89.02500162%，末50点mean88.96720158%；按population std (ddof=0)聚合全部400点，未用best值替代终点。参考值为PNG估读，不是历史原始指标。

| **诊断** | **实测误差 / 时间 std (百分点)** | **预先门限 (百分点)** | **门内** |
|---|---:|---:|---|
| endpoint | 0.01500162 | ≤1.0 | True |
| late_50_mean | 0.02279842 | ≤1.0 | True |
| anchor_mae | 0.27785788 | ≤1.0 | True |
| anchor_max | 0.59499842 | ≤2.0 | True |
| late_50_temporal_std | 0.06365856 | ≤0.5 | True |

| **槽 / optimizer_step** | **mean Accuracy (%)** | **population std (百分点)** | **PNG (%)** | **有符号差 (百分点)** |
|---|---:|---:|---:|---:|
| 50 / 10200 | 81.16500158 | 1.62521527 | 81.76 | -0.59499842 |
| 150 / 30200 | 86.54500155 | 0.55975428 | 86.01 | +0.53500155 |
| 200 / 40200 | 87.30750165 | 0.20091953 | 86.98 | +0.32750165 |
| 250 / 50200 | 87.91250162 | 0.32360284 | 87.62 | +0.29250162 |
| 300 / 60200 | 88.64500127 | 0.21195519 | 88.66 | -0.01499873 |
| 350 / 70200 | 88.96500154 | 0.11101799 | 88.80 | +0.16500154 |
| 399 / 80000 | 89.02500162 | 0.19397174 | 89.01 | +0.01500162 |

上述五项仅为该四seed完整组的诊断；如门外按实测记录，不调整门限。八组完整性由汇总审查核对，四模型排序与整体终验由COMPARISON应用；本单组合报告不能替代整体goal判定。

## 四、正式独立 eval 与 CPU 回放

| **seed / eval Run** | **own-best step** | **Accuracy (%) / 正确样本** | **完整性检查** | **CPU 回放** |
|---|---:|---|---|---|
| 0 / b357409255c9124a | 79000 | 89.00000153 / 8900 | 20/20，失败项 [] | 未执行 |
| 1 / 424e5ab0c4d2b71d | 75600 | 89.21000195 / 8921 | 20/20，失败项 [] | 未执行 |
| 2 / a17f1a2474d8425a | 73800 | 89.05000153 / 8905 | 20/20，失败项 [] | 未执行 |
| 3 / 51a9ff648cff3883 | 70000 | 89.45000114 / 8945 | 20/20，失败项 [] | 未执行 |

本次没有执行额外 CPU 推理或模型权重回放；当前 --replay 仅支持 Linear。正式 eval 未导出在线模型终态哈希，其来源核对以实际 resume_path、未变化的 canonical own-best、冻结加载器和正式结果/日志为依据；CPU 重载保存状态不等同独立模型推理。

## 五、证据与边界

[本组机器JSON](CIFAR_CNN_RESULT.json)、[八组完整性汇总](GROUP_INTEGRITY_AUDIT.md)、[整体比较记录](COMPARISON.json)、[预定曲线门](TARGET.md)、[原图估读](REFERENCE_CURVES.json)、[源码/展开配置 manifest](SOURCE_MANIFEST.json)、[数据 manifest](DATA_MANIFEST.json)。本报告的同名JSON包含完整400点、七锚点、每项检查、实际终态和哈希。

本次审计程序为 `D:\GitHub\diaoenmao\RPipe\studies\main_historical\verify_group.py`，实际 SHA-256 为 `2b9efcf86f2a7f1fe23c447be95f29cd38a8dd05ba1459954411c8e1068f9ce7`。原始 runs/checkpoint 不会随 Git clone 提供。冻结训练源与配置未改变。正式长训练未保存完整 RNG/迭代位置，本审查以无 resume 连续日志和正式终态为证据，不声称重新证明每一次未记录的 RNG 状态。
