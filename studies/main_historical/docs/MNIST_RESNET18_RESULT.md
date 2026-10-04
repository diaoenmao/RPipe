# MNIST / resnet18 四 seed 独立 CPU 审查

## 一、摘要

审查时间：2026-10-04T18:39:13.807740+08:00。固定 dev `8bccbac321d4c3ac1ea9892a5e774c114e0298c6`，历史来源 `4ccb28d0496110253e9f8e3f3df658853f07996b`。只读取正式 Run 的实际结果、完整 JSONL、tracker 和 CPU 重载的 canonical checkpoint；未启动、恢复或重启任何正式 Run，未使用 GPU。

训练四 seed complete=True，完整性审查 passed=True；正式独立 eval complete=True。本报告只审查该完整组合，不应用整体曲线门；自身字段`historical_reproduction_passed=null`。整体判定由COMPARISON记录。

全矩阵快照：`{"counts": {"train": {"succeeded": 32}, "eval": {"succeeded": 32}}, "completed_training_groups": 8, "completed_train_and_eval_groups": 8}`。

## 二、训练终态与来源

| **seed / Run** | **状态** | **test 点数** | **latest step** | **终点 Accuracy (%) / 正确样本** | **自身 global Accuracy-best step / (%)** | **完整性检查** |
|---|---|---:|---:|---|---|---|
| 0 / 6d6bd02aadf66eb0 | succeeded | 400 | 80000 | 99.53000126 / 9953 | 74400 / 99.59000130 | 41/41，失败项 [] |
| 1 / d4057e1cdf1c4dc4 | succeeded | 400 | 80000 | 99.60000057 / 9960 | 68800 / 99.66000042 | 41/41，失败项 [] |
| 2 / f9af547f2362dcf5 | succeeded | 400 | 80000 | 99.57000065 / 9957 | 43200 / 99.61000080 | 41/41，失败项 [] |
| 3 / 886b0201caa9fea8 | succeeded | 400 | 80000 | 99.57000065 / 9957 | 25000 / 99.65000038 | 41/41，失败项 [] |

四条均已正式 succeeded，各一次 Flow start、一次 fresh tracker start，无 resume/error/retry；400点显式 optimizer_step覆盖200–80000，train/test内部counter按200/40batch连续。每次test10000来自固定batch250、完整数据元信息与40batch差复算，正式worker未记录逐样本输入字节。latest global_step80000、tracker counter96000，所有train/test Loss/Accuracy历史与原始JSONL一致；best只取自身完整Accuracy曲线最大值，并核对对应历史前缀、scheduler和SGD momentum。保存的配置/shape/有限数检查不等同重新执行全部80000次更新。

来源审查：`{"files": {"files_checked": 99, "differences": []}, "plan_files": {"files_checked": 65, "differences": []}, "raw_data": {"files_checked": 16, "differences": []}, "passed": true}`。机器证据保存各result/config/log/scalars/tracker/latest/best的SHA-256，未读取分件镜像作为checkpoint权威。

## 三、固定曲线门的单组合诊断

本非线性组合的 SGD 检查覆盖保存配置、momentum 字段和有限数；未应用 Linear 参数形状与 momentum 的逐一映射检查，也未声称完成额外 CPU 推理。

同一步四seed mean终点99.56750078%，末50点mean99.58075070%；按population std (ddof=0)聚合全部400点，未用best值替代终点。参考值为PNG估读，不是历史原始指标。

| **诊断** | **实测误差 / 时间 std (百分点)** | **预先门限 (百分点)** | **门内** |
|---|---:|---:|---|
| endpoint | 0.02249922 | ≤0.25 | True |
| late_50_mean | 0.00924930 | ≤0.25 | True |
| anchor_mae | 0.01857146 | ≤0.25 | True |
| anchor_max | 0.04749911 | ≤0.6 | True |
| late_50_temporal_std | 0.00648556 | ≤0.15 | True |

| **槽 / optimizer_step** | **mean Accuracy (%)** | **population std (百分点)** | **PNG (%)** | **有符号差 (百分点)** |
|---|---:|---:|---:|---:|
| 50 / 10200 | 99.43250089 | 0.04918065 | 99.48 | -0.04749911 |
| 150 / 30200 | 99.53250103 | 0.04437048 | 99.57 | -0.03749897 |
| 200 / 40200 | 99.59000072 | 0.01414212 | 99.58 | +0.01000072 |
| 250 / 50200 | 99.59250078 | 0.02947433 | 99.59 | +0.00250078 |
| 300 / 60200 | 99.57750068 | 0.02585989 | 99.57 | +0.00750068 |
| 350 / 70200 | 99.58250074 | 0.02585994 | 99.58 | +0.00250074 |
| 399 / 80000 | 99.56750078 | 0.02487443 | 99.59 | -0.02249922 |

上述五项仅为该四seed完整组的诊断；如门外按实测记录，不调整门限。八组完整性由汇总审查核对，四模型排序与整体终验由COMPARISON应用；本单组合报告不能替代整体goal判定。

## 四、正式独立 eval 与 CPU 回放

| **seed / eval Run** | **own-best step** | **Accuracy (%) / 正确样本** | **完整性检查** | **CPU 回放** |
|---|---:|---|---|---|
| 0 / 48a3f005762cb11d | 74400 | 99.59000130 / 9959 | 20/20，失败项 [] | 未执行 |
| 1 / 23786b1c061e9ebe | 68800 | 99.66000042 / 9966 | 20/20，失败项 [] | 未执行 |
| 2 / 5887232578c33d3e | 43200 | 99.61000080 / 9961 | 20/20，失败项 [] | 未执行 |
| 3 / eb5a34a38e710dc7 | 25000 | 99.65000038 / 9965 | 20/20，失败项 [] | 未执行 |

本次没有执行额外 CPU 推理或模型权重回放；当前 --replay 仅支持 Linear。正式 eval 未导出在线模型终态哈希，其来源核对以实际 resume_path、未变化的 canonical own-best、冻结加载器和正式结果/日志为依据；CPU 重载保存状态不等同独立模型推理。

## 五、证据与边界

[本组机器JSON](MNIST_RESNET18_RESULT.json)、[八组完整性汇总](GROUP_INTEGRITY_AUDIT.md)、[整体比较记录](COMPARISON.json)、[预定曲线门](TARGET.md)、[原图估读](REFERENCE_CURVES.json)、[源码/展开配置 manifest](SOURCE_MANIFEST.json)、[数据 manifest](DATA_MANIFEST.json)。本报告的同名JSON包含完整400点、七锚点、每项检查、实际终态和哈希。

本次审计程序为 `D:\GitHub\diaoenmao\RPipe\studies\main_historical\verify_group.py`，实际 SHA-256 为 `2b9efcf86f2a7f1fe23c447be95f29cd38a8dd05ba1459954411c8e1068f9ce7`。原始 runs/checkpoint 不会随 Git clone 提供。冻结训练源与配置未改变。正式长训练未保存完整 RNG/迭代位置，本审查以无 resume 连续日志和正式终态为证据，不声称重新证明每一次未记录的 RNG 状态。
