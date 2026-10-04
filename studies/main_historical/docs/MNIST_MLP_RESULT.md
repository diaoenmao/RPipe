# MNIST / mlp 四 seed 独立 CPU 审查

## 一、摘要

审查时间：2026-10-04T03:16:25.604286+08:00。固定 dev `8bccbac321d4c3ac1ea9892a5e774c114e0298c6`，历史来源 `4ccb28d0496110253e9f8e3f3df658853f07996b`。只读取正式 Run 的实际结果、完整 JSONL、tracker 和 CPU 重载的 canonical checkpoint；未启动、恢复或重启任何正式 Run，未使用 GPU。

四 seed 训练均完成80000步且正式succeeded，每 seed41项保存状态/history/来源检查通过；原计划四条独立eval也均正式succeeded，每 seed20项实际own-best加载路径、结果及完整test检查通过。终点mean **98.1500014305%**，五项预定单组合曲线诊断均在门限内。本组没有额外CPU模型推理，未采用Linear参数/momentum形状映射。整项历史Study最终门未应用，`historical_reproduction_passed=null`。

本次03:16:25 +08:00全矩阵快照为train13 succeeded / 5 started / 14 pending、eval12 succeeded / 20 pending。MNIST linear、CIFAR10 linear和MNIST mlp合计 **3/8个组合完成四seed训练及独立eval**；多出的单条成功train尚不构成一个完整四seed组合。

## 二、训练终态与来源

| **seed / Run** | **状态** | **test 点数** | **latest step** | **终点 Accuracy (%) / 正确样本** | **自身 global Accuracy-best step / (%)** | **完整性检查** |
|---|---|---:|---:|---|---|---|
| 0 / d3d9d1c94f0981f3 | succeeded | 400 | 80000 | 98.08000126 / 9808 | 36200 / 98.13000164 | 41/41，失败项 [] |
| 1 / b44b3eb465b12318 | succeeded | 400 | 80000 | 98.26000156 / 9826 | 54400 / 98.31000137 | 41/41，失败项 [] |
| 2 / 568bad58b59b7728 | succeeded | 400 | 80000 | 98.17000122 / 9817 | 35800 / 98.24000092 | 41/41，失败项 [] |
| 3 / 6e63d83b28da4f6e | succeeded | 400 | 80000 | 98.09000168 / 9809 | 13400 / 98.15000153 | 41/41，失败项 [] |

全部四条需要正式 succeeded，各一次 Flow start、一次 fresh tracker start，无 resume/error/retry；400点显式 optimizer_step覆盖200–80000，train/test内部counter按200/40batch连续。每次test10000来自固定batch250、完整数据元信息与40batch差复算，正式worker未记录逐样本输入字节。latest global_step80000、tracker counter96000，所有train/test Loss/Accuracy历史与原始JSONL一致；best只取自身完整Accuracy曲线最大值，并核对对应历史前缀、scheduler和SGD momentum。保存的配置/shape/有限数检查不等同重新执行全部80000次更新。

来源审查：`{"files": {"files_checked": 99, "differences": []}, "plan_files": {"files_checked": 65, "differences": []}, "raw_data": {"files_checked": 16, "differences": []}, "passed": true}`。机器证据保存各result/config/log/scalars/tracker/latest/best的SHA-256，未读取分件镜像作为checkpoint权威。

## 三、固定曲线门的单组合诊断

本非线性组合的 SGD 检查覆盖保存配置、momentum 字段和有限数；未应用 Linear 参数形状与 momentum 的逐一映射检查，也未声称完成额外 CPU 推理。

同一步四seed mean终点98.15000143%，末50点mean98.15235143%；按population std (ddof=0)聚合全部400点，未用best值替代终点。参考值为PNG估读，不是历史原始指标。

| **诊断** | **实测误差 / 时间 std (百分点)** | **预先门限 (百分点)** | **门内** |
|---|---:|---:|---|
| endpoint | 0.00999857 | ≤0.25 | True |
| late_50_mean | 0.00764857 | ≤0.25 | True |
| anchor_mae | 0.01499860 | ≤0.25 | True |
| anchor_max | 0.02499865 | ≤0.6 | True |
| late_50_temporal_std | 0.00365409 | ≤0.15 | True |

| **槽 / optimizer_step** | **mean Accuracy (%)** | **population std (百分点)** | **PNG (%)** | **有符号差 (百分点)** |
|---|---:|---:|---:|---:|
| 50 / 10200 | 98.02500153 | 0.04821800 | 98.04 | -0.01499847 |
| 150 / 30200 | 98.10500131 | 0.07921481 | 98.12 | -0.01499869 |
| 200 / 40200 | 98.13500137 | 0.08321654 | 98.15 | -0.01499863 |
| 250 / 50200 | 98.15500135 | 0.06184659 | 98.18 | -0.02499865 |
| 300 / 60200 | 98.15500140 | 0.06946221 | 98.17 | -0.01499860 |
| 350 / 70200 | 98.16000142 | 0.06363952 | 98.17 | -0.00999858 |
| 399 / 80000 | 98.15000143 | 0.07245691 | 98.16 | -0.00999857 |

上述五项仅为该四seed完整组的诊断；如门外按实测记录，不调整门限。四模型排序及其余组合完整性尚待，不能声明整体goal完成。

## 四、正式独立 eval 与 CPU 回放

执行记录状态为 `succeeded`，四条child.wait退出0及正式succeeded已独立核对；执行来源审查 passed=True。

[原计划eval执行记录](EARLY_EVAL_MNIST_MLP_EXECUTION.json) 记录03:13:50–03:14:40 +08:00执行，控制器实际exit0、四条worker各exit0。实际 [helper v2](EARLY_EVAL_GROUP_EXECUTION_HELPER_V2.py) SHA-256为 `aea398fea16fb1b35ed2e2103742b95d778358a90e78fb86aa221c1019d5d552`，独立读取与执行记录完全相同；绑定的SOURCE_MANIFEST SHA也一致。此版本与此前CIFAR linear遇WinError5的旧helper版本分别保留。

| **seed / eval Run** | **own-best step** | **Accuracy (%) / 正确样本** | **完整性检查** | **CPU 回放** |
|---|---:|---|---|---|
| 0 / 7327aec21ab7437f | 36200 | 98.13000164 / 9813 | 20/20，失败项 [] | 未执行 |
| 1 / a9aea870e5109f87 | 54400 | 98.31000137 / 9831 | 20/20，失败项 [] | 未执行 |
| 2 / 8ddd32256875b483 | 35800 | 98.24000092 / 9824 | 20/20，失败项 [] | 未执行 |
| 3 / 764eaa17d56ff8a1 | 13400 | 98.15000153 / 9815 | 20/20，失败项 [] | 未执行 |

本次没有执行额外 CPU 推理或模型权重回放；当前 --replay 仅支持 Linear。正式 eval 未导出在线模型终态哈希，其来源核对以实际 resume_path、未变化的 canonical own-best、冻结加载器和正式结果/日志为依据；CPU 重载保存状态不等同独立模型推理。

## 五、证据与边界

[机器JSON](MNIST_MLP_RESULT.json) 包含完整400点、七锚点、每项检查、实际终态和哈希；[预定曲线门](TARGET.md)、[原图估读](REFERENCE_CURVES.json)、[源码/展开配置manifest](SOURCE_MANIFEST.json)、[数据manifest](DATA_MANIFEST.json) 提供固定判定与来源。

本次审计程序为 `D:\GitHub\diaoenmao\RPipe\studies\main_historical\verify_group.py`，实际 SHA-256 为 `2b9efcf86f2a7f1fe23c447be95f29cd38a8dd05ba1459954411c8e1068f9ce7`。原始 runs/checkpoint 不会随 Git clone 提供。冻结训练源与配置未改变。正式长训练未保存完整 RNG/迭代位置，本审查以无 resume 连续日志和正式终态为证据，不声称重新证明每一次未记录的 RNG 状态。
