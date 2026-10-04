# 历史完整组合的 CPU 审计入口与来源

## 一、摘要

2026-10-04，新增随 Git 携带的 [verify_group.py](../verify_group.py)，供将来对已完成四 seed 组合重新进行 CPU 来源、checkpoint、history、固定曲线门与独立 eval 审查。旧 `.tmp/historical_group_audit.py` 原样保留。当前有三项来源/报告适配：仓库根定位改为 `parents[2]`；生成Markdown引用当次真实审计路径/SHA；报告按实际model与replay状态区分保存状态审查和额外CPU推理，并明确非线性组合没有Linear参数/momentum形状映射。除仓库定位和 `report_text` 外，数值审计源码的AST与旧程序相同；冻结训练源、展开配置和manifest均未改变。

03:16:25 +08:00，当前SHA版本对 [MNIST / mlp](MNIST_MLP_RESULT.md) 的四条完整train及四条正式eval审查均通过；先train-only、后`--with-eval`两条命令均exit0，未使用`--replay`。Markdown明确没有额外CPU推理，未套用Linear形状映射；当次真实路径/SHA、eval helper v2源码与manifest绑定均核对。

02:56:53 / 02:56:58 +08:00的前版SHA `21958ded…` 审查快照，已对 MNIST / linear、CIFAR10 / linear 完成两次CPU重核，均exit0；旧入口02:49–02:50的科学baseline复用。全部400点、四seed聚合、七锚点、固定门、保存状态、own-best及CPU回放与来源证据均逐字段相同。两份Markdown均显示前版portable真实路径/SHA。本轮非线性措辞修改没有再次执行这两组推理，不把前版回归改称当前SHA的新实测。

除了审计脚本路径、SHA和记录时间，运行中的整个Study状态快照还真实变化为train succeeded8→9、started6→5；这两项动态上下文差异逐值保留，不作为完整组科学数值变化。[机器比较记录](GROUP_AUDIT_METHOD.json) 保存全部差异、原状态与新状态；不增加正式训练或GPU eval，也不改变整体终验未应用状态。

## 二、本次正式快照与新入口的区别

| **来源** | **实际 SHA-256** | **用途** |
|---|---|---|
| 本机旧 `.tmp/historical_group_audit.py` | `1b00c0eb77ea4ee25bbc7252f77502dc0497c29d810d7ce9dbb9cee2c8ec8c2f` | CIFAR10 / linear 02:43:12正式独立审查的实际程序；保留原样 |
| 当前 [verify_group.py](../verify_group.py) | `2b9efcf86f2a7f1fe23c447be95f29cd38a8dd05ba1459954411c8e1068f9ce7` | 数值审计逻辑相同；目录、真实报告来源与model/replay边界措辞适配 |

[CIFAR_LINEAR_RESULT.json](CIFAR_LINEAR_RESULT.json) 继续绑定实际旧入口SHA，原 [MNIST_LINEAR_RESULT.json](MNIST_LINEAR_RESULT.json) 也未覆盖。前版两份输出保留在 `.tmp/verify-group-template-fixed-{mnist,cifar}/`；当前MNIST/mlp输出保留在 `.tmp/verify-group-mnist-mlp-{train,complete}/` 并生成独立正式快照。机器记录中的 `previous_portable_template_repair_audit` 与 `initial_portability_audit` 保存前版来源、全部差异、输出哈希及比较范围，不倒写此前实测版本。

新入口每次输出的JSON与Markdown都引用当次真实 `audit_script_path` / `audit_script_sha256`。报告模板的措辞修复不改变checkpoint审查、指标、门限、CPU回放实现或正式快照来源。

## 三、使用与验收边界

```powershell
python -B studies/main_historical/verify_group.py --data CIFAR10 --model linear --with-eval --replay --output .tmp/cifar-linear-cpu-recheck
python -B studies/main_historical/verify_group.py --data MNIST --model linear --with-eval --replay --output .tmp/mnist-linear-cpu-recheck
```

默认输出位于 `.tmp/historical-group-audit`，重核使用独立输出目录，保留旧正式快照。入口只使用 CPU，未执行任何进程管理、训练、resume 或 GPU调用。四条正式 train 未全部 succeeded 时报告 incomplete、退出非零，不打开活动 checkpoint；`--with-eval` 只读取既有正式eval，`--replay` 需要四条既有eval成功且model=linear，使用官方原始test数据和归档Linear / Accuracy定义执行额外CPU评测。

每条训练的显式 optimizer_step、train/test内部counter、四类metric history、canonical latest / best、SGD与cosine终态、结果与index/config和冻结来源一致性分别核对。test10000样本来自固定batch250、完整数据元信息与40batch差复算，正式worker没有提供逐样本字节观测；有限数和保存状态检查不等同重跑80000次更新。CPU权重回放的正确样本数必须相同，Loss差仍按≤1e-6，不放宽既定曲线门。

入口可审查四模型的训练终态，额外CPU回放目前只支持Linear；不把其他模型标为已经回放。完整组合诊断不等于全部八组合终验，输出保持 `whole_study_final_gates_applied=false`、`historical_reproduction_passed=null`。CIFAR独立eval记录控制器的WinError5、退出1与仅恢复记录边界继续保留在 [完整结果报告](CIFAR_LINEAR_RESULT.md) 和 [执行记录](EARLY_EVAL_CIFAR10_LINEAR_EXECUTION.json)。

## 四、依赖与可取得证据

当前依赖为Python、NumPy、Torch和PyYAML；运行时先导入NumPy再导入Torch，CPU线程数为2。需在本仓库保有历史Git对象 `4ccb28d`、正式index/config、完整Run原始result/log/scalars/tracker/checkpoint、共享原始数据及SOURCE/DATA/REFERENCE manifests。入口不会下载或生成缺失训练数据，也不导入会导出归档文件的训练适配器。

新入口、方法说明和机器比较记录可随Git提供；原始Run/checkpoint/data和 `.tmp` 程序/回归输出仍是本机产物，不随Git clone提供。当前 `scientific_json_sha256` 对剔除4个审计来源/时间字段及整个Study运行状态快照后的全部JSON计算；运行状态快照单独保留两份原值与差异。除此之外全部字段必须精确相同，包含完整曲线、checkpoint/optimizer/scheduler/history、固定门与独立CPU评测，未只抽查末点。
