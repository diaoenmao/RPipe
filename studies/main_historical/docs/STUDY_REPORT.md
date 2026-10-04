# 历史 main 曲线复现报告

## 一、摘要

2026-10-04，本机在最新 dev `8bccbac321d4c3ac1ea9892a5e774c114e0298c6` 上完成全部 **32 条连续 80000-step 训练与 32 条对应 best 独立评估**。MNIST / CIFAR10 × linear / mlp / cnn / resnet18 × seeds0–3 全部 succeeded，每条训练有400个完整test点，每次test为10000个样本。主控制器于北京时间 **09:12:29** 成功结束，终态工具句柄37125退出0。

默认完整比较于 UTC `2026-10-04T10:38:37.965111+00:00` 执行退出0，[COMPARISON.json](COMPARISON.json) 为 `complete=true / final_gates_applied=true / passed=true / errors=[]`。八组的终点、末50点均值、七个固定锚点 MAE / 最大误差、末50点时间波动，以及两数据集的模型排序全部通过事先冻结的门限。终点相对 main README 原图估读的最大绝对差为 **0.207501 个百分点**。可以据此称为：**在预先固定的原图估读合同下，较好复现 main README 的历史曲线**。

本机归档原代码 / 当前链600-step八格对照通过，24个段的模型参数和整数buffer逐位一致。现代 main 的60-step八格 CUDA 对照也通过，16个段参数最大差0，optimizer / RNG / scheduler一致；它使用原 Stats 全精度 profile 与确定性配置，另见 [当前设备结果](../../main_reproduction/docs/CURRENT_DEVICE_RESULT.md)。这些是分别限定范围的证明，不把短探针对齐写成全部80k权重逐位对齐。

最终机器摘要见 [FINAL_RESULT.json](FINAL_RESULT.json)，逐项完成证据见 [COMPLETION_AUDIT.json](COMPLETION_AUDIT.json)。本轮未修改生产 `src/`，新增可重跑历史 Study、当前设备对照入口及审计/报告。所有预定验收已完成，无待运行的正式 Run。

## 二、完整结果

### （一）终点与 main 原图

以下为 step80000 的四 seed 同步 mean，单位为 Accuracy (%)；差距单位为百分点。原图数值是PNG像素估读，独立评测使用的较高 best 未替代训练曲线终点。

| **数据集** | **模型** | **原图终点估读** | **本机终点 mean** | **绝对差** | **末50点 mean** | **固定门** |
|---|---|---:|---:|---:|---:|---|
| MNIST | linear | 92.6700 | 92.6750 | 0.0050 | 92.6687 | 通过 |
| MNIST | mlp | 98.1600 | 98.1500 | 0.0100 | 98.1524 | 通过 |
| MNIST | cnn | 99.4100 | 99.4125 | 0.0025 | 99.4087 | 通过 |
| MNIST | resnet18 | 99.5900 | 99.5675 | 0.0225 | 99.5808 | 通过 |
| CIFAR10 | linear | 40.3500 | 40.3700 | 0.0200 | 40.3694 | 通过 |
| CIFAR10 | mlp | 63.2600 | 63.4675 | 0.2075 | 63.4733 | 通过 |
| CIFAR10 | cnn | 89.0100 | 89.0250 | 0.0150 | 88.9672 | 通过 |
| CIFAR10 | resnet18 | 92.9700 | 92.9250 | 0.0450 | 92.9227 | 通过 |

![完整四 seed 历史曲线与 main 原图估读](figures/historical_comparison.png)

实线和阴影为本机各步四 seed mean / population std（ddof=0），十字为原图估读。保留全部400点和真实瞬时波动，没有平滑或删除MNIST ResNet18的中段下跌。阴影的原图标准差未数字化验收；均值±标准差绘图区可能超过100%，不表示实际 Accuracy 大于100%。

### （二）固定锚点与末段诊断

锚点为step10200 / 30200 / 40200 / 50200 / 60200 / 70200 / 80000。末50点为slot350–399。MNIST门：终点及末50点均值差≤0.25pp、锚点MAE≤0.25pp、最大差≤0.60pp、末段mean曲线时间population std≤0.15pp；CIFAR10门分别为1.00 / 1.00 / 1.00 / 2.00 / 0.50pp。门限在训练前采用，未根据结果调整。

| **数据集** | **模型** | **末50点均值差 (pp)** | **锚点 MAE (pp)** | **锚点最大差 (pp)** | **末段时间 std (pp)** |
|---|---|---:|---:|---:|---:|
| MNIST | linear | 0.0013 | 0.0096 | 0.0475 | 0.0124 |
| MNIST | mlp | 0.0076 | 0.0150 | 0.0250 | 0.0037 |
| MNIST | cnn | 0.0113 | 0.0164 | 0.0475 | 0.0062 |
| MNIST | resnet18 | 0.0092 | 0.0186 | 0.0475 | 0.0065 |
| CIFAR10 | linear | 0.0094 | 0.0371 | 0.1600 | 0.0527 |
| CIFAR10 | mlp | 0.2333 | 0.2536 | 0.5175 | 0.0316 |
| CIFAR10 | cnn | 0.0228 | 0.2779 | 0.5950 | 0.0637 |
| CIFAR10 | resnet18 | 0.0373 | 0.2093 | 0.8825 | 0.0349 |

两数据集末50点均值均保持 `resnet18 > cnn > mlp > linear`。所有锚点中最大差为CIFAR10 / resnet18的0.882498pp，低于2pp门。初始段与非锚点孤立波动属于完整叠图保留的描述证据，不扩称原图所有400点均满足同一最大差门。

## 三、基线、配方与计算对照

main固定为 `98648f3a5c7db7dccf3ca806410d5b6fdee9484c`，README两张图与历史提交 `4ccb28d0496110253e9f8e3f3df658853f07996b` 的字节相同。main当前代码配方已变化，历史图片候选单独使用归档旧模型、CPU torchvision增强和常量归一化：MNIST mean/std0.1307/0.3081；CIFAR10 mean0.4914/0.4822/0.4465，std0.2023/0.1994/0.2010。旧CNN保留四层BN。归档源由Git blob原字节导出，不修改源码。

历史矩阵每Run连续80000次SGD更新，train/test batch250；lr0.01、momentum0.9、Nesterov、weight_decay0.0005、clip1、cosine T_max80000，每200step完整评测10000张test。采样总预算始终80000，包括600-step前缀探针。当前RPipe正确全程Accuracy-best保留，归档旧best比较缺陷未移植；图片比较读取train/test history，独立eval只核对自身best来源。

设备为RTX5090 D v2 24GB，Python3.13.9、Torch2.11.0+cu130、torchvision0.26.0+cu130、NumPy2.4.4；deterministic=true / benchmark=false / CUBLAS_WORKSPACE_CONFIG=:4096:8 / CPU threads2。16个原始数据文件SHA与历史清单一致，MNIST60000/10000、CIFAR10 50000/10000。99个数值源文件、65个展开计划文件在最终审查时仍与冻结清单一致。

历史原码 / 当前链同机探针为8格、seed0、step200/400/600三段；模型参数、整数buffer、SGD、scheduler、CPU/CUDA RNG、实际采样索引及增强输入均一致，train/test Loss累计差最大6.661338147750939e-16。独立CPU重载审计通过，见 [PREFLIGHT.json](PREFLIGHT.json)、[PREFLIGHT_VERIFICATION.json](PREFLIGHT_VERIFICATION.json)。这只证明600-step前缀数值桥，不证明全部80k原码/current权重相等。

现代main另以60-step/eval30、八格seed0验证。原Stats(dim1)对完整train、batch250计算的全精度profile，统一确定性设置；参数/buffer/optimizer/RNG/scheduler一致，独立全test评测一致，CPU重载来源审计通过。现代CNN在MNIST step60实测仅11.35%，该实际结果完整保留；它与历史旧CNN的80k结果属于不同配方。未验证无profile常量默认设置或benchmark=true模式，见 [CURRENT_DEVICE_PLAN.md](../../main_reproduction/docs/CURRENT_DEVICE_PLAN.md)、[CURRENT_DEVICE_CPU_RELOAD_AUDIT.json](../../main_reproduction/docs/CURRENT_DEVICE_CPU_RELOAD_AUDIT.json)。旧cu128证据保留自己的设备边界，本次结论只用cu130本机新证据。

## 四、执行与独立审计

主调度有且仅有50个不同Run的start和50个对应exit0，分别30train+20eval；另2条原计划CIFAR10/resnet18训练和12条已完成组的eval提前执行，合计全部64Run，未新增配方/seed。父调度曾由身份绑定guard等待提前训练，训练子进程继续连续执行；guard退出0并恢复父，主调度按正式succeeded跳过提前Run。第二波提前训练未启动。完整执行与64条Flow/tracker/checkpoint审计见 [FINAL_EXECUTION_AUDIT.md](FINAL_EXECUTION_AUDIT.md) 与 [机器JSON](FINAL_EXECUTION_AUDIT.json)。

各条train恰一次Flow start、fresh tracker start，无resume/retry或重复worker；latest步80000、counter96000、scheduler最后步80000。四类train/test Loss/Accuracy各400个显式坐标，test段每点40batch。best为自身全程Accuracy最大点，独立eval的actual resume_path与对应canonical best一致，各完整test10000 / 40batch。CPU审计核对保存状态、有限数、SGD/scheduler、tracker/TensorBoard scalar和原始result/log。八组汇总见 [GROUP_INTEGRITY_AUDIT.md](GROUP_INTEGRITY_AUDIT.md)。

- [MNIST_LINEAR_RESULT.md](MNIST_LINEAR_RESULT.md)：四 seed 训练完整性、对应 best 独立 eval 与来源审查。
- [MNIST_MLP_RESULT.md](MNIST_MLP_RESULT.md)：四 seed 训练完整性、对应 best 独立 eval 与来源审查。
- [MNIST_CNN_RESULT.md](MNIST_CNN_RESULT.md)：四 seed 训练完整性、对应 best 独立 eval 与来源审查。
- [MNIST_RESNET18_RESULT.md](MNIST_RESNET18_RESULT.md)：四 seed 训练完整性、对应 best 独立 eval 与来源审查。
- [CIFAR_LINEAR_RESULT.md](CIFAR_LINEAR_RESULT.md)：四 seed 训练完整性、对应 best 独立 eval 与来源审查。
- [CIFAR_MLP_RESULT.md](CIFAR_MLP_RESULT.md)：四 seed 训练完整性、对应 best 独立 eval 与来源审查。
- [CIFAR_CNN_RESULT.md](CIFAR_CNN_RESULT.md)：四 seed 训练完整性、对应 best 独立 eval 与来源审查。
- [CIFAR_RESNET18_RESULT.md](CIFAR_RESNET18_RESULT.md)：四 seed 训练完整性、对应 best 独立 eval 与来源审查。

两组linear额外CPU完整test推理与正式eval的整数正确样本数相同，Loss最大差分别1.0244548320770264e-08 / 2.682209010451686e-08。其他六组仅做保存状态/来源/指标的CPU审计，没有额外CPU推理，也没有对非线性模型套用linear参数/momentum映射。正式eval worker未导出在线终态权重指纹，权重来源依据为实际resume日志、未变canonical best及冻结加载器；不扩称已取得在线终态快照。

独立原图与比较合同终审见 [FINAL_REFERENCE_AUDIT.md](FINAL_REFERENCE_AUDIT.md)。[NUMBERS.md](NUMBERS.md) 来自native process聚合；其中通用标量std与曲线使用的population std口径分开，不用于替换原图验收。

## 五、保留的失败、恢复与验证边界

1. 首次旧Logger(None)探针因writer字段不存在失败；改用真实TensorBoard路径重新在新目录探针，归档源码不改，原失败保留在本机 `.tmp/historical-preflight-20261004/`。
2. 初次tests/run.py子进程测试退出1且无诊断；随后显式注入隔离小依赖并先导入NumPy，直接pytest核心unit c1/c2为266 passed / 23 deselected。证据 `.tmp/test-results/20261003T164247Z_38a72e/report.md` 不随clone提供；未声称全部integration/e2e已通过。
3. CIFAR10/linear提前eval四条worker全exit0/succeeded，但记录控制器10488因最终os.replace遭WinError5而退出1。只恢复执行JSON，没有重跑worker，也没有把原控制器退出码改成0。原始 [失败记录](EARLY_EVAL_CIFAR10_LINEAR_FAILED_CONTROLLER.json)、[未提交记录](EARLY_EVAL_CIFAR10_LINEAR_UNCOMMITTED_RECORD.json)、[实际helper](CIFAR_LINEAR_EVAL_EXECUTION_HELPER.py) 与 [恢复记录](EARLY_EVAL_CIFAR10_LINEAR_EXECUTION.json) 均保留并哈希绑定。后续helper只修复记录写入重试与执行去重，数值计算源不变。
4. 原图未找回逐点日志、实际seed集合、原Torch环境或阴影原始数据；图像估读不等于原指标。末点/七锚点/晚期统计通过不证明所有原图点逐位一致。未重建历史Torch2.0.1环境；本机原码/current确定性前缀对照和原图近似程度分别报告。
5. native resume不保存完整RNG/迭代位置，本矩阵全部连续且没有训练恢复。正式worker未做逐样本输入哈希；实际输入字节一致来自专门的600-step探针，不能转称正式80k全程逐样本观测。
6. 数据、64Run完整checkpoint/log/tracker、Git归档导出和隔离运行依赖均为本地忽略产物，不随Git clone提供。正式配置/运行入口、来源清单、数字与图保留在Study，异机需重新准备数据并执行。混合GPU负载时长不作算法性能排名。

## 六、可重跑入口与存档

先读 [PLAN.md](PLAN.md)、[TARGET.md](TARGET.md)、[REFERENCE_CURVES.json](REFERENCE_CURVES.json)，环境见 [ENVIRONMENT.json](ENVIRONMENT.json)，数据见 [DATA_MANIFEST.json](DATA_MANIFEST.json)，源码清单见 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)。适配/准备/调度入口为 [recipe.py](../recipe.py)、[prepare_data.py](../prepare_data.py)、[run.py](../run.py)。本机既有64Run已成功，launch只跳过已成功项，不应重新创建重复矩阵。

```powershell
python studies/main_historical/run.py status
python studies/main_historical/compare.py
python studies/main_historical/verify_group.py --data CIFAR10 --model resnet18 --with-eval --output .tmp/independent-audit
```

verify_group无GPU操作；只有linear支持额外 `--replay`。固定数值源及原门限不要在结果出来后修改。当前最终记录为 [ACTIVE_SESSION.json](ACTIVE_SESSION.json)、[COMPLETION_AUDIT.json](COMPLETION_AUDIT.json)，完整运行事件仍在未改写的 [EXECUTION.json](EXECUTION.json)。

原进度报告的全部时间快照、部分诊断和故障细节逐字保存在 [PROGRESS_REPORT_20261004.md](PROGRESS_REPORT_20261004.md)；原running元数据保存为 [ACTIVE_SESSION_PROGRESS_SNAPSHOT.json](ACTIVE_SESSION_PROGRESS_SNAPSHOT.json)、[COMPLETION_AUDIT_PROGRESS_SNAPSHOT.json](COMPLETION_AUDIT_PROGRESS_SNAPSHOT.json)、[EAGER_EXECUTION_PROGRESS_SNAPSHOT.json](EAGER_EXECUTION_PROGRESS_SNAPSHOT.json)。这些是历史时点，不代表当前尚有任务运行。最终结果未覆盖前三组原JSON快照；完整结论由本次COMPARISON与FINAL_RESULT提供。
