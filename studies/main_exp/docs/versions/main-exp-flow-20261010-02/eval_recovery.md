# 独立评测恢复约定

## 一、问题与边界

main-exp-flow-20261010-02 已完成全部连续训练。CIFAR10 CNN seed0 首次独立评测加载了正确的 best checkpoint，但少一个正确样本，原始终验为 invalid。原生评测链的隔离实验确认：允许 cuDNN TF32 时并发执行存在两种预测结果，输入及模型 hash 相同；FP32 对照一致。依据见 [EVAL_DIAGNOSTIC.json](EVAL_DIAGNOSTIC.json)。具体 kernel 选择尚未跟踪，不把此诊断写成库缺陷已经修复。

## 二、串行恢复的停止条件与实测

首先考虑原配方、原精度的串行评测恢复，库已有单条 `run-one` 重评入口，见 [Study 指南](../../../../README.md)。以下条件在正式恢复前确定。

1. 先用五个独立新进程串行执行原生 prepare / execute，检查正确样本数均与训练 best 一致。任一次不一致则停止正式恢复，继续调查；不循环重试直到出现通过值。
2. 在对应 Study 内归档第一次评测的 config、result、日志和 tracker，以及第一次 invalid 的 COMPARISON。核对归档字节 hash，保留其质量失败状态与复现条件。
3. 仅移开该 eval 已归档的输出，config 和来源冻结保持不变。确认没有本任务的其他 GPU 进程后，执行一次原生 `rpipe run-one studies/main_exp 00597cc7bfc1fb1d` 六阶段链。
4. 单条重评结束后执行 Study process，重新核验全部 32 个自身 best 对照、完整性与固定四 seed 曲线门。若失败，保留第二次输出并继续调查，不放宽门或再次盲重试。
5. 正式报告同时记录两个评测 attempt、实际调度条件和未解决的设备数值边界。新的精度策略若进入代码或配方，须另用新 version 做前置门和受影响训练复验，不能把此次原配方串行重评当作新精度训练验收。

2026-10-11 的五次独立新进程串行复查得到 8915、8915、8915、8915、8914。第一项停止条件触发，没有移走正式 eval 输出，也没有重写其 result 或再次试到通过。问题不能归结为并发调度；原 version02 仍为 invalid。

## 三、明确设备精度的新基线

1. 库 System runtime 增加可选布尔配置 `cudnn_allow_tf32` 与 `matmul_allow_tf32`。未配置时保留 Torch 当前值，避免改变既有使用者；配置后在构造 Data / Model 前生效，记录实际值。非法类型明确拒绝。
2. 两个正式 Study 显式关闭这两个 TF32 开关。main_exp 的原代码 / 当前链 preflight 与 main_probe 的原代码 / 当前链必须使用同一精度；prepare 拒绝遗漏或不匹配的精度控制，不能在内部静默更改配方。
3. 精度改变会影响浮点训练轨迹，不能复用 version02 训练冒充新基线。保留全部旧 Run、配置、来源清单、数字及 invalid 终验，使用 main-exp-flow-20261011-03 / main-probe-flow-20261011-03 的新 Run ID。
4. 先完成 runtime 回归、同 checkpoint 的受控并发 FP32 诊断及 main_probe 八格复验，再通过 main_exp 同设备 600-step 八格前置门。通过后执行新基线完整矩阵，原始数据清单和固定门不变。
5. 利用已确认的三重型任务并发经验，main_exp 以显式 `--round 3` 展开本轮计划；库已有配置入口，容量估算器 B-022 不在此处宣布修复。持续检查显存、唯一进程、完整预算及评测轨迹，不减少训练或 test 数量。

关闭 TF32 是此次受控数值环境方案，不是所有设备的逐位一致保证；B-023 需新基线完整独立评测验收后再决定关闭。
