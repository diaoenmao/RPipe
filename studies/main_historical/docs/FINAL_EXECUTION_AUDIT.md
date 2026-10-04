# 历史矩阵最终执行审查

## 一、结论

独立 CPU 审查通过：**64 个正式 Run 全部成功，32 条连续 80,000-step 训练与 32 条 own-best 独立完整评测齐全**。[完整审查 JSON](FINAL_EXECUTION_AUDIT.json) 顶层为 `passed=true`、`errors=[]`，包含逐 Run 检查、真实执行出处、终态记录及证据 SHA256。

主控制器工具句柄 `37125` 由执行者确认真实终态退出码 `0`；[EXECUTION.json](EXECUTION.json) 于北京时间 **2026-10-04 09:12:29** 结束，状态 `succeeded`。本审查没有启动 GPU、重新训练、重新评测或控制进程。

| **实际执行来源** | **训练 Run** | **评测 Run** | **worker 退出码** |
| --- | ---: | ---: | --- |
| 主控制器 | 30 | 20 | 50 个均为 0 |
| 第一波 eager | 2 | 0 | 两个均为 0 |
| 提前 MNIST / linear 评测 | 0 | 4 | 四个均为 0 |
| 提前 CIFAR10 / linear 评测 | 0 | 4 | 四个均为 0 |
| 提前 MNIST / mlp 评测 | 0 | 4 | 四个均为 0 |
| **合计** | **32** | **32** | **64 个均成功** |

执行完整性与预先曲线门分别判定。本审查 JSON 的 `historical_reproduction_passed=null`、`whole_study_curve_gates_applied=false` 表示本文件的审查范围；整轮曲线结论由 [COMPARISON.json](COMPARISON.json) 和 [STUDY_REPORT.md](STUDY_REPORT.md) 单独给出。

## 二、唯一执行与连续性

主控制器记录恰有 50 个互异 start、50 个对应 exit，每个 exit 均为 `0 / succeeded`；实际分配为 30 train、20 eval。所有 group 均有结束记录，组内 Run 集合与这 50 个启动完全相同。

两条 eager 训练由 [WAVE1_COMPLETION.json](WAVE1_COMPLETION.json) 绑定实际工具句柄 `11464 / 17242`、两个退出码 `0`、正式成功与精确身份退出。12 条提前评测分别由 [MNIST linear 记录](EARLY_EVAL_EXECUTION.json)、[CIFAR10 linear 记录](EARLY_EVAL_CIFAR10_LINEAR_EXECUTION.json)、[MNIST mlp 记录](EARLY_EVAL_MNIST_MLP_EXECUTION.json) 绑定。

这三类执行来源互不重叠，恰覆盖冻结 index 内的 64 个 Run。每条正式日志均只有一次 Flow start、一次 Flow succeeded、一次 execute finished；Flow start 的 PID 与所属执行记录一致。64 条 tracker 都只有一次 `start / keep_until=0`，没有 error / retry 或重复正式 Run 启动。32 条训练的 resume 事件均为零；32 条 eval 各有一次预期的 own-best 权重加载。

第二波计划中的 `b218a0bde4013fb8`、`3b6bdafdab45a4ef` 实际各由主控制器执行一次。执行者的只读 prelaunch 检查因主矩阵已经终态而安全拒绝，未启动第二波 ad-hoc worker 或新 guard；原 GUARD 与 WAVE1 归档仍完全相同。

## 三、训练、评测和 CPU 存档核对

### （一）32 条连续训练

- 每条最新 canonical checkpoint 为 optimizer step **80000**，tracker batch ledger 为 **96000**；train / test 的 Loss、Accuracy 各有完整 **400** 个 history。
- 原始 scalars 每条为一次 start 加 1600 个指标记录；optimizer 坐标严格为 `200, 400, …, 80000`，每槽 train ledger 增加 200 batch、test ledger 增加 40 batch。原始日志的 train / test 坐标与此完整序列一致。
- 每个 test 槽的 40 batch × 250 对应完整 10,000 样本。每条训练据账本推导为 train 20,000,000 样本、周期 test 4,000,000 样本；32 条共有 12,800 个周期完整 test 槽，每槽同时保留 train / test 指标。
- CPU 重载全部 32 对 `latest.pt / best.pt`，共 **64** 个 canonical checkpoint。模型与 SGD momentum 张量均有限；momentum / Nesterov / weight decay 和 80k cosine scheduler 对应冻结配置。存在 BatchNorm 计数 buffer 的模型，latest / best 的计数分别与各自 optimizer step 一致。
- best 步数对应实际 400 点 Accuracy history 的首次严格最大值；canonical bundle、目录 metadata、tracker、result 指标相互对应。

### （二）32 条 own-best 独立评测

每条 eval 的日志记录完整自家训练 Run 的 `best.pt` 路径；resume step、scalars optimizer 坐标和 eval tracker progress 均等于该 canonical best 步数。每条 eval 仅有 test split、40 个实际 batch ledger、一个 Loss / Accuracy history，覆盖 full test 10,000 样本。

各条 eval 与自家 best 所在训练 test 点的 **Loss 最大差为 0**，10,000 样本下推导的 correct 数严格相等。每份 `result.json` 的 control 严格等于各自冻结配置，artifact / result / tracker 路径属于自家正式 Run，指标来自相应 tracker；没有跨 seed 或跨模型借用结果。

样本总量依据实际 batch ledger、固定 batch 250、完整 10k 数据 metadata 与冻结的无截断 eval loop 交叉核对。长训没有保存逐批图像 trace；本审查没有用额外 GPU 回放补造该证据。

## 四、真实退出与异常证据

### （一）CIFAR10 linear 记录控制器

工具句柄 `10488` 的记录控制器实际退出码为 **1**：第四个 worker 的 wait 已返回 `0` 后，原子发布执行 JSON 遭遇 WinError5。四条 worker 均真实退出 `0 / succeeded`；后续只恢复执行记录，恢复程序退出 `0`，没有重跑 worker。

本审查继续保留 [原控制器已提交记录](EARLY_EVAL_CIFAR10_LINEAR_FAILED_CONTROLLER.json)、[原未提交记录](EARLY_EVAL_CIFAR10_LINEAR_UNCOMMITTED_RECORD.json)、[实际执行 helper](CIFAR_LINEAR_EVAL_EXECUTION_HELPER.py) 及恢复 helper 的原 SHA256。恢复记录仍明确 `controller_status=failed / controller_exit_code=1`，未改写为原控制器成功。

### （二）父调度协调与相关 PID

guard 工具句柄 `89871` 真实退出 `0`，[GUARD.json](GUARD.json) 为 `passed / succeeded`、`parent_suspension_owned=false`。事件顺序为暂停父、复核安全边界、验证 eager 成功并退出、恢复父。与 [WAVE1 归档](GUARD_WAVE1.json) 和完成记录的 SHA256 一致；已启动训练子进程保持连续。

独立 targeted unsandboxed、只读 psutil 查询仅核对本 goal 执行记录中的 **68 个去重 PID**，全部不存在。没有枚举其他进程、扫描无关命令或执行暂停 / 终止操作；查询原始数据包含在正式审查 JSON。

早期 EAGER_EXECUTION、ACTIVE_SESSION、COMPLETION_AUDIT 中的 `running` 属于其记录时刻的进展快照。执行者将旧字节归档为 PROGRESS_SNAPSHOT，并更新当前状态；旧快照不改变实际终态，也不构成科学失败。本审查的硬证据为终态 EXECUTION / WAVE1 / GUARD、正式 Run 资产与原失败恢复记录。

## 五、冻结来源与审查边界

审查前后独立重新哈希并确认：**99 个计算源文件、65 个 expanded plan 文件、16 个官方原始数据文件**均与冻结清单一致；共享历史归档的 **36 个文件**与原 Git blob 的 SHA256 一致。所有审查使用的 result、log、tracker、canonical checkpoint、执行记录及 helper 均在正式 JSON 中记录来源 SHA256，审查期间未变化。

独立 CPU 脚本位于本机 `.tmp/final_historical_execution_audit.py`，相关 PID 查询证据位于 `.tmp/final_execution_pid_observation.json`；二者不会随 Git clone 提供，其 SHA256 与查询内容已保存在 [FINAL_EXECUTION_AUDIT.json](FINAL_EXECUTION_AUDIT.json)。本审查只新增独立审查文件，没有改写冻结数值源、配置、index、Run 结果或原执行证据。
