# Bugs

> 这里只保留尚未解决的缺陷。编号只增不复用；修复并验证后移除条目，历史证据留在 Git 与对应 Study 报告。正常差异或决定不修的事项不列为开放缺陷。权威仍是 [concept.md](../code/concept.md) → [layout.md](../code/layout.md) → 代码。

新增一条用下面的块，**一条一事**。

```text
## B-NNN 短标题
- 状态：open | investigating
- 看见：怎么复现、期望 vs 实际
- 落点：源文件 / Study / 测试
- 方案：
- 验证：
```

---

## 开放

B-019 / B-020 已经 CPU 回归与 version02 八组合真实复验关闭，失败和修复依据见 [record.md](record.md) 与 [探针报告](../../studies/main_probe/docs/STUDY_REPORT.md)。

### B-021 checkpoint 未完整保存随机数和数据迭代状态

- 状态：open。2026-10-11 确认，按当前要求在正在执行的 Study 实验完成后修复。
- 看见：训练保存 checkpoint 后，在新进程中加载并续跑。期望后续样本、随机变换与训练状态沿中断点继续；实际只恢复部分训练状态，不能保证与连续训练等价。当前证据为源码核对，跨进程数值复现与回归尚未执行。
- 落点：[训练 checkpoint 构建](../../src/rpipe/structure/algorithm/train/__init__.py) 的 `_build_payload` 保存模型、优化器、调度器、进度和日志，但没有随机数与数据迭代状态；[Algorithm.resume](../../src/rpipe/structure/algorithm/base.py) 忽略 `data`，未恢复这两类状态；[Data.rebind_train_steps](../../src/rpipe/structure/data/factory.py) 按初始 seed 重建剩余预算的采样器，可能重放相同 shuffle 的前缀。[main_exp recipe](../../studies/main_exp/recipe.py) 已明确拒绝精确续跑。
- 影响：涉及随机采样、数据增强或模型随机行为的中断续跑不能作为连续训练的等价结果。旧版 main_exp 的中断原因仍未查明，本缺陷不作为该次进程终止的原因；当前 version02 按计划从头连续训练，见 [复跑记录](../../studies/main_exp/docs/RERUN.md)。
- 方案：定义 checkpoint 的续跑契约和格式版本，保存并恢复 Python、NumPy、PyTorch CPU / CUDA 随机数状态，以及数据采样器、迭代位置和相关 generator 状态。明确 worker 随机状态与预取队列的支持边界、恢复顺序及成功条件。旧 checkpoint 缺少必要状态时明确提示能力限制，不静默宣称精确续跑。
- 验证：以相同配置和 seed 对照连续训练与中断后新进程续跑，覆盖非 epoch 边界，检查后续样本序列、随机变换、随机数、参数、优化器和调度器状态及指标；按声明的支持范围覆盖 CPU / CUDA 和数据 worker 场景。保留失败证据，针对性回归通过后再复验受影响的 Study。

### B-022 GPU 容量估算与调度过于保守，重型训练被串行安排

- 状态：investigating。2026-10-11 按用户要求记录，当前连续实验结束后再调查和修复。
- 看见：main-exp-flow-20261010-02 在单张 RTX 5090 D v2 上展开 64 个任务。期望在保证显存安全和实验配方不变的条件下充分利用设备；实际 make 将 CIFAR10 ResNet18 的四个 seed 分成四个单任务 wait 组，逐个训练。make 显示 free 20.2GiB、usable 10.1GiB；2026-10-11 02:04 的 nvidia-smi 单次快照为 GPU 利用率 34%、显存 5521/24455MiB。近期每 200 个训练 step 加完整 test 约需 16 秒，该 seed 的总耗时估计约 90 分钟。
- 落点：[容量估算与装箱](../../src/rpipe/structure/make/capacity.py) 默认 `SAFETY = 0.5`，按估算显存和可用预算安排并发。该策略与当前串行安排已确认；安全系数、显存估算误差、CPU 数据处理和其他瓶颈各自的影响尚未量化。单次 GPU 快照不证明持续利用率，也不证明提高并发一定加速。
- 证据：.tmp/main-exp-flow-20261010-02-make.log 中的 14 个 wait 组及显存预算；Run 20b54c915faeb9b9 的 assets/logs/run.log 中的实际阶段时间，见 [复跑记录](../../studies/main_exp/docs/RERUN.md)。这些日志和设备快照只在本机取得，不随 Git clone 提供。
- 方案：先测量代表任务的显存峰值、持续 GPU 利用率、CPU / 数据处理耗时及吞吐，再核对容量估算与安全余量。依据测量改进并发决策与配置方式，保留合理余量；不直接取消显存保护。当前在跑任务的源码和配方保持不变。
- 速度校准要求（2026-10-11）：维护者指出当前耗时可能异常，要求在当前连续实验结束后与早年版本做同条件速度对照。先确认旧版本的实际提交与可取得的计时证据；旧日志缺少设备或配方信息时，不直接比较总耗时。使用同一设备、数据、模型、batch、seed、精度开关、评测频率、checkpoint 频率及线程条件，分别测量训练、完整 test、checkpoint 保存和 Flow 调度开销；串行单任务与当前并发方式分别比较，记录预热、CUDA 同步计时、吞吐、墙钟时间及显存峰值。历史默认精度或 benchmark 与当前条件不同的结果单列，不能把改变数值条件带来的速度差当作 Flow 开销。先做代表性短程测量，再按证据决定优化与复验范围；当前长训期间不启动竞争 GPU 的速度实验。
- 临时处理：随后用户明确授权直接并发。2026-10-11 02:18–02:19 撤下串行调度父进程，保留 seed 1 原训练，通过现有 run-one 流程并发启动 seed 2、3；源码和配方未变。首次设备快照为 GPU 利用率 97%、显存 9903MiB，长期吞吐和稳定性仍待观察，容量估算器尚未修复，本项不关闭。具体进程、Run ID 和后续评估接管见 [复跑记录](../../studies/main_exp/docs/RERUN.md)。
- 校准进展：完整 version03 已通过。随后接入 recipe calibrate 的同设备历史 / 当前训练链计时，CNN 三次独立进程的初始与最终参数精确一致，完整 test 数值门通过；ResNet18 三次对照也全部通过，六组最终参数精确一致；已将归档 blob 读取合并为 Git batch，文件字节、损坏响应拒绝和真实导入运行条件回归通过。报告和原始 JSON 见 [速度校准](../../studies/main_exp/docs/speed_calibration.md)。尚未测父调度开销和并发总体吞吐，不据单任务 PyTorch allocated 峰值直接降低装箱保护，也不据短程结果关闭本问题。
- 验证：以固定设备、配置、seed 和预算对比修改前后的并发安排、整体墙钟时间、显存峰值和吞吐，覆盖轻型、重型及混合任务；验证无 OOM、任务重复或遗漏，并保留失败证据。确认实验数值契约仍成立，不能以缩短训练或减少评测取得加速。

### B-023 CIFAR10 CNN 自身 best 独立评测差一个正确样本

- 状态：investigating。2026-10-11 完整矩阵终验发现。
- 看见：main-exp-flow-20261010-02 的 train 16ae89c5c90f7eca 与 eval 00597cc7bfc1fb1d，CIFAR10 / CNN / seed0 / test10000 / batch250 / CUDA。期望加载自身 best 后正确样本数一致；实际训练 step76800 best 为 8915，独立评测为 8914。加载路径、步数、40 个完整 test batch 和 result / tracker 一致性均通过，其余 31 个 best 对照通过。
- 落点：[Study 终验](../../studies/main_exp/process/curves.py)、[EvalAlgorithm](../../src/rpipe/structure/algorithm/eval/__init__.py)、历史配方与设备运行条件。当前尚未确认是评测链、并发运行或数值环境问题，不能据此归因于 checkpoint 丢失。
- 证据：[version02 COMPARISON.json](../../studies/main_exp/docs/versions/main-exp-flow-20261010-02/COMPARISON.json) 的 best_source_audit；[复跑记录](../../studies/main_exp/docs/RERUN.md) 第 73 项。正式 Run 原始 result / tracker / 日志在本机 runs 下；只读 checkpoint 重评的 GPU topk 正确数为 8915，重复前向及训练前向 / 反向后恢复模型均一致，临时诊断不随 Git clone 提供。
- 调查：原生 Flow 八进程默认精度两批分别有 5/8、4/8 重现 8914，其余为 8915；串行五次也有一次 8914，不能归因于并发调度。允许 cuDNN TF32 时逐样本 logits 差异定位到 test index4970。禁用 TF32 的 FP32 对照 8/8 为 8915，输入和模型 hash 与默认精度对照完全一致。详见 [EVAL_DIAGNOSTIC.json](../../studies/main_exp/docs/EVAL_DIAGNOSTIC.json)。证据指向设备数值执行，具体 kernel 选择仍待确认；FP32 诊断不替代原矩阵验收。
- 方案：保留原失败，明确可复现的设备精度与并发契约，跟踪具体 kernel 选择；确认根因后明确修复边界与新版本复验方案。不放宽原有正确样本一致门，不用临时重评覆盖正式结果。
- 验证：同 checkpoint 与完整固定 test 的多次新进程评测，覆盖串行和并发条件，检查逐样本预测与输入哈希；针对根因做回归后按计划复验受影响结果，再应用固定曲线门。
- 新发现：归档 train_model 在导入时设置 cudnn.benchmark=True，覆盖 Flow 已应用的 false；旧数值诊断实际 benchmark 为 true。适配器现保存并恢复导入前开关，Study prepare 核对真实设备条件，register 拒绝旧源码前置门。真实导入保留与异常拒绝回归通过；TF32 允许、实际 benchmark=false 的八次同 checkpoint 并发诊断均为 8914，但不匹配旧训练 best8915，不能作为恢复验收。证据见 [运行条件回归](../../studies/main_exp/docs/ARCHIVE_RUNTIME_REGRESSION.json)、[隔离诊断](../../studies/main_exp/docs/BENCHMARK_ISOLATION_DIAGNOSTIC.json)。
- 修复进展：System runtime 增加可选 TF32 布尔配置与实际值记录，未配置保留 Torch 当前值；两个 Study 显式关闭 TF32并检查两侧条件。runtime 针对性回归 13 passed，核心回归 295 passed / 40 deselected。main_probe version03 八格已通过；main_exp 在导入修复基线 09e495b 上重新通过八格前置门，完整 version03 的 32 train / 32 eval、全部自身 best 一致性与固定曲线门于 19:48 通过。本次显式 FP32 基线的修复复验完成；默认精度的具体 kernel 根因仍未完全确定，本项保留调查状态，旧失败归档保留。

下次新增问题从 **B-024** 起编号。

已关闭项不在此处重复列出；修复、验证及维护关闭的不同口径见 [record.md](record.md)、[brainstorm.md](brainstorm.md) §4 和对应 Study 报告。没有开放条目不等于所有历史根因已查明或所有场景已验证。
