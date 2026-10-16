# v0.3.0 本机成对探针执行记录

## 一、摘要

- 发布基线：`v0.3.0` / `804b3823ab6d2af46cbda1a65688ca8b76c60b22`；历史原版保持 `98648f3a5c7db7dccf3ca806410d5b6fdee9484c`。
- 目标：在本机独立验收八个 60-step 成对组合，并测量不同 Run 的并发吞吐与资源边界。原版先 / 当前版后、RNG 恢复、完整 test 与全部门限保持 [PLAN.md](<../../../docs/research/研究计划 RPipe.md#study-main-probe>) 不变。
- 当前状态：新的正式 `main-probe-v030-20261013-final` 八个 Run 与 Study 终验均通过。环境、数据与共同顺序见 [main_exp 本轮执行记录](../../main_exp/docs/v030_execution.md)。历史报告与本机可取得清单已逐字节保全在 [快照清单](versions/v030-before-local-20261013/manifest.json)，不把历史完整通过当作本次验收。

## 二、执行与并发校准

1. 两处 version 同步：`study.yaml fixed.version` 与 `experiment_config.yaml version`。每次实测新建 version / Run ID，由新 index 选 Run，不覆盖旧工作区。
2. 首先做 MNIST / linear 单格真实 Flow smoke，再验证 Study process 因缺七格而拒绝。该拒绝是预期完整性边界，不作为完整通过。
3. 随后用原生 make / launch 顺序完成同一八组合串行基线。通过后，依据本机 GPU、reserved / 整卡峰值和 RAM 观察，逐级测 round2、资源允许时测 round3。每一级相同八任务、seed0、60-step、test10000，保存职责一致；不同 Run 的原版配置与资产完全隔离。
4. 每条成对门与 Study 终验均须通过；并发轨迹的原生 latest / best checkpoint 与 test history 对串行做严格零容差对照。父墙钟、吞吐、GPU / CPU / 主存 / I/O 分开记录。缺资源时保留未执行理由，不机械执行更高并发。
5. 显存预算依据本 Study 串行实测留出 context / 驱动余量及波动边界，不能使用 main_exp 的 4.5 GiB。GPU / RAM 准入不取消保护；正式执行继续用原生装箱的 safety=0.5。
6. 最终资源配置确定后，重新设置正式 version、fresh make、核对 provenance，再完整执行八格并定稿 Study 报告。校准 Run 不回填成最终配置的验收。

## 三、证据边界

校准报告与数字保存到本 Study `docs/`；临时观察脚本、安装缓存和命令日志在 `.tmp/v030-20261013/`，Run / 原版归档 / 观测权重在原生 Run 目录。这些本机原始产物不会随 clone 提供。WDDM 未提供可用逐进程 GPU 显存时，整卡增量只辅助判断 context / 驱动及背景开销，不声称精确归因。

## 四、2026-10-13 本轮结果

1. [smoke](v030_smoke.json)：MNIST / linear 成对数值门通过，单格完整性拒绝保留；不是完整八组。
2. [手动受控校准](v030_capacity.json)：每次同样八任务，串行 246.60 秒、round2 141.75 秒，约 1.74 倍总体吞吐。八个成对门、Study 终验以及 strict latest / best / history 串行对照通过。round3 被原准入保护挡住，未执行；不当作通过。
3. [原生自动分组校准](v030_capacity_auto.json)：依据 reserved 峰值、整卡增量及 context 余量取整到 5 GiB，保留 safety=0.5，同模型两任务并发。185.0 秒完成八任务，约为串行吞吐的 1.33 倍，严格数值与完整性门通过。较快的手动混合分组没有替代正式原生自动策略；未把调试准入预算 6.25 GiB 当作另一 Study 的预算。
4. [正式八格验收](v030_final_gate.json)：资源配置确定后重新声明 version、fresh make，8 succeeded / 0 failed / 0 pending，Study complete / passed，来源冻结未变；本轮软件基线为指定发布提交 `804b3823`，运行声明和资源预算为本轮新配置。总父墙钟约 171.20 秒，[资源观察](v030_final_telemetry.json)保留。正式数字由原生 report 生成在 [NUMBERS.md](NUMBERS.md)，校准 Run 没有回填成正式 Run。
