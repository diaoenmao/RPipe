# Checkpoint / 重试恢复验收计划

## 1. 验收问题

针对 B-009–B-012，验证真实 `launch → run-one → train → checkpoint → retry → eval → process` 链路。
成功不只看 Run 数量：必须核对失败时旧存档、恢复学习率和最终 eval 的来源。它不是 MNIST 精度研究。

## 2. 固定条件与因素

- 2 个 Experiment（train / eval）× seed 0 / 1，共 4 个 Run；使用独立 `version: recovery-20261002`，不覆盖旧 Study。
- CPU、MNIST 前 256 个训练样本、linear、batch 32、6 optimizer steps、SGD momentum 0.9、lr 0.03、cosine。
- 每 2 step 保存 latest / 评测；按 Loss 选 best。test 仅前 2 个 batch，共 256 个样本，两侧口径相同。
- 复用已有本地 MNIST 数据副本；不联网、不启动 GPU / 大网格。

## 3. 高效率排班与故障控制

- 同类 train 一组，重试串行，再放行符合依赖条件的 eval；CPU 每个进程限制 1 个计算线程。
- 仅在 `.tmp/` 验收 harness 中拦截 checkpoint 最终 bundle 替换，不向生产配置添加故障开关。
- seed 0：第一次提交 latest step 4 时模拟 `PermissionError`，重试应从 latest step 2 恢复并完成。
- seed 1：首轮每次提交 latest step 4 都失败，耗尽一次重试后其 eval 必须不启动，Study 必须非 complete；其他 seed 的 eval 可以完成。
- 第二次 launch 关闭故障，seed 1 续训成功后才执行其 eval；seed 0 已完成部分应跳过。
- 失败证据与首轮 process 快照保留到 `.tmp/checkpoint-recovery-20261002/`；正式 Run 日志和 checkpoint 沿用 Study 目录。

## 4. 时长预估

首次执行前预算：每条 train（含进程启动）约 10 s，eval 约 5 s；首轮含两次串行 train 重试约 35 s，二次恢复约 15 s。以实际测量更新报告，不把预算当性能门槛。make 后补各 Run ID。

| 波次 | mode | seed | Run ID | 单次预估 |
|---|---|---:|---|---:|
| 1 | train | 0 | `41e0510c18eecdb7` | 10 s |
| 1 | train | 1 | `71adf4bdc64aba7a` | 10 s |
| 2 | eval | 0 | `d3efc98cdf282c41` | 5 s |
| 2 | eval | 1 | `2a951eb86bb6f824` | 5 s |

## 5. 成功标准

1. 任一模拟提交失败后，原 latest bundle 的字节哈希不变，仍为 step 2，scheduler 进度为 2；镜像不被误认成新提交。
2. 重试日志从 step 2 恢复；第 3 步使用 cosine 的正确下一步 lr（0.0225），而不是重用第 2 步 lr。
3. 首轮 seed 1 eval 未执行且不成功；第二轮只恢复未完成部分，最终 4/4 成功。
4. 两个 eval 的 step、Accuracy、Loss 与各自最终 best 对齐，Loss 选择的 checkpoint 无 `best_accuracy` 伪字段。
5. 小范围故障注入回归与 core 测试通过，检查 `.gitignore` 和变更范围；确认通过的 BUGS 条目才移出开放列表。

## 6. 边界

不承诺恢复与不中断训练数值相同（sampler 前缀重播、完整 RNG / 迭代位置未保存）。本次模拟验证错误处理，不证明已经找到历史 Windows `WinError 5` 的环境根因。不修旧权重、不更改历史 Study 结论，也不提交或推送。
