# Study Report: my_study

## 摘要

| **项目** | **内容** |
| --- | --- |
| 研究目标 | 填写本轮要回答的问题，链接 [PLAN.md](PLAN.md) |
| 记录日期与实验日期 | 分别填写，回顾旧实验时注明实际执行日期 |
| 主要结果 | 按 Experiment 汇总，注明指标、单位、统计口径和判定门限 |
| 当前状态 | 区分执行完成、质量通过、跳过、失败和未执行 |
| 验证边界与未完成项 | 填写不可比条件、缺失证据与下一步运行条件 |

数字来源为 Study 根的 `process.json`，图放在 `docs/figures/`，日志引用各 Run 的 `assets/logs/run.log`。正式结论应能回溯到配置、环境、seed、预算和实际证据。

## 1. 怎么跑的（§3）

```bash
python -m rpipe make studies/<name> --num-gpus 1 --init-gpu 0
python -m rpipe launch studies/<name> --num-gpus 1 --init-gpu 0
```

make 打印的 `pack N waits:` （贴实际输出）。整轮预估 ___，实际 ___。launch 不应再印 pack。error / resume：（有则记 `run_id`）。每条 Run 的预估和实际写在 Runs 表，不另开一份时长记录。

## 2. 结果与结论

（按 Experiment 聚合，不要扁平 run 列表。数字用 `process.json` 里该格子的 mean / std / min / max。）

## 3. 学习曲线

生成实际曲线后，将图保存在 `docs/figures/`，再按下面的示例添加链接；模板本身不包含实验图片。

```markdown
[打开 learning_curves.png](./figures/learning_curves.png)

[![learning curves](./figures/learning_curves.png)](./figures/learning_curves.png)
```

## 4. Run 明细

| **factors** | **seed** | **id** | **metrics** | **est** | **actual** | **log** |
|---------|------|----|---------|-----|--------|-----|
|  |  |  |  |  |  | 填写实际 Run 的日志链接 |

Run 表的日志链接相对于本报告，格式为 `../runs/<id>/assets/logs/run.log`。填写实际 ID 后使用 Markdown 链接。

## 5. 复现条件

记录源码提交、环境与依赖、数据来源、seed、实际预算，以及上方 make / launch 参数。已 succeeded 的 Run 默认跳过；独立重跑使用新 version，保留旧证据。
