# Studies

通用 Study 指南已迁入[根 README](../README.md#study-使用指南)。源码复制、项目与 Study 的 structure/flow 对齐见[项目接入](../README.md#在其他项目中使用-rpipe)；分支与可选 CI 按 [CI/CD](../docs/development/cicd.md) 执行。这里说明如何从宏观研究形成 Study，并保留本仓库的实例导航和目录迁移记录。

## 从宏观研究提取具体 Study

研究项目先在 `docs/research/` 维护宏观问题、研究方向、假设与方案，再从其中提取可独立执行和验收的 Study，不从现有脚本倒推研究问题，也不把整份宏观方案直接当作一个实验矩阵。一个宏观方向可以拆成多个 Study，每个只回答边界明确的问题。

1. 选择宏观方案中已经确认的一个问题，写明它与研究方向的对应关系；探索想法先确认，不直接成为执行合同。
2. 在目标 Study 的 `docs/PLAN.md` 提取对照、变量、固定条件、数据与模型来源、实际精度、seed、预算、指标、成功门和停止规则，并引用实际存在的宏观文档。
3. 将所需研究能力实现于 `src/<项目名>/` 的对应 structure/flow 层；Study 在同名目录导入和组装，不另写一套研究算法。structure 层名和 flow 六个阶段与模板一致，目录图见根 README。
4. 用声明表达该问题的具体因素和复测次数，先做小规模正确性与失败边界验证；并发与资源系数按该项目、设备和负载校准，不固定统一比例。
5. 用项目自己的实际入口执行，再将证据和结论写入 Study 报告，说明回答了哪个宏观问题、哪些假设成立或不成立及未验证范围。

宏观资料尚不存在时，先明确研究问题并补充真实方案，不编造来源链接。本仓库的 main_exp / main_probe 是模板的实例验证；它们的历史曲线与探针结果不是下游项目的新研究结论。

## 接续状态

**2026-10-12 目录重构边界：** `main_exp` / `main_probe` 的阶段包位于 `flow/`，专用建构与成对计算位于 `structure/`。库已加载 `flow/<phase>/` 并自动将两柱 Python 源码纳入 provenance；空目录和没有 `__init__.py` 的手动工具目录不自动执行。旧根目录阶段包须迁移。小规模流程验证与正式数值验收分开，原 version03 的历史通过记录不能替代本次重构的验收；正式复跑前使用新 version、重新 make 和执行计划规定的前置门。

2026-10-13 发布后本机复验：软件固定为 v0.3.0 / `804b3823`。main_probe 新 version 八格真实 Flow 与完整数值终验通过；main_exp 独立并发校准与新来源八格 600-step 门通过，完整 32 train / 32 eval 及固定曲线终验均已通过，外部中断与同源码接续分开记录。两项使用各自实测预算、原生 auto 两并发与 safety=0.5；最新证据见 [main_probe](main_probe/docs/v030_execution.md) / [main_exp](main_exp/docs/v030_execution.md)。相同 seed / 轴值不得重复产生同一 Run，重复实测使用新 version。严格数值对照显式约定 TF32 / 混合精度，不能只依靠 seed 和确定性开关。

## Study 与历史导航

以下导航保留目录调整时的记录；其中旧 version 和“待执行”描述属于对应日期，不代表最新验收。当前来源、数字与未完成项以各 Study 的报告和复跑记录为准。

现成 Study：

| **Study** | **研究或验收范围** | **计划 / 报告** | **本机原始产物** |
|---|---|---|---|
| `_template` | 新 Study 的配置与报告起点 | [计划](_template/docs/PLAN.md) / [报告模板](_template/docs/STUDY_REPORT.md) | 尚未执行 |
| `main_exp` | 历史 main `4ccb28d` 配方的四 seed、80000-step 曲线复现，对原图估读门终验 | [入口](main_exp/README.md) / [计划](main_exp/docs/PLAN.md) / [报告](main_exp/docs/STUDY_REPORT.md) | FP32 version03 的 32 train / 32 eval、完整性及固定曲线终验全部通过；串行历史速度校准已完成、容量调查继续，version02 失败保留 |
| `main_probe` | 固定 main `98648f3` 原代码与当前 RPipe 的 60-step 同设备数值探针 | [入口](main_probe/README.md) / [计划](main_probe/docs/PLAN.md) / [报告](main_probe/docs/STUDY_REPORT.md) | 历史 version03 八组合通过；2026-10-12 version05 仅 MNIST / linear、cnn 两组通过，其余六组未执行，子集未通过完整终验 |

2026-10-10 两个 Study 由 `main_historical`、`main_reproduction` 改名，按 [flow.md](../docs/code/flow.md) §14 整理为声明、recipe 与同名阶段扩展，旧结果与证据从当前树删除，见提交 [`18cd76c`](https://github.com/diaoenmao/RPipe/tree/18cd76c/studies)。

2026-10-10 只保留以上三个 Study。其余 Study（`mnist_train_size`、`mnist_native_vs_hf`、`cifar_grid`、`main_base`、`checkpoint_recovery`、`mnist_cnn_lr`、`mnist_cnn_budget`、`mnist_cnn_budget_repeat`、`local_model_matrix`、`support_data_smoke`、`support_model_smoke`）已删除，计划与报告见提交 [`71143ab`](https://github.com/diaoenmao/RPipe/tree/71143ab/studies)。测试要用的 `mnist_train_size`、`mnist_native_vs_hf`、`cifar_grid` 声明在 [`tests/_data/studies/`](../tests/_data/studies/)。

Study 保存声明、recipe、阶段包、专用 Algorithm 及研究证据，调度一律复用 `rpipe make / launch`。`main_exp` 的 `historical_4ccb28d` 由 `study.yaml` 的 `recipe` 注册；`main_probe` 由成对 Algorithm 计算、阶段扩展验收。库加载 `flow/<phase>/`，2026-10-12 已完成 main_probe 两组真实 60-step Run 与 main_exp 单组手动 200-step 前置对照；没有新完整矩阵验收。实测后 collect 重复入口清理通过针对性回归，后续真实复跑须新 version / make，按各自计划重新建立基线。文件说明、结果与边界见两个 Study 的 README。

声明、研究计划、报告、图、正式数字随 Git 提供。`runs/`、`shared/`、`scripts/`、`index.json`、`process.json`、`provenance.json` 和 `.tmp/` 按仓库忽略规则留在本机；clone 只有报告，不自动拥有报告内引用的原始 checkpoint、日志和数据。
