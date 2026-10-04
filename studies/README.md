# Study 导航

## 一、当前结果与阅读顺序

2026-10-04，历史 README 曲线的 32 条连续 80000-step 训练和 32 条独立评估已全部完成，固定原图估读门通过；本机现代 main 的 60-step 确定性八格对照也已通过。两个目标的配方、步数和证据范围分别记录。

1. [main_historical 专用入口](main_historical/README.md)：历史配方、准备与审计命令、正式及本机证据导航。
2. [历史完整曲线报告](main_historical/docs/STUDY_REPORT.md)：四 seed 的完整结果、预先门限、执行与失败证据。
3. [当前设备现代 main 结果](main_reproduction/docs/CURRENT_DEVICE_RESULT.md)：seed0、60-step、全精度统计 profile 和确定性控制下的跨实现对照。
4. [Study 使用指南](../docs/STUDY_GUIDE.md)：通用声明、调度、产物与报告约定。

## 二、全部 Study

| **Study** | **研究或验收范围** | **计划 / 报告** | **本机原始产物范围** |
|---|---|---|---|
| `_template` | 新 Study 的配置与报告起点；图片和 Run 行为占位 | [计划](_template/docs/PLAN.md) / [报告模板](_template/docs/STUDY_REPORT.md) | 尚未执行 |
| `mnist_train_size` | MNIST linear 的训练样本量扫描，train 与独立 eval | [计划](mnist_train_size/docs/PLAN.md) / [报告](mnist_train_size/docs/STUDY_REPORT.md) | 本机未取得旧 Run 目录 |
| `mnist_native_vs_hf` | 相同超参下 native 与 HF Trainer 的 CPU 对照 | [计划](mnist_native_vs_hf/docs/PLAN.md) / [报告](mnist_native_vs_hf/docs/STUDY_REPORT.md) | 本机未取得旧 Run 目录 |
| `cifar_grid` | CIFAR10、train_size=1024 的四模型小网格 | [计划](cifar_grid/docs/PLAN.md) / [报告](cifar_grid/docs/STUDY_REPORT.md) | 本机未取得旧 Run 目录 |
| `main_base` | main `--mode base` 接口的短探针与 60-step 可视化验收 | [计划](main_base/docs/PLAN.md) / [报告](main_base/docs/STUDY_REPORT.md) | 本机未取得旧 Run 目录 |
| `checkpoint_recovery` | CPU checkpoint 提交失败、重试与 sibling eval 依赖验收 | [计划](checkpoint_recovery/docs/PLAN.md) / [报告](checkpoint_recovery/docs/STUDY_REPORT.md) | 本机未取得旧 Run / 诊断目录 |
| `mnist_cnn_lr` | MNIST CNN 多 seed 学习率筛选，保留首轮故障 | [计划](mnist_cnn_lr/docs/PLAN.md) / [报告](mnist_cnn_lr/docs/STUDY_REPORT.md) | 本机未取得旧 Run 目录 |
| `mnist_cnn_budget` | MNIST CNN 600-step 三 seed 研究，保留保存失败与恢复 | [计划](mnist_cnn_budget/docs/PLAN.md) / [报告](mnist_cnn_budget/docs/STUDY_REPORT.md) | 本机未取得旧 Run 目录 |
| `mnist_cnn_budget_repeat` | seed2 无中断补测与后续文件占用诊断结论 | [计划](mnist_cnn_budget_repeat/docs/PLAN.md) / [报告](mnist_cnn_budget_repeat/docs/STUDY_REPORT.md) | 本机未取得旧 Run / 诊断目录 |
| `local_model_matrix` | MNIST / CIFAR10 × CNN / ResNet18 × 三 seed 的 600-step 研究 | [计划](local_model_matrix/docs/PLAN.md) / [报告](local_model_matrix/docs/STUDY_REPORT.md) | 本机未取得旧 Run 目录 |
| `support_data_smoke` | FashionMNIST / CIFAR100 / SVHN 的 30-step 数据支持验收 | [计划](support_data_smoke/docs/PLAN.md) / [报告](support_data_smoke/docs/STUDY_REPORT.md) | 本机未取得旧 Run 目录 |
| `support_model_smoke` | ResNet10 与两种 WideResNet 的 30-step 模型验收；保留 Accuracy 复算差异 | [计划](support_model_smoke/docs/PLAN.md) / [报告](support_model_smoke/docs/STUDY_REPORT.md) | 本机未取得旧 Run 目录 |
| `main_reproduction` | 固定现代 main 的 60-step 计算对照、历史来源审计与 200-step 前缀探针；新机结果见 CURRENT_DEVICE | [计划](main_reproduction/docs/PLAN.md) / [原报告](main_reproduction/docs/STUDY_REPORT.md) / [本机结果](main_reproduction/docs/CURRENT_DEVICE_RESULT.md) | 本机新对照在 `.tmp/`；旧设备原始目录未取得 |
| `main_historical` | 历史 PNG 候选配方的四 seed、80000-step 完整曲线复现；使用专用入口 | [入口](main_historical/README.md) / [计划](main_historical/docs/PLAN.md) / [完整报告](main_historical/docs/STUDY_REPORT.md) | 本机完整 Run / data / preflight / 诊断证据保留 |

“本机未取得”只说明 2026-10-04 的目录可用范围；旧报告及正式 JSON 保留自己的实际日期、来源和判定。不能用本机新结果补写旧实验的原始证据。

## 三、Git 与本机证据

声明、研究计划、报告、图、正式数字与来源清单随 Git 提供。`runs/`、`shared/`、`scripts/`、`index.json`、`process.json` 和 `.tmp/` 按仓库忽略规则留在本机；clone 只有报告，不自动拥有报告内引用的原始 checkpoint、日志和数据。

旧报告中的 Run 日志或 `.tmp` 链接可能在本机或 clone 中不可取得，应按报告记录的原始路径查找对应证据。正式失败快照、恢复记录、来源 SHA 和日期快照继续保留。运行与聚合入口的完整约定见 [LAYOUT](../docs/LAYOUT.md) 和 [STUDY_GUIDE](../docs/STUDY_GUIDE.md)。
