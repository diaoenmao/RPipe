# Bugs

> 这里只保留尚未解决的缺陷。编号只增不复用；修复并验证后移除条目，历史证据留在 Git 与对应 Study 报告。正常差异或决定不修的事项不列为开放缺陷。权威仍是 [CONCEPT.md](CONCEPT.md) → [LAYOUT.md](LAYOUT.md) → 代码。

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

B-007 已于 2026-10-02 按维护决定关闭，不再安排排查；关闭口径见 [BRAINSTORM.md](BRAINSTORM.md) §4。下次新增问题从 B-016 起编号。

B-013 的有界原子替换修复已通过定向、core、integration、真实 Windows 句柄验证及 seed 2 无中断补测；历史失败、原因边界与修复证据见 [补测报告](../studies/mnist_cnn_budget_repeat/docs/STUDY_REPORT.md) §3，360 开关诊断见 §5。原事件的外部占用进程未识别。

## B-014 错误的数据 / 模型配置会静默回退
- 状态：open
- 看见：未知数据名或 source 会回退 stub；未知模型产生 module=None / ready=True，native train / eval 可走占位路径；已知模型名配未知 source 会回落 custom_torch，同时仍报告原 source。正式实验应明确拒绝未注册配置，避免误把占位结果当成真实训练。
- 落点：[data/factory.py](../src/rpipe/structure/data/factory.py)、[model/factory.py](../src/rpipe/structure/model/factory.py)、[train](../src/rpipe/structure/algorithm/train/__init__.py)、[eval](../src/rpipe/structure/algorithm/eval/__init__.py)。
- 方案：明确显式 stub 的使用边界；正式配置按注册的 name / source 精确匹配，不静默改来源或进入占位计算。
- 验证：2026-10-03 本地最小探针已复现三种回退，证据 `.tmp/main-support-audit-20261003/evidence.json`。尚未修复；后续需覆盖错误名 / 错误来源拒绝、合法 stub 和原有正式配置。

## B-015 SVHN 的 scipy 运行依赖未声明
- 状态：open
- 看见：当前 torchvision SVHN 读取 .mat 文件时导入 scipy.io；其基础依赖不自动包含 scipy。旧 main 的 requirements 曾声明 scipy，当前项目依赖未声明，干净安装后不能保证 SVHN 可用。
- 落点：[pyproject.toml](../pyproject.toml)、[data/factory.py](../src/rpipe/structure/data/factory.py) 的 SVHN 构造路径。
- 方案：补齐与所声明 SVHN 能力一致的依赖，并验证真实数据加载、短训练和独立 eval。
- 验证：已核对本机 torchvision 的 SVHN 实现、包依赖元数据及 main 依赖清单；现有 fake dataset 单测不能覆盖此问题。尚未修复，未进行干净环境安装或真实 SVHN 验收。
