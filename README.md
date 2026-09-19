# RPipe

RPipe（Research Pipeline）是一个**文档先行、可重复、可编排、可序列化**的研究执行底座，Python 包名为 `rpipe`。

它负责把一轮研究从声明展开为可执行的 Run，并把配置、结果、日志、曲线和聚合摘要落到稳定的 Study 目录中。它不负责研究选题、UI 或自动决定下一步实验。

## 核心模型

```text
Study
  └─ Experiment（axes 的一个取值组合，不含 seed）
       └─ Run（Experiment × seed 的一次实测）

src/rpipe/
  ├─ structure/   # control、data、model、algorithm、system、artifact、make
  └─ flow/        # prepare → execute → collect → summarize → write → process
```

- **structure** 定义一次 Run 如何组成，并由 `make` 将 Study 声明展开成 config、index 和调度计划。
- **flow** 服务整个 Study，对每个 Run 执行固定阶段链，最后按 Experiment 跨 seed 聚合。
- **artifact** 是磁盘契约：Study 级 `docs/`、`shared/`、index、process，以及 Run 级 config、result、tracker、log、checkpoint。

## 文档顺序

设计与代码冲突时，先更新文档，再更新实现。权威阅读顺序是：

1. [CONCEPT.md](docs/CONCEPT.md)：概念、职责与边界
2. [LAYOUT.md](docs/LAYOUT.md)：仓库和 Study 的目录契约
3. [CODE_STRUCTURE.md](docs/CODE_STRUCTURE.md)：模块边界与依赖方向
4. [structure.md](docs/code_structure/structure.md) / [flow.md](docs/code_structure/flow.md)：两柱的详细契约
5. [STUDY_GUIDE.md](docs/STUDY_GUIDE.md)：如何设计和运行一轮 Study
6. [TESTING.md](docs/TESTING.md)：测试策略、标签和结果持久化

已知缺陷与尚未兑现的设计见 [BUGS.md](docs/BUGS.md)，历史交接记录不覆盖上述文档。

## 安装

要求 Python 3.10 或更高版本。

```bash
python -m pip install -e ".[dev]"
```

使用 Hugging Face Trainer 通路时安装对应可选依赖：

```bash
python -m pip install -e ".[dev,hf]"
```

依赖的唯一事实源是 `pyproject.toml`；`requirements.txt` 仅保留为兼容入口。

## 快速开始

先复制 `studies/_template/`，完成 `docs/PLAN.md`、`study.yaml` 和 `experiment_config.yaml`，再运行：

```bash
# 只展开 Experiment × seed，检查生成的 config 与 index
python -m rpipe run studies/<name> --skip-launch

# 生成调度计划；默认按任务类型和显存估计组成 wait 组
python -m rpipe make studies/<name> --num-gpus 1

# 复用 scripts/jobs.json，运行未完成的 Run，最后执行 Study 聚合
python -m rpipe launch studies/<name> --num-gpus 1

# 只重建 Study 级聚合与曲线
python -m rpipe process studies/<name>
```

需要最短的顺序执行路径时：

```bash
python -m rpipe run studies/<name>
```

完整字段、并行排班、失败续跑和报告约定见 [STUDY_GUIDE.md](docs/STUDY_GUIDE.md)。

## Study 中什么进 Git

| 内容 | 是否入库 | 原因 |
|------|----------|------|
| `study.yaml`、`experiment_config.yaml` | 是 | 可复现实验声明 |
| `docs/PLAN.md`、`docs/STUDY_REPORT.md`、报告图片 | 是 | 人写的研究计划与结论 |
| `runs/`、`shared/`、`scripts/` | 否 | 可重新生成或体积较大的运行产物 |
| `index.json`、`process.json` | 否 | 由 make / process 重建 |
| `.tmp/` | 否 | 本地测试、缓存和临时验证 |

不要在仓库根重新创建旧式 `data/`、`output/`；数据与产物都归属具体 Study。

## 开发与 CI

```bash
# 快速门：unit + p1，不跑 slow / external
python tests/run.py --fast

# 当前 PR 门：全部 unit，不跑 slow / external
python tests/run.py --core

# 显式运行所有已收集测试
python tests/run.py --all
```

pytest 的 base temp、cache 和测试结果都写入 `.tmp/`。GitHub Actions 当前有两条只读门禁：

- `Unit Tests`：安装 `.[dev]` 后执行 core 测试；
- `Package Check`：构建 wheel / sdist，并验证 wheel 可以独立导入。

## 当前实现范围

- 数据：stub，以及 torchvision 的 MNIST、FashionMNIST、CIFAR10、CIFAR100、SVHN；
- 模型：`custom_torch` 的 linear、MLP、CNN、ResNet、WideResNet；
- 算法：原生 PyTorch train/eval，以及 `transformers_trainer`；
- 运行：顺序执行、按 GPU/wait 组并行、checkpoint/resume、Run 日志与 Study 聚合。

文档中列出的其他下游生态是扩展边界，不代表已经实现；以 Registry 和 [BUGS.md](docs/BUGS.md) 为准。

## Acknowledgements

[Federated Learning Platform](https://github.com/IBM/federated-learning-lib),
[EasyFL](https://github.com/EasyFL-AI/EasyFL/),
[FedLab](https://github.com/SMILELab-FL/FedLab),
[Flower](https://flower.dev/),
[NIID-Bench](https://github.com/Xtra-Computing/NIID-Bench),
[FedTorch](https://github.com/OPTML-Group/FedTorch)
