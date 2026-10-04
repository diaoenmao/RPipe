# RPipe

RPipe（Research Pipeline）是一个研究执行库，Python 包名为 `rpipe`。用 YAML 声明实验因素与 seed，展开并运行实验，再将配置、结果、日志、checkpoint 和跨 seed 汇总保存到同一 Study 中。

当前包版本为 **0.2.0（阶段版）**：提供可安装的 CLI、原生 PyTorch / Hugging Face Trainer 通路、运行与恢复保护，以及 main 基线的可复跑验证入口。阶段成果见 [v0.2.0 版本说明](docs/releases/v0.2.0.md)，旧脚本、配置和 checkpoint 的迁移见 [MAIN_MIGRATION.md](docs/MAIN_MIGRATION.md)。

## 旧 main → 当前实现

以下对比旧 main `98648f3` 与当前 0.2.0 实现。旧版已有 YAML、调度、聚合和 checkpoint，变化在于统一的接口与组织方式。

| **范围** | **旧 main** | **当前实现** |
| --- | --- | --- |
| 调用与编排 | 从 `src/` 运行独立脚本，共享全局 `cfg`，make 生成 Bash / wait 调度 | 安装 `rpipe` 包，通过 make / launch / process 等公共 CLI 执行 Study |
| 配置与身份 | `config.yml` 与 `control_name` 解析配置，seed 与配置拼成 tag | Study YAML 声明 axes / seeds，展开 Experiment × seed 的 Run，配置内容生成稳定 ID |
| 产物与读取 | 配置、实验、结果和图分布在 `output/` 各目录；checkpoint / best 使用旧分件格式 | 按 Study / Run 保存 config、result、tracker、log 与新 checkpoint；status / logs / report 读取同一套产物 |
| 曲线与验证 | 专用 process 脚本聚合固定 base 矩阵，图横轴按观测序号标作 Epoch | 新曲线记录实际 optimizer step / epoch，处理恢复后的回滚分支；正式 Study 保留来源、配方、门限与独立评测证据 |

dev 中的指标、来源匹配、依赖和 checkpoint 修复另见 [改前 / 改后摘要](docs/SUMMARY.md)。恢复保护不等于 RNG / sampler 逐位续训，旧 checkpoint 也不能直接作为新 Run resume。

## 本机实测（2026-10-04）

### 历史 main 长曲线

MNIST / CIFAR10 × linear / mlp / cnn / resnet18 × seeds 0–3：**32 条连续 80000-step 训练与 32 条自身 best 独立评测全部成功**。每条训练每 200 个 optimizer steps 完整评测 10000 张 test，共 400 个点。下图为本机实际训练曲线，横轴是 optimizer step，实线为四 seed mean，阴影为 **population std（ddof=0）**，保留全部点和瞬时波动。

![MNIST：本机四 seed test Accuracy，mean ± population std](asset/MNIST_Accuracy_mean.png)

![CIFAR10：本机四 seed test Accuracy，mean ± population std](asset/CIFAR10_Accuracy_mean.png)

两图使用训练中的完整 test history，独立 eval 的 best 没有替代曲线终点。八组通过事先固定的原图估读合同，终点最大绝对差小于 **0.207502 个百分点**。原 main 图缺少逐点日志、实际 seed 与原环境记录，因此结论限定于这套图像估读门，不表示全部原始点或 80000-step 权重逐位相同；本机阴影也不是原图阴影的验收结果。

原参考图另行保留：[MNIST 原图](studies/main_historical/docs/reference/MNIST_Accuracy_mean_4ccb28d.png)、[CIFAR10 原图](studies/main_historical/docs/reference/CIFAR10_Accuracy_mean_4ccb28d.png)，其来源与新图对应关系见 [图像发布记录](studies/main_historical/docs/reference/README.md)。完整数值、环境和科学边界见 [历史复现报告](studies/main_historical/docs/STUDY_REPORT.md)。

### 现代 main 的计算对照

固定 main `98648f3` 的 seed 0、60-step / eval30 八组合在本机受控条件下 **8/8 通过**：step30/60 共 16 段参数最大差 0，optimizer / RNG / scheduler 一致，完整 test 跨实现一致，独立 eval 与各自 best 指标一致。当前 Factory 读取归档原 Stats 对完整 train、batch250 重算的全精度 profile；两边统一确定性配置、`benchmark=false`。本次不覆盖无 profile 默认常量或原默认非确定性条件，也不代替上面的历史长曲线验收。详见 [本机现代对照](studies/main_reproduction/docs/CURRENT_DEVICE_RESULT.md)。

## 核心模型

```text
Study
  └─ Experiment（axes 的一个取值组合，不含 seed）
       └─ Run（Experiment × seed 的一次实测）

src/rpipe/
  ├─ structure/   # control、data、model、algorithm、system、artifact、make
  └─ flow/        # prepare → execute → collect → summarize → write → process
```

- **structure** 定义一次 Run 如何组成，并由 `make` 将 Study 声明展开成 config、index 和调度计划。`artifact/readout/` 再把落下的 index、result、日志和 `process.json` 读成表。
- **flow** 是跑 Study 的入口：对每个 Run 执行固定阶段链，最后按 Experiment 跨 seed 聚合。`status` / `logs` / `report` 从 cli 进来，转给 readout。
- **artifact** 是磁盘契约：Study 级 `docs/`、`shared/`、index、process，以及 Run 级 config、result、tracker、log、checkpoint。

## 文档顺序

Agent 的文档导航与通用工作约定见 [AGENTS.md](AGENTS.md)。

设计与代码冲突时，先更新文档，再更新实现。权威阅读顺序是：

1. [CONCEPT.md](docs/CONCEPT.md)：概念、职责与边界
2. [LAYOUT.md](docs/LAYOUT.md)：仓库和 Study 的目录契约
3. [CODE_STRUCTURE.md](docs/CODE_STRUCTURE.md)：模块边界与依赖方向
4. [structure.md](docs/code_structure/structure.md) / [flow.md](docs/code_structure/flow.md)：两柱的详细契约
5. [STUDY_GUIDE.md](docs/STUDY_GUIDE.md)：如何设计和运行一轮 Study
6. [TESTING.md](docs/TESTING.md)：测试策略、标签和结果持久化

已知缺陷与尚未兑现的设计见 [BUGS.md](docs/BUGS.md)，历史交接记录不覆盖上述文档。

开发过程与 main 结果复现进度见 [SUMMARY.md](docs/SUMMARY.md)，阶段性方案见 [BRAINSTORM.md](docs/BRAINSTORM.md)。

正式研究入口见 [Study 导航](studies/README.md)。旧 main 的命令、配置与 checkpoint 迁移见 [MAIN_MIGRATION.md](docs/MAIN_MIGRATION.md)，本轮仓库整理及交付验收见 [REPOSITORY_CLEANUP.md](docs/REPOSITORY_CLEANUP.md)。

## 安装

要求 Python 3.10 或更高版本。

```bash
python -m pip install -e ".[dev]"
```

使用 Hugging Face Trainer 通路时安装对应可选依赖：

```bash
python -m pip install -e ".[dev,hf]"
```

依赖声明以 `pyproject.toml` 为准；`requirements.txt` 是兼容入口，不是旧科学环境的版本锁定文件。

复跑两种 main 对照还需要归档 Logger / Metric 的依赖。新环境先让 pip 解析 TensorBoard / evaluate 的传递依赖，再按实测版本补充隔离 runtime：

```bash
python -m pip install tensorboard==2.21.0 evaluate==0.4.6
python -m pip install --target .tmp/runtime --no-deps -r studies/main_historical/runtime-requirements.txt
```

第二条命令只补固定版本；`--no-deps` 以第一步已解析的依赖为基础。本机安装验收复用了已有科学依赖，全新环境的完整安装仍待实测；本次科学环境版本见 [ENVIRONMENT.json](studies/main_historical/docs/ENVIRONMENT.json)。

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

# 查看完成状态与事件行；从已有聚合生成 docs/NUMBERS.md
python -m rpipe status studies/<name>
python -m rpipe logs studies/<name>
python -m rpipe report studies/<name>
```

需要最短的顺序执行路径时：

```bash
python -m rpipe run studies/<name>
```

完整字段、并行排班、失败续跑和报告约定见 [STUDY_GUIDE.md](docs/STUDY_GUIDE.md)。

### 复跑 main 验证

历史配方使用 Study 专用 Registry，通用 `rpipe run/launch` 不自动注册。先按 [历史 Study 入口](studies/main_historical/README.md) 准备依赖和数据，在独立 clone 或工作树中依次执行：

```bash
python -B studies/main_historical/prepare_data.py
python -B studies/main_historical/run.py make
python -B studies/main_historical/run.py preflight
python -B studies/main_historical/verify_preflight.py
python -B studies/main_historical/run.py launch
python -B studies/main_historical/compare.py
```

现代 60-step 对照使用独立的新工作区：

```bash
python -B studies/main_reproduction/current_device.py prepare --workspace .tmp/main-current-device-new-run
python -B studies/main_reproduction/current_device.py run --workspace .tmp/main-current-device-new-run --device cuda
```

其准备条件与门限见 [现代对照计划](studies/main_reproduction/docs/CURRENT_DEVICE_PLAN.md)。已有运行目录和原失败记录应保留；数据、checkpoint、tracker、日志与 `.tmp/` 不随 Git clone 提供，正式报告保留来源与结果。

## Study 中什么进 Git

| 内容 | 是否入库 | 原因 |
|------|----------|------|
| `study.yaml`、`experiment_config.yaml` | 是 | 可复现实验声明 |
| `docs/PLAN.md`、`docs/STUDY_REPORT.md`、报告图片 | 是 | 人写的研究计划与结论 |
| `docs/NUMBERS.md` | 可以 | `rpipe report` 从 `process.json` 生成的数字表，不是结论 |
| `runs/`、`shared/`、`scripts/` | 否 | 可重新生成或体积较大的运行产物 |
| `index.json`、`process.json`、`activity.json` | 否 | make / process 重建；`activity.json` 只在 make 进行中存在 |
| `.tmp/` | 否 | 本地测试、缓存和临时验证 |
| `docs/*.tmp`、`docs/*.claim` | 否 | 本地事务暂存和执行占用标记；正式恢复/失败快照单独保留 |

不要在仓库根重新创建旧式 `data/`、`output/`；数据与产物都归属具体 Study。

`studies/` 保留正式研究和可复用验收案例。一次性的文件占用、环境开关等排错放 `.tmp/diagnostics/`，结论合并到相关 Study 报告；本地证据链接不会随 Git clone 提供。

## 开发与 CI

```bash
# 快速门：unit + p1 + c1，不跑 slow / external / gpu
python tests/run.py --fast

# 当前 PR 门：unit 的 c1 与 c2，不跑 slow / external / gpu
python tests/run.py --core

# 本地执行链：符合 c1/c2 的 integration/e2e，不跑 slow / external / gpu
python tests/run.py --all --cost-class c1 --cost-class c2 -- -m "(integration or e2e) and not external and not gpu and not slow"

# 显式运行所有已收集测试
python tests/run.py --all
```

pytest 的 base temp、cache 和测试结果都写入 `.tmp/`。本轮本机 CPU 门为 **288 passed / 3 deselected**；wheel / sdist 安装后各 9 条 CLI 命令通过，各完成 4 个 Toy/Stub Run。安装验收复用了本机已有科学依赖，Toy/Stub 验证安装与产物合同，不能代替全新依赖环境、GPU 或可选 HF 验收。详见 [交付验收记录](docs/REPOSITORY_CLEANUP_RESULT.json)。

GitHub Actions 配置包括：

- `Unit Tests`：Ubuntu/Windows × Python 3.10/3.13，安装 `.[dev]` 后执行 core 和本地 CPU 流程门；
- `Package Check`：构建 wheel / sdist，在源码目录之外安装两种分发包，验证模块入口、console script 与离线 Toy/Stub Study 的 make / launch / process / readout。

新增 CI 配置的实际通过状态以提交后的 Actions 为准；安装验收不等于所有 GPU 或可选 HF 场景通过。

## 当前实现范围

- 数据：stub，以及 torchvision 的 MNIST、FashionMNIST、CIFAR10、CIFAR100、SVHN；
- 模型：`custom_torch` 的 linear、MLP、CNN、ResNet、WideResNet；
- 算法：原生 PyTorch train/eval，以及 `transformers_trainer`；
- 运行：顺序执行、按 GPU/wait 组并行、checkpoint/resume、Run 日志与 Study 聚合。

文档中列出的其他下游生态是扩展边界，不代表已经实现；以 Registry 和 [BUGS.md](docs/BUGS.md) 为准。

上述清单表示已有实现，不代表全部模型与数据组合已完成真实运行验收。六个指定组合的 30-step 验收见 [数据报告](studies/support_data_smoke/docs/STUDY_REPORT.md) 与 [模型报告](studies/support_model_smoke/docs/STUDY_REPORT.md)，模型报告保留 ResNet10 的 Accuracy 复算差异；未解决缺陷见 BUGS。

## Acknowledgements

[Federated Learning Platform](https://github.com/IBM/federated-learning-lib),
[EasyFL](https://github.com/EasyFL-AI/EasyFL/),
[FedLab](https://github.com/SMILELab-FL/FedLab),
[Flower](https://flower.dev/),
[NIID-Bench](https://github.com/Xtra-Computing/NIID-Bench),
[FedTorch](https://github.com/OPTML-Group/FedTorch)
