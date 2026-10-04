# 一、摘要

RPipe 已从旧 main 的独立训练脚本演进为可安装的研究执行库。本节按内容归纳当前 dev 与本机新增成果的累计变化；具体实施日期、失败和阶段边界保留在第二节。

## （一）改前与改后对比

### 1. 结构与公共接口

本表以旧 main `98648f3` 对比当前 dev。旧版已经有 YAML、命令参数、调度、聚合和 checkpoint；变化在于组织方式、公共接口与产物合同。

| **范围** | **改前** | **改后** | **依据** |
| --- | --- | --- | --- |
| 调用方式 | 从 `src/` 运行 `train_model.py`、`test_model.py`、`make.py` 等独立脚本，共享全局 `cfg` | 安装 `rpipe` 包，以统一 CLI 执行 make / launch / process，源码集中到 `src/rpipe/` | [代码结构](CODE_STRUCTURE.md) |
| 配置与编排 | `config.yml` 配合 `control_name` 解析；按 seed 与配置拼 tag，make 生成 Bash / wait 调度 | Study YAML 声明 axes / seeds，展开 Experiment × seed 的 Run；配置内容生成 Run ID，launch 执行固定阶段链 | [概念](CONCEPT.md)、[Study 指南](STUDY_GUIDE.md) |
| 运行产物 | 配置、脚本、实验、结果和图分别落到 `output/config`、`script`、`exp/<tag>`、`result`、`vis`；checkpoint / best 为旧分件格式 | 按 Study / Run 统一落盘 config、result、tracker、log 和新 checkpoint；index / process 支持 status / logs / report | [目录约定](LAYOUT.md) |

### 2. 计算与恢复修复

本表的“改前”是 dev 中对应问题修复前的行为；这些问题不能统称为旧 main 的缺陷。

| **范围** | **改前** | **改后** | **依据** |
| --- | --- | --- | --- |
| 来源与依赖（B-014 / B-015） | 未知 name / source 可能落入默认或占位路径；缺正式输入仍可能产出结果；SVHN 所需 scipy 未列入基础依赖 | Factory 严格匹配，未知/空 source 报错，省略时保留正常默认；占位必须显式 stub，缺输入失败落盘；声明并核验 scipy | 第二节 §1、[数据验收](../studies/support_data_smoke/docs/STUDY_REPORT.md) |
| train 均值（B-016） | step 模式跨评测段累计，反例最后一段应约10却报告约5 | 每个评测周期收口并重置 train 段，同一反例报告约10；空段不覆盖最后有效均值 | 第二节 §4 |
| topk 计数（B-017） | topk=2 将3个样本的候选展平为6个索引，触发形状/计数错误 | 保留候选轴，每个样本命中任一候选只计一次；修复既有函数，未新增 CLI topk 配置 | 第二节 §5 |
| 零评测预算（B-018） | `eval_num_steps=0` 仍读取并评测第一批 | 读取数据前报错；正数限批、缺省/负数完整 test 的语义保留 | 第二节 §7 |
| 曲线进度与回滚 | 横轴按 history 记录序号；记录频率变化不能直接解释为训练步数，恢复后的回滚分支未单独排除 | 按 optimizer_step / epoch 对齐；保留原始日志，聚合时排除回滚分支、处理重复坐标并记录逐点 n | [曲线修复记录](BRAINSTORM.md#4-已经做的) |
| Windows checkpoint 保存（B-013） | 文件占用时保存出现 WinError5，训练中断；原占用者未查明 | 原子替换对特定 Windows 错误有限重试，持续失败保留旧档并报错；真实句柄探针与训练补测通过，原占用者仍未查明 | [保存补测](../studies/mnist_cnn_budget_repeat/docs/STUDY_REPORT.md) |

### 3. 本机复现与交付整理

本表对比本次接续工作前后的成果；新增验收与新增 CI 配置分别记录。

| **范围** | **改前** | **改后** | **依据** |
| --- | --- | --- | --- |
| 现代 main 本机对照 | 已有旧设备受控60-step结果，当前 cu130 设备尚缺同条件执行与存档核验 | 本机8组合、60-step受控对照通过，16段参数最大差0，并完成独立 CPU 重载；原默认条件失败仍保留 | [本机结果](../studies/main_reproduction/docs/CURRENT_DEVICE_RESULT.md) |
| 历史曲线复现 | 只完成 seed0、200-step前缀探针；接入主要在 `.tmp/`，四 seed 完整曲线未运行 | 新增正式历史 Study 与复跑入口；32条80000-step训练、32条自身 best 评测成功，固定图像估读门通过 | [历史复现](../studies/main_historical/README.md) |
| 统一测试入口 | 子进程隐式继承环境与输出；当前沙箱曾有3项 Kornia 导入失败，正常本机原入口为286项通过 | 显式传递环境和 stdout/stderr，补2个真实子进程回归；当前统一 CPU 门288项通过，原失败记录保留 | [测试入口](../tests/README.md) |
| 包与 CI | Ubuntu / Python3.10 只跑 core；包门构建后仅对 wheel 做无依赖安装和轻量 import | Linux / Windows、Python3.10 / 3.13 的 CPU及本地 integration 全部通过；两种 OS 全新环境中的 wheel / sdist 安装和公开 CLI 全部通过 | [远端测试](https://github.com/diaoenmao/RPipe/actions/runs/37210453756)、[远端包验收](https://github.com/diaoenmao/RPipe/actions/runs/37210453815) |
| 导航与清理 | Study 入口分散；SUMMARY 先按日期叙述；有13个冗余占位和未忽略的本地事务标记，模板直接链接未生成图片 | 增加14项 Study 导航和摘要对照表；移除13个冗余占位、保留7个空目录占位；忽略并保留事务标记，模板图片改为代码示例 | [Study 导航](../studies/README.md)、[整理报告](REPOSITORY_CLEANUP.md) |

## （二）已经验证到哪一步

| **验证对象** | **最新实际结果** |
| --- | --- |
| CPU unit / integration | 288 passed / 3 deselected；c1/c2，排除 external / gpu / slow，包含两个新测试入口回归 |
| 安装后的 wheel / sdist | 各9条公开 CLI 命令退出0，各4个 Toy/Stub Run、2个 Experiment完整；配置不变，重复 launch 跳过成功任务 |
| 远端发布准备 CI | PR #12 的四个 OS / Python 测试矩阵、两个 OS 包矩阵及 Unit tests / Build package 聚合门全部成功，绑定提交 `216e2b8`；实际运行链接见上表 |
| 现代 main 计算对照 | 8组合、seed0、60-step；step30/60共16段参数最大差0，optimizer / RNG / scheduler与完整 test一致，独立 CPU 重载通过 |
| 历史 main 完整曲线 | 32条四 seed 连续80000-step训练、32条自身 best 独立评测成功；8组固定图像估读门通过，最大终点差0.207502个百分点 |
| 整理后的证据保全 | 772个唯一科学证据文件哈希一致，原始失败与旧快照保留；正式相对链接、代码语法和 CI YAML 检查通过 |

现代对照使用原 Stats 的全精度 profile 与统一确定性条件；原默认条件未通过的结果仍保留。历史图缺少原始逐点指标，长曲线结论限定于预先固定的图像估读门，前缀逐位对齐不扩称为全80000步权重逐位一致。

本机包验收复用了已有科学依赖，Toy/Stub 只验证安装与产物合同。发布准备提交 `216e2b8` 的新跨平台 CI 已实际通过；后续提交的状态以其对应 Actions 为准。整理阶段的固定验收快照见 [机器可读结果](REPOSITORY_CLEANUP_RESULT.json)，发布验证与环境修复见 [v0.3.0 说明](releases/v0.3.0.md)。

# 二、开发记录

按日期保留阶段事实，最新日期在前。较早记录中的“尚未完成”只描述当时范围；当前累计状态以上方摘要为准。

## （一）2026-10-04

最新远端 dev 基线为 `8bccbac`。本机完成 [历史 main 曲线复现](../studies/main_historical/docs/STUDY_REPORT.md)：32 条四 seed 连续80000-step训练与32条自身best独立评测全部成功，每条400个完整test点。八组的终点、末50点均值、七锚点、末段波动及两数据集模型排序通过预先冻结的原图估读合同，最大终点差0.207502个百分点。

[现代 main 本机对照](../studies/main_reproduction/docs/CURRENT_DEVICE_RESULT.md) 在原Stats全精度profile与统一确定性配置下60-step八格通过，step30/60共16段参数最大差0；本机历史600-step八格前缀桥及两类独立CPU存档核验也通过。原默认条件的失败、旧设备结果和原图缺失运行记录的边界保留；不把前缀对齐扩称为80k权重逐位相同。

本轮另整理全仓库导航、正式 Study 入口、冗余占位和事务标记，补充安装后 CLI 与 CPU 验收。完整变更与实际验证见 [REPOSITORY_CLEANUP.md](REPOSITORY_CLEANUP.md)。

以下2026-10-03记录保留当时“仅探针、未执行长矩阵”的真实范围，不再代表最新完成状态。

### 1. 当前设备的完整验收

| **验收对象** | **实际完成** | **证据边界** |
|---|---|---|
| 历史候选同机计算桥 | 8组合、seed0、step200/400/600，24段参数/buffer逐位一致，SGD/scheduler/RNG/实际输入一致 | 600-step前缀，不能代替全80k原码/current权重证明 |
| 历史四seed长曲线 | 32train × 80000step，每条400次完整test；32own-best eval；完整固定门通过 | PNG估读，未知原seed与环境，未验收原图阴影std及全400点逐位差 |
| 现代main受控计算对照 | 8格seed0、60step/eval30、16段参数maxdiff0，独立完整test一致，CPU重载通过 | 原Stats B250全精度profile、deterministic条件；未覆盖原默认benchmark=true/无profile |
| 本轮整理与迁移准备 | 首页与14个Study导航、专用Registry入口、旧接口迁移说明、CPU和安装交付门 | 具体执行结果及远端待执行项见整理报告；未实际合并main |

历史矩阵主控制器于2026-10-04 09:12:29 +08:00结束；完整曲线及独立审计于当日晚间收口。[最终结果](../studies/main_historical/docs/FINAL_RESULT.json)与[执行审计](../studies/main_historical/docs/FINAL_EXECUTION_AUDIT.json)保留真实来源。提前排班去重成功、无训练resume；CIFAR linear评测四worker成功，但记录控制器因WinError5退出1，仅恢复记录，没有重跑或改写原退出码。

旧进度和前三组JSON快照保留原样。[Study导航](../studies/README.md)区分正式可交付成果与本机runs/shared/.tmp证据，清理没有删除数据、checkpoint或失败记录。旧main无扩展checkpoint与新的.pt/bundle、模型key和Stats对象存在实际格式差异，普通新CLI不直接resume旧任务。

## （二）2026-10-03

### 阶段摘要

当天工作由 Cursor 的实现与 Codex 的接续复核组成，重点是修复基础问题、核对 main 的计算结果，并完成限定范围的历史超参探针。

#### 1. 推进过程

1. 修复 B-014～B-018 的配置入口、依赖、训练均值、topk 计数和零评测预算问题，调整曲线的真实进度与回滚分支处理
2. 补验 FashionMNIST、CIFAR100、SVHN，以及 ResNet10 和两种 WideResNet 的真实数据训练与评测
3. 对照当前 main 的 60-step 配方，核对数据、初始化、采样、参数、scheduler、best checkpoint 和独立 eval
4. 追溯 README 历史图的候选旧配方，保留旧 BN、CPU 增强、归一化和裁剪，仅执行 seed 0 的 200-step/eval200 前缀探针

#### 2. 验证结果

- 本地回归：**286 passed / 3 deselected**，接续复核确认 91 个生产源文件与最新实验快照一致
- 支持范围短验收：**12/12 执行成功**，严格 Accuracy 复算 5/6 一致。ResNet10 相差一个正确样本，保留未通过判定
- 当前 main 对照：默认条件严格数值门 **4/8**，原代码重复也有分歧。统一 CUDA 确定性条件后 **8/8** 对齐
- 历史超参探针：计算对照及存档复核均 **8/8**，参数差值为 0，内部耗时约 **149 秒**。[探针报告](../studies/main_reproduction/docs/HISTORICAL_PREFIX_PROBE.md)保留完整原值与边界

#### 3. 当前状态

- 没有开放缺陷，本轮探针未发现新生产 bug。维护关闭与根因修复的区别保留在详细记录中
- 本轮到探针结束，**完整历史曲线尚未复现**。没有执行 80000-step/4-seed 长实验，也未重建旧依赖环境
- Git 交付从 `feat/main-base` 集成至 `dev`，远程电脑使用 `dev` 的实际提交作为测试基线。交付范围和发布前回归见详细记录 §12；临时脚本、缓存和原始验证输出仍放在 `.tmp/`，不随 Git 同步

### 1. 本轮收尾与交接

本轮接续复核、历史超参前缀探针和bug列表整理已收口。用户限定的200-step/eval200探针8/8通过；固定当前main的确定性对照8/8，原默认严格门仍为4/8。README历史图未复现，不把本轮探针完成写成完整结果复现。

| **交付** | **内容与证据** |
|---|---|
| B-014 / B-015 | Data/Model Factory严格匹配name/source，拒绝隐式stub；补齐SVHN的scipy依赖及验证 |
| B-016 / B-017 / B-018 | 按评测段保存train均值；topk按样本计数；零评测预算在读取数据前报错 |
| 用户选定3.1 | 曲线按optimizer step / epoch对齐，排除回滚分支，按共同坐标聚合并记录逐点n；3.2/3.3不做 |
| 最新本地回归 | unit + integration、c1/c2：286 passed / 3 deselected，排除external/slow/gpu；[Codex 接续测试报告](../.tmp/test-results/20261003T110214Z_95a976/report.md) |
| 当前源码完整对照 | 8 train + 8 eval全部成功；确定性门8/8，step30/60参数与test指标差值0，独立eval/best一致；[逐格证据](../studies/main_reproduction/docs/DETERMINISTIC_COMPARISON_AFTER_B018.json) |
| 历史超参前缀探针 | seed0、200-step/eval200、scheduler T_max80000；原代码/当前链8/8，存档独立复核8/8，参数差值0，内部148.740s；[探针报告](../studies/main_reproduction/docs/HISTORICAL_PREFIX_PROBE.md) |
| 当前bug清单 | 没有开放缺陷；移除“开放”标题下的已关闭说明，本探针未发现新生产bug，不把维护关闭说成根因修复 |
| 研究报告 | [main对照](../studies/main_reproduction/docs/STUDY_REPORT.md)、[数据支持验收](../studies/support_data_smoke/docs/STUDY_REPORT.md)、[模型支持验收](../studies/support_model_smoke/docs/STUDY_REPORT.md)；各报告保留未通过门限和证据边界 |

探针收口时的交接位置：分支`feat/main-base`，HEAD `d938874`，当时后续改动尚未提交或推送，之后的 Git 交付见 §12。正式记录在本文件、[BRAINSTORM](BRAINSTORM.md)与各Study的docs；原始日志、探针和固定源码副本在`.tmp/main-reproduction-20261003/`等临时目录，保留以供核对，不作为已提交成果。当前91个生产源文件与B-018快照一致；接续阶段新增的训练仅为用户明确要求的200-step前缀探针，没有长实验或重复相同回归。

当前决定（2026-10-03）：用户选择README历史图对应的main超参路线，随后明确本轮只做到探针，并要求处理实际开放bug。200-step/eval200、seed0的8格前缀对照与计时已完成并收口；不跑80000-step/4-seed长实验。原默认4/8、确定性8/8及历史图未复现结论保留；原历史运行记录仍缺失。详细范围见[本轮计划](../studies/main_reproduction/docs/PLAN.md#用户确认的历史超参前缀探针2026-10-03执行前)。

### 2. 目标与对照

最终目标是复现 main 的研究结果。当前本地 main / origin/main 均为 `98648f3a5c7db7dccf3ca806410d5b6fdee9484c`，与已有 main_base PLAN 固定的对照一致。新架构的 Run 成功、更多训练步数或更高精度都不能替代对照验证。

main 的 base 矩阵是 MNIST / CIFAR10 × linear / mlp / cnn / resnet18，seed 0；batch 250、test batch 1000、60 optimizer steps、完整 test 每 30 steps；SGD lr=0.1 / momentum=0.9 / Nesterov / weight decay=0.0005、cosine T_max=60；按 test Loss 选择 best，独立 test 加载 best。

main Git 中只有 README 的曲线图，没有对应的可读原始结果、权重和依赖锁定。先在本机运行固定提交建立可核对的新对照，逐项注明与历史图结果的证据边界；不从图片猜出精确数值。

### 3. 已完成阶段

| **阶段** | **已验证的内容** | **尚不能证明的内容** |
|---|---|---|
| 结构迁移与执行链 | Study / Experiment / Run、make / launch / process、日志、结果及 checkpoint 路径 | 原 main 数值结果已复现 |
| checkpoint 修复 | B-009–B-013 的恢复、原子写入与 sibling eval 保护；真实 Windows 保存问题补测 | RNG / sampler 逐位恢复 |
| 真实训练验证 | main_base 16/16，600-step 多模型矩阵 24/24；报告保留各自配方 | main_base 每 5 步 eval 与 main 每 30 步 eval 等价 |
| 配置入口修复 | B-014 / B-015：拒绝未知来源、显式 stub、SVHN scipy 依赖与真实数据短验收 | 所有数据 / 模型组合与全新环境安装 |
| 曲线进度修复 | 新记录按 optimizer_step / epoch 对齐，回滚分支保留原始日志、有效轨迹单独聚合，逐点 n；本地门 279 passed / 3 deselected | 历史没有进度字段的曲线可以自动恢复真实坐标 |
| 训练均值修复 B-016 | native step 模式按评测段保存训练均值，空段不覆盖最后有效值；合并回归 282 passed / 3 deselected | 完整 60-step 真实数据结果与 main 等价 |
| 完整 main 源码对照 | 真实数据60-step / eval30，原默认CUDA配方两套各16/16执行成功、严格数值门4/8；确定性控制两套各16/16成功、数值门8/8，step30/60参数与test指标差值均为0 | 原默认benchmark配方逐位一致；历史README图已复现 |
| 指标修复B-017 | topk参数按样本计数；本地门283 passed / 3 deselected；修复后CPU探针8/8、当前16次真实数据CUDA执行与原main确定性存档数值对照8/8 | 新增topk配置入口；原默认非确定性/历史图结果的额外复现 |
| 评测预算修复B-018 | eval_num_steps=0在读取数据前报错；本地门286 passed / 3 deselected，main原代码CPU探针8/8；当前源码完整16次CUDA执行成功，确定性数值门8/8 | 原默认严格门4/8的额外验收；历史图复现 |

详细记录与原始验证入口见 [BRAINSTORM](BRAINSTORM.md)、各 Study 报告；未解决缺陷见 [BUGS](BUGS.md)。本文件记录开发事实，不取代设计文档。

### 4. main 配方与行为审计

- 工作区 HEAD `d938874`，保留已有未提交修复与两份支持范围验收 Study。
- 已核对 main 固定提交、base 矩阵、数据增强、模型结构 / 初始化、采样器及优化器；原代码只读导出在 `.tmp/main-reproduction-20261003/reference/`，不切换或覆盖工作区。
- B-016 已确认并修复：固定权重、前两步 Loss≈0 / 后两步 Loss≈10、eval_period=2，原实现最后 train_loss≈5，main 的最后段应≈10。修复前 step 反例失败、epoch 对照通过（1 failed / 1 passed），证据 `.tmp/main-red-38ebbacd7ca6403f87b14f187885a238/`；修复后两种模式通过。native step 每次周期 test 前保存并重置 train 段，收尾跳过空段；epoch 模式与参数更新顺序保留。
- 固定 main 原代码与新实现的 CPU 探针 8/8 通过：MNIST / CIFAR10 × linear / mlp / cnn / resnet18，合成固定输入、4 steps / eval2。初始化和采样顺序一致，step2 / 4 的参数、train / test Loss、Accuracy 差值均为 0，scheduler 状态一致。脚本与结果见 `.tmp/main-reproduction-20261003/cpu_parity.py`、`cpu-parity.json`、`cpu-parity.log`。这不是完整真实数据训练验收。
- 原代码导出未修改；缺失依赖 evaluate / datasets / multiprocess / xxhash 仅安装到 `.tmp/main-reproduction-20261003/deps/`。Windows 启动时先导入 numpy 再导入 torch，以适配本机运行环境；未重建 main 的历史依赖版本。
- 定向 algorithm / process 回归 110 passed；最终 unit + integration c1 / c2 本地门 282 passed / 3 deselected（排除 external / slow / gpu），[验证报告](../.tmp/test-results/20261002T213434Z_f4a444/report.md)，临时工作目录 `.tmp/main-final-715e6cdaac9c49faa00315c0761604d0/`。
- 完整 main 对照已进入 [main_reproduction 计划](../studies/main_reproduction/docs/PLAN.md)：复制已缓存的真实原始数据，32 个文件副本 SHA-256 一致；旧 main_base CIFAR10 目录读取被拒绝，改用已有 support_model_smoke 的可读缓存副本，不改目录权限。原 main 的完整 train Stats 以 batch250、dim1 计算，完整精度写入新 Study；两套完整 train / test 像素和标签逐项一致。
- 真实数据初始化预检8/8：当前Factory重建的初始参数与原main实际训练前存档相同，初始化后Torch CPU RNG相同；每个数据集的全部15000个训练采样索引一致。两套原默认60-step配方各16/16执行成功，严格数值门4/8（linear / mlp通过，CNN / ResNet18未通过）。原main子进程计时合计205.609s，Rpipe launch外部计时125.277s，计时边界不同，不作为算法加速结论。
- 隔离重复原代码3条训练，发现同一源码、数据、seed在默认benchmark配方下也出现参数与指标分歧。额外完整确定性控制：两套同改deterministic=true / cudnn.deterministic=true / benchmark=false / CUBLAS_WORKSPACE_CONFIG=:4096:8，其余条件和门限不变；各16/16成功、数值门8/8，step30/60参数、test Loss、test Accuracy差值为0，scheduler一致、best均step60、独立eval与best一致。训练Loss最大差6.66e-16、训练Accuracy最大差1.42e-14个百分点，为浮点均值累加顺序差异。原默认4/8判定保留，未把控制条件改成原配方结论。所有数字、来源与图见 [main_reproduction报告](../studies/main_reproduction/docs/STUDY_REPORT.md)。
- README历史图核对：两个PNG与最后修改它们的`4ccb28d`（2024-01-08）逐字节相同。该提交候选配方为80000 steps / eval200 / lr0.01 / batch250 / test250 / 4 seeds、梯度裁剪1、CNN含BN、torchvision CPU增强与旧常量统计；与当前main明显不同。旧图Epoch轴为评测段序号，不代表MNIST完整遍历400轮。没有历史原始运行配置或结果，旧提交配方仍为候选依据。已向用户说明最终对照选择，未启动长实验或擅自更改模型；当前main源码计算对齐与历史图复现分别记录。
- 核对2402个可读历史Study文件size/mtime与当前源码/声明94文件哈希，均未改变；main_base的CIFAR10解压目录无法枚举，未访问或修改。生产代码本轮未新增改动，沿用已通过的282项本地回归。
- 已取消的 1800-step 扩展、自动环境留档功能及独立 eval 数值诊断项目仍不作为本轮任务。此次手工记录复现环境只服务实际对照。

### 5. 后续历史审计与指标修复

- 历史CPU模型探针核对8个组合：linear / mlp / resnet18初始化参数与固定输入logits一致；旧CNN有4层BN、当前无BN，两个数据集logits最大差0.64609 / 0.60425。只扩大预算不能复现历史结构。
- 直接执行旧提交的Metric.compare，Accuracy序列95→90→92最终保存92%的第三个checkpoint，覆盖95%的全程最佳；原因是旧方法即使没有改善也更新比较基准。当前main与Rpipe均保留95%的第一个checkpoint。历史独立eval的权重来源必须考虑这个旧缺陷，归档代码不修改；见 [历史审计](../studies/main_reproduction/docs/HISTORICAL_AUDIT.md)。
- 当前工作树B-017已修复：`accuracy_value(topk=2)`原本将3个样本展开为6个索引并抛RuntimeError，现在保留候选轴，命中任一候选即计为该样本正确。没有增加配置入口；默认top1保留。最小回归修复前1 failed，`.tmp/metric-red-ab3d83106e99449eacd5d33ced97a9e1/`。
- 修复后本地门283 passed / 3 deselected，[测试报告](../.tmp/test-results/20261002T220008Z_f2b7db/report.md)；默认top1 CPU原代码探针8/8。当前源码以新临时Study重跑完整8 train + 8 eval，16/16成功、对照原main既有确定性存档数值门8/8，step30/60参数与test指标差值0，独立eval与best一致，launch86.401s。源码和逐格证据见 [报告后续章节](../studies/main_reproduction/docs/STUDY_REPORT.md)。此前SOURCE_MANIFEST和所有原始Run保留，未覆盖旧证据。

### 6. 历史候选短接入对照

- 现有DataRegistry / ModelRegistry的临时builder复用归档dataset/model，样本dict转tuple，`f(x)`返回logits；旧BN、CPU增强/常量归一化和clip1均保留，无生产源码或新配置改动。
- MNIST / CIFAR10 × linear / mlp / cnn / resnet18，seed0、batch250/test250、SGD lr0.01、cosine T_max80000，两边仅运行4个optimizer steps，在step2/4各评测完整10000张test。数值门8/8通过；初始化/RNG、1000个实际训练索引、增强后输入哈希、scheduler相同，参数与buffer差值0。train指标差值0；test Loss最大差8.88e-16、Accuracy最大差3.55e-15个百分点，正确样本数一致。
- 16个原始数据文件复制SHA256一致，旧dataset自行生成pickle缓存；原归档与既有证据不改写。进程内数据复制+对照36.841s，启动导入不计。91个当前源文件与2402个可读旧Study文件保护复核通过；归档历史36文件、当前main40文件逐字节相同。正式证据见 [HISTORICAL_BRIDGE](../studies/main_reproduction/docs/HISTORICAL_BRIDGE.json) 与 [历史审计](../studies/main_reproduction/docs/HISTORICAL_AUDIT.md)。
- 这支持继续复用Registry承载候选旧配方，不先给生产CNN增加BN开关。eval2改变了随机数消费，这不是历史完整运行的前4步；未执行eval200/80000-step/4-seed，未验证历史独立eval或复现旧best缺陷。该短对照未发现新的生产bug。最终对照选择及长实验仍待明确，已拒绝的两项功能仍不做。

### 7. 评测预算修复B-018

发现`eval_num_steps=0`仍先评测第一批，然后才判断上限，产生与预算不符的Loss/Accuracy。先在structure / STUDY_GUIDE明确：正整数限batch，缺省/负数表示完整test，零预算无法产生有效指标，应报错。共享`eval_test_split`在迭代前拒绝非正的显式上限，训练中test与独立eval共同覆盖；配置负数经已有`eval_batch_limit`转为None，完整test配方仍有效。

最小回归修复前1 failed，[失败报告](../.tmp/test-results/20261002T221547Z_b97619/report.md)；首次测试另有全局pytest缓存写入警告，最终验证改用独立临时cache。修复后unit + integration、c1/c2、排除external/slow/gpu：**286 passed / 3 deselected**，无警告，[通过报告](../.tmp/test-results/20261002T221619Z_6a712b/report.md)。回归直接验证零预算报错且不访问数据，并验证完整split及1/2批上限。

main原代码CPU探针以新输出目录重跑8/8，step2/4参数、train/test Loss与Accuracy差值0、scheduler一致。证据见 [CPU_PARITY_AFTER_B018](../studies/main_reproduction/docs/CPU_PARITY_AFTER_B018.json)；脚本`.tmp/main-reproduction-20261003/cpu_parity_after_B018.py`，目录`.tmp/main-reproduction-20261003/cpu-after-B018/`。这是4-step固定合成输入对照，该修复阶段先完成CPU验证，后续完整GPU复核见下节。91个源文件中仅eval_hook.py相对B-017快照改变，[新快照](../studies/main_reproduction/docs/SOURCE_AFTER_B018_MANIFEST.json)保留；2402个可读旧Study文件及两份归档逐字节/size/mtime保护核对通过。B-018已从开放缺陷移除，下个编号B-019；已拒绝的自动留档和独立eval诊断项目未实施。

### 8. 当前源码完整收口复核

新的临时Study/version `main-98648f3-deterministic-after-B018`运行8 train + 8 eval，60-step/eval30、完整10000张test、原固定Stats及CUDA确定性控制。当前16/16执行成功，与已有原main确定性存档比较数值门8/8：step30/60全部参数和test Loss/Accuracy差值0、scheduler与best步数一致、独立eval与本实现best一致；train Loss/Accuracy仅约6.66e-16/1.42e-14的浮点均值累加差异。launch外部墙钟86.176s，make0.410s，无train恢复或retry，原代码存档不重复执行。

逐格证据见 [DETERMINISTIC_COMPARISON_AFTER_B018](../studies/main_reproduction/docs/DETERMINISTIC_COMPARISON_AFTER_B018.json)，[当前学习曲线](../studies/main_reproduction/docs/figures/learning_curves_after_B018.png)已目检，test坐标30/60、n=1。91个源文件与B-018快照相同，2402个可读旧Study文件及两份归档保护核对通过；生产源码未再改动，沿用286项本地回归。历史候选长实验没有执行，原默认严格门4/8不改写。

### 9. 历史依据与实验工作量核对

本地非shallow仓库的13个可达refs包含150个提交；main祖先91个，图最后更新提交的祖先39个。旧历史曾跟踪61个路径，按原始结果/权重后缀及output/data路径检查，只找到`src/config.yml`，没有找回运行指标或权重；未fetch远端或搜索外部存储，不能宣称外部记录不存在。两张PNG的Software字段均为Matplotlib3.7.1，而旧requirements固定3.7.0，依赖文件不能证明实际绘图环境；这不提供训练Torch版本的证据。

在隔离目录执行归档make.py的生成阶段，4seed、单GPU/round1，32条train + 32条test命令及wait数通过核对，生成命令未执行。一套80000-step候选需256万次optimizer更新、6.4亿train样本处理、1.28亿曲线test样本处理，双实现对照翻倍；不按短探针混合墙钟猜时长。正式证据见 [历史搜索与工作量](../studies/main_reproduction/docs/HISTORICAL_EVIDENCE_SEARCH.json)、[历史审计](../studies/main_reproduction/docs/HISTORICAL_AUDIT.md)。

已再次以明确选项询问最终按当前main源码的60-step确定性对照验收，还是继续README历史图路线；答案未到。后者的条件下一步是先规划200-step/eval200前缀对照及分项计时，再制定完整预算；当前不执行依赖目标选择的实验。此次只有证据搜索、调度草稿与文档改动，没有新生产代码或bug修复。

### 10. Codex 接续复核

回到 Codex 后，按当前工作树重新核对交接：HEAD 仍为 `d938874`，后续代码、文档和三份 Study 尚未提交。当前 91 个生产源文件的 SHA-256 全部与 `SOURCE_AFTER_B018_MANIFEST.json` 相同，最近的完整 GPU 对照证据仍对应当前源码。

重新执行 unit + integration、c1 / c2、排除 external / slow / gpu 的本地门：**286 passed / 3 deselected**，无警告，[本次报告](../.tmp/test-results/20261003T110214Z_95a976/report.md)。首次调用未创建临时父目录，导致 pytest setup 错误；补建父目录后原测试集合通过，未修改生产代码或测试。

复用原 `compare_after_B018.py` 的比较逻辑，直接读取两套已保存的 step30 / 60、best checkpoint、scheduler、独立 eval 结果及日志，将输出写入新的临时目录。重算 **8/8 通过**，逐格数值与正式 `DETERMINISTIC_COMPARISON_AFTER_B018.json` 完全一致；本次是存档证据复核，没有重新训练或覆盖历史结果。脚本及新输出在 `.tmp/codex-handoff-2d8d26a628d5420290defad76e254941/`。

接续时曾需明确最终验收对象：当前 main 源码的计算对照，或 README 历史图复现。连续三个目标轮次未收到选择，2026-10-03 曾将 goal 标记为 blocked（等待验收范围决定），不是完成；当时只核对源码哈希与既有报告，没有重跑测试或训练。用户随后选择历史超参路线并限制本轮只做探针，该阻塞已解除；原默认严格门4/8、确定性控制8/8及历史图未复现的结论保留。

### 11. 完成判定

收口审计按实际目标区分交付与未完成范围：

| **要求** | **当前证据与判定** |
|---|---|
| 查找并修复影响执行/验收的bug | B-014–B-018修复、最小反例与回归已完成；最新本地门286 passed / 3 deselected |
| 阶段性brainstorm与开发summary | 两份文档已同步到当前源码复核、历史证据搜索及本轮200-step前缀探针 |
| 用户选定3.1曲线进度 | 已落地，真实当前矩阵按optimizer step记录；3.2/3.3保持不做 |
| 固定当前main计算对照 | 当前源码16次执行成功，确定性门8/8；原默认严格门4/8仍未通过 |
| README历史图 | 候选配方与短接入验证已核对；原运行记录未找回，长实验未执行，未复现 |
| 本轮验收对象 | 用户已选历史图对应超参，并限制只做200-step/eval200前缀探针；不执行80000-step/4-seed长实验 |
| 历史超参前缀探针 | 8格执行结束并通过，独立读取checkpoint/optimizer/tracker复核8/8，分项计时和正式报告已保存；没有新生产bug |

本轮用户要求的接续理解、历史超参前缀探针、实际开放bug核对及报告已完成。长实验不属于用户最后确认的本轮范围；未来继续历史曲线或长期收敛需另行决定，不自动执行。

此前当前main对照逐格核查数据、结构、初始化、采样/增强、更新次数、scheduler、step30/60、best和独立eval；原默认严格门未通过，确定性控制8/8，历史README图未复现。上次等待范围决定的阻塞已解除。本轮按用户明确的探针范围完成8格200-step/eval200前缀对照、分项计时、报告和实际当前bug处理，不要求长实验或完整历史曲线通过。

### 12. Git 交付与异机测试基线

2026-10-03 用户确认将现有本地成果推送至工作分支，再合并至开发分支。远端开发分支实际名称为 `dev`，集成路径为 `feat/main-base → dev`，具体交付提交及合并状态以 Git / PR 为准。

- 交付内容：B-014～B-018 修复、曲线进度与恢复处理、对应回归测试、三份 Study 的正式声明与报告，以及更新后的 `AGENTS.md` 和项目文档
- 发布前本地回归：unit + integration、c1/c2，排除 external / slow / gpu，**286 passed / 3 deselected**。首次执行仅有旧 pytest cache 写入权限警告，改用 `.tmp/` 内独立缓存后复测无警告，不涉及生产代码修改
- 不随 Git 交付：`.tmp/`、数据缓存、checkpoint、生成调度脚本及 Study 运行产物。正式报告中指向这些本地证据的链接不会随 clone 提供
- 历史配方边界：200-step 探针的临时 Registry 接入与归档脚本仍在 `.tmp/`，尚未整理为正式长实验入口。拉取本次代码不能直接替代完整历史配方交付
- 远程验收：目标环境的安装、GPU、正式 Study 执行与中断恢复尚未验证，本次 Git 集成不等于异机测试通过，也没有启动长实验

# 三、后续合并 main

本阶段版本为 [v0.3.0](releases/v0.3.0.md)，将正式复跑入口、声明、报告、图和仓库整理改动一并交付。长期分支仅保留 `main` 与 `dev`，同步本阶段内容；实际标签、Release 与分支状态以远端为准。

发布前核对的旧 main 是 dev 的祖先，没有 main 独有提交。先通过受保护 dev 的 PR 与跨平台 CPU / 隔离安装门，再将通过验收的成果进入 main 并发布阶段版本；临时发布和已合并功能分支在确认提交已保留后清理。

第二节、科学报告和 [仓库整理验收](REPOSITORY_CLEANUP.md) 中的 refs 与“未合并”描述保留其记录时点；它们不是发布后 main 的动态引用。原 source SHA、数值与失败判定不随发布改写。

本次是公共接口和产物格式的迁移：旧脚本调用需要更新，旧 checkpoint 不能直接作为新 Run resume；旧提交和原产物保留用于读取与回退。具体命令、配置、Stats精度、std及恢复边界见 [MAIN_MIGRATION.md](MAIN_MIGRATION.md)。
