# Brainstorm

> 未拍板的想法。**不**当合同。权威是 [CONCEPT.md](../code/concept.md) → [LAYOUT.md](../code/layout.md) → [README.md](../../studies/README.md)。缺陷进 [BUGS.md](bugs.md)。

**对照** git **`main`** 的执行形状。**借鉴** DeepScientist 的账本纪律，不当对照物，不做研究 OS。

前面只写规则和还要做的。已经落地的在文末，不占对照清单。

新想法追加在 **§3**。拍板后写入 CONCEPT / LAYOUT / STUDY_GUIDE，并从这里删掉。

---

## 1. 对照 `main`（硬性）

对照包括调度、Study 收口、metric / checkpoint 习惯；按当前持续工作目标，还需验证固定 main 源码在相同配方与环境下的数值结果，不能用执行形状或 Run 成功代替结果复现。

**不对照：**「只做 `custom_torch`」。旧 `main` 只有这一支；这边 Registry 并列挂多个 `source`。Trainer 特有键不进 Control 必须表。

---

## 2. 借鉴 DeepScientist（硬性）

只借：写下的东西还能被指认（index、result、status）。

**不做：** Quest、Canvas、Findings、daemon、Web、决策器。

---

## 3. 待执行想法

当前没有待执行的实验扩展。仓库整理和 main 迁移准备按 REPOSITORY_CLEANUP 与 MAIN_MIGRATION执行，发布前的提交、远端CI与合并另按实际授权处理。

2026-10-03新一轮§3.1「曲线记录并对齐真实训练进度」已落实；§3.2「自动记录复现信息」与§3.3「独立eval复算判定」按用户决定不做。取消的CIFAR10 1800-step实验保持原决定。新的未确认想法再追加于本节，不把已经完成的长矩阵列为待执行任务。

---

## 4. 已经做的

### 2026-10-04

持续目标已完成：历史32条80000-step四seed训练与32条own-best eval，八组完整曲线通过事先原图估读门；本机现代60-step受控八格和历史600-step前缀桥也通过。正式入口和独立CPU审计进入 [main_historical](https://github.com/diaoenmao/RPipe/blob/18cd76c/studies/main_historical/README.md)。图像、原默认失败及原环境未知的证据边界保留，科学结论不扩称所有原始逐点数据一致。

本轮整理全仓库导航、正式成果及事务标记，补充迁移文档和发布验收，见 整理报告。生产数值源和原存档没有改动，不再以临时脚本作为唯一重跑入口。

### 2026-10-03 阶段计划备查

以下保留当时的只探针授权和候选工作量；2026-10-04另获持续运行授权并完成长矩阵，不倒写旧记录。

2026-10-03 main复现阶段已有结果，见 [main_reproduction报告](https://github.com/diaoenmao/RPipe/blob/18cd76c/studies/main_reproduction/docs/STUDY_REPORT.md)：原默认60-step配方各16/16执行成功、严格数值门4/8；原代码重复也分歧，完整确定性控制则8/8对齐。两个历史PNG最后更新于`4ccb28d`，该提交为80000-step / eval200 / 4-seed候选配方，CNN含BN、梯度裁剪1、CPU增强和统计也不同。用户已选择历史图对应超参路线，随后明确**本轮只做到探针**；200-step/eval200、seed0的8格前缀对照和计时进入[执行计划](https://github.com/diaoenmao/RPipe/blob/18cd76c/studies/main_reproduction/docs/PLAN.md#用户确认的历史超参前缀探针2026-10-03执行前)，不是长实验授权。

此前4-step/eval2短接入验证已经8/8通过，新200-step/eval200前缀探针及存档复核也8/8通过，见§4；候选旧运算可以复用现有Registry，暂不提生产模型兼容开关。本轮已结束；探针不能替代历史实际运行依据、4-seed长期曲线或独立eval证据。没有自动进入长实验的待执行任务。

历史证据搜索已完成本地可达Git范围，未找到原始指标/权重；PNG绘图版本还与旧requirements不同。已生成供核对的64条原调度命令，未执行。单套完整候选即256万次更新、6.4亿train样本处理，两套对照翻倍。详细依据见 [历史审计](https://github.com/diaoenmao/RPipe/blob/18cd76c/studies/main_reproduction/docs/HISTORICAL_AUDIT.md)。用户本轮限制为探针，后续是否制定长实验预算另行决定，不自动执行。

### 2026-10-03 实施记录

历史超参前缀探针（2026-10-03）：按用户“只做到探针”的范围，MNIST / CIFAR10 × linear / mlp / cnn / resnet18，seed0、200 optimizer steps、eval200完整test10000，保留旧BN/CPU增强/常量统计/clip1及scheduler T_max80000。两边8/8通过，参数和buffer差值0、初始/最终RNG及50000个采样与增强输入相同；训练/test正确样本数相同，Loss差仅浮点累计。独立加载checkpoint/optimizer/tracker复核8/8，内部148.740s；91当前源文件及36历史归档文件不变。见[探针报告](https://github.com/diaoenmao/RPipe/blob/18cd76c/studies/main_reproduction/docs/HISTORICAL_PREFIX_PROBE.md)。未发现新生产bug；BUGS清除已关闭项的重复说明，明确无开放缺陷。本轮结束，不执行80000-step/4-seed长实验，历史图仍未复现。

当前源码完整收口复核（2026-10-03）：B-018后的新临时Study完整8 train + 8 eval成功，与原main未修改的确定性存档数值门8/8，step30/60参数及test指标差值0，独立eval/best一致，launch86.176s。91个源文件与当前快照一致，286项本地回归有效，原默认4/8和历史图未复现结论保留。证据见 [最新GPU对照](https://github.com/diaoenmao/RPipe/blob/18cd76c/studies/main_reproduction/docs/DETERMINISTIC_COMPARISON_AFTER_B018.json)；最终验收对象仍待用户选择，不继续追加无关bug或扩大实验来替代这个决定。

历史依据搜索（2026-10-03）：本地13 refs / 150 commits、旧祖先39 commits / 61路径未找到原结果或权重；两张PNG记录Matplotlib3.7.1，旧requirements为3.7.0，不能据依赖文件认定历史实际环境。原make.py仅生成32 train + 32 test命令供核对，没有执行训练。工作量与搜索范围见 [HISTORICAL_EVIDENCE_SEARCH](https://github.com/diaoenmao/RPipe/blob/18cd76c/studies/main_reproduction/docs/HISTORICAL_EVIDENCE_SEARCH.json)；目标选择仍待用户明确。

B-018零评测预算（2026-10-03）：eval_num_steps=0原本仍评第一批并生成指标，现由共享入口在读取数据前报错；缺省/负数完整test及正数限批保持。最小反例先失败，本地门286 passed / 3 deselected，main原代码CPU对照8/8、参数/指标差值0。修复与验证见 [record.md](record.md)，开放项已移除；该修复阶段先完成CPU验证，后续完整GPU复核见上方收口记录。

历史候选短接入对照（2026-10-03）：临时Registry builder直接复用归档模型/dataset，旧CNN BN、CPU增强和clip1接入当前原生训练循环，真实8组合4-step/eval2数值门8/8。step2/4参数与buffer差值0、采样与增强后输入哈希一致。已有Registry足够承载候选旧运算，暂不增加生产CNN兼容开关。证据见 [历史审计](https://github.com/diaoenmao/RPipe/blob/18cd76c/studies/main_reproduction/docs/HISTORICAL_AUDIT.md)；eval2不是历史eval200轨迹，未启动80000-step长实验，最终对照选择仍待明确。

B-017与历史候选审计（2026-10-03）：修复Accuracy已有topk参数的样本轴丢失，最小反例先失败，本地门283 passed / 3 deselected；默认top1 CPU探针8/8，修复后新临时Study完整16次CUDA执行成功，对原main确定性存档数值门8/8，参数/test指标差值0。历史探针证明CNN需要旧版4层BN；旧Accuracy best比较还会用上一轮代替全程最佳，95→90→92会覆盖真正最佳。归档缺陷不改，候选历史路线不能只加步数。证据见 [main报告](https://github.com/diaoenmao/RPipe/blob/18cd76c/studies/main_reproduction/docs/STUDY_REPORT.md) 与 [历史审计](https://github.com/diaoenmao/RPipe/blob/18cd76c/studies/main_reproduction/docs/HISTORICAL_AUDIT.md)。B-017已从开放缺陷移除，未启动长实验。

完整main源码对照（2026-10-03）：新 [main_reproduction Study](https://github.com/diaoenmao/RPipe/blob/18cd76c/studies/main_reproduction/docs/STUDY_REPORT.md)，真实数据与原Stats完整精度、初始化/RNG/15000个采样索引核对通过；两套原默认60-step / eval30各16/16执行成功，数值门4/8。相同原代码的3条重复训练也出现分歧；在隔离目录同改CUDA确定性条件后，两套完整矩阵各16/16成功，数值门8/8，step30/60参数与test Loss/Accuracy差值为0；训练均值仅有约1e-16 / 1e-14的浮点累加差异。原默认失败判定保留，历史README图未复现。2402个可读旧Study文件与94个源码/声明文件保护检查通过；未改生产实现或扩大预算。

B-016 于 2026-10-03 修复：native step 训练摘要按评测段收口，空段不覆盖最后有效均值。固定 main 原代码 CPU 探针 8/8 通过，step2 / 4 参数与指标差值均为 0；合并本地 unit + integration c1 / c2 回归 **282 passed / 3 deselected**。最小反例、原代码对照证据与完整 60-step 真实数据验收的剩余边界见 [record.md](record.md)。用户本轮确认仍只实施 §3.1 曲线进度，§3.2 / §3.3 不做。

B-014 / B-015 最终合并回归：unit + integration 的 c1 / c2 本地门 **264 passed / 3 deselected**（排除 external / slow / gpu），无警告；日志 `.tmp/bugs-final-9c5b65c635bd4d53bfd397214bf250c9/final.log`。wheel 构建及 scipy 声明核对通过，产物 `.tmp/bugs-package-f8f84b9bab544d81a18d954326c38db1/dist/rpipe-0.3.0-py3-none-any.whl`。以上证据不包含官方 SVHN 下载或全新环境完整安装。

| 能力 | 口径 |
|------|------|
| 曲线按真实训练进度对齐（新一轮 §3.1） | 2026-10-03：native / HF 报告新增 optimizer_step / epoch，batch counter 保留；checkpoint 记录日志位置与实际保存进度，原 JSONL 不截断，学习曲线排除已回滚分支、同坐标重复报告取最后一条。process / 图共用坐标聚合，缺失点不插值、逐点 n_at_point；step / epoch / observation 分开，旧记录不猜单位。覆盖不同记录频率、重复恢复、未知 / 迁移日志、半条 JSONL、真实 checkpoint 保存失败后续跑、无当步报告的 checkpoint、HF 半 epoch 与恢复坐标；失败 Run 不参加曲线。最终本地 unit + integration c1 / c2 **279 passed / 3 deselected**（排除 external / slow / gpu，无警告），[验证报告](../../.tmp/test-results/20261002T212208Z_c1ea50/report.md)；[坐标与逐点 n 示例](../../.tmp/curves-final-1b40e02c4d07468aa2f66f0be9a63091/base/test_study_progress_summary_an0/docs/figures/learning_curves.png) 已目检。只做小数据验证，未重训或重绘历史 Study；HF 预算 / 调度与数值恢复边界不变 |
| 支持范围 30-step 验收（原 §3.1） | 2026-10-03：两份 Study 共 12/12 succeeded，6 train 均 30 step、参数更新且有限、无训练错误 / 重试 / resume；6 组 checkpoint 来源和 Loss 核对通过，严格 Accuracy 一致 5/6。ResNet10 为 20.02% / 独立 eval 20.01%，相差 1 个正确样本；同一权重的数值敏感性有额外诊断，历史逐样本差异未复原，保留原值与未通过的严格判定。FashionMNIST / CIFAR100 / SVHN × cnn 的 [数据报告](https://github.com/diaoenmao/RPipe/blob/71143ab/studies/support_data_smoke/docs/STUDY_REPORT.md)：6/6、严格 3/3，launcher 31.176s；CIFAR10 × resnet10 / wresnet28x2 / wresnet28x8 的 [模型报告](https://github.com/diaoenmao/RPipe/blob/71143ab/studies/support_model_smoke/docs/STUDY_REPORT.md)：6/6、严格 2/3，launcher 107.944s。7 个官方数据资源 MD5 一致；下载 / 构造 / make 735.735s，与模型运行重叠。旧 Study 的 2078 文件 size / mtime 及非 checkpoint 内容哈希未变，CIFAR10 复制缓存 10 文件 SHA-256 一致；当前源码 92 文件 snapshot / manifest 和原始证据在 `.tmp/support-smoke-20261003/`。单 seed、6 个指定组合不代表全组合或收敛验收；未做全新环境完整安装，未追加长预算实验 |
| B-014 / B-015 配置与依赖修复 | 2026-10-03：data / model Factory 按 name / source 精确匹配，未知或空 source 明确报错；省略时保留 torch / custom_torch 默认。占位计算仅接受显式 stub 数据；native 及 HF 回退路径缺少正式输入时拒绝，Flow 留 failed result / prepare 错误日志。新增拒绝、合法 stub 和失败落盘回归在修复前 24 项失败，修复后定向 51 项、core 245 项、本地 integration 19 项通过。SVHN 经真实 torchvision 构造器读取本地合成 `.mat`（含标签 10→0），一步 train / best checkpoint / 独立 eval 对齐；scipy 已进入基础依赖，并核对所构建 wheel 的 Requires-Dist。未下载官方 SVHN，也未做全新环境完整安装。原始验证输出在 `.tmp/bugs-red-*`、`.tmp/bugs-green-*`、`.tmp/bugs-core-*`、`.tmp/bugs-integration-*`、`.tmp/bugs-package-*`；长期回归见 `tests/rpipe/structure/{data,model,algorithm}/`、`tests/rpipe/flow/test_runner.py` 和 `test_svhn.py`。已从 BUGS 开放项移除 |
| 360 开关诊断收口 | 两轮各 2/2 succeeded，关闭 / 开启分别 380 / 390 次 checkpoint 替换、均零错误；不能确定原占用者。一次性诊断完整归入 `.tmp/diagnostics/360-retest-20261003/`，结论合并至 [B-013 补测报告](https://github.com/diaoenmao/RPipe/blob/71143ab/studies/mnist_cnn_budget_repeat/docs/STUDY_REPORT.md) §5，不再单列正式 Study |
| 本地 600-step 多模型矩阵 | `local_model_matrix`，MNIST / CIFAR10 × cnn / resnet18 × 三 seed × train / eval，24/24 succeeded、12 组 best / eval 对齐。全部 train 到 600，无训练恢复或保存错误；两条 MNIST ResNet18 首次在 prepare 后启动会话中断，继续后从头训练，保留两次 start。第二次 launcher 503.028s，全轮 flow 包络 860.246s 含会话间隔，不称单次连续 launch；详见 [矩阵报告](https://github.com/diaoenmao/RPipe/blob/71143ab/studies/local_model_matrix/docs/STUDY_REPORT.md) |
| B-013 有界原子替换与 seed 2 补测 | Windows 真实句柄复现 WinError 5，统一 artifact 原子替换最多 6 次 / 750ms，仅 Windows 5/32/33；持续失败保旧档并抛错。core 225 / integration 14 通过，分件与整包真实短暂 / 持续占用四探针通过；新 version 补测 2/2、train 无中断、97.12%、best/eval 一致，launcher 28.486s。代码鲁棒性修复完成，原占用进程未识别；证据见 [补测报告](https://github.com/diaoenmao/RPipe/blob/71143ab/studies/mnist_cnn_budget_repeat/docs/STUDY_REPORT.md) |
| MNIST CNN 600-step 本地实验 | `mnist_cnn_budget`，lr 0.03 × seed 0 / 1 / 2 × train / eval，6/6 succeeded；最终 Accuracy 95.58% / 98.07% / 97.14%，三组 best / eval 对齐、best step 600。seed 0 / 1 无中断，seed 2 保存失败后恢复；launcher 72.311s。原始恢复证据保留，不能称三条无中断复测；B-013 后续已修复并补测，见上一行 |
| B-007 关闭（维护决定） | 2026-10-02 按用户决定标记完成，从 BUGS 开放项移除，不再主动排查。最后确认发生于 2026-09-26；后续实验未复现，最新检查现存 216 个 Run 日志 / result 文件无同类记录。这是“不再追查”的关闭，不是根因修复证明；保留现有代码，仅在实际再次出现时重新评估 |
| checkpoint / 重试可靠性 | B-009–B-012：scheduler 先推进再存档；best_metric/best_value 与 best_accuracy 区分；整包原子提交且旧存档保全；train 波内重试、sibling eval 阻断/失效。生成 Bash 脚本复用同一调度器。CPU 小 Study 首轮正确保留 2 个失败，二轮恢复至 4/4，耗时 13.992s + 7.083s；详见 checkpoint_recovery 报告 |
| MNIST CNN 学习率诊断 | 新 Study `mnist_cnn_lr`，0.1 / 0.03 / 0.01 × seed 0 / 1 / 2 × train / eval。首轮故障证据保留，不混入比较；`cnn-lr-clean-20261002` 干净轮 18/18 成功。只修 lr 日志缓存，不改变训练更新顺序；35 项定向、168 项 core 通过。main_base 370 个 Run / index / process 文件哈希未变 |
| 本机多点曲线验证 | `local-curves-20261002`：16/16 succeeded，99.827s；8 条 train 均有 13 个 train / 12 个 test 观测，8 组 best / 独立 eval 指标一致。绘图按既有契约优先读 scalars.jsonl，缺失或无有效记录时回退 history，不改训练或 tracker 语义。定向 17 passed / 1 deselected，core 166 passed / 17 deselected；旧 8 条 Run 的 160 个文件哈希未变，四步探针图单独保留 |
| Accuracy 学习曲线（B-008 已修复） | 百分制纵轴和单位、单点 marker、history 序号横轴，图例移到图外。已从 main_base 原 tracker 重绘并目检，无重训；index / process、Run 产物及脚本共 167 个文件前后哈希一致。定向 11 passed / 1 deselected（`.tmp/test-results/20261002T133258Z_a36770/`），core 160 passed / 17 deselected（`20261002T133206Z_859836/`）。历史 0–1 Study 未重绘，不按数值猜单位；已从 BUGS 开放项移除 |
| 对照与本机执行计划 | `main_base/docs/PLAN.md` 固定旧 main `98648f3`，明确 CIFAR 增强、train stats、best / last / eval、百分制指标及分层计时；当前配置已切换为本机 60-step 可视化验证轮，评测频率差异已记录，不用 33s × 15 估时 |
| version 与当前清单 | 沿用 Config extras，经基底顶层或 `fixed.version` 进入 hash；省略保持原有 ID，新 version 换 Run，旧文件保留。已移除 timestamp 后缀 helper；Study process 缺失/损坏 index 明确报错，不再扫描历史目录。回归覆盖身份、合并往返、index 换轮、聚合和 sibling checkpoint 隔离；操作约定见 STUDY_GUIDE §6–7 |
| `&` / `wait` | 一组结束才开下一组。`launch` 和 make 脚本 |
| Study process | launch 全部 wait 完后再收口；也可 `rpipe process`。信封在根 `process.json`；跨 seed 的 mean / std / min / max 在 `experiments[]`。Experiment 无文件夹 |
| metric / checkpoint | 当前 Accuracy 0–100，latest / best、shuffle / 续训。历史 0–1 报告保留当时单位。实际算法 source 有 `custom_torch` 与 `transformers_trainer`；其它来源不能因文档提及就视为已实现 |
| `--mode train` / `eval` | 只滤这一次 launch，不改 `jobs.json` |
| skip / `--include-done` / `--remake` | 已成功默认跳过；`--include-done` 复用清单；`--remake` 才再 make |
| 失败再试 | train 波内 `retry … (resume latest)` 完成后才放行依赖 eval；最终失败不伪装全部成功。不改 yaml |
| 算法 resume | train `latest`；eval `best`（sibling） |
| 失败 log | 同一份 `run.log` 加 traceback；`result` 有 `status` / `error` |
| config 身份 | Run `id` 就是 config hash。`index.id` 是整张清单。不再做第二套 |
| `rpipe status` | 只读，实现在 `structure/artifact/readout/`。`pending` 的 `note` 是最后一条 `[epoch]` 或 `[error]`；`failed` 的 `note` 是最后一条 `[error]` 摘要，`error` 列仍是 result。`succeeded` 的 `note` 是 `-` |
| launch 阶段 | 每组开始打 `launch: wait i/n mode=`；再试打 `launch: retry`。结束打 `planned` / `succeeded` / `failed` / `pending`。不改 `jobs.json` |
| `rpipe logs` | 只读，同一份 readout。各 Run 事件行按时间打到终端。不写 Study 级总 log |
| `rpipe report` | 同一份 readout。从 `process.json` 写 `docs/NUMBERS.md`：Experiment mean / std / min / max 和 Run 表。不改 `STUDY_REPORT.md` 的结论 |
| `split-round` | make 已有 |
| CIFAR 小网格 | `studies/cifar_grid/`。2026-09-30 从头重跑，8/8 succeeded，没有 `torchvision::nms`。结论在该 Study 的 `docs/STUDY_REPORT.md` |
| `run.log` 行格式 | `时间 级别 Run id [事件] 内容`。时间是 RFC 3339 毫秒+时区。事件：`[flow]` `[error]` `[warn]` `[epoch]` `[split]` `[metric]` `[time]` `[ckpt]` `[resume]`。traceback 每一行都是 `[error]`。不改 `result.json`。PR #8，`e9a2f6b`，已进 `dev` |
| 测试规范 | `docs/development/testing.md` 采用 2026-09-28 正式规范，用例声明 `cost_class` 和 `result_type`。2026-10-02 当前工作区（HEAD `3597137` 加未提交改动）定向 26 passed / 1 deselected，core 160 passed / 17 deselected。报告在 `.tmp/test-results/20261002T132526Z_0e3a85/` 与 `20261002T132548Z_17665f/`；manifest 只记 HEAD，不包含 dirty diff。未运行 external / GPU 网格，不等于全量验证 |
| main_base 历史探针 | 4 train step、结束时 4 个 test batch，8/8 succeeded，约 33s。原始 Run 与单点图保留；本机 60-step 验证轮另用 version，不覆盖探针证据；详见该 Study 的 PLAN / STUDY_REPORT |

不造 `main` 的 `resume_mode` 同名开关。
