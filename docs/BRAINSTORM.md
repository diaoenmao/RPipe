# Brainstorm

> 未拍板的想法。**不**当合同。权威是 [CONCEPT.md](CONCEPT.md) → [LAYOUT.md](LAYOUT.md) → [STUDY_GUIDE.md](STUDY_GUIDE.md)。缺陷进 [BUGS.md](BUGS.md)。

**对照** git **`main`** 的执行形状。**借鉴** DeepScientist 的账本纪律，不当对照物，不做研究 OS。

前面只写规则和还要做的。已经落地的在文末，不占对照清单。

新想法追加在 **§3**。拍板后写入 CONCEPT / LAYOUT / STUDY_GUIDE，并从这里删掉。

---

## 1. 对照 `main`（硬性）

对照的是形状：调度、Study 收口、metric / checkpoint 习惯。

**不对照：**「只做 `custom_torch`」。旧 `main` 只有这一支；这边 Registry 并列挂多个 `source`。Trainer 特有键不进 Control 必须表。

---

## 2. 借鉴 DeepScientist（硬性）

只借：写下的东西还能被指认（index、result、status）。

**不做：** Quest、Canvas、Findings、daemon、Web、决策器。

---

## 3. 要做的

2026-10-03：已完成 B-013 的代码修复、seed 2 无中断补测及 24-Run 本地多模型矩阵，证据与执行边界见 §4。当前已授权阶段收口；以下仍是建议，不自动授权追加实验。

### 3.1 先补齐 main 支持范围的可用性验证

2026-10-03 对照本地 `main=98648f3`：5 个数据集（MNIST / FashionMNIST / CIFAR10 / CIFAR100 / SVHN）和 7 个模型（linear / mlp / cnn / resnet10 / resnet18 / wresnet28x2 / wresnet28x8）均已有实现。MNIST / CIFAR10 × linear / mlp / cnn / resnet18 已有真实 train / eval；其余三个数据集只有替身构造测试，resnet10 / 两个 WideResNet 只有默认网络前向测试，不能称全组合实测完成。

优先处理 [BUGS](BUGS.md) 的 B-014（错误配置回退到占位路径）和 B-015（SVHN 缺 scipy 依赖声明），再为剩余数据集和模型安排短 train → checkpoint → eval 验收。FashionMNIST / SVHN 当前需使用 foreign 下载入口。上述缺口尚未修复，本次仓库整理只更新记录，不将已有实现等同于安装和运行验收。

### 3.2 然后研究 CIFAR10 的预算敏感性

600-step 矩阵中，MNIST CNN / ResNet18 最终 Accuracy 为 96.8700±1.2933% / 99.3833±0.0929%；CIFAR10 为 56.0100±0.4232% / 74.2733±0.3855%（三个 seed，样本 std）。12 组 best / 独立 eval 均对齐，未再发生 checkpoint 替换错误。完整走势与启动中断说明见 [矩阵报告](../studies/local_model_matrix/docs/STUDY_REPORT.md)。

CIFAR10 两个模型的三个 seed 在 step 480→600 都仍有改善，所有 Loss-best 在 600。建议另建 1800-step Study：同数据、两模型、三 seed × train / eval，共 12 Run，沿用现有排班与报告能力，先写 PLAN / 逐 Run 估时再启动。优先回答“更长配方能否继续改善、seed 差异如何”，暂不扫 lr 或扩大数据集。

若把 cosine T_max 随预算改为 1800，这比较的是两套训练预算 / lr 配方，不是纯粹步数因果实验；同 step 的不同模型也不是等算力比较。需要隔离步数时，先明确一致的 lr 轨迹与 best 评测候选口径，不将旧 600-step 对照强行解释成纯预算效应。

### 3.3 保持研究边界

只复用现有 make / launch / process / report；正式新 Study 配置与报告沿用项目目录，临时探针、缓存和证据放 .tmp/。跨 seed 的原生曲线按观测序号对齐；恢复产生重复 step 时，用 Study 内逐 seed 的实际 step 图解释，不将错位均值冒充同一步的统计。

恢复仍不是精确重放：采样前缀会重播，完整 RNG / 迭代位置和 early-stop stall 未恢复；手工改权重或绕过 launch 的 run-one 也不承诺自动失效旧 eval。需要严格的中断 / 不中断对照时再单独设计，不为当前研究扩成通用工作流系统。

**先不做：** 数据 / 库指纹；自动追加训练预算；通用多图报表框架。

**明确不做：** 把 inference 当对照义务；DDP；vLLM；TensorBoard。Kornia 只用于模型入口：Normalize，以及训练态的 flip / crop。

---

## 4. 已经做的

| 能力 | 口径 |
|------|------|
| 360 开关诊断收口 | 两轮各 2/2 succeeded，关闭 / 开启分别 380 / 390 次 checkpoint 替换、均零错误；不能确定原占用者。一次性诊断完整归入 `.tmp/diagnostics/360-retest-20261003/`，结论合并至 [B-013 补测报告](../studies/mnist_cnn_budget_repeat/docs/STUDY_REPORT.md) §5，不再单列正式 Study |
| 本地 600-step 多模型矩阵 | `local_model_matrix`，MNIST / CIFAR10 × cnn / resnet18 × 三 seed × train / eval，24/24 succeeded、12 组 best / eval 对齐。全部 train 到 600，无训练恢复或保存错误；两条 MNIST ResNet18 首次在 prepare 后启动会话中断，继续后从头训练，保留两次 start。第二次 launcher 503.028s，全轮 flow 包络 860.246s 含会话间隔，不称单次连续 launch；详见 [矩阵报告](../studies/local_model_matrix/docs/STUDY_REPORT.md) |
| B-013 有界原子替换与 seed 2 补测 | Windows 真实句柄复现 WinError 5，统一 artifact 原子替换最多 6 次 / 750ms，仅 Windows 5/32/33；持续失败保旧档并抛错。core 225 / integration 14 通过，分件与整包真实短暂 / 持续占用四探针通过；新 version 补测 2/2、train 无中断、97.12%、best/eval 一致，launcher 28.486s。代码鲁棒性修复完成，原占用进程未识别；证据见 [补测报告](../studies/mnist_cnn_budget_repeat/docs/STUDY_REPORT.md) |
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
| 测试规范 | `docs/TESTING.md` 采用 2026-09-28 正式规范，用例声明 `cost_class` 和 `result_type`。2026-10-02 当前工作区（HEAD `3597137` 加未提交改动）定向 26 passed / 1 deselected，core 160 passed / 17 deselected。报告在 `.tmp/test-results/20261002T132526Z_0e3a85/` 与 `20261002T132548Z_17665f/`；manifest 只记 HEAD，不包含 dirty diff。未运行 external / GPU 网格，不等于全量验证 |
| main_base 历史探针 | 4 train step、结束时 4 个 test batch，8/8 succeeded，约 33s。原始 Run 与单点图保留；本机 60-step 验证轮另用 version，不覆盖探针证据；详见该 Study 的 PLAN / STUDY_REPORT |

不造 `main` 的 `resume_mode` 同名开关。
