# MNIST / linear 完整训练独立 CPU 审查

## 一、摘要

2026-10-04。完整训练的首次审查快照为 01:31:28 +08:00；新增独立 eval 的 CPU 复核快照为 **01:58:40 +08:00**。MNIST / linear 的 seed0–3 已分别连续完成 80000 optimizer steps，四条 train result 均为 succeeded；对应原计划的四条 sibling-best eval 随后均正式 succeeded。独立读取原始 JSONL、tracker state 和 canonical latest.pt / best.pt，**四条训练与四条 eval 的完整性审查全部通过**。本审查没有启动 GPU 或新的正式 Run；为核对权重关联，另外在 CPU 严格加载相同 best 并重算完整 test10000。原始数值与文件哈希见 [MNIST_LINEAR_RESULT.json](MNIST_LINEAR_RESULT.json)。

四 seed 的最终 test Accuracy mean 为 **92.67500124%**，距 PNG 估读 92.67% **0.00500124 个百分点**；末 50 点均值为 **92.66865129%**，距图像估读均值 92.67% **0.00134871 个百分点**。预定七锚点、终点、末段与稳定性的五项单组合诊断均在原门限内。

**这仅是八个数据/模型组合中一个组合的 train + eval 完成与数值诊断。** 01:58:40 快照中全矩阵为 train4 succeeded / 6 started / 22 pending，eval4 succeeded / 28 pending；四模型排序尚不可核对，整个 Study 的最终 gate 尚未应用，历史曲线整体复现尚未完成。

## 二、独立审查方法与完整性

固定当前代码基线 `8bccbac321d4c3ac1ea9892a5e774c114e0298c6`、历史候选提交 `4ccb28d0496110253e9f8e3f3df658853f07996b`。独立比对 SOURCE_MANIFEST 的 99 个源码/声明文件和 65 个展开计划文件，0 差异；没有修改冻结源码、配置、index 或 manifests。

1. 按 index 的 MNIST / linear / train 因素选取 seed0–3，检查 result/config 的 seed 和数据模型身份。每条日志只有一次 Flow start、tracker 只有一次 keep_until=0 的 start marker；没有 resume、error 或 retry 记录。
2. 每条 test Accuracy / Loss 和 train Accuracy 均有 400 个完整点，显式 optimizer_step 为 200、400、…、80000。内部 tracker counter 的 train 行为 200、440、…、95960，test 行为 240、480、…、96000。每段差 40 个 test batches；结合不可变 full-eval 配置和结果元数据 test10000 / batch250，复算每次评测 10000 样本。该证据是计数器/配置复算，正式 worker 未新增逐样本输入字节哈希。
3. 仅在 CPU 读取已提交的整包 latest.pt / best.pt，没有使用诊断分件镜像。latest 的 global step 和 tracker progress 均为 80000，内部 counter96000；整包 history 与全部 400 条原始 scalar 值相同。scheduler T_max=80000、last_epoch=80000、eta_min=0，scheduler 与 optimizer 的最终 lr 为 0。
4. 每条 best 均属于自身完整 Accuracy 曲线的全局最大值，best step 命中该最大点、best history 是原始 history 的相应前缀，best scheduler 的 last_epoch 等于 best step。result 的最后 Accuracy、best_accuracy 和最新整包记录与原始轨迹一致。没有把 best 的较高 Accuracy 替代 step80000 终点。

| **seed / Run** | **终点 Accuracy (%)** | **终点正确数 / 10000** | **best step** | **best Accuracy (%)** | **best 正确数 / 10000** | **审查** |
|---|---:|---:|---:|---:|---:|---|
| 0 / `f7b6c59223a46803` | 92.65 | 9265 | 46800 | 92.80 | 9280 | 通过 |
| 1 / `93a684aa9d0631f5` | 92.69 | 9269 | 26800 | 92.77 | 9277 | 通过 |
| 2 / `51fd2cbf55af3719` | 92.68 | 9268 | 33000 | 92.77 | 9277 | 通过 |
| 3 / `b667e3702ad664b4` | 92.68 | 9268 | 59000 | 92.80 | 9280 | 通过 |

Accuracy 原值保留了 batch 浮点计算产生的约 1e-6 个百分点余量；表中四舍五入到 0.01%。正确数按完整 test10000 的比例复算，不是另执行一次 test。各 Run 原始指标、scheduler、完整性布尔检查和文件 SHA-256 均保存在机器 JSON。日志、tracker 与 checkpoint 为本地 ignored 产物，不随 Git clone 提供。

## 三、四 seed 曲线与预定门的单组合诊断

逐个真实 optimizer step 对齐四条轨迹，直接重算 seed mean / population std（ddof=0），没有重用 compare.py 的计算结果。与 [COMPARISON.json](COMPARISON.json) 中 MNIST / linear 的全部 400 点交叉核对，mean 和 population std 最大差均为 0，每点 seed 数均为 4。终点 population std 为 0.01500006 个百分点。

### （一）七个固定锚点

| **槽 / step** | **当前 mean (%)** | **跨 seed population std (百分点)** | **PNG 估读 (%)** | **绝对差 (百分点)** |
|---|---:|---:|---:|---:|
| 50 / 10200 | 92.567502 | 0.051660 | 92.52 | 0.047502 |
| 150 / 30200 | 92.552501 | 0.051174 | 92.55 | 0.002501 |
| 200 / 40200 | 92.520002 | 0.102713 | 92.52 | 0.000002 |
| 250 / 50200 | 92.597501 | 0.049181 | 92.60 | 0.002499 |
| 300 / 60200 | 92.632501 | 0.046570 | 92.63 | 0.002501 |
| 350 / 70200 | 92.677501 | 0.016394 | 92.67 | 0.007501 |
| 399 / 80000 | 92.675001 | 0.015000 | 92.67 | 0.005001 |

### （二）单组合诊断数值

沿用 [TARGET.md](TARGET.md) 与 [REFERENCE_CURVES.json](REFERENCE_CURVES.json) 的预先门限，没有因本轮数值改门。以下“门内”仅说明该已完成训练组合的诊断值，不是整个 Study 的终验结论。

| **诊断** | **实测偏差 / 时间 std (百分点)** | **原门限 (百分点)** | **单组合诊断** |
|---|---:|---:|---|
| 终点与 PNG 估读绝对差 | 0.00500124 | ≤0.25 | 门内 |
| 末 50 点均值与 PNG 估读绝对差 | 0.00134871 | ≤0.25 | 门内 |
| 七锚点 MAE | 0.00964393 | ≤0.25 | 门内 |
| 七锚点最大绝对差 | 0.04750174 | ≤0.60 | 门内 |
| mean 曲线末 50 点时间 std | 0.01238150 | ≤0.15 | 门内 |

末 50 点为槽350–399，即 step70200–80000。时间 std 衡量四 seed mean 曲线的末段稳定性，和上表各点的跨 seed std 是不同统计量。PNG 数值是图片估读，名义 MNIST 竖直误差 ±0.05 个百分点，原历史运行指标与实际 seed 数仍未知；新结果接近图片不证明找回了原运行记录。

## 四、2026-10-04 01:58:40 +08:00：四条计划内独立 eval 的 CPU 复核

主线程先在 PLAN 记录安排，于 01:53:34–01:54:29 +08:00 提前串行完成原计划内 MNIST / linear 的四条 eval；只调整开始顺序，seed、配置、完整 test 和 sibling-best 口径不变。[EARLY_EVAL_EXECUTION.json](EARLY_EVAL_EXECUTION.json) 的 controller 状态为 succeeded，各 child exit0，四条正式 result 均 succeeded。本审查是在这些正式状态齐全后执行，没有把未结束的 eval 写成通过。

1. 四条各只有一次 Flow start、一次 fresh tracker start 和一次加载 best 的 resume 事件；没有 error/retry。eval 的一次 resume 是独立评测加载权重，不是被禁止的 train 中途恢复。
2. 每条 test Accuracy / Loss 各一行，tracker counter40、full test10000、batch250；JSONL 的 optimizer_step 与父 train 自身 best step 分别为 46800 / 26800 / 33000 / 59000。tracker history、result Accuracy / Loss 和父 best 的相应记录一致。
3. 初始 `[resume] target=best` 只记录 stem；实际 test report 的 `resume_path` 分别精确命中同 seed、同因素父 Run 的 canonical best.pt。每个 best 整包 SHA-256 与首次完整训练审查保存值一致，source/展开计划 hash 仍为 0 差异。
4. 从固定历史提交提取真实 Linear 类，在 CPU 用 `load_state_dict(strict=True)` 加载相同 canonical best，所有参数与 best state_dict 逐 tensor `torch.equal`。按原始 IDX 顺序、float32 ToTensor / Normalize、batch250 重算完整10000样本；四条正确样本数与正式 eval 全部相同，Loss 最大差 `1.0244548320770264e-08`，小于预先 1e-6 数值口径。此重算没有触发新的正式 Flow Run。

| **seed / 正式 eval Run** | **父 best step** | **正式 eval Accuracy (%)** | **正确数 / 10000** | **CPU 复算正确数** | **CPU / 正式 Loss 绝对差** | **独立审查** |
|---|---:|---:|---:|---:|---:|---|
| 0 / `ba0a42f0131ebd4a` | 46800 | 92.80 | 9280 | 9280 | 1.02445e-08 | 通过 |
| 1 / `0005897719db9384` | 26800 | 92.77 | 9277 | 9277 | 9.96515e-09 | 通过 |
| 2 / `f8567981f878e200` | 33000 | 92.77 | 9277 | 9277 | 3.16650e-09 | 通过 |
| 3 / `81d5e2fdf7e8d89c` | 59000 | 92.80 | 9280 | 9280 | 5.82077e-09 | 通过 |

权重证据包括实际加载源路径、canonical 包 checksum 与冻结 loader 契约，以及独立 CPU 重建严格相同权重后复现正确样本数和 Loss。**正式 eval worker 没有导出运行时 model state / fingerprint**，因此上述复核不是该 worker 权重的直接字节快照；CPU 加载与复算是本次真实执行的独立证据。逐参数指纹、19 项/条检查和完整文件 SHA-256 留在机器 JSON 的 `formal_eval_audit`。

这些 eval 指标对应各自 best，不能替换前文 step80000 的终点 mean92.67500124%；首次训练曲线的五项诊断数值不变。当前完成的是 MNIST / linear 一个数据/模型组合的 train + eval，整个八组合门仍未应用。

## 五、补充：CIFAR10 / ResNet18 单 seed 前段核对

另按要求只读提前 Run `4b6781cffa2c2b8c` 的已提交完整 test JSONL 行，核对颜色、指标和横轴，未打开该活动 Run 的 checkpoint。匹配模型后，step10200 的当前 seed0 为 85.80000210%，参考 ResNet18 图像估读为 86.28%，差 **−0.47999790 个百分点**；step5200 的当前 seed0 为 80.79000111%，参考为 82.43%，差 **−1.63999889 个百分点**。保留这两个前段偏差，不以 MNIST 尾段相近代替其他组合的曲线检查。

固定 `4ccb28d0496110253e9f8e3f3df658853f07996b` 的静态源码给出以下口径，原行号和 Git blob 留在机器 JSON 的 static_historical_source_evidence：

1. src/process.py 的 color_dict 将 CNN 映射到 blue / dotted，ResNet18 映射到 dodgerblue / dash-dot。slot50 的 81.76% 是 CNN 估读，**ResNet18 的对应值是 86.28%**，不能混用两条蓝色曲线。
2. src/test_model.py 将最终训练 checkpoint 的 logger 放入结果的 logger_state_dict['train']；process.py 从这个 train logger 选择 test/Accuracy 的 history。因此图画的是训练期间反复完整 test 的 Accuracy，不是 train Accuracy，也不是独立 best eval 的单一 mean。
3. src/module/hyper.py 固定 eval_period200 / num_steps80000；train_model.py 从 iteration0 开始，先训练满200步再 test，process.py 用 x=np.arange(len(y))。因此候选配方的槽 j 映射到 step(j+1)×200；slot50 对应 step10200。当前 JSONL 在该点的内部 tracker counter 是12240，它包含 train 与 test batches，不能拿它作为 optimizer 横轴。

这些源码证据支持固定候选配方的对齐口径，PNG 自身缺少原始 Run 元数据，不能据此证明图片实际训练环境和 seed 数。当前 CIFAR10 / ResNet18 仍只有 seed0 的早期点，这是 **单 seed 与参考均值的描述性比较**；其余三 seed 和完整长曲线未完成，不应用四 seed 整组 gate，也不据两个点判断该组合整体通过或失败。

## 六、未完成事项与验证范围

首次 01:31:28 训练审查时的矩阵为 train4 succeeded / 5 started / 23 pending，eval32 pending；保留这个历史快照。新增 01:58:40 eval 复核快照中为 train4 succeeded / 6 started / 22 pending，eval4 succeeded / 28 pending。尚需完成另外七个数据/模型的 train + eval 组合、其余28条独立 eval 的 best 来源与数值核对、两数据集的四模型末段排序，以及整体预先曲线门。

本次只写入本报告与同名机器 JSON，全部计算、checkpoint 读取与额外完整 test 复算在 CPU 完成；没有启动 GPU 操作或新的正式 Run，没有修改冻结源码、配置、index 或 manifests。主线程发起的四条原计划正式 eval 与本审查 CPU 复算分别记录，whole_study_final_gates_applied=false、historical_reproduction_passed=null 保持。
