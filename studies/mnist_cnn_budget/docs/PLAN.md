# Study Plan: mnist_cnn_budget

2026-10-02 用户授权执行。依据 [STUDY_GUIDE](../../../docs/STUDY_GUIDE.md)；背景见 [60-step 学习率诊断](../../mnist_cnn_lr/docs/STUDY_REPORT.md)。

## 1. 研究问题

lr 0.03 的现有 CNN 在 600 optimizer steps 内能否持续学习？观察完整 Loss / Accuracy 走势、seed 差异，以及 best 与 last 是否明显分离。低分或回落本身不等于代码缺陷。

## 2. Study / Experiment / Run

Study 为 `mnist_cnn_budget`；mode 轴产生 train / eval 两个 Experiment，每个 seed 0 / 1 / 2，共 6 个 Run。新 version `cnn-600step-20261002`，从头训练。旧 Study 和 Run 保留。

## 3. 高效率排班

单张 GPU 0，同模型最多 2 条并发，2 个 train wait 组后再执行 2 个 eval wait 组。组末等待全部进程退出；失败 train 在训练波内重试 latest，最终失败阻断对应 eval，无关 seed 继续。若发生恢复，报告明确标记。

执行前 GPU 为 RTX 5090 D v2，24455 MiB 总显存，4096 MiB 已用、3% 利用率。沿用本机已验证的双并发起点。MNIST 缓存复用上一轮并核对 SHA-256；不改变共享旧数据。

```powershell
python -m rpipe make studies/mnist_cnn_budget --num-gpus 1 --init-gpu 0 --round 2 --console shared
python -m rpipe launch studies/mnist_cnn_budget --num-gpus 1 --init-gpu 0 --round 2 --console shared
python -m rpipe report studies/mnist_cnn_budget
```

## 4. 固定条件与观测

MNIST 全量 train 60000 / test 10000，Normalize 沿用 train stats；CNN [64,128,256,512]，无 BN / dropout，MNIST augment=true 仍仅 Normalize。SGD lr 0.03、momentum 0.9、Nesterov、weight decay 0.0005、不裁剪梯度，cosine T_max=600、eta_min=0。batch 250，test batch 1000。

600 steps 即每 seed 150000 次采样，约 2.5 个训练集规模，不预先宣称收敛。train 每 10 step 记录，test 每 30 step 全量评估，latest 每 30 step；按 test Loss 最低选 best。独立 eval 默认加载同 seed 的 sibling best。deterministic=false，cudnn_benchmark=true，不承诺逐位重现。

与旧 60-step 轮比较时，T_max 60→600 改变前 60 步 lr 轨迹，评测频率 5→30 改变 best 候选集合。因此只能描述两个配方的观测差异，不能把全部提升归因于步数。

## 时长预估

make 后、launch 前填写 Run ID 和内置估时。另以每 train 120s、eval 8s 作为本机保守排班参考：四个 wait 组的 max 加总为 256s，含初始化但不包含事后诊断 / 报告。参考旧 60-step 的 train 约 12s、eval 约 4s；本轮训练量和评测次数均改变，实际成本待测，未调整通用估时模型。

| wait | mode | seed | id | est（本机参考秒） |
|---:|---|---:|---|---:|
| 1 | train | 0 | cb3e0009eae4d66b | 120 |
| 1 | train | 1 | 7a73df1b02869b59 | 120 |
| 2 | train | 2 | 032cf3b849f5673a | 120 |
| 3 | eval | 0 | dcb6b5e904073e97 | 8 |
| 3 | eval | 1 | c88be77461e7326c | 8 |
| 4 | eval | 2 | d18a81b715f60ab8 | 8 |
| | 整轮 | | | 256 |

make 内置估时为 train 58s / eval 5s，四组共 126s。preflight 已确认两 train 组后两 eval 组、6 个新 Run 无 result / latest、配置和 seed 完整；缓存 10 个文件哈希一致。上述 256s 是人工参考，与内置值分别保留。

## 5. 验收与交付

执行完成于 2026-10-03：6/6 succeeded；seed 0 / 1 无中断，seed 2 在 step 510 保存时出现一次 WinError 5，自动从 latest step 480 恢复。无中断三 seed 标准未完全满足，恢复状态明确列入 [报告](STUDY_REPORT.md)。实际 launcher 72.311s，3 组 best / eval 对齐，旧文件保护检查通过；新的保存问题列为 B-013。

核对 6/6 succeeded、每条 train 到 step 600、20 次全量 test；检查 train 记录、实际使用 lr 与 checkpoint scheduler 状态。报告每 seed last / best Loss、Accuracy、best step，以及配对 eval 的权重来源和指标一致性；聚合用样本 std（n−1）。保留可点击曲线、Run 日志、逐 Run 估时与实际 flow 耗时及整轮墙钟。核对旧 Study 文件未变化；运行产物忽略，临时脚本与验证结果留 `.tmp/`。

先用已有曲线判断是否塌缩 / 反复回落；有需要才做只读预测分布检查。完成后写 STUDY_REPORT 与 NUMBERS，再按结果更新 brainstorm。新的扫参或预算扩展另定计划。
