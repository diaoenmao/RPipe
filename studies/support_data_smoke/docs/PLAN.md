# Study Plan: support_data_smoke

2026-10-03，用户批准 BRAINSTORM 原 §3.1。与另一 support smoke Study 合计 6 个组合、12 Run；本目录只展开下表 3 个组合、6 Run。设计依据：[STUDY_GUIDE](../../../docs/STUDY_GUIDE.md)。

## 1. 验收问题与范围

验证真实数据 → 原生模型 → 30 次 optimizer 更新 → latest / Loss-best → 独立 eval 的可用性。一个 seed 的短训练不评价收敛或模型优劣，不代表全部 5×7 组合验收。

## 2. Study / Experiment / Run 与固定配方

| data.name | seed | optimizer step | mode |
|---|---:|---:|---|
| FashionMNIST | 0 | 30 | train + eval |
| CIFAR100 | 0 | 30 | train + eval |
| SVHN | 0 | 30 | train + eval |

每个 train / eval 为独立 Experiment，n=1。batch=64、test batch=256、完整 test；step 10 / 20 / 30 评测和更新 latest，按 test Loss 选 best。SGD lr=0.03、momentum=0.9、Nesterov、weight decay=0.0005；cosine T_max=30、eta_min=0，无梯度裁剪。使用默认模型结构，数据 Normalize / 训练增强沿用现有实现；有复制的 stats.yaml 时优先使用，否则使用 Factory 常数。train 默认 latest、eval 默认 sibling best。

version `support-30step-20261003`。Python 3.13.9、torch 2.11.0+cu128、torchvision 0.26.0+cu128、scipy 1.16.3；HEAD d938874 加 B-014 / B-015 未提交修复，不能只凭 HEAD 重建。RTX 5090 D v2，启动前空闲显存约 22 GiB、磁盘约 834 GiB。scipy 导入及含 scipy 的 wheel 元数据已核对；本轮使用现有环境，不声称全新环境完整安装。

## 3. 高效率排班与估时

先准备共享官方数据。FashionMNIST / SVHN 使用 foreign 入口；必要时从同一项目官方发布地址下载并按 torchvision 官方 MD5 验证。CIFAR10 从 local_model_matrix 的可读缓存复制到新 Study 并核对 SHA-256，不修改旧文件。首次下载 / 初始化预留 10 分钟，记录实测，不计入 launcher 计算估时。

make 使用 auto 同模型装箱；CNN 的三个条件可同组，模型 Study 三种模型各一组，WideResNet28×8 单并发。每组 wait 后再开下一组，全部 train 完成再 eval；失败沿用 launcher 一次 resume 重试，保留失败和实际步骤记录。make 后补全每个 Run ID、wait、内置预估，再 launch。启动前核对实际空闲显存与 jobs 清单。

| 条件 | mode | seed | 人工预估（s / Run） |
|---|---|---:|---:|
| FashionMNIST / cnn | train | 0 | 90 |
| FashionMNIST / cnn | eval | 0 | 30 |
| CIFAR100 / cnn | train | 0 | 90 |
| CIFAR100 / cnn | eval | 0 | 30 |
| SVHN / cnn | train | 0 | 120 |
| SVHN / cnn | eval | 0 | 30 |

以上含启动、30 步训练、三次完整 test 及 checkpoint IO；仅为保守资源参考，不设性能质量阈值。数据 Study 若三个 CNN 同组，参考 launcher 墙钟 150s；模型 Study 三种模型分组，参考 525s。真实成本写入报告，不能用历史 600-step 总时间简单按比例外推。

```powershell
python -m rpipe make studies/support_data_smoke --num-gpus 1 --init-gpu 0
python -m rpipe launch studies/support_data_smoke --num-gpus 1 --init-gpu 0 --console shared
python -m rpipe report studies/support_data_smoke
```

## 4. 验收标准与交付

核对官方数据完整性、split 张数、NCHW / dtype / 像素范围及完整标签范围；初始参数哈希与 latest 参数不同、所有参数与 Loss 有限、训练确有 30 个 optimizer step。核对 latest / best step 与 scheduler 进度、best 选择口径；独立 eval 的 sibling train ID / checkpoint 来源及 Accuracy / Loss 与选中 best 对齐。全部 Run succeeded、Study complete、NUMBERS 和嵌图报告同源，记录每条预估和实测成本。

检查旧 Study config / result / tracker / log 内容哈希与 checkpoint size / mtime 未变。临时验证证据放 `.tmp/support-smoke-20261003/`；配置、PLAN、报告与图留在本目录。研究边界沿用当前文档。本轮不追加长预算实验。


## 5. 实际启动前清单

已核对 6 个 Run 的 content ID、seed=0、num_steps=30、没有旧 result。所有真实 split 的样本数、标签范围及类数、test batch 形状、像素范围、初始参数有限性通过。GPU 当前空闲 22.30 GiB。

| wait | 条件 | mode | seed | Run ID | 内置预估（s） | 内置显存估计（GiB） |
|---:|---|---|---:|---|---:|---:|
| 1 | FashionMNIST / cnn | train | 0 | c0781b98329e2173 | 9 | 0.74 |
| 1 | CIFAR100 / cnn | train | 0 | cb5270edd40d52f2 | 9 | 1.06 |
| 1 | SVHN / cnn | train | 0 | bdb9054f3029855c | 16 | 1.06 |
| 2 | FashionMNIST / cnn | eval | 0 | 311740c40612d085 | 5 | 0.50 |
| 2 | CIFAR100 / cnn | eval | 0 | f38b8e8751159448 | 5 | 0.50 |
| 2 | SVHN / cnn | eval | 0 | 177b25ca3b231bdd | 8 | 0.50 |

实际 make 为 2 wait，内置墙钟预估 24s（各组 max 后相加）。三个 CNN 的 train 同一组、eval 同一组；实际结果与成本见报告。


## 6. 执行收口

2026-10-03 已完成 6/6 Run；每条 train 到 30 optimizer step，无训练错误、重试或 resume。checkpoint 来源与 Loss 核对通过，严格 Accuracy 结果和实测成本见 [STUDY_REPORT](STUDY_REPORT.md)。新配方使用当前工作区源码，本地 source snapshot 与 manifest 留 `.tmp/support-smoke-20261003/`。
