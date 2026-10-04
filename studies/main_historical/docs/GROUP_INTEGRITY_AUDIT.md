# 八组历史长曲线的独立 CPU 完整性汇总

## 一、摘要

汇总审查时间：2026-10-04T18:47:06.576253+08:00。原主控制器正式结束于2026-10-04 09:12:29 +08:00，EXECUTION.status=succeeded；32条train与32条独立eval的正式result全部succeeded。八组各四seed、每条400个完整test点，完整性与自身global Accuracy-best来源审查全部通过。

原三组正式JSON快照原样保留；剩余五组于2026-10-04 18:39:13 +08:00用portable入口独立完成CPU重载审查。原始artifact哈希再次核对；源前后99个Python/训练文件、65个index/config文件及16个原始数据文件均无差异。非线性六组未执行额外CPU模型推理，两个Linear组保留已有CPU回放证据。

八组共40项预定单组合诊断均在固定门限内；[整体COMPARISON](COMPARISON.json)已由主线程实际执行，complete=True、passed=True，包含两数据集四模型排序。本汇总只引用整体结果，不自行应用或放宽整体门。

## 二、400点与独立eval完整性

| **组合** | **每seed train / eval检查数** | **4×400同点曲线** | **自身best step：seeds0/1/2/3** | **完整性** |
|---|---|---|---|---|
| [MNIST / linear](MNIST_LINEAR_RESULT.md) | 24/24/24/24 / 19/19/19/19 | True | 46800 / 26800 / 33000 / 59000 | True |
| [MNIST / mlp](MNIST_MLP_RESULT.md) | 41/41/41/41 / 20/20/20/20 | True | 36200 / 54400 / 35800 / 13400 | True |
| [MNIST / cnn](MNIST_CNN_RESULT.md) | 41/41/41/41 / 20/20/20/20 | True | 40000 / 39400 / 52600 / 48600 | True |
| [MNIST / resnet18](MNIST_RESNET18_RESULT.md) | 41/41/41/41 / 20/20/20/20 | True | 74400 / 68800 / 43200 / 25000 | True |
| [CIFAR10 / linear](CIFAR_LINEAR_RESULT.md) | 41/41/41/41 / 20/20/20/20 | True | 73000 / 74600 / 72200 / 73400 | True |
| [CIFAR10 / mlp](CIFAR_MLP_RESULT.md) | 41/41/41/41 / 20/20/20/20 | True | 71000 / 74400 / 61200 / 67800 | True |
| [CIFAR10 / cnn](CIFAR_CNN_RESULT.md) | 41/41/41/41 / 20/20/20/20 | True | 79000 / 75600 / 73800 / 70000 | True |
| [CIFAR10 / resnet18](CIFAR_RESNET18_RESULT.md) | 41/41/41/41 / 20/20/20/20 | True | 78800 / 78200 / 69800 / 74800 | True |

训练step精确覆盖200–80000；latest step80000、tracker counter96000，400个test10000由固定batch250及40batch计数复算。每条train唯一fresh start、无resume；canonical latest/best、optimizer/scheduler、完整metric history与自身global Accuracy最大点已各自审查。独立eval只加载自身sibling-best、记录同一optimizer step和Accuracy，每次10000样本/40batch。早期报告不同检查数反映当时审计版本，未把后续检查倒写成旧审查。

## 三、固定门的真实偏差

| **组合** | **终点mean (%)** | **末50mean (%)** | **终点差 / tail差 (pp)** | **七锚点MAE / max (pp)** | **末50时间std (pp)** | **五诊断** |
|---|---:|---:|---|---|---:|---|
| MNIST / linear | 92.67500124 | 92.66865129 | 0.00500124 / 0.00134871 | 0.00964393 / 0.04750174 | 0.01238150 | True |
| MNIST / mlp | 98.15000143 | 98.15235143 | 0.00999857 / 0.00764857 | 0.01499860 / 0.02499865 | 0.00365409 | True |
| MNIST / cnn | 99.41250086 | 99.40870091 | 0.00250086 / 0.01129909 | 0.01642899 / 0.04749906 | 0.00622975 | True |
| MNIST / resnet18 | 99.56750078 | 99.58075070 | 0.02249922 / 0.00924930 | 0.01857146 / 0.04749911 | 0.00648556 | True |
| CIFAR10 / linear | 40.37000083 | 40.36940076 | 0.02000083 / 0.00940076 | 0.03714316 / 0.16000062 | 0.05270570 | True |
| CIFAR10 / mlp | 63.46750147 | 63.47330134 | 0.20750147 / 0.23330134 | 0.25357255 / 0.51750111 | 0.03157709 | True |
| CIFAR10 / cnn | 89.02500162 | 88.96720158 | 0.01500162 / 0.02279842 | 0.27785788 / 0.59499842 | 0.06365856 | True |
| CIFAR10 / resnet18 | 92.92500148 | 92.92265154 | 0.04499852 / 0.03734846 | 0.20928553 / 0.88249846 | 0.03489777 | True |

MNIST终点/tail/锚点MAE门为≤0.25pp、锚点max≤0.6pp、tail时间std≤0.15pp；CIFAR10对应≤1.0/1.0/1.0/2.0/0.5pp。所有值来自四seed同一步mean与population std，未用较高best替代训练终点，未更改门限。七锚点逐值和偏差见各组JSON。

## 四、来源、失败与证据边界

[汇总机器JSON](GROUP_INTEGRITY_AUDIT.json)保留各组哈希、再核对文件数/差异、每seed来源检查和实际门误差；[CPU审计入口](../verify_group.py)及[方法版本记录](GROUP_AUDIT_METHOD.md)说明不同审查版本。当前portable SHA为`2b9efcf86f2a7f1fe23c447be95f29cd38a8dd05ba1459954411c8e1068f9ce7`；本汇总临时程序SHA另存机器JSON，临时文件与原始runs/checkpoints/data不随Git clone提供。

此前CIFAR linear的eval记录控制器因WinError5写报告失败而实际exit1，四worker均exit0/succeeded；记录仅恢复、没有worker重跑，原失败和源码版本继续保留在[该组报告](CIFAR_LINEAR_RESULT.md)与[恢复记录](EARLY_EVAL_CIFAR10_LINEAR_EXECUTION.json)。其他已完成组不因这一记录错误被改写为worker失败。

非线性组只核对保存状态、history与正式eval的实际加载来源和结果，未声称额外CPU推理或套用Linear momentum形状映射。正式eval未保存在线模型终态哈希，权重来源由实际resume日志、canonical best校验和与冻结加载器绑定；正式训练也未逐样本记录输入字节或保存完整RNG，不声称重建每次未记录更新。

参考值为[历史PNG估读](REFERENCE_CURVES.json)，不是已恢复的历史原始metrics；图的实际seed数不可证明。完整四seed和固定门通过支持曲线相近复现，保留这层来源限制。
