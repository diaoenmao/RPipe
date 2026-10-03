# 历史图候选配方审计

2026-10-03。两张README图最后更新的提交是`4ccb28d0496110253e9f8e3f3df658853f07996b`；源码审计只定位候选配方，不把它当成已找回的原运行记录。本阶段未启动80000-step长实验，未更改当前模型以适配历史图。

## 配方差异

| 项 | 图最后更新提交 | 当前main `98648f3` |
|---|---|---|
| 训练 / 评测 | 80000 steps / 每200 steps | 60 steps / 每30 steps |
| seed | 0、1、2、3 | 0 |
| train / test batch | 250 / 250 | 250 / 1000 |
| SGD lr / momentum / wd | 0.01 / 0.9 / 0.0005，Nesterov | 0.1 / 0.9 / 0.0005，Nesterov |
| cosine T_max | 80000 | 60 |
| 梯度裁剪 | 每次optimizer.step前max_norm=1 | 无 |
| MNIST归一化 | mean0.1307 / std0.3081常量 | 完整train的原Stats |
| CIFAR归一化 | mean(0.4914,0.4822,0.4465) / std(0.2023,0.1994,0.2010) | 完整train的原Stats，std约(0.2470,0.2435,0.2616) |
| CIFAR随机增强 | torchvision在CPU逐样本flip / reflect crop | Kornia在模型前按batch处理 |
| CNN | 4层BatchNorm2d | 无BatchNorm2d |
| best指标 | test Accuracy越大越好，但比较有下述缺陷 | test Loss全程最小 |

旧代码的Epoch标签是评测段序号，80000/200=400；对MNIST，这并非完整数据遍历400轮。原`process.py`聚合4个seed的训练中test history；图片缺少原始配置、历史权重和逐点数字，无法据此证明实际图就是上述配方所产出。

## 模型探针

执行导出的旧模型代码与当前Factory；seed0、固定合成CPU输入，当前模型不附加归一化，避免混入已经单列的数据差异。仅将旧输出层的`linear`字段名映射为当前`output_proj`进行参数核对，不改变运算。

| 模型 | MNIST / CIFAR10初始化参数与buffer | 固定输入训练模式logits | 结论 |
|---|---|---|---|
| linear | 相同 | 差值0 / 0 | 核心结构可复用 |
| mlp | 相同 | 差值0 / 0 | 核心结构可复用 |
| cnn | 字段与buffer不同 | 最大差0.64609 / 0.60425 | 旧模型4层BN，当前0层，不能只扩大步数 |
| resnet18 | 相同 | 差值0 / 0 | 核心结构可复用；两边均17层BN |

这是初始化/forward探针，不是完整训练验收。原值见 [HISTORICAL_AUDIT.json](HISTORICAL_AUDIT.json)，脚本位于`.tmp/main-reproduction-20261003/historical_probe.py`。

## 旧原代码的best回退缺陷

旧`metric/metric.py::Metric.compare`不论是否改善都更新`self.best`，因此实际比较基准是上一轮，而非全程最佳。对Accuracy序列95→90→92，旧代码的决定为保存→不保存→保存，最终best变成92%，覆盖此前95%的权重。原训练入口在`logger.compare('test')`为真时确实覆盖best目录，因此不是孤立的辅助方法问题。

已直接执行两个导出提交的compare方法，并与当前Rpipe选择逻辑核对：当前main `98648f3`与Rpipe都保留95%的首个checkpoint，旧图提交选择第三个checkpoint。该缺陷属于归档源码；本阶段不修改归档，不把它列为当前工作树待修bug，也不让当前正确的全程best逻辑退回旧行为。

这会影响历史独立test的权重来源；训练中的完整曲线不因best目录选择而直接改变。后续若选择历史图路线，需明确对照曲线与独立test的不同来源，不能把“旧代码best”写成“全程最大Accuracy”。

## 下一阶段的边界

历史路线需先明确最终对照对象，再验证BN、CPU增强/统计、裁剪、采样和checkpoint语义，最后规划长实验。当前source路线已完成60-step确定性对照；该结果不能替代历史曲线复现。原依赖固定为torch2.0.1等旧版本，当前RTX5090/Python3.13环境并未重建这些版本，候选配方与历史实际环境仍有证据缺口。

## 旧运算接入当前训练循环的短对照

2026-10-03，临时DataRegistry / ModelRegistry builder直接复用归档dataset与model：样本dict转tuple，模型`f(x)`返回logits。归档源码、生产模型及Factory均未修改；旧归一化在dataset内完成，模型前不重复Normalize。使用复制并核对SHA256的16个原始数据文件，旧dataset自行生成pickle格式缓存，没有拿当前main的torch格式缓存冒充旧缓存。

两边seed0、train/test batch250、SGD lr0.01/momentum0.9/Nesterov/wd0.0005、每次更新前clip_grad_norm1、cosine T_max80000；同用CUDA确定性控制。每条训练仅4个optimizer steps，在step2/4各评测完整10000张test。归档采样器仍按80000步生成长度预算，当前短预算的实际1000个样本是相同前缀。

| 数据 | 模型 | step2/4参数与buffer最大差 | 实际采样与增强后输入 | 数值门 |
|---|---|---:|---|---|
| MNIST | linear / mlp / cnn / resnet18 | 各0 | 各一致 | 4/4通过 |
| CIFAR10 | linear / mlp / cnn / resnet18 | 各0 | 各一致 | 4/4通过 |

全部初始化参数、CPU/CUDA RNG、scheduler及整数buffer一致；每边每组合训练1000样本、测试20000样本，输入/标签哈希一致。训练Loss/Accuracy差值均0，test Loss最大差8.88e-16、test Accuracy最大差3.55e-15个百分点，10000张test的正确样本数一致。进程内计时36.841s，包含数据复制与对照执行，不含启动导入。原始逐格数据见 [HISTORICAL_BRIDGE.json](HISTORICAL_BRIDGE.json)，复制文件哈希见 [HISTORICAL_BRIDGE_DATA_MANIFEST.json](HISTORICAL_BRIDGE_DATA_MANIFEST.json)。

执行命令`python .tmp/main-reproduction-20261003/historical_bridge.py`；脚本哈希固定在JSON，临时目录`.tmp/main-reproduction-20261003/historical-bridge-7ec0aaaec0e1/`保存日志、原代码step2/4快照和当前checkpoint/tracker。当前step2参数比较在运行中完成，临时latest后续会被step4替换，未给全部当前step2参数额外留副本。保护复核：91个当前源文件、2402个可读旧Study文件不变；旧提交36个、当前main40个归档文件逐字节不变。目录不可枚举的旧CIFAR缓存不在核对范围，本轮未访问。

此探针证明已有Registry和原生训练循环能承载旧BN、CPU增强及裁剪，无需先给生产模型加兼容开关。它没有执行80000-step/eval200/4-seed，缩短的eval间隔会改变后续随机数消费，因此不是历史完整运行的前4步证据；也没有复现旧best缺陷或验证独立eval。历史实际运行配置、依赖环境及完整曲线仍未找回，最终对照选择仍待明确。

## Git历史证据搜索与候选工作量

2026-10-03，检查本地13个可达ref、150个提交；仓库非shallow，main祖先91个，图最后更新提交的祖先39个。旧祖先历史曾跟踪61个路径，按序列化结果/权重/表格后缀及output/data目录搜索，候选仅`src/config.yml`，没有找到原始指标或checkpoint路径；所有本地可达历史同样没有pt/pth/pickle/numpy/Excel/CSV或旧output目录的产物路径。这是本地Git路径搜索，不证明远端其他分支、未提交磁盘文件或外部存储没有原记录。完整范围和路径见 [HISTORICAL_EVIDENCE_SEARCH.json](HISTORICAL_EVIDENCE_SEARCH.json)。

两张PNG都嵌入`Software: Matplotlib version3.7.1`、约300 DPI；该提交requirements固定`matplotlib==3.7.0`。因此依赖文件不能作为实际绘图环境的完整证明。PNG只标识绘图软件，不能由此推断训练使用的Torch版本或硬件。

在新临时目录仅执行归档make.py，指定4 seeds、单GPU、round1，生成32条train与32条独立test命令，各自32个wait；**未执行生成命令**。原make.sh的4 seeds和process.py的4-seed聚合相互印证候选矩阵，但依然不是实际历史运行配置。命令草稿与逐项工作量保存在上述JSON；临时目录`.tmp/main-reproduction-20261003/provenance-search-7bb294786394/`。

| 完整候选工作量 | 一套实现 | 原代码与Rpipe两套 |
|---|---:|---:|
| train / 独立test任务 | 32 / 32 | 64 / 64 |
| optimizer更新 | 2,560,000 | 5,120,000 |
| train样本处理 | 640,000,000 | 1,280,000,000 |
| 曲线test样本处理 | 128,000,000 | 256,000,000 |
| 独立test样本处理 | 320,000 | 640,000 |

数量由32组合×80000steps×250batch及32组合×400评测×10000test直接计算，不用4-step探针的混合计时猜长实验时长。若用户选择历史图路线，下一阶段先提出200-step/eval200的真实采样前缀对照与分项计时，再据实际证据制定完整实验预算和曲线验收；该计划仍待目标选择，未启动实验。即使候选长实验运行成功，也须区分候选配方重跑、跨实现数值对齐和历史图片复现三个结论。

## 后续用户决定与前缀探针

2026-10-03，用户随后选择历史超参路线，并明确本轮只做到探针。200-step/eval200、seed0的8格对照及独立存档复核已8/8通过，详见[探针报告](HISTORICAL_PREFIX_PROBE.md)。这补齐真实首个评测段的数值证据；上方“待选择”是审计当时状态，不再是当前阻塞。未执行80000-step/4-seed长实验，没有因此证明历史图来源或曲线复现，探针完成后停止。
