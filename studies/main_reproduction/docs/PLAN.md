# Study Plan: main_reproduction

## 目标

固定旧 main `98648f3a5c7db7dccf3ca806410d5b6fdee9484c`，运行其原始 base 配方，在本机相同环境、真实数据和统计值下，与当前 RPipe 逐格比较。历史 main README 只有图，没有原始指标或 checkpoint；本轮可以验证固定源码在当前环境中的复现，历史图数值复现仍需原始证据。

## 配方与执行

- MNIST / CIFAR10 × linear / mlp / cnn / resnet18，seed 0；每套实现 8 train + 8 eval。
- 全量 train / test，batch 250 / 1000；60 optimizer steps，step_period=1；step30 / 60 完整 test。
- SGD lr=0.1、momentum=0.9、Nesterov、weight_decay=0.0005；cosine T_max=60、eta_min=0；不裁剪梯度。
- main 的原 Stats(dim=1) 按顺序、batch250 计算完整 train 统计；浮点值原精度写入本 Study stats.yaml，模型前 Kornia 增强与归一化一致。已有 Study 不改写。
- CIFAR10 训练 flip p=0.5 → crop32 / padding4 / reflect → Normalize；MNIST 与所有 test 仅 Normalize。
- Loss-best，候选只有 step30 / 60；独立 eval 加载对应 best。
- CUDA、cudnn benchmark=true、deterministic=false；单进程串行，保留其他 GPU 工作。各套先 8 train，再 8 eval。
- 原源码导出与依赖安装均在 `.tmp/main-reproduction-20261003/`；源文件不改写。启动时先导入 numpy，以适配本机 Windows OpenMP 环境。运行包装只在原 check 写完后复制 step30 / 60 存档并记录初始化，不改变训练计算。
- RPipe 沿用 make / launch / process，`--round 1 --console shared`。checkpoint 使用已有 percent 模式，只在 50% / 100% 多保留 step30 / 60 快照，同时保留 latest / best；与 main 的保存语义差异仅为额外证据副本。
- RPipe log_period=7，main 在每个 30-step 段的 1 / 8 / 15 / 22 / 29 步记录。记录位置不同不参与 best 判断；直接比较每个评测段的训练摘要和 step30 / 60 test。

## 验收

1. 原始数据文件复制哈希一致，全部 train / test 像素与标签顺序一致；train stats 为同一完整精度值。8 组模型结构、初始参数相同。
2. 两套各 16 次成功，全部 train 恰好 60 steps，test 只有 step30 / 60、每次 10000 张；无静默 stub、失败重试或恢复。若失败保留原日志，续跑另行标注。
3. 列 step30 / 60 train Loss、test Loss / Accuracy、best step、独立 eval Loss / Accuracy；列两套参数最大差值及 scheduler 一致性。指标数值门：Loss 绝对差 ≤1e-6，Accuracy 正确样本数严格相同；参数浮点比较 atol=1e-6 / rtol=1e-5，整数 buffer 严格相同。所有差异均保留实际值，超过门限不得宣称通过。
4. 独立 eval checkpoint 的参数与父 train best 相同，best step 与 Loss 选择一致。学习曲线使用新 progress 坐标；process / NUMBERS / STUDY_REPORT 与原始结果对应。
5. 报告环境、源码快照与 manifest、运行命令、错误 / retry / resume、各 Run 及整轮墙钟。执行成功与数值通过分别计数；不把 CPU 合成数据探针当作完整验收。

## 时长与证据

执行前按 make 的串行估计补入报告。原 main 的逐样本拼接 collate 与 TensorBoard IO 会增加耗时；耗时只用于记录，不用于宣称算法加速。依赖安装在临时目录，本机环境不等于 main 历史环境；不增加训练预算或新模型。

### 原配方分歧后的控制（2026-10-03）

原默认配方执行完成后，完整数值门4/8通过；在原代码自身的3条重复运行中也看到超过门限的数值分歧。为区分实现差异和非确定性，隔离目录额外运行两套完整60-step配方，统一 deterministic=true、cudnn.deterministic=true、benchmark=false、CUBLAS_WORKSPACE_CONFIG=:4096:8；其余条件与数值门不变。该控制不替代原默认配方，也不放宽原判定。当前源码与历史README图的来源差异单独记录在报告，80000-step候选历史配方未执行。

### make 初始估计（2026-10-03，执行前）

单进程串行16个wait；估计仅为调度参考。

| data | model | mode | run | 估计秒 |
|---|---|---|---|---:|
| MNIST | linear | train | `68161cb21047d108` | 7 |
| MNIST | mlp | train | `72a8a161b6aa3b5c` | 7 |
| MNIST | cnn | train | `37c734f1fa7b1d3a` | 9 |
| MNIST | resnet18 | train | `ea91b2cd520b3aa3` | 13 |
| CIFAR10 | linear | train | `8ad44b7d5c4eb2a7` | 7 |
| CIFAR10 | mlp | train | `7bbdc6c4f22c59b4` | 7 |
| CIFAR10 | cnn | train | `7b943d4fb96d2514` | 9 |
| CIFAR10 | resnet18 | train | `92f28741143d768e` | 13 |
| MNIST | linear | eval | `280e512ddbe67ad1` | 4 |
| MNIST | mlp | eval | `e504d1f31fc48e57` | 4 |
| MNIST | cnn | eval | `bc0d5be09075a93a` | 5 |
| MNIST | resnet18 | eval | `37708b22766853a5` | 6 |
| CIFAR10 | linear | eval | `388aaf49b65a8433` | 4 |
| CIFAR10 | mlp | eval | `e44a4ffb1cc04d93` | 4 |
| CIFAR10 | cnn | eval | `a272fa4cdb665561` | 5 |
| CIFAR10 | resnet18 | eval | `cf823481ff5f542b` | 6 |

串行估计合计110秒。原代码缺乏本次collate / IO实测估计，不据此承诺其耗时。

### 新环境恢复本轮归一化统计

shared不入Git。首次make前，按已存档的原Stats值写入以下文件；make会准备完整原始数据。不要用rpipe data重新覆盖成本机profile的另一种统计。

```python
import json
from pathlib import Path
import yaml
study = Path("studies/main_reproduction")
for row in json.loads((study / "docs/DATA_PARITY.json").read_text(encoding="utf-8")):
    folder = study / "shared/data" / row["data"].lower()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "stats.yaml").write_text(yaml.safe_dump({"mean": row["mean"], "std": row["std"]}), encoding="utf-8")
```

统计来源已在DATA_PARITY和报告固定；这是手工恢复步骤，不是新增自动环境留档功能。

### 历史候选配方短接入对照（2026-10-03，执行前）

在新的临时目录运行旧提交`4ccb28d`的真实数据、模型和train/test函数；通过现有DataRegistry / ModelRegistry注册临时适配器，再运行当前原生TrainAlgorithm。归档源码和当前生产模型不改动，不新增长期配置。旧dict样本转为tuple，旧模型的`f(x)`输出logits；归一化与CPU增强仍由旧dataset执行，Factory不重复附加Normalize。

MNIST / CIFAR10 × linear / mlp / cnn / resnet18，seed0、batch250/test250、SGD lr0.01/momentum0.9/Nesterov/wd0.0005、梯度裁剪1、cosine T_max80000。两边采用相同CUDA确定性控制。仅执行4次optimizer更新，评测间隔临时缩为2，每次完整10000张test；原采样器保留80000-step预算，当前短预算仅取相同初始采样前缀。这不是完整历史配方执行，也不证明历史图片结果。

验收初始化、实际1000个训练索引及增强后输入哈希、step2/4的全部参数与buffer、scheduler、train/test指标。参数门沿用atol1e-6/rtol1e-5，整数buffer严格一致；Loss绝对差≤1e-6，Accuracy正确样本数严格一致。归档旧best比较缺陷仍单独保留，本探针不将其当成当前正确best选择的验收要求。保存逐格证据和脚本哈希，既有Study与原始对照证据保留。

### B-018之后的当前源码完整复核（2026-10-03，执行前）

收口审计发现当前完整GPU证据早于B-018共享eval入口改动。以新的临时Study/version执行8 train + 8 eval，60-step/eval30、eval_num_steps=-1、真实完整数据和已固定Stats、相同CUDA确定性控制；与既有未修改的原main确定性step30/60及独立eval存档比较，沿用原门限、检查best/checkpoint来源与无retry/resume。此前Run与报告原值保留，原代码存档不重复执行。这只补齐当前源码验证，不改变默认配方4/8判定或历史图路线范围。

### 用户确认的历史超参前缀探针（2026-10-03，执行前）

用户先选择 main README 历史曲线路线，随后明确本轮**只做到探针**，并要求处理实际开放缺陷。本轮不执行 80000-step 长实验、不展开 4-seed 矩阵、不把探针判为历史图复现。

- 沿用图最后更新提交 `4ccb28d` 的候选模型、真实完整数据、CPU增强/常量统计、batch250/test250、SGD lr0.01/momentum0.9/Nesterov/wd0.0005、clip1。CNN仍含旧BN；归档源码不改。
- MNIST / CIFAR10 × linear / mlp / cnn / resnet18，seed0；每边只执行200次optimizer更新，step200完整test一次（10000张）。原采样器保留80000-step长度预算，两边scheduler T_max=80000，不改成短训练退火。
- 原代码调用其train/test函数执行首个eval200段；当前链沿用已验证的临时DataRegistry / ModelRegistry适配器和原生TrainAlgorithm。两边统一CUDA确定性控制；这不是历史默认benchmark环境的重建。
- 比较初始化/RNG、实际50000个训练采样索引及增强后输入哈希、step200参数/整数buffer、scheduler、train/test Loss和正确样本数。参数门atol1e-6/rtol1e-5，整数buffer相同，Loss绝对差≤1e-6，Accuracy正确样本数相同；不得放宽门限以换通过。
- 单进程串行，每组合完成原代码与当前链后再下一组合，避免相互干扰计时。记录数据准备、原train、原test、当前run/eval hook/checkpoint分项；CUDA计时前后同步。分项边界与探针哈希/存档IO开销明确，不据此宣称算法加速。
- 脚本在`.tmp/main-reproduction-20261003/`，复制数据、checkpoint、tracker与日志在新的`historical-prefix-*`子目录；既有证据与生产源码不覆盖。计划、逐格小JSON及人写的结果报告留在本Study的docs。
- 失败时保留证据，先区分适配脚本问题、当前生产缺陷与归档旧行为。当前生产缺陷需最小反例、文档约定与回归验证后关闭；历史best回退缺陷不移植或冒充当前开放bug。本探针只有一个test点，不验证独立eval、best全程选择、长期收敛或历史图曲线。

完成条件：8格探针全部结束并报告实际通过/失败、分项时间与证据边界；存在当前生产缺陷时修复并在相关测试及新探针中复核。本轮到此停止，不自动启动后续长实验。

执行结果：8/8通过，独立存档核对8/8；未发现新的当前生产缺陷。本轮已按限定范围结束，见[探针报告](HISTORICAL_PREFIX_PROBE.md)。
