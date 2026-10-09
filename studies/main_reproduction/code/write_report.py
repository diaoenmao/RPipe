from pathlib import Path
import json,shutil
s=Path('.tmp/main-reproduction-20261003');docs=Path('studies/main_reproduction/docs')
for name in ['repeat-comparison','historical-provenance','gpu-deterministic-parity']:
    shutil.copy2(s/f'{name}.json',docs/f'{name.upper().replace("-","_")}.json')
b=json.loads((s/'full-comparison.json').read_text(encoding='utf-8'))
env=json.loads((s/'environment.json').read_text(encoding='utf-8'))
execution=json.loads((s/'original-execution.json').read_text(encoding='utf-8'))
current=json.loads((s/'current-execution.json').read_text(encoding='utf-8'))
repeat=json.loads((s/'repeat-comparison.json').read_text(encoding='utf-8'))
lines=['# Study Report: main_reproduction\n\n','2026-10-03。原 main 与 RPipe 各 **16/16 执行成功**；原默认 CUDA 配方的完整数值门 **4/8 通过**，因此尚未完成严格数值复现。本文保留失败指标，不放宽执行前门限。\n\n',
'## 配方与证据\n\n',
'固定 main `98648f3a5c7db7dccf3ca806410d5b6fdee9484c`，本工作区 HEAD `d938874` 加未提交修复；精确源码 SHA-256 见 [SOURCE_MANIFEST](SOURCE_MANIFEST.json)。60 optimizer steps / step30、60 完整 test，batch250 / test1000，seed0，SGD lr0.1 / momentum0.9 / Nesterov / wd0.0005，cosine T_max60；Loss-best、独立 eval 加载 best。完整声明与执行前门限见 [PLAN](PLAN.md)。\n\n',
'32 个原始数据文件副本 SHA-256 一致；MNIST 60000/10000、CIFAR10 50000/10000 的全部像素和标签顺序一致。完整 train 按原 Stats(dim1)、顺序 batch250 重算，未使用六位小数替代原精度。8 个模型的初始化参数和初始化后 CPU RNG 一致，两个数据集的全部15000个采样索引一致；预检重建使用与 Flow 相同的 Factory / seed 设置。见 [DATA_PARITY](DATA_PARITY.json)、[PREFLIGHT_PARITY](PREFLIGHT_PARITY.json)。\n\n',
'原源码 40 个文件与 git archive 字节一致；包装仅记录初始化、复制 step30/60 存档。当前 checkpoint percent 仅增加同一时刻的证据快照。main 的段内记录时刻与 RPipe log_period7 不同，不改变评测、best 判断或预算。两套均单进程串行，每条首次执行；训练无恢复、无错误、无 launcher retry。独立 eval 的 resume 指向各自 best。\n\n',
'本机 Python3.13.9、torch2.11.0+cu128、torchvision0.26.0+cu128、Kornia0.8.3、NumPy2.3.5、RTX5090 D v2。原源码缺少的 evaluate / datasets / multiprocess / xxhash 仅安装在 .tmp；启动先导入 numpy，CPU threads2；原默认配方 benchmark=true / deterministic=false。其他约11.1GiB GPU负载保留。详细版本见 [ENVIRONMENT](ENVIRONMENT.json)。这不是历史依赖环境重建。\n\n',
'## 完整数值对照\n\n','两套 best 均选 step60。以下 Accuracy 为百分制；逐步 train/test Loss、参数差值、scheduler 状态与独立 eval 原值见 [FULL_COMPARISON](FULL_COMPARISON.json)，不从日志的四位显示值复算。\n\n','| 数据 | 模型 | main step30 Acc | RPipe step30 Acc | main step60 Acc | RPipe step60 Acc | 最大 test Loss 绝对差 | 最大参数/Buffer差 | 数值门 |\n|---|---|---:|---:|---:|---:|---:|---:|---|\n']
for r in b['rows']:
    a,z=r['steps'];lines.append(f"| {r['data']} | {r['model']} | {a['original']['test/Accuracy']:.2f} | {a['current']['test/Accuracy']:.2f} | {z['original']['test/Accuracy']:.2f} | {z['current']['test/Accuracy']:.2f} | {max(abs(x['delta']['test/Loss']) for x in r['steps']):.9g} | {max(x['parameter_max_delta'] for x in r['steps']):.9g} | {'通过' if r['passed'] else '未通过'} |\n")
lines+=['\n所有16个 step30/60 scheduler 状态一致；linear / mlp 的参数和指标差值为0。CNN / ResNet18 超过门限。CIFAR10 ResNet18 的 RPipe best 周期评测为46.95%，独立 eval 为46.97%，相差2个正确样本；原代码自己的独立 eval Loss 也未通过 ≤1e-6 的同权重门。原值和失败判定保留，这里不额外开展已取消的独立 eval 诊断项目。\n\n','![两套原默认配方的 test 对照](figures/original_comparison.png)\n\n','Rpipe 自动过程曲线按真实 optimizer step 绘制，test 坐标为30/60；每点 n=1、std=0仅描述单seed，不表示跨seed统计。见 [学习曲线](figures/learning_curves.png)、[NUMBERS](NUMBERS.md)。\n\n','## CUDA 差异控制实验\n\n','在隔离目录重复原 main 的3条训练，保持原默认 benchmark 配方。相同源码、数据和seed也产生超过门限的参数与指标差异；这证明原配方自身不保证逐位重现，不能仅据跨实现差异就认定为实现 bug。见 [REPEAT_COMPARISON](REPEAT_COMPARISON.json)。\n\n','| 原代码重复 | step | 第一次 Accuracy | 重复 Accuracy | 最大参数/Buffer差 |\n|---|---:|---:|---:|---:|\n']
for r in repeat:lines.append(f"| {r['data']} / {r['model']} | {r['step']} | {r['test_accuracy_first']:.2f} | {r['test_accuracy_repeat']:.2f} | {r['parameter_max_delta']:.9g} |\n")
lines+=['\n4-step 合成数据 GPU 对照开启确定性后8/8通过；这只提供定位证据，不能替代完整配方。完整60-step确定性控制目前运行中：仅两套同改 deterministic=true / cudnn.deterministic=true / benchmark=false / CUBLAS_WORKSPACE_CONFIG=:4096:8，数据、统计、预算、评测和数值门保持。控制不替换原默认配方的4/8结论。\n\n','## README 历史图的来源\n\n',
'逐字节核对发现，main README 的两张 Accuracy 图自 `4ccb28d0496110253e9f8e3f3df658853f07996b`（2024-01-08）以来未变，MNIST图展示约400个Epoch。该提交源码为 **80000 steps / eval200 / lr0.01 / batch250 / test250 / 4 seeds**，CNN含BatchNorm，CIFAR在CPU使用torchvision随机增强与std(0.2023,0.1994,0.2010)。这些条件与当前main60-step配方、无BN的CNN、Kornia增强和原Stats明显不同。\n\n',
'这能定位候选历史配方，不能证明这些PNG当时一定来自该配置：Git中没有原运行配置、原始指标和checkpoint。已存档旧源码与 [HISTORICAL_PROVENANCE](HISTORICAL_PROVENANCE.json)。最终对照选择已向用户说明；尚未启动80000-step长实验，也未更改当前模型以猜测历史实现。\n\n',
'## 时间与可重跑入口\n\n',f"原main 16个子进程墙钟合计 **{sum(r['elapsed_seconds'] for r in execution):.3f}s**；Rpipe launch外部计时 **{current['elapsed_seconds']:.3f}s**（包含父进程和聚合）。两者统计边界不同，且有共享GPU负载，不据此宣称算法加速。逐命令记录见 [ORIGINAL_EXECUTION](ORIGINAL_EXECUTION.json)、[CURRENT_EXECUTION](CURRENT_EXECUTION.json)。\n\n",
'```powershell\npython -m rpipe make studies/main_reproduction --round 1 --num-gpus 1 --init-gpu 0 --console shared\npython -m rpipe launch studies/main_reproduction --round 1 --num-gpus 1 --init-gpu 0 --console shared\npython -m rpipe report studies/main_reproduction\n```\n\n',
'这些命令默认复用已成功清单。独立从头复测需新version及重新make；shared原始数据和本轮精确Stats须按来源恢复，不能用profile命令覆盖成另一种统计。原源码、包装、诊断脚本及原始日志在 `.tmp/main-reproduction-20261003/`，正式数字证据在本目录；所有已有Study保留。当前工作区本地unit + integration c1/c2回归仍为282 passed / 3 deselected，本轮未改生产源码。\n']
(docs/'STUDY_REPORT.md').write_text(''.join(lines),encoding='utf-8')
print('Wrote baseline report with explicit failed numerical gate and historical evidence boundary')
