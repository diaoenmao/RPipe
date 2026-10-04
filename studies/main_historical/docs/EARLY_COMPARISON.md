# MNIST linear 早期独立核对

## 一、当前结论

2026-10-04。仅核对正在运行的 MNIST / linear 四个 seed。首 600 步的 seed 0 train/test 指标与本机 600-step 原代码/当前链探针一致，静态重建输入前缀与探针实际输入哈希一致。四条运行目前仍在训练，不能据此宣称完整历史曲线复现。

## 二、首 600 步与连续性

读取各 Run 已完整换行的 scalars.jsonl 和 run.log；忽略未提交末行，不读取正在写入的 checkpoint 或分件。四条记录均只有一次 start，未发现 resume/error/retry，test 点保持 step 200 的连续前缀。冻结源码与 SOURCE_MANIFEST.json 一致。

首 200 / 400 / 600 步的 seed 0 train/test Loss、Accuracy 与当前链探针原值完全相同；和原代码探针之间只保留已经存在的浮点均值累加微差。

从原始 MNIST IDX 只读重建 seed 0 前 150000 个采样索引、归一化图片与标签，三个 SHA256 均匹配探针观测值。live workers 未记录输入哈希，因此这是固定数据与配置的静态重建支持，不能写成 live 输入字节已经直接观测。

## 三、四 seed 同步数与图片估读

按相同 optimizer step 对齐四个 seed，Accuracy 为百分制；std 为跨 seed population std（ddof=0）。图片数字来自 REFERENCE_CURVES.json，MNIST 的标称竖直读取误差为 ±0.05 个百分点。

| **槽** | **step** | **n** | **本轮 mean (%)** | **跨 seed std (pp)** | **PNG 估读 (%)** | **绝对差 (pp)** |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 200 | 4 | 91.417501 | 0.122958 | 91.48 | 0.062499 |
| 10 | 2200 | 4 | 92.320001 | 0.182346 | 92.30 | 0.020001 |
| 25 | 5200 | 4 | 92.542502 | 0.062600 | 92.49 | 0.052502 |
| 50 | 10200 | 4 | 92.567502 | 0.051660 | 92.52 | 0.047502 |

槽 50 / step 10200 的四 seed mean 为 92.567502%，与固定 PNG 估读 92.52% 相差 0.047502 个百分点。该锚点只提供早期相近程度；本次不执行或改写完整验收门。

槽 0 / 10 / 25 按既定合同仅描述，不进入终验。完整七个锚点、step80000 末点、末 50 点稳定性、全部模型次序和独立 eval 尚未满足核对条件。

## 四、证据与边界

逐 Run 的完整记录数量、读取快照 SHA256、首段差值、静态输入哈希、全部四 seed 同步数均值和检查结果见 [EARLY_COMPARISON.json](EARLY_COMPARISON.json)。JSON 中 complete=false、final_gates_applied=false、historical_reproduction_passed=null，保留运行中的事实。该文件是本次读取快照，不跟随 live 数据自动更新。
