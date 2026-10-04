# main 原图复现独立终审

## 一、摘要

2026-10-04。独立审查通过，没有发现阻断问题。[机器可读审计](FINAL_REFERENCE_AUDIT.json)保留完整文件哈希检查与已完成结果的核对范围。本审查仅读取已有源码、Git blob、清单和结果，并新增这两份报告；没有重跑 compare、合成 fixture、训练或 GPU 推理，没有修改冻结源、配置、门限或原始报告。

在预先固定的原图估读合同下，本机四 seed × 八组合的完整 80000-step 训练均值曲线较好复现 main README 目标。实际矩阵为 32 个 train + 32 个 matching-best eval，64 个不同 Run 全部 succeeded；八组每组五个固定数值门与两个数据集模型排序门全部通过。独立 eval 用于核对自身 best 来源，没有替代训练曲线终点或四 seed 逐步均值。

现代 main 60-step 八格的参数及状态精确对照另行成立，使用其声明的实测 Stats profile 和确定性条件；不将其视为历史 80k 的替代证据。

## 二、目标来源与冻结绑定

原 main 固定为 `98648f3a5c7db7dccf3ca806410d5b6fdee9484c`，历史候选配方固定为 `4ccb28d0496110253e9f8e3f3df658853f07996b`，当前 dev/HEAD 为 `8bccbac321d4c3ac1ea9892a5e774c114e0298c6`。直接读取两提交与本地 PNG 字节，确认 main README 引用这两张图片，图片 Git blob 与 SHA256 均一致。`4ccb28d` 是候选历史配方来源，原 PNG 没有原始运行元数据。

| **原图** | **main / 历史 Git blob** | **main / 历史 / 本地 SHA256** | **与数字化参考绑定** |
| --- | --- | --- | --- |
| MNIST | `39fc4cd04e63a72c2a458aba436b5e1f8200a613` | `58744e8721035ad523c82e7019f395f41b2b5acad3097ecca29c0c4ccb48fa26` | 一致 |
| CIFAR10 | `61a8e92ff461176c2fe2150e08d45802ba2b57d9` | `0a892f04c1c044b43cf7db2a0386b8e7c7dc07bc38225e0e8c2878a8eb925f3a` | 一致 |

[REFERENCE_CURVES.json](REFERENCE_CURVES.json)与 [TARGET.md](TARGET.md)声明图像估读口径，[PLAN.md](PLAN.md)在判断长实验结果前采用该合同。比较结果中的 reference SHA、compare SHA、source SHA 及 acceptance_contract 均与当前冻结文件精确一致。

| **证据** | **本次直接计算的 SHA256** |
| --- | --- |
| [REFERENCE_CURVES.json](REFERENCE_CURVES.json) | `4f8ec3a2e0e2d39a62fa71f2563f00a454ab55e619ee2a817548fe520f896d11` |
| [compare.py](../compare.py) | `89ccfee6b97242ea0d6c41db368b03534e62d302fed9b1e042164754cba9489b` |
| [COMPARISON.json](COMPARISON.json) | `a9f7dd5e8da467bcb7acbe8968624229f03891f31803abae44338e95471f2dd4` |
| [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) | `744c3ded41a0f3ca69b798bc43f18c5b64d918d5e482f5778a4ecb31fd58925a` |
| [TARGET.md](TARGET.md) | `3e783bd522cd48bdc00d01ce8a20dbe2a02dc6c642e75d3c8cca0d82b0b546a5` |
| [PLAN.md](PLAN.md) | `adb09eed0ee7a602766de71b80e4eff4a4b45b37ad05d48316f596946d120f27` |
| [EXECUTION.json](EXECUTION.json) | `5b100bdbd67a5bc5721bda5d464ef77b8809f7bf04f72dd3027454d8c2691f07` |
| [CURRENT_DEVICE_RESULT.json](../../main_reproduction/docs/CURRENT_DEVICE_RESULT.json) | `b9bfd62caa9908c76ccba3a4fa359040e8630ff0d7da29c0aebe9421b9a32f56` |
| [CURRENT_DEVICE_CPU_RELOAD_AUDIT.json](../../main_reproduction/docs/CURRENT_DEVICE_CPU_RELOAD_AUDIT.json) | `f265f9db4a4ee123e50ae3d66cd5631c48e51503552679c7c69bb03e3caca3f7` |

重新逐文件计算 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) 的 99 个 source 文件与 65 个 index/config 文件：99/99、65/65 一致，无缺失、无差异。JSON 保留各文件预期与实际 SHA。新增报告没有加入或改写这份冻结清单。

## 三、完整结果与比较逻辑审查

读取本次 [COMPARISON.json](COMPARISON.json)：`partial_mode=false`，`complete=true`，`final_gates_applied=true`，`passed=true`，`errors=[]`，`missing_runs=[]`。64 个不同 Run 精确覆盖两数据集、四模型、train/eval 和 seed0–3。

32 个 train 的 test/Accuracy history 都有 400 点，optimizer step 精确为 200、400、…、80000；latest checkpoint 与 tracker progress 都到 80000，history 与 curve 一致，训练没有 resume 事件。八组全部每步四 seed 齐全，按同 optimizer step 计算四 seed mean 与 population std（ddof=0）。32 个 eval 各有一个完整观测和一个 resume 事件，实际来源为同因素同 seed 的 train best，step 和 Accuracy 与该 best 一致。以上 checkpoint/history 核验引用已执行比较程序的记录，本独立审查没有再次加载全部 `.pt`。

[compare.py](../compare.py)在缺 Run、重复因素、非连续曲线、训练 resume、错误 best 来源或其他完整性错误时禁止最终门；末尾来源错误会清除此前已应用的门。partial 模式不会得出完整终验通过。每组五门全部通过后，还要求两个数据集各自满足 `resnet18 > cnn > mlp > linear` 的末 50 点均值顺序。

图横轴是从零开始的评测槽，槽 j 映射 optimizer step `(j + 1) × 200`。固定七锚点为 50、150、200、250、300、350、399，末窗口为 350–399 共 50 点。差值和标准差单位均为百分点：

| **固定门** | **MNIST 上限** | **CIFAR10 上限** |
| --- | ---: | ---: |
| 末点绝对误差 | 0.25 | 1.00 |
| 末 50 点均值绝对误差 | 0.25 | 1.00 |
| 七锚点 MAE | 0.25 | 1.00 |
| 七锚点最大误差 | 0.60 | 2.00 |
| 末 50 点四 seed 均值的时间 population std | 0.15 | 0.50 |

下表显示已有真实长训练结果，展示时保留六位小数，JSON 保留原值。末段时间 std 与单个 step 的跨 seed std 含义不同。

| **数据集** | **模型** | **step80000 mean %** | **末 50 点 mean %** | **末点误差 pp** | **末段均值误差 pp** | **锚点 MAE pp** | **锚点最大误差 pp** | **末段时间 std pp** | **数值门** |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MNIST | linear | 92.675001 | 92.668651 | 0.005001 | 0.001349 | 0.009644 | 0.047502 | 0.012381 | 5/5 |
| MNIST | mlp | 98.150001 | 98.152351 | 0.009999 | 0.007649 | 0.014999 | 0.024999 | 0.003654 | 5/5 |
| MNIST | cnn | 99.412501 | 99.408701 | 0.002501 | 0.011299 | 0.016429 | 0.047499 | 0.006230 | 5/5 |
| MNIST | resnet18 | 99.567501 | 99.580751 | 0.022499 | 0.009249 | 0.018571 | 0.047499 | 0.006486 | 5/5 |
| CIFAR10 | linear | 40.370001 | 40.369401 | 0.020001 | 0.009401 | 0.037143 | 0.160001 | 0.052706 | 5/5 |
| CIFAR10 | mlp | 63.467501 | 63.473301 | 0.207501 | 0.233301 | 0.253573 | 0.517501 | 0.031577 | 5/5 |
| CIFAR10 | cnn | 89.025002 | 88.967202 | 0.015002 | 0.022798 | 0.277858 | 0.594998 | 0.063659 | 5/5 |
| CIFAR10 | resnet18 | 92.925001 | 92.922652 | 0.044999 | 0.037348 | 0.209286 | 0.882498 | 0.034898 | 5/5 |

[EXECUTION.json](EXECUTION.json)状态为 `succeeded`，完成 UTC 时间 `2026-10-04T01:12:29.005702+00:00`（北京时间 `2026-10-04T09:12:29.005702+08:00`）。根线程明确确认主工具句柄 37125 的 terminal exit_code=0；本独立审查没有重新取得该终端输出。compare 本身检查文件与数值合同，不单独证明真实进程退出；统一终报告应同时引用实际 worker/控制器执行审计。

## 四、此前合成 fixture 的审查范围

读取此前的[合成核验脚本](../../../.tmp/historical_target/check_compare.py)，SHA256 为 `0ea6ebfe474681498f7b5fedba4393d924afe3386089d773ad37b27fcb8de820`。脚本构造由参考点插值的合成 64 Run，断言完整通过、partial 不启用终门、训练 resume 使终验无效、最后一组错误 sibling-best 来源使所有门无效、固定末点阈值违反导致科学失败、缺一个 seed 禁止完整通过。保留的 fixture 最终 index 为缺 seed 负例的 63 Run；它不是实际 64 Run 矩阵。本轮仅静态阅读脚本和已存证据，没有重新执行，合成值也不作为训练测量。这些 `.tmp` 证据不会随 Git clone 提供。

## 五、独立的现代 main 60-step 证明

[现代 main 本机正式报告](../../main_reproduction/docs/CURRENT_DEVICE_RESULT.md)和[独立 CPU 重载审计](../../main_reproduction/docs/CURRENT_DEVICE_CPU_RELOAD_AUDIT.json)是另一项验收：seed0、八组合、60 optimizer steps、step30/60 共 16 段，参数最大差为 0，整数 buffer、SGD optimizer 状态、scheduler 与 Torch CPU/CUDA RNG 一致，实际输入和样本计数门通过；完整 test Loss 差为 0，train/test 段摘要最大 Loss 差为 6.661338147750939e-16。各自独立 eval 加载 own-best，均为 step60。

归档原 Stats(dim=1) 在完整 train 按顺序 batch250 实测，完整精度保存为 `native-data/<dataset>/stats.yaml`，当前 DataFactory 读取该 profile。配方使用 deterministic=true、benchmark=false、CUBLAS_WORKSPACE_CONFIG=:4096:8。本轮没有验证无 profile 时的默认常量、Factory 自动 Stats 重算或原默认 benchmark=true 非确定性，也没有通过该 seed0 短实验推断历史四 seed 长曲线。本进程 Torch allocator reserved 峰值 3630 MiB 不是设备总显存峰值；混合负载下墙钟时间不是算法 benchmark。

## 六、科学边界与完成记录

1. 历史目标是 raster PNG 均值线估读；原始逐点训练日志、实际 seed 集合及运行环境尚未找回，4ccb28d 是候选历史配方来源，不能声称恢复了原始运行身份。

2. 原图聚合源码计划四 seed，但遇缺文件会减少实际 seed 数；本轮严格四 seed 完整，不能据此证明原图实际也是四 seed。

3. 原图阴影 population std 未数字化与验收；本轮跨 seed std 是新实测值。末 50 点 temporal population std 衡量四 seed 均值曲线的时间波动，与跨 seed std 是不同统计量。

4. 固定门覆盖末点、末 50 点均值、七锚点 MAE/最大误差、末段时间波动及模型排序；没有对原始 400 点逐点一致进行验收。初始段及 MNIST ResNet 历史孤立下降没有纳入数值门。

5. PNG 竖直估读容差为 MNIST ±0.05、CIFAR10 ±0.20 个百分点，虚线空隙可引入约三槽横向偏移；这些是图像读取容差，不是统计置信区间。

6. 本机 original/native 600-step 八格前缀对照不能证明完整 80000-step 原码/native 参数逐位相同；本轮完整长训练使用 native 链并复用历史 dataset/model。

7. 现代 main 60-step seed0 八格数值对照是独立验收，限定于完整 train 按 batch250 实测的归档 Stats(dim=1) 精度 profile 和 deterministic 控制；不覆盖无 profile 默认常量、Factory 自动 Stats 重算、benchmark=true 默认非确定性或其他 seed。

8. compare.py 检查记录、完整轨迹、checkpoint 及 sibling-best 来源一致性和图像合同；它本身不提供真实进程 terminal exit 证据，也不逐字段验收完整长训 optimizer 最终状态或全部 train Loss。真实进程退出与数值源冻结应连同独立执行审计引用。

9. 本次独立审查读取已完成的 COMPARISON；其中 checkpoint/history 核验是已运行比较程序的结果，本审查未再加载全部 .pt 重算该比较，也没有重跑 compare、fixture、训练或 GPU 推理。

10. 原始 Run 资产和合成 fixture 位于被忽略的 runs/.tmp 本机目录，不随 Git clone 提供；正式报告和来源哈希保留可取得证据的出处，不能保证 clone 后所有原始文件仍可访问。

本审查未发现为当前固定合同补充 GPU 实验或扩大门限的明确理由。统一终报告应保留主控制器真实退出、64 worker 完成/来源审计、冻结源检查与这里的科学边界；历史 PNG 估读目标与现代 main 精确数值对照分别表述。
