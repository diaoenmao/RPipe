# Study Plan: mnist_main_recipe

> 状态：已对照。4-seed last test acc mean **0.902**，与 main 旧图饱和约 0.90 同量级。

## 1. 研究问题

现在的 native 循环 + yaml extras，用 **main 当年的 MNIST_linear 超参**（不是 `mnist_train_size` 那套 20 epoch）再训一遍，test Accuracy 是否和旧图同一量级（饱和约 **0.90**）？

## 2. 变量轴

无研究因素。1 个 Experiment × seeds `0, 1, 2, 3`（对齐 main `process.py` 的 4 次复测）= **4 次 train Run**。

## 3. 固定条件（对 `src/module/hyper.py` + `config.yml`）

| 项 | 取值 |
|----|------|
| `data.name` | `MNIST`（全量 train，不切 `train_size`） |
| `batch_size` | `250` |
| `test_batch_ratio` | `4` |
| `num_steps` | `60` |
| `progress_unit` | `step` |
| `eval_period` | `30` |
| `checkpoint_period` | `30` |
| `optimizer` | `SGD`，`lr=0.1`，`momentum=0.9`，`nesterov`，`weight_decay=5e-4` |
| `max_grad_norm` | `0` |
| `scheduler` | `cosine`，`T_max` 跟 60 step |
| `best_metric` | `Loss`（min），对齐 main `best_metric_name: Loss` |
| `model.config.zero_bias` | `true`（对齐 main `init_param` 对 Linear bias 置零） |
| `system.device` | `cpu`（main 默认 cuda；数字同学级即可） |

## 4. 成功标准

- 4 Run `succeeded`
- 最后一次 test Accuracy 的 mean 落在旧图饱和区附近（约 0.88–0.92）
- 有学习曲线图

## 5. 刻意不做什么

- 不扫 `train_size` / 不切 HF
- 不对 bit-identical（cuda vs cpu、kornia Normalize vs torchvision）
- 不做 TensorBoard
