# Study Report: mnist_train_size

> Plan: [PLAN.md](PLAN.md)
> Date: 2026-09-19
> Recipe: native `custom_torch`；MNIST + linear；20 epoch；SGD + cosine；cuda。axes = `train_size` × `{train, eval}`。
> Location: Study `docs/`。数字读 `../process.json`。日志按 index 的 `log` 链到各 Run 的 `run.log`。

## 1. 怎么跑的（§3）

保留 `shared/data`，移走旧 `runs/scripts/index/process` 后，按 [STUDY_GUIDE.md](../../../docs/STUDY_GUIDE.md) 从空 Run 目录重跑。基底不写 `resume`。

```bash
python -m rpipe make studies/mnist_train_size --num-gpus 1 --init-gpu 0
python -m rpipe launch studies/mnist_train_size --num-gpus 1 --init-gpu 0 --console shared
```

机器：1× RTX 5090 D v2。make：`pack 2 waits: 9[linear×9], 9[linear×9] est wall 1m34s`。实测约 40s。18/18 `succeeded`；train 全部从 epoch 1 跑到 epoch 20，随后 eval 加载 sibling train 的 `best.pt`。eval accuracy mean 对齐同格子 train 的 `best_accuracy` mean。error / resume：无。

## 2. Conclusion

最终口径用 **eval**（跨 seed mean ± std；n=3）：

| train_size | accuracy mean | std | min | max |
|------------|---------------|-----|-----|-----|
| 500 | 0.8355 | 0.00173 | 0.8344 | 0.8375 |
| 2000 | 0.8842 | 0.00081 | 0.8835 | 0.8851 |
| 8000 | 0.9073 | 0.00042 | 0.9070 | 0.9078 |

同一预算下，测试准确率随训练集变大而升，seed 间离散变小。

## 3. Learning curves

[打开 learning_curves.png](./figures/learning_curves.png)

[![learning curves](./figures/learning_curves.png)](./figures/learning_curves.png)

## 4. Runs

点表格里的 **run.log** 打开该次日志（源码视图用 Ctrl+点击；预览里直接点）。结论仍按 Experiment。

| factors | seed | id | accuracy | log |
|---------|------|----|----------|-----|
| train_size=500 train | 0 | `8176170127f91cf0` | 0.8328 | [run.log](../runs/8176170127f91cf0/assets/logs/run.log) |
| train_size=500 train | 1 | `fc81e18a5207a45d` | 0.8341 | [run.log](../runs/fc81e18a5207a45d/assets/logs/run.log) |
| train_size=500 train | 2 | `442977f37212ef68` | 0.8359 | [run.log](../runs/442977f37212ef68/assets/logs/run.log) |
| train_size=500 eval | 0 | `e957d8a364952d1f` | 0.8346 | [run.log](../runs/e957d8a364952d1f/assets/logs/run.log) |
| train_size=500 eval | 1 | `dff1613a7bc1fd5f` | 0.8344 | [run.log](../runs/dff1613a7bc1fd5f/assets/logs/run.log) |
| train_size=500 eval | 2 | `56bc9bd79216fbba` | 0.8375 | [run.log](../runs/56bc9bd79216fbba/assets/logs/run.log) |
| train_size=2000 train | 0 | `d48fd72e60c7f6ac` | 0.8818 | [run.log](../runs/d48fd72e60c7f6ac/assets/logs/run.log) |
| train_size=2000 train | 1 | `bb1c4bab027af782` | 0.8828 | [run.log](../runs/bb1c4bab027af782/assets/logs/run.log) |
| train_size=2000 train | 2 | `31c40cec0f6d9cd5` | 0.8846 | [run.log](../runs/31c40cec0f6d9cd5/assets/logs/run.log) |
| train_size=2000 eval | 0 | `478225eadad22061` | 0.8841 | [run.log](../runs/478225eadad22061/assets/logs/run.log) |
| train_size=2000 eval | 1 | `b42b4fd9d969a3d6` | 0.8835 | [run.log](../runs/b42b4fd9d969a3d6/assets/logs/run.log) |
| train_size=2000 eval | 2 | `02214314ba92307e` | 0.8851 | [run.log](../runs/02214314ba92307e/assets/logs/run.log) |
| train_size=8000 train | 0 | `ae8cf11290ed6b54` | 0.9065 | [run.log](../runs/ae8cf11290ed6b54/assets/logs/run.log) |
| train_size=8000 train | 1 | `8217ff232db48b12` | 0.9056 | [run.log](../runs/8217ff232db48b12/assets/logs/run.log) |
| train_size=8000 train | 2 | `778bd5b434d18762` | 0.9062 | [run.log](../runs/778bd5b434d18762/assets/logs/run.log) |
| train_size=8000 eval | 0 | `92abeedb98737917` | 0.9070 | [run.log](../runs/92abeedb98737917/assets/logs/run.log) |
| train_size=8000 eval | 1 | `9d9e6347dee60c86` | 0.9072 | [run.log](../runs/9d9e6347dee60c86/assets/logs/run.log) |
| train_size=8000 eval | 2 | `a83a100692ba2930` | 0.9078 | [run.log](../runs/a83a100692ba2930/assets/logs/run.log) |

## 5. Reproduce

同上 make / launch。已 succeeded 的默认跳过。
