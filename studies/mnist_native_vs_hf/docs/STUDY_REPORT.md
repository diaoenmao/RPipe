# Study Report: mnist_native_vs_hf

> Plan: [PLAN.md](PLAN.md)
> Date: 2026-09-19
> Recipe: MNIST + linear；同一套键：SGD lr=0.1、cosine、`max_grad_norm: 0`、20 epoch、`progress_unit: epoch`。axes = `algorithm.source` × `train_size`。`system.device: cpu`。
> Location: Study `docs/`。数字读 `../process.json`。日志按 index 的 `log` 链到各 Run 的 `run.log`。

## 1. 怎么跑的（§3）

保留 `shared/data`，移走旧 `runs/scripts/index/process` 后，按 [STUDY_GUIDE.md](../../../docs/STUDY_GUIDE.md) 从空 Run 目录重跑。基底不写 `resume`。

```bash
python -m rpipe make studies/mnist_native_vs_hf
python -m rpipe launch studies/mnist_native_vs_hf --console shared
```

机器：RTX 5090 D v2 主机，但本 Study 全部使用 CPU。make：`pack 5 waits: 4[linear×4], 4[linear×4], 4[linear×4], 4[linear×4], 2[linear×2] est wall 5m20s | CPU`；18 个 job 均为 `device: cpu`，没有 `gpu` 字段，launch 标签为 `+ cpu`。实测约 2m31s。18/18 `succeeded`，全部从 epoch 1 跑到 epoch 20。error / resume：无。

## 2. Conclusion

最终口径用 **last accuracy** 跨 seed mean（n=3）。2026-09-19 的全新重跑复现了原结论：500 / 2000 上 native 与 HF **逐点相同**（accuracy、best_accuracy、train_loss）；8000 上 HF last accuracy mean 高约 **0.0008**（seed 标准差量级），train_loss 仍对齐。

| train_size | native acc mean | HF acc mean | Δ (HF − native) | native best | HF best |
|------------|-----------------|-------------|-----------------|-------------|---------|
| 500 | 0.8343 | 0.8343 | **0** | 0.8355 | 0.8355 |
| 2000 | 0.8831 | 0.8831 | **0** | 0.8842 | 0.8842 |
| 8000 | 0.9061 | 0.9069 | **+0.0008** | 0.9073 | 0.9076 |

排序一致：8000 > 2000 > 500。8000 差多半来自 Trainer 内部步进 / cosine 包一层 `_EpochScheduler`，不是 clip、也不是两套 shuffle。

`max_grad_norm` 是算法层超参：缺省 / `0` = 不裁。不写的话 HF 会用 Trainer 自带的 `1.0`，train_loss 会对不齐。

## 3. Learning curves

[打开 learning_curves.png](./figures/learning_curves.png)

[![learning curves](./figures/learning_curves.png)](./figures/learning_curves.png)

## 4. Runs

点表格里的 **run.log** 打开该次日志。结论仍按 Experiment。

| factors | seed | id | accuracy | best | log |
|---------|------|----|----------|------|-----|
| custom_torch train_size=500 | 0 | `7c48352c847b2096` | 0.8328 | 0.8346 | [run.log](../runs/7c48352c847b2096/assets/logs/run.log) |
| custom_torch train_size=500 | 1 | `5662540f720e8797` | 0.8341 | 0.8344 | [run.log](../runs/5662540f720e8797/assets/logs/run.log) |
| custom_torch train_size=500 | 2 | `5bb24f8a4b518994` | 0.8359 | 0.8375 | [run.log](../runs/5bb24f8a4b518994/assets/logs/run.log) |
| transformers_trainer train_size=500 | 0 | `6b3522139506555d` | 0.8328 | 0.8346 | [run.log](../runs/6b3522139506555d/assets/logs/run.log) |
| transformers_trainer train_size=500 | 1 | `7036783459b4dee3` | 0.8341 | 0.8344 | [run.log](../runs/7036783459b4dee3/assets/logs/run.log) |
| transformers_trainer train_size=500 | 2 | `e006258b46e5e54c` | 0.8359 | 0.8375 | [run.log](../runs/e006258b46e5e54c/assets/logs/run.log) |
| custom_torch train_size=2000 | 0 | `c96de76cbb910155` | 0.8818 | 0.8841 | [run.log](../runs/c96de76cbb910155/assets/logs/run.log) |
| custom_torch train_size=2000 | 1 | `7d8d11439fd02689` | 0.8828 | 0.8835 | [run.log](../runs/7d8d11439fd02689/assets/logs/run.log) |
| custom_torch train_size=2000 | 2 | `71a8ba2c73de2c4f` | 0.8846 | 0.8851 | [run.log](../runs/71a8ba2c73de2c4f/assets/logs/run.log) |
| transformers_trainer train_size=2000 | 0 | `529a5bf98413e51e` | 0.8818 | 0.8841 | [run.log](../runs/529a5bf98413e51e/assets/logs/run.log) |
| transformers_trainer train_size=2000 | 1 | `c5847d1f3f2da766` | 0.8828 | 0.8835 | [run.log](../runs/c5847d1f3f2da766/assets/logs/run.log) |
| transformers_trainer train_size=2000 | 2 | `6ab372aeaa9bd64a` | 0.8846 | 0.8851 | [run.log](../runs/6ab372aeaa9bd64a/assets/logs/run.log) |
| custom_torch train_size=8000 | 0 | `e971cf5220878f98` | 0.9065 | 0.9070 | [run.log](../runs/e971cf5220878f98/assets/logs/run.log) |
| custom_torch train_size=8000 | 1 | `a34e2f965d427746` | 0.9056 | 0.9072 | [run.log](../runs/a34e2f965d427746/assets/logs/run.log) |
| custom_torch train_size=8000 | 2 | `8de79ab58f66b2bb` | 0.9062 | 0.9078 | [run.log](../runs/8de79ab58f66b2bb/assets/logs/run.log) |
| transformers_trainer train_size=8000 | 0 | `021d59955431eda9` | 0.9067 | 0.9069 | [run.log](../runs/021d59955431eda9/assets/logs/run.log) |
| transformers_trainer train_size=8000 | 1 | `38b06b38804eb60e` | 0.9069 | 0.9082 | [run.log](../runs/38b06b38804eb60e/assets/logs/run.log) |
| transformers_trainer train_size=8000 | 2 | `71be2847ca254e8e` | 0.9071 | 0.9078 | [run.log](../runs/71be2847ca254e8e/assets/logs/run.log) |

## 5. Reproduce

同上 make / launch。已 succeeded 的默认跳过。需要 `pip install -e ".[hf]"`。
