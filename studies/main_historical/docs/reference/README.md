# 发布实测图与历史原图来源

## 一、2026-10-04 的发布图

仓库首页的 [MNIST 图](../../../../asset/MNIST_Accuracy_mean.png) 与 [CIFAR10 图](../../../../asset/CIFAR10_Accuracy_mean.png) 现在展示本次已完成的历史配方实测结果。绘图读取冻结 [COMPARISON.json](../COMPARISON.json) 的全部 32 条训练曲线，分别核对八个组合的四 seed 向量与已记录 mean / population std；32 条独立 own-best eval 的成功状态也作为完整性前置条件。

每模型完整保留 400 个观测点，横轴为 optimizer step200–80000、间隔200，每个点有 seeds0–3。纵轴为完整 test10000 张的 Accuracy (%)；线为四 seed arithmetic mean，阴影为 population std（ddof=0），不使用独立 best 值替代训练曲线终点。

没有平滑、插值、删点、绘图路径简化或裁剪标准差带。MNIST ResNet18 的中段下跌保留；mean±std 可超过100%，这不是实际 Accuracy 观测超过100%。旧 PNG 的阴影未用于新图，也未声称已取得旧图的实际 seed 集合。

完整400点 mean / std、四条源 Run ID、聚合重算偏差、原/新图SHA、绘图程序与输入SHA、NumPy/Matplotlib版本见 [PUBLISHED_FIGURES.json](PUBLISHED_FIGURES.json)。绘图程序为 [publish_figures.py](../../publish_figures.py)，仅用 NumPy/Matplotlib 和存储的 JSON，没有训练、模型推理或 GPU 操作。

## 二、逐字节保留的历史原图

| **历史记录的原路径** | **本次可交付归档** | **原 SHA-256** | **原 Git blob** |
|---|---|---|---|
| `asset/MNIST_Accuracy_mean.png` | [MNIST_Accuracy_mean_4ccb28d.png](MNIST_Accuracy_mean_4ccb28d.png) | `58744e8721035ad523c82e7019f395f41b2b5acad3097ecca29c0c4ccb48fa26` | `39fc4cd04e63a72c2a458aba436b5e1f8200a613` |
| `asset/CIFAR10_Accuracy_mean.png` | [CIFAR10_Accuracy_mean_4ccb28d.png](CIFAR10_Accuracy_mean_4ccb28d.png) | `0a892f04c1c044b43cf7db2a0386b8e7c7dc07bc38225e0e8c2878a8eb925f3a` | `61a8e92ff461176c2fe2150e08d45802ba2b57d9` |

两份归档分别为253403 / 293559 bytes、1849×1366 / 1849×1368 pixels；SHA-256与Git blob均核对，内容与图最后更新提交 `4ccb28d0496110253e9f8e3f3df658853f07996b` 的原图逐字节相同。main `98648f3` 中也保存这两份原图。它们与被忽略的 Git 导出目录分开，随正式 Study 一同交付。

[TARGET.md](../TARGET.md)、[REFERENCE_CURVES.json](../REFERENCE_CURVES.json)、[FINAL_REFERENCE_AUDIT.json](../FINAL_REFERENCE_AUDIT.json) 和先前 [HISTORICAL_PROVENANCE.json](../../../main_reproduction/docs/HISTORICAL_PROVENANCE.json) 中的 `source_path` / `original_images.path` / `asset_sha256` 保留当时的 root asset 路径与原哈希，没有将旧估读或旧审查改写成本次实测图。阅读这些旧路径时，以本节和机器JSON的 `historical_reference_mapping` 解析到原图归档；本次首页 root asset 已使用新的实测图。

## 三、复跑与发布后核验

在仓库根运行：

```powershell
# 从冻结完整 JSON 重画首页实测图；会更新发布图及本目录的新来源 JSON。
python -B studies/main_historical/publish_figures.py

# 只读验证完整点、聚合、输入/程序绑定、两张原图归档和两张实测图。
python -B studies/main_historical/publish_figures.py --verify
```

核验输出默认写 `.tmp/publish-figures/VERIFICATION.json`，不覆盖旧科学报告。发布入口复用已存在且原 SHA 一致的归档；归档若有不同字节即拒绝。尚未归档时先核对原 asset bytes，或从原 Git blob 恢复，然后才替换首页图。

冻结训练/准备/比较/CPU verifier 不从当前 root asset 像素取训练数据或门限；它们继续使用固定 Git blob 和 `REFERENCE_CURVES.json`。`compare.py` 支持独立输出，完成态工作区需要重核时可避免覆盖旧正式比较：

```powershell
python -B studies/main_historical/compare.py --output .tmp/historical-comparison-recheck
```

若旧临时图像提取程序或独立审计器只按 `asset/...` 读取像素，它将读到新实测图，不能继续把该路径视为历史原图。新增 `publish_figures.resolve_historical_reference(data)` 按冻结 reference 的原 SHA/Git blob 返回本目录归档，可供这类新核验包装使用；原提取参数、科学JSON、99个数值源与旧审核记录保持原样。

整体科学结论及原图估读边界见 [完整报告](../STUDY_REPORT.md)，复跑历史训练的专用入口见 [Study README](../../README.md)。
