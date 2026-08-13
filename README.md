# RPipe

Research Pipeline — **可重复、可编排、可序列化的研究执行底座**（包名 `rpipe`）。

设计文档：

- [docs/CONCEPT.md](docs/CONCEPT.md)
- [docs/LAYOUT.md](docs/LAYOUT.md)

---

## 安装

```bash
pip install -e ".[dev]"
```

---

## 目录（摘要）

```
src/rpipe/{structure,flow,artifact}
examples/studies/…
examples/experiments/<exp>/{launch,grid,artifact}
```

---

## 快速运行

```bash
# 生成 / 刷新 Config，再跑 Flow
python examples/studies/mnist_seeds/run_study.py

# 或直接 launch 已有 Artifact Config
python examples/experiments/mnist_linear/launch/__init__.py --slugs seed_0

pytest
```

---

## Acknowledgements

[Federated Learning Platform](https://github.com/IBM/federated-learning-lib),
[EasyFL](https://github.com/EasyFL-AI/EasyFL/),
[FedLab](https://github.com/SMILELab-FL/FedLab),
[Flower](https://flower.dev/),
[NIID-Bench](https://github.com/Xtra-Computing/NIID-Bench),
[FedTorch](https://github.com/OPTML-Group/FedTorch)
