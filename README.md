# RPipe

Research Pipeline — **可重复、可编排、可序列化的研究执行底座**（包名 `rpipe`）。

设计文档：

- [docs/CONCEPT.md](docs/CONCEPT.md) — 概念与边界
- [docs/LAYOUT.md](docs/LAYOUT.md) — 目录
- [docs/CODE_STRUCTURE.md](docs/CODE_STRUCTURE.md) — 库内两柱
- [docs/STUDY_GUIDE.md](docs/STUDY_GUIDE.md) — 怎么开一轮 Study

---

## 安装

```bash
pip install -e ".[dev]"
```

---

## 目录（摘要）

```
src/rpipe/{structure,flow}   # artifact 在 structure/artifact/
studies/<name>/{docs,shared,runs,study.yaml,experiment_config.yaml}
```

---

## 快速运行

```bash
# 薄 CLI：读 study.yaml → Config + index → Flow
python -m rpipe run studies/mnist_train_size

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
