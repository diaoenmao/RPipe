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
src/rpipe/{structure,flow}   # make 在 structure；cli 在 flow
studies/<name>/{docs,shared,runs,study.yaml,experiment_config.yaml}
```

---

## 快速运行

```bash
# 入口：study.yaml → make → 阶段链
python -m rpipe run studies/mnist_train_size

# 同一套 flow：写出调度脚本后按 GPU 与 round 并行
python -m rpipe make studies/<name> --num-gpus 1 --round 4
python -m rpipe launch studies/<name> --num-gpus 1 --round 4

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
