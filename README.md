# RPipe

Research Pipeline — **可重复、可编排、可序列化的研究执行底座**。

设计文档：[docs/CONCEPT.md](docs/CONCEPT.md)（概念定稿后再写布局与代码结构文档）

历史交接：[docs/HANDOVER.md](docs/HANDOVER.md)

---

## 安装

```bash
pip install -e ".[dev]"
```

---

## 快速运行（当前实现，待与 CONCEPT 阶段模型对齐）

```bash
pytest
python -m experiments --suite smoke --device cpu
python -m experiments --list-suites
```

Suite 定义：`configs/suites/default.yaml`

---

## Acknowledgements

*Enmao Diao*
