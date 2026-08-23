# 代码结构

前置阅读 [CONCEPT.md](CONCEPT.md)、[LAYOUT.md](LAYOUT.md)、[STUDY_GUIDE.md](STUDY_GUIDE.md)。

本文是**代码结构总览**：依赖规则、三柱文档入口、包外 studies / tests 要点。  
**每个 folder 内应有哪些模块与叶文件**，按柱拆开写（最细粒度；不列 `__init__.py`）：


| 柱         | 细则                                                         |
| --------- | ---------------------------------------------------------- |
| Structure | [code_structure/structure.md](code_structure/structure.md) |
| Flow      | [code_structure/flow.md](code_structure/flow.md)           |
| Artifact  | [code_structure/artifact.md](code_structure/artifact.md)   |


测试目录与标签见 [TESTING.md](TESTING.md)、LAYOUT。

---

## 1. 依赖规则

`rpipe` 库一级：**structure**、**flow**、**artifact**、**study**（编排入口）。不设库顶层 `schema/`、`defaults/`、`provider/`。

- `studies/` 只 import `rpipe`，不反向被库依赖
- `flow` 可 import `structure`、`artifact`；Structure 内跨层只经 `structure.api`
- `study` 可 import `artifact`、`flow`、`structure.control`（展开 Config / launch）
- `structure` 不 import `flow` / `study`；仅在需要路径约定时可 import `artifact`
- `artifact` 不 import `structure` 四层业务，不 import `flow` / `study`
- 第三方运行时适配写在 **Structure 各层实现内部**

**编排：** Study →（配方）→ Run（见 CONCEPT）。  

**Config 链路：** `experiment_config.yaml` ⊕ study 展开补丁 → 完整 Config（含内容 hash 的 `id`）写入 `runs/<id>/` → prepare 读回构造 `Control`。Flow 不修改 Config。  

---

## 2. 库内文件树（只到子包）

```
src/rpipe/
  structure/
    api/
    control/
    data/
    model/
    algorithm/
    system/
  flow/
    context.py
    runner.py
    prepare/
    execute/
    collect/
    summarize/
    persist/
    process/
    index/                 # 兼容别名 → persist
  study/
    expand.py
    runner.py
  artifact/
    layout.py
    config/
    result/
    asset/
  cli.py
  __main__.py
```

---

## 3. studies（包外）

见 LAYOUT。摘要：

```
studies/<study>/
  study.yaml
  experiment_config.yaml   # 配方（原 Experiment 概念）
  index.json
  docs/{PLAN,STUDY_REPORT}.md
  shared/{data,model}/
  runs/<id>/{config.yaml,result.json,assets/}
```

入口：`python -m rpipe study run studies/<study>`。

---

## 4. tests（摘要）

镜像 `tests/rpipe/` ↔ `src/rpipe/`。e2e 指向 `studies/`。不再镜像已删除的 `examples/`。

---

## 5. 读法

1. CONCEPT 定概念 → LAYOUT 定目录 → **本总览**定依赖
2. 实现某柱时打开对应分册
3. 包外 Study 只依赖库公开入口
