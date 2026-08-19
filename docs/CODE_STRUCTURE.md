# 代码结构

前置阅读 [CONCEPT.md](CONCEPT.md)、[LAYOUT.md](LAYOUT.md)。

本文是**代码结构总览**：依赖规则、三柱文档入口、包外 examples / tests 要点。  
**每个 folder 内应有哪些模块与叶文件**，按柱拆开写（最细粒度；不列 `__init__.py`）：


| 柱         | 细则                                                         |
| --------- | ---------------------------------------------------------- |
| Structure | [code_structure/structure.md](code_structure/structure.md) |
| Flow      | [code_structure/flow.md](code_structure/flow.md)           |
| Artifact  | [code_structure/artifact.md](code_structure/artifact.md)   |


测试目录与标签见 [TESTING.md](TESTING.md)、LAYOUT §6。

---

## 1. 依赖规则

`rpipe` 库一级只有三柱：**structure**、**flow**、**artifact**。不设库顶层 `schema/`、`defaults/`、`provider/`。

- `examples` 只 import `rpipe`，不反向被库依赖
- `flow` 可 import `structure`、`artifact`；Structure 内跨层只经 `structure.api`
- `structure` 不 import `flow`；仅在需要路径 / Asset 约定时可 import `artifact`
- `artifact` 不 import `structure` 四层业务，不 import `flow`
- 第三方运行时适配写在 **Structure 各层实现内部**，经对应 `*_api` 对外；不设库顶层 `provider/`

**编排：** Study → Experiment → Run（见 CONCEPT）。  

**Config 链路：** Experiment 基底 ⊕ Run 补丁 → 完整 Config（含内容 hash 的 `id`）经 `grid/` 落入 Artifact → prepare 读回构造 `Control`。Flow 不修改 Config。  

**契约 / schema：** 由 Control 侧代码声明与校验，视为 Config 能力；不单立包。

---

## 2. 三柱文件树（只到子包，叶文件见分册）

```
src/rpipe/
  structure/
    api/                 # data_api / model_api / … 层间门面
    control/             # ExperimentConfig / RunConfig / Control
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
    index/
  artifact/
    layout.py
    config/
    result/
    asset/
```

编排入口与叶文件见各柱分册。Structure 更下层（含 Control 字段、`id` hash）见 [structure.md](code_structure/structure.md)，不在本总览展开。

---

## 3. examples（包外，摘要）

目录约定见 LAYOUT §4。摘要：

```
examples/
  studies/<study>/
    run.py                 # 选定 Experiment；调 grid / launch
  experiments/<experiment>/
    grid/                  # 展开变量轴 → 写 Artifact Config（含 id）
    launch/                # 按 run_dir 驱动 Flow
    artifact/
      <id>/                # 或 <id>_<timestamp>/
        config.yaml
        result.json
        assets/
```


| 模块 | 职责 |
|------|------|
| `grid/` | 基底 ⊕ 补丁 → hash `id` → `write_config` 到 `artifact/<run_dir>/` |
| `launch/` | `artifact_layout(experiment_dir, run_dir)` → 确认 Config → `FlowContext` + `FlowRunner` |
| Study `run.py` | 只编排；不实现 Structure / Flow |


---

## 4. tests（摘要）

镜像 `tests/rpipe/` ↔ `src/rpipe/`，`tests/examples/` ↔ `examples/`。层级用 marker，不设 `unit/` 等分类目录。覆盖映射与强制标签见 TESTING；分册末可列与该柱对应的测试落位。

---

## 5. 读法

1. CONCEPT 定概念 → LAYOUT 定目录 → **本总览**定依赖
2. 实现某柱时打开对应分册，按模块 → 类 → 叶文件落地
3. 包外 Study / Experiment / Run 只依赖库公开入口，细节不进三柱分册
