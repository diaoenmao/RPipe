# 代码结构

前置：[CONCEPT.md](CONCEPT.md)、[LAYOUT.md](LAYOUT.md)。

库内两柱：**structure** + **flow**。artifact 在 `structure/artifact/`，make 在 `structure/make/`，cli 在 `flow/cli.py`。Study 声明在包外 `studies/`。

| 柱 | 分册 |
|----|------|
| structure（api / control / 四层 / artifact / **make**） | [code_structure/structure.md](code_structure/structure.md) |
| flow | [code_structure/flow.md](code_structure/flow.md) |

测试约定见 [TESTING.md](TESTING.md)、LAYOUT。

---

## 1. 依赖

- `studies/` 只 import `rpipe`；库不反向依赖包外
- `flow` 可 import `structure`（含 `artifact`、`make`、`control`）
- structure 跨层只经 `structure.api`；四层实现互不直接 import
- `structure` 只被 flow 调用；**make** 调用 control 与 artifact
- 第三方运行时适配写在 structure 各层内部

**编排：** Study → Experiment → Run。  
**config：** 基底 ⊕ make 展开的补丁 → `runs/<id>/` 下的 config → prepare 读回构造 **control**。Flow 不改 config。

---

## 2. 库内树

与 [LAYOUT.md](LAYOUT.md) 一致。`src/rpipe/` ↔ `tests/rpipe/`。

```
src/rpipe/
  structure/
    api/
    control/
    data/
    model/
    algorithm/
    system/
    artifact/
    make/
  flow/
    cli.py
    context.py
    runner.py
    prepare/
    execute/
    collect/
    summarize/
    write/
    process/
  __main__.py
```

| 包 | 职责 |
|----|------|
| `structure.api` | 四层对外门面 |
| `structure.control` | control 对象、config 合并、id hash、契约 |
| `structure.data` / `model` / `algorithm` / `system` | 四层实现。algorithm 含 **AlgorithmTracker** 与 **AlgorithmHook**；system 含 **Logger** 与 prepare 时的 seed / deterministic |
| `structure.artifact` | Study 树路径与 config / result / asset / index 的读写 |
| `structure.make` | 展开声明，写 config 与 index，生成调度脚本 |
| `flow.*` | 服务 Study：cli；每个 Run 的 prepare → … → process |

---

## 3. 包外 studies

见 LAYOUT。写 `study.yaml` 与基底；经 `python -m rpipe` 调用 make 与阶段链。

```
studies/<study>/
  study.yaml
  index
  experiment_config.yaml
  docs/
  shared/{data,model}/     # asset
  runs/<id>/{config, result, assets}/
  scripts/
```

---

## 4. 测试

| 树 | 对应 |
|----|------|
| `tests/rpipe/structure/` | `src/rpipe/structure/` |
| `tests/rpipe/flow/` | `src/rpipe/flow/` |
| `tests/e2e/` | `studies/` |

artifact 单测在 `tests/rpipe/structure/artifact/`。make 单测在 `tests/rpipe/structure/make/`。

---

## 5. 读法

1. [CONCEPT.md](CONCEPT.md) → [LAYOUT.md](LAYOUT.md) → **本总览**
2. 改某柱打开对应分册（structure / flow）
3. 开实验看 [STUDY_GUIDE.md](STUDY_GUIDE.md)
4. 开实验看 [STUDY_GUIDE.md](STUDY_GUIDE.md)；入口是 `python -m rpipe`
