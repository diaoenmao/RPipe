# Code structure · Flow

前置：[CONCEPT.md](../CONCEPT.md) §5、[LAYOUT.md](../LAYOUT.md) §5、[CODE_STRUCTURE.md](../CODE_STRUCTURE.md)。

本文规范 `src/rpipe/flow/` 下**模块与叶文件**、阶段职责与上下文契约。不列 `__init__.py`。

**本柱边界：** 编排 Experiment 动态过程；可 import `structure`、`artifact`；**不修改 Config**；不实现四层业务细节（调用 Structure 入口）。

---

## 0. 目录总树

```
flow/
  context.py
  runner.py
  errors.py
  prepare/
    run.py
    land_structure.py
  execute/
    run.py
    schedule.py
  collect/
    run.py
    buffer.py
  summarize/
    run.py
    draft.py
  index/
    run.py
    finalize.py
```

五阶段包均以 `run.py` 为 Runner 调用的唯一入口；辅助逻辑拆到同目录其它模块。

---

## 1. 编排核心

### 1.1 `context.py`

| 符号 | 职责 |
|------|------|
| `FlowContext` | 一次运行的可变上下文 |

建议字段：

| 字段 | 类型意图 | 谁写入 |
|------|----------|--------|
| `experiment_dir` | Path | launch 构造时 |
| `slug` | str | launch |
| `layout` | `ArtifactLayout` | launch / prepare 前 |
| `config` | mapping \| None | prepare 读入 |
| `control` | `Control` \| None | prepare |
| `state` | dict | prepare / execute / collect… |
| `result_draft` | mapping \| None | collect / summarize |
| `result_path` | Path \| None | index 后 |

`state` 至少约定键（与 structure `algorithm/state.py` 对齐），例如：`data`、`model`、`system`、`observations`、`metrics`。

### 1.2 `runner.py`

| 符号 | 职责 |
|------|------|
| `PHASES` | `("prepare", "execute", "collect", "summarize", "index")` |
| `FlowRunner` | `run(ctx, phases=None) → Path`（定稿 result 路径或 layout.result_path） |

行为：

- 按序加载 `rpipe.flow.<phase>.run`（或包内 `run` 函数）并调用 `run(ctx)`
- `phases` 子集用于裁剪重跑；缺省跑全链
- Runner **不含**业务逻辑，只编排与错误上浮

### 1.3 `errors.py`

| 符号 | 职责 |
|------|------|
| `FlowError` | 阶段失败基类 |
| `PrepareError` / `ExecuteError` / … | 可选细分，便于 Result / 日志标记失败阶段 |

---

## 2. 五阶段读写矩阵

与 CONCEPT §5 一致：

| 阶段 | Config | Asset | Result | Structure |
|------|--------|-------|--------|-----------|
| prepare | **读** | 读写 | — | 落地 Control + 四层 prepare |
| execute | — | 读写 | — | algorithm 语义 |
| collect | — | — | 写缓冲 | 读 state 观测 |
| summarize | — | — | 写草稿 | 读 Control / 快照 |
| index | — | 读路径 | **定稿写** | 可选 contract 校验 |

---

## 3. `flow/prepare/`

**读取** Config，校验并落地 Structure；**不修改** Config。

| 文件 | 模块职责 | 主要符号 |
|------|----------|----------|
| `run.py` | 阶段入口 | `run(ctx)` |
| `land_structure.py` | 编排四层 prepare | `land(ctx)`：`control_from_config` → `prepare_data/model/system` → 填 `ctx.state` |

`run(ctx)` 建议步骤：

1. `load_config(ctx.layout.config_path)` → `ctx.config`
2. `validate_config`（control.contract，若启用）
3. `control_from_config` → `ctx.control`
4. `land_structure(ctx)`
5. 需要时写/读 Asset（缓存、resume 探测）

测试：`tests/rpipe/flow/prepare/test_prepare.py`（unit）；与 codec 联调可标 integration，落在 prepare 或 flow 根。

---

## 4. `flow/execute/`

按 Control 声明的 semantics 驱动真实计算；写 Asset（checkpoint、日志、样本）；**不写 Result**。

| 文件 | 模块职责 | 主要符号 |
|------|----------|----------|
| `run.py` | 阶段入口 | `run(ctx)` |
| `schedule.py` | 解析 semantics 顺序与跳过策略 | `resolve_semantics(control)`、`run_all(ctx)` |

`run` 调用 `structure.algorithm.dispatch.run_semantics(...)`，把观测写入 `ctx.state["observations"]`（及 metrics 草稿键）。

测试：`tests/rpipe/flow/prepare/` 对侧；`tests/rpipe/flow/execute/test_execute.py`；多阶段子图 integration 放在 `tests/rpipe/flow/test_prepare_execute_path.py`。

---

## 5. `flow/collect/`

从 `state` 收集可进入 Result 的观测；**不操作 Asset 文件**。

| 文件 | 模块职责 | 主要符号 |
|------|----------|----------|
| `run.py` | 阶段入口 | `run(ctx)` |
| `buffer.py` | 观测 → 结构化缓冲 | `collect_observations(state) → draft_fragment` |

写入 `ctx.result_draft`（或等价缓冲），供 summarize 合并。

测试：`tests/rpipe/flow/collect/test_collect.py`

---

## 6. `flow/summarize/`

整理 Control 指派、Structure 快照、聚合 metric、执行元数据；仍不读写 Asset。

| 文件 | 模块职责 | 主要符号 |
|------|----------|----------|
| `run.py` | 阶段入口 | `run(ctx)` |
| `draft.py` | 合并草稿字段 | `build_result_draft(ctx) → mapping` |

典型草稿键：`control`、`structure_snapshot`、`metrics`、`meta`（时间、phases、git 等由 launch/runner 注入的部分）。

测试：`tests/rpipe/flow/summarize/test_summarize.py`

---

## 7. `flow/index/`

编入 Asset 路径，契约校验，**定稿**写入 `result.json`（或约定叶名）。

| 文件 | 模块职责 | 主要符号 |
|------|----------|----------|
| `run.py` | 阶段入口 | `run(ctx)` |
| `finalize.py` | 路径登记 + 校验 + 写盘 | `finalize_result(ctx) → Path` |

步骤建议：

1. 扫描 / 登记 `layout.assets_dir` 下需暴露的路径
2. `validate_result`（control.contract）
3. `write_result(layout.result_path, draft)`
4. 设置 `ctx.result_path`

测试：`tests/rpipe/flow/index/test_index.py`

---

## 8. 阶段包的统一约定

每个阶段包必须导出：

```text
run(ctx: FlowContext) -> None
```

- 只通过 `ctx` 通信；禁止隐式全局单例承载运行态
- 失败抛 `FlowError` 子类；不静默吞掉
- 不在阶段内 `import` Experiment 包外代码

---

## 9. 与 launch 的边界

`examples/.../launch/run.py`：

1. 解析 `experiment_dir`、`slug`、可选 `phases`
2. `artifact_layout(...)` → 确认 Config 存在
3. 构造 `FlowContext`、`FlowRunner(phases=...).run(ctx)`

Flow 库代码不依赖某个具体 Experiment 路径结构之外的约定（统一为 `experiment_dir/artifact/<slug>/`）。

---

## 10. 测试落位摘要

| 镜像位置 | 层级 | 覆盖 |
|----------|------|------|
| `tests/rpipe/flow/prepare/` 等五阶段 | unit | 单阶段 `run(ctx)` |
| `tests/rpipe/flow/test_runner.py` | unit / integration | PHASES 顺序、子集 phases |
| `tests/rpipe/flow/test_prepare_execute_path.py` | integration | prepare→execute 子图 |
| `tests/examples/.../launch/` | integration / e2e | 外部入口驱动全链 |
