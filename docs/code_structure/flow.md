# Code structure · Flow

前置：[CONCEPT.md](../CONCEPT.md) §5、[LAYOUT.md](../LAYOUT.md)、[CODE_STRUCTURE.md](../CODE_STRUCTURE.md)。

并列：[structure.md](structure.md)、[artifact.md](artifact.md)。

**边界：** 编排一次 Run 的 prepare → execute → collect → summarize → index；可 import `structure`、`artifact`；**不修改 Config**；不实现四层业务。

---

## 1. 目录

与现实现一致：

```
flow/
  context.py
  runner.py
  prepare/
  execute/
  collect/
  summarize/
  index/
```

各阶段包导出 `run(ctx)`。

---

## 2. 核心类型

| 符号 | 位置 | 职责 |
|------|------|------|
| `FlowContext` | `context.py` | 一次 Run 的上下文 |
| `FlowRunner` / `PHASES` | `runner.py` | 按序调用各阶段 `run(ctx)`；可裁剪 `phases` |

`FlowContext` 字段（与现实现对齐）：

| 字段 | 说明 |
|------|------|
| `experiment_dir` | Experiment 目录 |
| `layout` | `ArtifactLayout`（`artifact/<run_dir>/`） |
| `config` | prepare 读入的 Config mapping |
| `control` | prepare 构造的 `Control` |
| `state` | 阶段间传递的运行时 dict（落地对象、观测、草稿等） |

---

## 3. 读写矩阵（同 CONCEPT §5）

| 阶段 | Config | Asset | Result |
|------|--------|-------|--------|
| prepare | 读 | 读写 | — |
| execute | — | 读写 | — |
| collect | — | — | 缓冲 |
| summarize | — | — | 草稿 |
| index | — | 登记路径 | 定稿写入 |

---

## 4. 各阶段职责

| 阶段 | 做什么 |
|------|--------|
| **prepare** | `load_config` → `control_from_config` → 落地 Structure（经现有 prepare 入口）；不改 Config |
| **execute** | 按 Control 的 `algorithm.mode`（`train` / `eval` / `inference`）跑计算；可写 Asset |
| **collect** | 从 `state` 收观测 / metrics 缓冲；不碰 Asset 文件 |
| **summarize** | 整理 Control、快照、metrics、**`status: succeeded`** 等进 Result 草稿 |
| **index** | 登记 Asset 路径，写入 `result.json` 定稿（成功时保留 `status: succeeded`） |

### 4.1 Runner 与失败 Result

`FlowRunner.run`：

- 正常跑完所选阶段 → 由 summarize / index 定稿，`status: succeeded`
- 任一阶段抛错 → **尽量**写入 `status: failed` 与简短 `error`（及已知 `paths` / 已有草稿字段），再重新抛出原异常
- 失败写入不得掩盖原异常；写盘本身再失败时以原异常为准向上抛

不在此引入中间态 status。统一 `index.json`（Study 编排清单）不属于 Flow，见 CONCEPT §6.4 / artifact 分册。

细节（四层如何 build、`mode` 语义、Config/`id`）见 Structure / Artifact 分册与 CONCEPT，本分册不重复。

---

## 5. 与 launch

`examples/.../launch/`：选定 `experiment_dir` 与 `run_dir` → `artifact_layout` → 确认 Config → `FlowContext` + `FlowRunner().run(ctx)`。

---

## 6. 测试

镜像 `tests/rpipe/flow/`（或现有 `tests/unit/flow/`）：Runner 全链、单阶段 `run(ctx)`。不在此重复 Structure / Artifact 契约测试。
