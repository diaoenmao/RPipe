# Code structure · Flow

前置：[CONCEPT.md](../CONCEPT.md) §5、[LAYOUT.md](../LAYOUT.md)、[CODE_STRUCTURE.md](../CODE_STRUCTURE.md)、[STUDY_GUIDE.md](../STUDY_GUIDE.md)。

并列：[structure.md](structure.md)、[artifact.md](artifact.md)。

**边界：** 编排一次 Run 的  
prepare → execute → collect → summarize → **persist** → **process**；  
可 import `structure`、`artifact`；**不修改 Config**；不实现四层业务。

> 阶段 **`persist`** 即原 **`index`**：定稿并序列化写入 `result.json`。Study 的 `index.json` 是另一回事。

---

## 1. 目录（目标）

```
flow/
  context.py
  runner.py
  prepare/
  execute/
  collect/
  summarize/
  persist/          # 原 index/
  process/
```

迁移期可保留包名 `index` 作 `persist` 别名，但文档与新代码用 `persist`。

---

## 2. 核心类型

| 符号 | 位置 | 职责 |
|------|------|------|
| `FlowContext` | `context.py` | 一次 Run 的上下文 |
| `FlowRunner` / `PHASES` | `runner.py` | 按序调用各阶段；可裁剪 |

`FlowContext` 宜含：`study_dir`（共享 Asset 根）、`experiment_dir`、`layout`（指向 `artifact/runs/<id>/`）、`config`、`control`、`state`。

---

## 3. 读写矩阵

| 阶段 | Config | Asset | Result |
|------|--------|-------|--------|
| prepare | 读 | 读写 shared + run | — |
| execute | — | 读写 | — |
| collect | — | — | 内存缓冲 |
| summarize | — | — | 可序列化草稿 |
| persist | — | 登记路径 | **写 `result.json`** |
| process | — | 可选读 | 读定稿；派生 / 回填 |

---

## 4. 各阶段职责

| 阶段 | 做什么 |
|------|--------|
| **prepare** | 读 Config → Control → 落地 Structure；共享 cache 优先 |
| **execute** | 按 `algorithm.mode` 计算 |
| **collect** | 收 metrics 缓冲 |
| **summarize** | 只投影可 JSON 字段进草稿（见 structure.md §9.1） |
| **persist** | 定稿写入 Result |
| **process** | 如相对 baseline tag 的 Δ；更新 Study 级摘要 |

### 4.1 Runner 与失败 Result

成功：summarize → persist（`status: succeeded`），再 process。  
失败：尽量 persist 一份 `failed` Result，再抛出原异常。

---

## 5. 与 launch / Study

launch 作用于 Study Artifact 下的 `runs/<id>`；共享根来自 `study_dir/artifact/shared/`。

---

## 6. 测试

Runner 全链、失败落盘、**Result 不含非 JSON 对象**。真 MNIST 放 e2e；unit 用 `source: stub` / `Toy`。
