# Layout

本文定义 **RPipe** 仓库的全部目录约定（不含最底层文件）。前置阅读 [CONCEPT.md](CONCEPT.md)。模块职责见 [CODE_STRUCTURE.md](CODE_STRUCTURE.md)。

可安装库包名为 **`rpipe`**，源码根为 `src/rpipe/`。目录只映射 CONCEPT，不按现状实现或第三方名单倒推。

---

## 1. 导读

本文一次定清目录树：哪些目录存在、概念落在哪一层。不列叶文件（如具体 `.py` / `.yaml` / `.json`）。

对齐 CONCEPT，并补充实现落点：

- 库内一级目录对应 **Structure**、**Flow**、**Artifact**
- **Control** 属于 Structure（与四层并列），对象代码在 `structure/control/`
- Config、Result、Asset 都在 Artifact 下；Study 落盘 Config，**prepare 读取** Config，Flow 不修改 Config
- 每个 Experiment 自带 `launch/`、`grid/`；外面 Study 脚本与之沟通，不设跨 Experiment 的 `_common`

---

## 2. 完整目录树

下列为仓库目标目录。实例名用示例 slug（可换），叶文件不列出。

```
RPipe/
  src/
    rpipe/
      structure/
        control/
        data/
        model/
        algorithm/
          train/
          eval/
          inference/
        system/
      flow/
        prepare/
        execute/
        collect/
        summarize/
        index/
      artifact/
        config/
        result/
        asset/
  examples/
    studies/
      mnist_seeds/
    experiments/
      mnist_linear/
        launch/
        grid/
        artifact/
          seed_0/
            assets/
          seed_1/
            assets/
  docs/
  tests/
    unit/
      structure/
      flow/
      artifact/
    integration/
    e2e/
    fixtures/
      artifact/
        seed_0/
          assets/
```

根下另有工程元数据（版本库忽略、打包配置等），不进入概念映射。

---

## 3. 概念与路径映射

概念关系以 CONCEPT 为准。本节说明每个概念在仓库里落在哪条路径，以及职责边界。

**Study** 只做编排，不实现 Structure / Flow。路径在 `examples/studies/` 下按 Study 取名（如 `mnist_seeds/`）：展开变量轴并落盘 Config、调用某个 Experiment 的 `launch/` 或 `grid/`。

**Experiment** 是可运行单元，路径在 `examples/experiments/` 下（如 `mnist_linear/`）。每个 Experiment 目录内自带 `launch/`、`grid/` 与本实验的 `artifact/`。同一套 Experiment 代码可服务多个 Control，不为每个 Control 再建代码目录。

**Control** 属于 Structure（与 data / model / algorithm / system 并列），**没有**独立的 examples 代码目录。落盘上，每次运行对应一棵 Artifact 子树（如 `artifact/seed_0/`）。库内 Control 对象（从 Config 解析变量指派、slug、与四层取值的对应等）在 **`src/rpipe/structure/control/`**。概念图上 Control 不连接 Config；由 prepare 读取 Config 后构造 Control。

**Config / Result / Asset** 都是 Artifact 成员，实体落在各次运行的 Artifact 子树内（Config、Result 为叶文件；Asset 在 `assets/`）。库侧读写在 `src/rpipe/artifact/config/`、`result/`、`asset/`。Study 写 Config；prepare 读 Config；Flow 写 Result 与 Asset，不改 Config。

**Structure** 在 `src/rpipe/structure/`：含 `control/` 与四层 `data/`、`model/`、`algorithm/`、`system/`。`algorithm/` 下再分 `train/`、`eval/`、`inference/`。

**Flow** 在 `src/rpipe/flow/`：五阶段 `prepare/`、`execute/`、`collect/`、`summarize/`、`index/`。Experiment 的 `launch/` import 库内 Flow，并使用 Structure（含 Control 对象）与 Artifact IO。


| 概念 | 路径 |
|------|------|
| Study | `examples/studies/…` |
| Experiment | `examples/experiments/…`（含 `launch/`、`grid/`、`artifact/`） |
| Control（对象） | `src/rpipe/structure/control/` |
| Config / Result / Asset（IO） | `src/rpipe/artifact/{config,result,asset}/` |
| Structure | `src/rpipe/structure/` |
| Flow | `src/rpipe/flow/` |

---

## 4. examples：Study 与 Experiment

### 4.1 分工

**Study**（如 `examples/studies/mnist_seeds/`）是外侧编排：展开 Control、往 Artifact 写入 Config、选定并调用某个 Experiment 的 `launch/`。Study 目录放编排脚本与 Study 级约定。

**Experiment**（如 `examples/experiments/mnist_linear/`）是内侧可运行单元，各自独立持有：

- **`launch/`** — 指定 Artifact 子树，调用库内 Flow；prepare 读 Config 构造 Control，写回 Result / Asset
- **`grid/`** — 按本 Experiment 的 Structure 字段展开变量轴，写入各次运行子树下的 Config
- **`artifact/`** — 各次运行的 Artifact 子树

外面 Study 脚本与里面 Experiment 的 `launch/` / `grid/` 沟通；Experiment 之间不共享包外公共编排目录。共性逻辑若抽离，进 `rpipe` 库，而不是 `_common`。

### 4.2 Artifact 子树

```
examples/experiments/mnist_linear/artifact/seed_0/
  assets/
```

同一 slug 目录下还承载 Config、Result 等叶文件（与 `assets/` 同级）。Config / Result / Asset 同树，不在 Artifact 外另设平行配置目录。

### 4.3 调用关系

```
examples/studies/mnist_seeds/
        │
        ▼
examples/experiments/mnist_linear/grid/     → 写 Config
examples/experiments/mnist_linear/launch/   → 跑 Flow（prepare 读 Config → Control），写 Result / Asset
        │
        ▼
src/rpipe/structure/   （含 control/ 与四层）
src/rpipe/flow/
src/rpipe/artifact/
```

Study 可只调 `launch/`（Config 已在），或先 `grid/` 再 `launch/`。绑定由 Study 持有。

---

## 5. 可安装库 `src/rpipe/`

仓库名 **RPipe**，包名 **`rpipe`**。库目录与 CONCEPT 对齐，不平行挂第二套四层实现树，不设 `provider/` 等与概念词表无关的一级目录。


| 目录 | 对应 CONCEPT | 职责 |
|------|------|------|
| `structure/control/` | Control | 变量指派对象；prepare 读 Config 后构造；与四层取值对应 |
| `structure/data/` 等 | Structure 四层 | 静态能力落地 |
| `structure/algorithm/` | algorithm 语义 | `train/` `eval/` `inference/` |
| `flow/` | Flow | 五阶段 |
| `artifact/` | Artifact | `config/` `result/` `asset/` IO |

第三方运行时在 Structure 各层内部按需适配。Result 契约校验落在 `artifact/result/`，不单独占库顶层目录。

---

## 6. tests

`unit/` 对齐 `structure/`、`flow/`、`artifact/`（含对 `structure/control` 的单测）。`integration/`、`e2e/` 覆盖跨模块与整链。`fixtures/artifact/` 模拟 Control 子树。

---

## 7. docs

| 文档 | 内容 |
|------|------|
| CONCEPT.md | 概念与关系 |
| LAYOUT.md | 本文：目录树与落盘 |
| CODE_STRUCTURE.md | 模块与实现职责 |
| HANDOVER.md | 历史交接 |

---

## 8. 版本库与忽略

忽略缓存与 Artifact 下大体积 Asset（`assets/` 等）。Config 等 declarative 小文件是否进 git 由 Study / Experiment 约定。
