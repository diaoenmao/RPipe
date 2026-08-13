# 代码结构

> **前置：** [CONCEPT.md](CONCEPT.md)、[LAYOUT.md](LAYOUT.md)。  
> 目录以 LAYOUT 为准；本文说明当前骨架职责。

---

## 1. 导读

`rpipe` 库只有三块：`structure`、`flow`、`artifact`。examples 侧每个 Experiment 自带 `launch/`、`grid/`、`artifact/`；Study 在外侧调用它们。

依赖：`flow` 可 import `structure` 与 `artifact`；`structure` 不 import `flow`；`artifact` 不 import 四层业务。

---

## 2. 库内职责

**`structure/control`** — `Control`、`control_from_config`；prepare 读 Config 后构造。

**`structure/{data,model,system}`** — 各层 prepare 落地钩子（当前为可替换 stub）。

**`structure/algorithm/{train,eval,inference}`** — execute 按 Control 声明的 semantics 调用。

**`flow`** — `FlowContext`、`FlowRunner`；阶段包 `prepare` / `execute` / `collect` / `summarize` / `index` 各提供 `run(ctx)`。

**`artifact`** — `ArtifactLayout`；`config` / `result` / `asset` IO。磁盘形态：`artifact/<slug>/{config.yaml,result.json,assets/}`。

---

## 3. examples

**`experiments/mnist_linear/grid`** — 按 seed 写入 Config。

**`experiments/mnist_linear/launch`** — 发现 `artifact/*/config.yaml`，对每个 slug 跑 Flow。

**`studies/mnist_seeds`** — 先 grid 再 launch。

---

## 4. tests

`tests/unit/{structure,flow,artifact}`、`tests/e2e`、`tests/fixtures/artifact`。
