# Layout

本文定义 **RPipe** 仓库目录约定（不含最底层叶文件）。前置阅读 [CONCEPT.md](CONCEPT.md)。模块细则见 [CODE_STRUCTURE.md](CODE_STRUCTURE.md)。

可安装包名 **`rpipe`**，源码根 `src/rpipe/`。目录只映射 CONCEPT。

---

## 1. 导读

库内两柱：`structure/`、`flow/`。artifact 在 **`structure/artifact/`**。Study 在包外 `studies/`。

| 概念 | 目录落点 |
|------|----------|
| **Study** | `studies/<name>/` |
| **Experiment** | 逻辑分组（**index**）；无顶层文件夹 |
| **Run** | `studies/<name>/runs/<id>/` |
| **structure** | `src/rpipe/structure/`：`api`、`control`、四层、**artifact** |
| **flow** | `src/rpipe/flow/`：每阶段一个子包 |

**Run 目录名：** config 的 `id` = 除 `id` / `description` 外内容的 hash（含 tags、seed）。同 id 多次存储可用时间戳后缀。

读写：

- **config**：编排写入 `runs/<id>/`；prepare 只读
- **result**：summarize / **write** 写入；process 可派生
- **asset**：data / model 的磁盘文件（数据集、权重、checkpoint）以及日志等，落在 `shared/` 或 `runs/<id>/assets/`

---

## 2. 概念与路径

```mermaid
flowchart TB
  subgraph outside [包外]
    studies[studies/ Study]
  end
  subgraph lib [src/rpipe/]
    structure[structure/]
    flow[flow/]
  end
  studies -->|展开 config / 调 Flow| flow
  flow --> structure
  flow -->|读 config 写 result 等| studies
```

| 概念 | 路径 | 说明 |
|------|------|------|
| Study | `studies/<study>/` | 编排壳 + artifact 根 |
| 基底配置 | Study 目录下的 experiment 基底文件 | Study 默认值 |
| Run | `…/runs/<id>/` | config / result / assets |
| 共享与文档 | `…/shared/`、`…/docs/` | Study 级。`shared/` 里的 data / model 文件属于 **asset** |
| structure | `src/rpipe/structure/` | api、control、data、model、algorithm、system、artifact |
| flow | `src/rpipe/flow/<phase>/` | prepare / execute / collect / summarize / write / process |

---

## 3. 库内目录树

`src/rpipe/` 与 `tests/rpipe/` 同构。`structure/` 只展到下一层。

```
RPipe/
  src/
    rpipe/
      structure/
        api/
        control/
        data/
        model/
        algorithm/
        system/
        artifact/
      flow/
        prepare/
        execute/
        collect/
        summarize/
        write/
        process/
      cli.py
  studies/
  docs/
  tests/
    rpipe/
```

| flow 子包 | 阶段 |
|-----------|------|
| `prepare/` | 读 config，落地 structure |
| `execute/` | 计算 |
| `collect/` | 收观测 |
| `summarize/` | result 草稿 |
| `write/` | 把 result 写入 artifact |
| `process/` | write 后派生（可空） |

---

## 4. Study 目录

```text
studies/<study>/
  study.yaml
  index
  experiment_config.yaml
  docs/
    PLAN.md
    STUDY_REPORT.md
  shared/
    data/
    model/
  runs/
    <run_id>/
      config.yaml
      result.json
      assets/
```

| 成员 | 说明 |
|------|------|
| `docs/` | 计划与报告；可入库 |
| `shared/` | Study 内共享 asset（data / model 文件）；默认不入库 |
| `runs/` | 每次 Run；默认不入库 |
| `index` | launch 前编排清单 |
| `shared/data/`、`shared/model/` | structure data / model 的落盘 |
| `runs/<id>/assets/` | 本 Run 的 asset（checkpoint、日志等） |

---

## 5. 调用关系

```
studies/<name>/study.yaml
        │
        ▼
  包外编排 / 薄 CLI
        │
        ├─► 写 runs/<id>/config.yaml
        ├─► 写 index
        └─► FlowRunner（structure + flow）
                    │
                    ▼
            runs/<id>/ 下的 result 与 assets
            shared/ 下的 data / model asset
```

---

## 6. 测试镜像

| 测试树 | 对应 |
|--------|------|
| `tests/rpipe/structure/` | `src/rpipe/structure/` |
| `tests/rpipe/flow/` | `src/rpipe/flow/` |
| e2e | `tests/e2e/` → `studies/…` |
