# Code structure · Artifact

前置：[CONCEPT.md](../CONCEPT.md) §6、[LAYOUT.md](../LAYOUT.md) §3–§5、[CODE_STRUCTURE.md](../CODE_STRUCTURE.md)。

本文规范 `src/rpipe/artifact/` 下**模块与叶文件**、磁盘约定与 IO 边界。不列 `__init__.py`。

**本柱边界：** 持久化根的库内 IO 与路径布局；**不**解析 Control 业务语义；**不** import `flow` 或 Structure 四层业务。结构校验调用方传入的 contract 函数，或只做最小 JSON/YAML 形状检查。

---

## 1. 目录总树

```
artifact/
  layout.py
  paths.py
  errors.py
  index.py
  config/
    io.py
    format.py
  result/
    io.py
    format.py
  asset/
    io.py
    tree.py
    kinds.py
```

磁盘上一次 Artifact（对应一次 Run）落在 Experiment 下。目录名用 Config 的 **`id`**（内容 hash），或同 `id` 多次落盘时用 **`id` + timestamp**：

```
<experiment_dir>/artifact/<id>/
  config.yaml            # Config（grid 写；prepare 读；含 id、推荐 description）
  result.json            # Result（含 status；Flow index 定稿或 Runner 失败落盘）
  assets/                # Asset 根
    ...

<experiment_dir>/artifact/<id>_<timestamp>/   # 可选：同 id 再次存储
  ...
```

**统一编排清单**（不属于单次 Artifact 子树）：

```
examples/studies/<study>/index.json   # Study 写；launch 之前；见 §5
```

叶文件名可通过 `paths.py` 常量配置，**同树原则不变**（Config / Result / Asset 不平行拆到 Artifact 外）。`id` 的生成规则见 [structure.md](structure.md) §8；本柱只负责路径拼装与 IO。

---

## 2. 布局与路径

### 2.1 `layout.py`

| 符号 | 职责 |
|------|------|
| `ArtifactLayout` | 一次 Artifact 的路径句柄 |
| `artifact_layout(experiment_dir, run_dir) → ArtifactLayout` | 工厂；`run_dir` 为 `<id>` 或 `<id>_<timestamp>` |

`ArtifactLayout` 建议属性 / 方法：

| 成员 | 含义 |
|------|------|
| `root` | `…/artifact/<run_dir>/` |
| `config_path` | Config 叶路径 |
| `result_path` | Result 叶路径 |
| `assets_dir` | Asset 根目录 |
| `ensure()` | 创建 root / assets（及约定子目录） |
| `exists_config()` / `exists_result()` | 存在性 |

### 2.2 `paths.py`

| 符号 | 职责 |
|------|------|
| `CONFIG_NAME` | 默认 `config.yaml` |
| `RESULT_NAME` | 默认 `result.json` |
| `ASSETS_DIRNAME` | 默认 `assets` |
| `INDEX_NAME` | 默认 `index.json`（位于 **Study 目录**，非 `artifact/<run_dir>/`） |
| `run_dir_path(experiment_dir, run_dir)` | 拼 root |
| `make_run_dir(id, timestamp=None)` | 可选：拼 `<id>` 或 `<id>_<timestamp>` |
| `index_path(study_dir)` | 拼 `…/studies/<study>/index.json` |

集中改名，避免 layout / IO 魔法字符串散落。不用 `slug` 命名。

### 2.3 `errors.py`

| 符号 | 职责 |
|------|------|
| `ArtifactError` | IO / 布局错误基类 |
| `MissingConfigError` | prepare / launch 发现无 Config |
| `CorruptArtifactError` | 无法解析的 Config / Result |

---

## 3. `artifact/config/`

Config 是 Artifact 成员：declarative 落盘；由 Control / grid 得到；Flow 不修改。正文通常含 **`id`**（hash）及四层等字段。

| 文件 | 模块职责 | 主要符号 |
|------|----------|----------|
| `io.py` | 读写 | `load_config(path) → dict`、`write_config(path, mapping) → Path` |
| `format.py` | YAML（或 JSON）编解码细节 | `dumps` / `loads`、编码与排序键策略 |

调用方：

| 谁 | 操作 |
|----|------|
| Experiment `grid/` | `write_config` |
| Flow `prepare` | `load_config` |
| Study | 间接经 grid；不直接实现格式 |

规则：

- `write_config` 原子写（临时文件 + replace）为宜，避免半截 Config
- **不做** Control 字段业务校验；校验在 `structure.control.contract`
- 读入后返回纯 mapping，由 `control_from_config` 解释

测试：`tests/rpipe/artifact/config/test_config_io.py`

---

## 4. `artifact/result/`

Result 由 collect → summarize → index 形成；成功路径下 index 后定稿。失败时可由 `FlowRunner` 直接 `write_result`（见 flow 分册）。

定稿（含失败落盘）宜含：

| 键 | 要求 |
|----|------|
| `status` | 必选：`succeeded` \| `failed` |
| `error` | `failed` 时宜有；字符串摘要 |
| `control` / `structure` / `metrics` / `paths` | 成功路径宜有；失败时可部分缺失 |

| 文件 | 模块职责 | 主要符号 |
|------|----------|----------|
| `io.py` | 读写 | `load_result(path) → dict`、`write_result(path, mapping) → Path` |
| `format.py` | JSON（默认）编解码、缩进、浮点策略 | `dumps` / `loads` |

调用方：

| 谁 | 操作 |
|----|------|
| collect / summarize | 通常只持有内存草稿；也可写草稿文件（若约定） |
| index | `write_result` 定稿（`status: succeeded`） |
| FlowRunner | 失败时尽量 `write_result`（`status: failed`） |
| Study / autoresearch | `load_result` 消费 |

规则：

- 定稿前业务契约用 `structure.control.contract.validate_result`（至少检查 `status` ∈ 允许集合）
- `artifact.result` 可做「必须是 object、顶层键为 str」等最小校验，不替代 contract
- 同样建议原子写

测试：`tests/rpipe/artifact/result/test_result_io.py`

---

## 5. `artifact/index.py`（统一 `index.json`）

Study 编排清单（CONCEPT §6.4）。模块路径为 `rpipe.artifact.index`，**不是** Flow 的 `rpipe.flow.index` 阶段；也**不是** Experiment `artifact/index.json`。

| 符号 | 职责 |
|------|------|
| `build_index(...) → dict` | 由 Study 描述 + Experiment 描述 + 已规划 Run（Config 路径等）组装；写入前算 **`id`** |
| `write_index(study_dir, mapping) → Path` | 写入 `studies/<study>/index.json` |
| `load_index(study_dir) → dict` | 读清单 |
| `compute_index_id(mapping) → str` | content hash（排除 `id`）；可复用 control hashing |

建议 shape：

```json
{
  "id": "<hash>",
  "description": "mnist seed sweep",
  "study": "mnist_seeds",
  "experiments": [
    {
      "name": "mnist_linear",
      "description": "MNIST linear train stub",
      "path": "…/experiments/mnist_linear",
      "runs": [
        {
          "id": "<run hash>",
          "description": "seed=0",
          "tags": ["baseline"],
          "run_dir": "<run hash>",
          "config": "…/artifact/<run_dir>/config.yaml"
        }
      ]
    }
  ]
}
```

规则：

- **先于 launch** 写出；条目来自编排 / Config，**不要求**已有 `result.json`
- Study 与每个 Experiment 都有短 **`description`**
- 每个 Run Artifact 的 Config 自带 **`description`** / 可选 **`tags`**（如约定 tag `baseline`）；index 中的 Run 条目与之对齐
- 可选：跑完后回填 `status` / metrics；默认不以 Result 扫描作为建清单手段

测试：`tests/unit/artifact/test_artifact_io.py`（或现有 unit 镜像）

---

## 6. `artifact/asset/`

文件型产物根；prepare / execute 读写；collect / summarize / index **不操作 Asset 文件内容**（index 只登记路径）。

| 文件 | 模块职责 | 主要符号 |
|------|----------|----------|
| `io.py` | 读写辅助 | `ensure_assets(layout)`、`read_bytes` / `write_bytes`、`copy_into` |
| `tree.py` | 子树约定与列举 | `list_assets(assets_dir)`、`relpaths_for_index` |
| `kinds.py` | 逻辑种类 → 相对路径约定 | checkpoint、cache、logs、samples 等常量或函数 |

建议相对路径约定（可演进，集中在 `kinds.py`）：

```
assets/
  cache/           # data prepare 缓存
  weights/         # 外部权重副本或软链说明
  checkpoints/     # train 周期 checkpoint
  logs/            # 训练 / 系统日志
  samples/         # inference 生成物
```

Structure 的 data/model/system/algorithm 经 `kinds` 解析路径，避免硬编码散落。

测试：`tests/rpipe/artifact/asset/test_asset_io.py`、`test_tree.py`

---

## 7. 成员与阶段权限（复述）

| Phase | config IO | asset IO | result IO |
|-------|-----------|----------|-----------|
| prepare | 读 | 读写 | — |
| execute | — | 读写 | — |
| collect | — | — | 内存草稿 |
| summarize | — | — | 内存草稿 |
| index | — | 读路径列表 | 写定稿 |
| Runner（失败） | — | — | 尽量写 `failed` Result |

`grid/`（及 Study 触发）是 Config 的**唯一常规写入方**；Flow 禁止改 Config。统一 `index.json` 由 Study 在 launch **之前**写入 Study 目录。

---

## 8. 与 Control 契约的分工

| 层次 | 位置 | 做什么 |
|------|------|--------|
| 字节 / 格式 | `artifact/*/format.py`、`io.py` | 能否解析为 mapping、原子写 |
| 业务契约 | `structure/control/contract.py` | 字段、类型、Result 必选键（含 `status`） |
| 消费方 | Study / autoresearch | 读 Result / Study `index.json`；需要时再读同树 Config |

不在 `artifact/` 下建 `schema/` 包。

---

## 9. 布局单测与 location

| 镜像位置 | 层级 | 覆盖 |
|----------|------|------|
| `tests/rpipe/artifact/test_layout.py` | unit | `ArtifactLayout` 路径拼法（`id` / `id_timestamp`）、`ensure` |
| `tests/rpipe/artifact/test_paths_location.py` | unit + location | 模块路径 / 公开符号仍在约定位置 |
| `tests/rpipe/artifact/config/` | unit | load/write 往返 |
| `tests/rpipe/artifact/result/` | unit | load/write 往返 |
| `tests/unit/artifact/test_artifact_io.py` | unit | build / write / load 统一 `index.json`（先于 Result） |
| `tests/rpipe/artifact/asset/` | unit | ensure、kinds、列举 |

集成「grid 写 Config → prepare 读」路径起点在 `tests/examples/.../grid/` 或 `tests/rpipe/flow/prepare/`，不在 artifact 内重复造 Flow。
