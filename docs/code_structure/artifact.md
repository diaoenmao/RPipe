# Code structure · Artifact

前置：[CONCEPT.md](../CONCEPT.md) §6、[LAYOUT.md](../LAYOUT.md) §3–§5、[CODE_STRUCTURE.md](../CODE_STRUCTURE.md)。

本文规范 `src/rpipe/artifact/` 下**模块与叶文件**、磁盘约定与 IO 边界。不列 `__init__.py`。

**本柱边界：** 持久化根的库内 IO 与路径布局；**不**解析 Control 业务语义；**不** import `flow` 或 Structure 四层业务。结构校验调用方传入的 contract 函数，或只做最小 JSON/YAML 形状检查。

---

## 0. 目录总树

```
artifact/
  layout.py
  paths.py
  errors.py
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

磁盘上一次运行（概念 Artifact 子树）落在 Experiment 下：

```
<experiment_dir>/artifact/<run_slug>/
  config.yaml          # Config（grid 写；prepare 读）
  result.json          # Result（index 定稿）
  assets/              # Asset 根
    ...
```

叶文件名可通过 `paths.py` 常量配置，**同树原则不变**（Config / Result / Asset 不平行拆到 Artifact 外）。

---

## 1. 布局与路径

### 1.1 `layout.py`

| 符号 | 职责 |
|------|------|
| `ArtifactLayout` | 一次 run 的路径句柄 |
| `artifact_layout(experiment_dir, slug) → ArtifactLayout` | 工厂 |

`ArtifactLayout` 建议属性 / 方法：

| 成员 | 含义 |
|------|------|
| `root` | `…/artifact/<run_slug>/` |
| `config_path` | Config 叶路径 |
| `result_path` | Result 叶路径 |
| `assets_dir` | Asset 根目录 |
| `ensure()` | 创建 root / assets（及约定子目录） |
| `exists_config()` / `exists_result()` | 存在性 |

### 1.2 `paths.py`

| 符号 | 职责 |
|------|------|
| `CONFIG_NAME` | 默认 `config.yaml` |
| `RESULT_NAME` | 默认 `result.json` |
| `ASSETS_DIRNAME` | 默认 `assets` |
| `slug_path(experiment_dir, slug)` | 拼 root |

集中改名，避免 layout / IO 魔法字符串散落。

### 1.3 `errors.py`

| 符号 | 职责 |
|------|------|
| `ArtifactError` | IO / 布局错误基类 |
| `MissingConfigError` | prepare / launch 发现无 Config |
| `CorruptArtifactError` | 无法解析的 Config / Result |

---

## 2. `artifact/config/`

Config 是 Artifact 成员：declarative 落盘；由 Control 得到；Flow 不修改。

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

## 3. `artifact/result/`

Result 由 collect → summarize → index 形成；index 后定稿。

| 文件 | 模块职责 | 主要符号 |
|------|----------|----------|
| `io.py` | 读写 | `load_result(path) → dict`、`write_result(path, mapping) → Path` |
| `format.py` | JSON（默认）编解码、缩进、浮点策略 | `dumps` / `loads` |

调用方：

| 谁 | 操作 |
|----|------|
| collect / summarize | 通常只持有内存草稿；也可写草稿文件（若约定） |
| index | `write_result` 定稿 |
| Study / autoresearch | `load_result` 消费 |

规则：

- 定稿前业务契约用 `structure.control.contract.validate_result`
- `artifact.result` 可做「必须是 object、顶层键为 str」等最小校验，不替代 contract
- 同样建议原子写

测试：`tests/rpipe/artifact/result/test_result_io.py`

---

## 4. `artifact/asset/`

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

## 5. 成员与阶段权限（复述）

| Phase | config IO | asset IO | result IO |
|-------|-----------|----------|-----------|
| prepare | 读 | 读写 | — |
| execute | — | 读写 | — |
| collect | — | — | 内存草稿 |
| summarize | — | — | 内存草稿 |
| index | — | 读路径列表 | 写定稿 |

`grid/`（及 Study 触发）是 Config 的**唯一常规写入方**；Flow 禁止改 Config。

---

## 6. 与 Control 契约的分工

| 层次 | 位置 | 做什么 |
|------|------|--------|
| 字节 / 格式 | `artifact/*/format.py`、`io.py` | 能否解析为 mapping、原子写 |
| 业务契约 | `structure/control/contract.py` | 字段、类型、Result 必选键 |
| 消费方 | Study / autoresearch | 读 Result；需要时再读同树 Config |

不在 `artifact/` 下建 `schema/` 包。

---

## 7. 布局单测与 location

| 镜像位置 | 层级 | 覆盖 |
|----------|------|------|
| `tests/rpipe/artifact/test_layout.py` | unit | `ArtifactLayout` 路径拼法、`ensure` |
| `tests/rpipe/artifact/test_paths_location.py` | unit + location | 模块路径 / 公开符号仍在约定位置 |
| `tests/rpipe/artifact/config/` | unit | load/write 往返 |
| `tests/rpipe/artifact/result/` | unit | load/write 往返 |
| `tests/rpipe/artifact/asset/` | unit | ensure、kinds、列举 |

集成「grid 写 Config → prepare 读」路径起点在 `tests/examples/.../grid/` 或 `tests/rpipe/flow/prepare/`，不在 artifact 内重复造 Flow。
