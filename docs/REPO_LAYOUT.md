# 仓库目录布局

本文定义本仓库的目录与落盘约定。前置阅读 [CONCEPT.md](CONCEPT.md)。模块职责与依赖见 [CODE_STRUCTURE.md](CODE_STRUCTURE.md)。

## 一、导读

本文回答顶层目录如何划分、概念如何映射到路径、可安装库一级结构、Experiment 下 Config 与 Artifact 如何摆放。不写类与方法实现。

Phase 1 最小闭环：一个 Experiment 目录（如 `mnist_linear`）读取 `configs/` 下多份 YAML，自动跑多个 Control，产物落在 `artifact/<control_slug>/` 下。

## 二、顶层目录总览

```
<repo>/
  src/<lib>/
  examples/
  docs/
  tests/
  pyproject.toml
  README.md
```

仓库根目录不设全局 `configs/`、不设全局 `output/` 或 `artifact/`。声明与产物归属到具体 Experiment 目录。`examples/` 通过安装后的可安装库 import 使用。

**路径职责摘要：** `src/<lib>/` 承载 Structure、Flow、Artifact IO、schema、库内默认配置。`examples/` 含 experiments 类示例。`docs/` 为设计文档。`tests/` 为测试。

## 三、概念与路径映射

Study 为概念层，无专属目录。一次 Study 可由多个 Experiment 目录、或同一 Experiment 下多份 Config 共同表达。

**Experiment** 对应 `examples/experiments/<slug>/`：一套代码与 Flow，**不为每个 Control 复制目录**。

**Config** 对应 `examples/experiments/<slug>/configs/*.yaml`：每文件声明一个 Control（含 seed 等变量）。可由 `grid.py` 批量生成。

**Control** 无独立顶层文件夹；其 slug 用于 **Artifact 分区名**（如 `artifact/lr0.01_seed0/`）。

**Structure** 写在每份 Config YAML 的四层字段中，并与库内 `default.yaml` 合并。

**Flow** 由库内 `flow` 执行；examples 只编排，不单独落盘。

**Artifact** 对应 `examples/experiments/<slug>/artifact/`，其下按 Control slug 分区。每个分区内含 Result、Asset 文件及可选 Config 冻结副本，**不使用**名为 `output/` 的目录。

## 四、examples 目录

`experiments` 是 `examples` 的一种。编排与 Study 级脚本放在 examples 内，通过 import 可安装库调用。

### 4.1 单 Experiment 目录形态

```
examples/experiments/
  _common/
    launch.py
    grid.py
  mnist_linear/
    configs/
      base.yaml
      seed_1.yaml
    run.py
    artifact/
      base/
        assets/
        result/
        frozen.yaml
      seed_1/
        assets/
        result/
        frozen.yaml
```

`mnist_linear` 仅为助记 slug，不编码 seed。多个 Control 靠 `configs/` 内多份 YAML 区分，**不**靠 `mnist_linear_seed1/` 这类额外代码目录。

`configs/` 为运行前输入。`artifact/<control_slug>/` 为运行后产物根：其下 `assets/` 供 prepare、execute 读写；`result/` 内存放定稿 Result；`frozen.yaml`（名可定）为 prepare 后写入的 Config 冻结副本，属 Artifact 的一种。

### 4.2 多 Control 与存储

同一 `run.py` 扫描 `configs/`（或通过 CLI 指定列表），对每个 Config 解析出 Control slug，构造独立 Artifact 分区路径，依次或并行跑 Flow。每个 Control 一份 Result，互不覆盖。

`grid.py` 可根据 `study.yaml` 在 `configs/` 生成多份 YAML，仍共用同一套 Experiment 代码。

### 4.3 与 Flow 的关系

`run.py` 调用 `_common.launch`：加载全部或选定 Config，对每个 Control 构造上下文，以 preset `full` 或 `--modules` 调用 `FlowRunner`。Flow 为可注册模块，按依赖排序，非固定串行脚本。

## 五、可安装库包结构

```
src/<lib>/
  structure/
  flow/
  artifact/
  schema/
  config/
  provider/
  data/
  model/
  algorithm/
  system/
```

**structure** 与 flow 同级：四层落地、Control 解析。**flow** 为模块化 FlowRunner。**artifact** 为 Asset / Result 的 IO 门面（库内包名，与 Experiment 下 `artifact/` 目录概念一致）。**schema** 仅 Result JSON 契约校验。**config** 为库内 `default.yaml` 与 Runtime 过渡，**不是** Experiment 的 `configs/`。**provider** 为薄注册。`data/`、`model/`、`algorithm/`、`system/` 为第三方集成实现，由 structure 各层委托。

## 六、tests 目录

测试在临时目录模拟 `artifact/<slug>/` 结构，不依赖仓库根落盘。夹具 Config 可放在 `tests/fixtures/configs/`。

## 七、docs 目录

CONCEPT、本文、CODE_STRUCTURE、HANDOVER。

## 八、版本库与忽略项

`.gitignore` 覆盖各 Experiment 下 `artifact/`、`__pycache__/`、`.pytest_cache/`。大体积 Asset 不进 git。`configs/` 中小文件可进 git，`artifact/` 整体忽略。
