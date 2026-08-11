# 代码结构

> **状态：** 大纲 v3（文字叙述 + 文末摘要表），待审。  
> **前置：** [CONCEPT.md](CONCEPT.md)、[REPO_LAYOUT.md](REPO_LAYOUT.md)。

## 一、导读

本文用**文字**说明目标代码文件、类与方法职责。正文不用表格展开细节；文末附一张总览表。

核心布局原则：

- **一个 Experiment 目录 = 一套代码**，通过 `configs/` 里多份 YAML 自动处理多个 Control，**不为每个 Control 建代码目录**。
- 产物写在 Experiment 下的 **`artifact/<control_slug>/`**，内含 `assets/`、`result/`、可选 `frozen.yaml`（Config 冻结副本）。**不用** `output/` 作目录名。
- 概念 **Config**（YAML 输入）已纳入 CONCEPT；库内 `config/` 包仅表示**包级默认**，与 Experiment 的 `configs/` 不同。

Phase 1： `mnist_linear` 下至少两份 Config（如不同 seed），共用 `run.py`，各 Control 分区落 Result。

**依赖总则：** examples 只 import 库。`flow` 可 import `structure`、`artifact`、`schema`、库内 `config`。`structure` 不 import `flow`。`artifact` 不 import 四层业务。`schema` 仅在 Result 定稿时校验。

## 二、Experiment 目录与编排（examples）

### 2.1 目录约定

`examples/experiments/mnist_linear/` 含 `configs/`、`run.py`、`artifact/`。`configs/base.yaml` 与 `configs/seed_1.yaml` 各声明一个 Control。`run.py` 仅把本目录传给共享启动逻辑。

slug 来自 Config 内显式字段 `control_slug`，或默认取 YAML 文件名（不含扩展名）。slug 决定 `artifact/<slug>/` 路径，**不把 seed 拼进 Experiment 目录名**。

### 2.2 `examples/experiments/_common/launch.py`

`run_experiment(experiment_dir, device=None, preset="full", modules=None, config_names=None)` 是主入口。它发现 `configs/` 下全部 YAML，或在 `config_names` 限定子集；对每个文件调用 `run_one_control`；返回各 Control 的 Result 路径列表。

`run_one_control(experiment_dir, config_path, device, preset, modules)` 加载单份 Config，解析 Control slug，构造 `ControlContext`，创建 `FlowRunner` 并 `run()`。

`discover_configs(experiment_dir)` 列出 `configs/*.yaml`。

`parse_modules` 解析 CLI `--modules`，覆盖 preset。

`main` 提供 argparse：`--device`、`--preset`、`--modules`、`--configs`（逗号分隔文件名）。

### 2.3 `examples/experiments/_common/grid.py`

`expand_study(study_yaml, configs_dir)` 按 Study 变量轴在 `configs/` **生成**多份 YAML，不生成新代码目录。

`cartesian(axes)` 做笛卡尔积，供 expand 使用。

### 2.4 `examples/experiments/<slug>/run.py`

`main()` 调用 `run_experiment(Path(__file__).resolve().parent)`。

## 三、库内文件树（Phase 1）

```
src/<lib>/
  structure/
    config.py
    control.py
    data.py
    model.py
    algorithm.py
    system.py
  flow/
    context.py
    state.py
    module.py
    registry.py
    runner.py
    presets.py
    modules/
      prepare.py
      execute.py
      collect.py
      summarize.py
      index.py
  artifact/
    asset.py
    result.py
    layout.py
  schema/
    validator.py
    contracts/result_v1.py
  config/
    defaults.py
    runtime.py
  provider/
  data/ model/ algorithm/ system/
```

`data/` 等集成目录 Phase 1 保留现状，由 `structure.*Layer` 委托。

## 四、structure 包

与 `flow` 同级，负责 **Config 解析 → Structure + Control** 以及四层 prepare / execute。

### 4.1 `structure/config.py`

`StructureConfig` 为四层 dict 容器，提供 `from_mapping`、`merge`、`frozen_dict()`。

`load_config_file(path, package_defaults)` 读取单份 Experiment Config YAML，与库内 `default.yaml` 合并，返回 `StructureConfig`。

注意：此处的「从文件加载 Structure」对应概念 **Config**，文件名避免与库内 `config/` 包混淆，故函数名用 `load_config_file`。

### 4.2 `structure/control.py`

`Control` 含 `assignments`（含 seed）与 `slug`。

`from_config(structure, slug)` 从 Structure 提取有意变量。

`to_dict()` 供 Result 写入。

### 4.3 各层 `structure/data.py` 等

`DataLayer.prepare(ctx, assets)` 校验数据、写缓存 Asset，返回 `DataHandles`。

`DataLayer` 的 `train_batches` / `eval_batches` 供 execute 使用。

`ModelLayer`、`AlgorithmLayer`、`SystemLayer` 同理：prepare 构图与读权重，execute 训练 / 评测 / checkpoint 节奏。细节见 v2 大纲，此处不重复表格。

## 五、flow 包（模块化，非固定串行）

Flow 由可注册 **FlowModule** 组成。每个模块声明 `name`、`requires`、`provides`、`caps`（是否接触 Asset / Result）。`FlowRunner` 对模块列表做拓扑排序后执行；默认 preset `full` 与 CONCEPT 五阶段同名，但实现上顺序由依赖决定，也可用 `--modules` 跳过 prepare 等。

### 5.1 `flow/context.py`

`ControlContext` 描述**单次 Control 运行**：`experiment_dir`、`config_path`、`structure`、`control`、`artifact_root`（即 `artifact/<slug>/`）、`assets_dir`、`result_dir`、`device`。

`from_config_file(experiment_dir, config_path, device)` 完成路径计算与 Config 加载。`result_file` 属性指向 `artifact/<slug>/result/result.json`。

Experiment 级共享代码路径在 `experiment_dir`；**每次 Control 运行**有独立 `ControlContext`。

### 5.2 `flow/state.py`

`FlowState` 为模块间共享状态袋，键如 `prepared`、`trace`、`draft_partial`、`draft`、`timings`。提供 `get`、`set`、`has`、`require`。

### 5.3 `flow/module.py`

`ModuleCaps` 标记 `TOUCHES_ASSET`、`READ_ASSET`、`TOUCHES_RESULT`。

`FlowModule` 协议：`run(ctx: ControlContext, state: FlowState, resources: FlowResources)`。

`FlowResources` 按 caps 注入 `AssetStore` 或 `ResultWriter`。

### 5.4 `flow/registry.py` 与 `flow/presets.py`

`register` / `get` 管理内置模块。

`PRESET_FULL` 列表为 prepare、execute、collect、summarize、index。

`PRESET_EXECUTE_ONLY` 仅 execute。

`resolve_modules(preset, modules)`：`modules` 非空时优先，否则解析 preset。**无 flow.yaml 文件**。

### 5.5 `flow/runner.py`

`FlowRunner(ctx, preset="full", modules=None)`。

`run()` 解析模块列表 → `_execution_order()` 拓扑排序 → 逐模块 `run()` → 若包含 index 模块则返回 Result 路径。

`_build_resources(module)` 按 caps 构造 `FlowResources`。

### 5.6 `flow/modules/*.py`

每个文件一个 `FlowModule` 实现类：

`PrepareModule` 产出 `prepared`，caps 为 TOUCHES_ASSET。内部调用各 `structure.*Layer.prepare`，并在 `artifact/<slug>/frozen.yaml` 写入 Config 冻结副本（通过 `artifact.layout` 路径助手）。

`ExecuteModule` 需要 `prepared`，产出 `trace`，TOUCHES_ASSET。`ExecuteCoordinator` 组织 data → algorithm ← model 与 system。

`CollectModule` 需要 `trace`，产出 `draft_partial`，仅 TOUCHES_RESULT。

`SummarizeModule` 合并 Control、Structure 快照、metric、元数据到 draft。

`IndexModule` 需要 `draft`，READ_ASSET + TOUCHES_RESULT，编入 Asset 路径并 `finalize`。

collect 与 summarize 是否合并为一个模块留作开放问题，Phase 1 可保持拆分。

## 六、artifact 包（库内 IO）

库内 `artifact` 包实现 Artifact 的读写门面，与 Experiment 目录名 `artifact/` 对应。

### 6.1 `artifact/layout.py`

`artifact_root(experiment_dir, control_slug)` → `experiment_dir / "artifact" / control_slug`。

`assets_dir`、`result_dir`、`frozen_config_path` 等路径构造函数，避免各处手写路径。

### 6.2 `artifact/asset.py`

`AssetStore(root)` 绑定某 Control 的 `assets/`。

`write_text`、`write_bytes`、`open_read`、`register`、`list_registered` 与 v2 一致。仅 prepare / execute 模块经 `FlowResources` 获得写权限。

### 6.3 `artifact/result.py`

`ResultWriter(result_dir)` 累积 `ResultDraft`，`finalize` 写 `result.json` 并调用 schema 校验。

## 七、schema 包（是什么）

**schema 不是业务模块**，不训练、不编排 Flow。它只定义 **Result JSON 长什么样**，并在 `ResultWriter.finalize` 前做校验，保证 autoresearch 可稳定消费。

`schema/contracts/result_v1.py` 定义 `RESULT_V1_ID`、`validate_result_v1`。

`schema/validator.py` 提供轻量 `validate_against_schema`。

遗留 `result_blob.v1` 可放 `schema/legacy.py`，Phase 1 不写。

## 八、库内 config 包（与概念 Config 区分）

| 概念 | 路径 | 含义 |
|------|------|------|
| **Config**（概念） | `examples/.../configs/*.yaml` | 一次 Control 的输入声明 |
| **config**（库包） | `src/<lib>/config/` | 包级 `default.yaml`、`RuntimeConfig` 过渡 |

`defaults.load_package_defaults()` 读库内默认。

`runtime.build_runtime_from_context(ctx)` 为 MNIST Phase 1 适配旧 Trainer，逐步收敛到纯 Structure 驱动。

## 九、单次 Control 调用链（文字）

`run.py` 调用 `launch.run_experiment`，扫描 `configs/`。对每个 YAML，`run_one_control` 构建 `ControlContext`，其中 `artifact_root` 指向 `artifact/<slug>/`。`FlowRunner.run()` 按模块依赖执行：Prepare 写 `frozen.yaml` 与 assets 缓存；Execute 训练并写 checkpoint；Collect / Summarize 写 Result 草稿；Index 登记 assets 路径并定稿 `result.json`。

同一 `run.py` 第二次处理下一 YAML 时，slug 不同，写入另一 `artifact/<other_slug>/`，代码路径不变。

## 十、现状迁移（摘要）

旧 `experiments/runner.py` 的 suite 多 control 循环 → `launch.run_experiment` 扫 `configs/`。旧顶层 `output/` → Experiment 下 `artifact/<slug>/`。旧 `experiment.yaml` 单文件 → `configs/*.yaml` 多 Control。旧 `ResearchPipeline` 固定 STAGES → `FlowRunner` + preset / modules。

## 十一、开放问题

1. Control slug 默认用文件名还是 YAML 内必填字段？
2. Config 冻结副本固定名 `frozen.yaml` 还是 `config.yaml`？
3. collect 与 summarize 是否合并为一个 FlowModule？
4. 库内 `config/` 包是否改名为 `defaults/` 以避免与概念 Config 混淆？

## 十二、总览表（摘要）

| 概念 | 代码载体 | 磁盘位置 |
|------|----------|----------|
| Experiment | `examples/.../<slug>/` + `run.py` | 代码目录 |
| Config | `configs/*.yaml` → `load_config_file` | 输入，可进 git |
| Control | `structure.control.Control` | slug → `artifact/<slug>/` |
| Flow | `flow.runner.FlowRunner` + modules | 不落盘 |
| Result | `artifact.result.ResultWriter` | `artifact/<slug>/result/` |
| Asset | `artifact.asset.AssetStore` | `artifact/<slug>/assets/` |
| Config 冻结副本 | prepare 模块写入 | `artifact/<slug>/frozen.yaml` |
| schema | `schema.contracts.result_v1` | 无独立目录 |

---

*大纲 v3 结束。*
