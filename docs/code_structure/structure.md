# Code structure · Structure

前置：[CONCEPT.md](../CONCEPT.md) §4、[LAYOUT.md](../LAYOUT.md) §5、[CODE_STRUCTURE.md](../CODE_STRUCTURE.md)。  
并列分册：[flow.md](flow.md)、[artifact.md](artifact.md)。

本文是 Structure 柱的**最细规范**：每个 folder 内模块 / 叶文件、公开符号、字段与句柄契约、层内依赖、backend 落位、测试镜像。不列 `__init__.py`。

第三方运行时适配写在**各层 `backends/`（或该层子模块）内部**，禁止另立库顶层 `provider/`、`schema/`、`defaults/`。

---

## 1. 柱边界与原则

### 1.1 做什么 / 不做什么

| 做 | 不做 |
|----|------|
| 承载 **Control** 与 data / model / algorithm / system | 编排 Flow 五阶段 |
| 由 Config mapping 构造 Control；导出 Config mapping | 直接读写 Result 文件 |
| prepare 落地句柄；execute 期由 algorithm 使用句柄 | 修改已落盘 Config |
| 在 Control 侧声明 Config / Result **契约校验** | 单立 `schema/` 包 |
| 经 Asset 路径读写缓存 / 权重 / checkpoint / 样本 / 日志 | 解析 Artifact 布局业务（那是 `artifact`） |

### 1.2 能力 vs 取值（对齐 CONCEPT §4）

| 概念 | 谁声明 | 落在哪 |
|------|--------|--------|
| **能力**（能接什么数据、模型、哪些语义） | Experiment 代码 + 本层 registry / backends | `structure/*` 已注册的 name |
| **取值**（这次跑什么） | Study / grid 展开写入 | Artifact `config.yaml` → prepare → `Control` |

同一 Experiment 代码服务多次运行；差异只在各 run 的 Config / Control 取值，**不为每个 Control 再建代码目录**。

### 1.3 柱内依赖规则

```
control          ← 无 structure 兄弟依赖（可被所有层与 flow 使用）
data             ← 可读 control.fields 类型；可 import artifact.asset 路径约定
model            ← 同上；不 import data 的 loader 实现
system           ← 同上；不 import algorithm
algorithm        ← 可依赖 data/model/system 的 *Handle 协议与 state 键；经 dispatch 被 flow.execute 调用
backends/*       ← 只被所在层 registry / prepare / run 引用；backends 互不强制依赖
```

硬约束：

- `structure` **禁止** `import rpipe.flow`
- `algorithm` **禁止**调用 `artifact.result` 写盘（观测进 `state`，由 Flow collect/summarize/index 落 Result）
- `data` / `model` / `system` **禁止**互相循环 import；共享常量可放在本层或 `control/fields.py`
- backend 文件 **禁止**被 `examples/` 直接 import（examples 只经 Experiment 配置 name，由 registry 解析）

### 1.4 文件必选等级

| 标记 | 含义 |
|------|------|
| **必须** | 骨架即应存在；缺则不符合本规范 |
| **应该** | 默认实现路径需要；可推迟但目录职责保留 |
| **可以** | 按接入的第三方 / 范式增减；未用则不建叶文件 |

`backends/` 下未列出的适配器可以新增，命名 `snake_case`，一文件一后端家族；不得抬到 `structure/` 外。

---

## 2. 目录总树

```
structure/
  control/
    control.py                 # 必须
    fields.py                  # 必须
    codec.py                   # 必须
    contract.py                # 必须
    errors.py                  # 应该
  data/
    prepare.py                 # 必须
    handle.py                  # 必须
    registry.py                # 必须
    dataset.py                 # 应该
    transforms.py              # 应该
    sampling.py                # 应该
    errors.py                  # 可以
    backends/
      native.py                # 应该（MNIST/CIFAR 等本地路径）
      torchvision_datasets.py  # 可以
      hf_datasets.py           # 可以
  model/
    prepare.py                 # 必须
    handle.py                  # 必须
    registry.py                # 必须
    build.py                   # 应该
    weights.py                 # 应该
    errors.py                  # 可以
    backends/
      native/
        linear.py              # 应该（最小可跑通）
        mlp.py                 # 可以
        cnn.py                 # 可以
        resnet.py              # 可以
      torchvision.py           # 可以
      transformers.py          # 可以
      timm.py                  # 可以
      gguf.py                  # 可以
      peft.py                  # 可以（adapter 包装，非独立语义）
  algorithm/
    dispatch.py                # 必须
    state.py                   # 必须
    errors.py                  # 应该
    train/
      run.py                   # 必须
      loop.py                  # 应该
      optim.py                 # 应该
      checkpoint.py            # 应该
      loss.py                  # 可以
      backends/
        native.py              # 应该
        accelerate.py          # 可以
    eval/
      run.py                   # 必须
      metrics.py               # 应该
      aggregate.py             # 应该
      backends/
        native.py              # 应该
        torchmetrics.py        # 可以
        evaluate_hf.py         # 可以
        lm_eval.py             # 可以
        opencompass.py         # 可以
    inference/
      run.py                   # 必须
      decode.py                # 应该
      sample_io.py             # 应该
      backends/
        native.py              # 应该
        diffusers.py           # 可以
        llama_cpp.py           # 可以
  system/
    prepare.py                 # 必须
    handle.py                  # 必须
    device.py                  # 必须
    precision.py               # 应该
    parallel.py                # 应该
    io.py                      # 应该
    resume.py                  # 应该
    profile.py                 # 可以
    errors.py                  # 可以
    backends/
      native.py                # 应该
      accelerate.py            # 可以
```

说明：

- `model/backends/native/` 用子目录承载多个小模型实现；`registry` 仍只暴露 `name → builder`。
- algorithm 语义目录**只有** `train` / `eval` / `inference`（无顶层 `metric/`、`generate/`；评测在 eval，生成在 inference）。
- 叶文件名可微调，**目录职责与公开入口签名**不得漂移。

---

## 3. `structure/control/`

变量指派对象；Config 由其得到；契约在此，算 Config 能力的一部分。

### 3.1 模块表

| 文件 | 等级 | 职责 | 主要符号 |
|------|------|------|----------|
| `control.py` | 必须 | Control 数据类 | `Control`、`to_dict()` |
| `fields.py` | 必须 | 四层字段形状（TypedDict / dataclass） | `DataFields`、`ModelFields`、`AlgorithmFields`、`SystemFields` |
| `codec.py` | 必须 | Control ↔ declarative mapping | `control_from_config`、`control_to_config` |
| `contract.py` | 必须 | Config / Result 校验 | `validate_config`、`validate_result`、`ConfigContract`、`ResultContract` |
| `errors.py` | 应该 | 控制面错误 | `ControlError`、`ContractError`、`CodecError` |

### 3.2 `Control` 形状

| 字段 | 含义 | 来源 |
|------|------|------|
| `slug` | 与 `run_slug` 对齐的标识 | Config / 调用方传入 |
| `seed` | 复现种子；与其它变量同质 | Config |
| `data` | data 层取值 | Config.`data` |
| `model` | model 层取值 | Config.`model` |
| `algorithm` | algorithm 层取值（含 semantics 列表） | Config.`algorithm` |
| `system` | system 层取值 | Config.`system` |
| `raw` | 可选：完整 Config mapping 备份 | codec 填入；`to_dict()` 默认不含或显式决定 |

`to_dict()` 供 summarize 写入 Result 的 Control 块；应稳定、可 JSON 化。

### 3.3 Config mapping 约定（codec）

`control_to_config` / 落盘 YAML 的推荐顶层键：

```text
slug: <run_slug>
seed: <int | null>
data: { ... }          # DataFields
model: { ... }         # ModelFields
algorithm: { ... }     # AlgorithmFields
system: { ... }        # SystemFields
```

规则：

- `control_from_config(cfg, slug=None)`：`slug` 参数优先，否则 `cfg["slug"]`
- 缺省层键视为 `{}`，不得静默塞入库级「隐藏 defaults 包」；缺省值若需要，写在 **Experiment grid / Control 构造处** 或 fields 的显式 default_factory
- 磁盘 IO 只走 `rpipe.artifact.config`；codec 只做对象 ↔ mapping

### 3.4 契约（contract）

| 符号 | 何时调用 | 校验什么 |
|------|----------|----------|
| `validate_config(mapping)` | prepare 读入后、grid 写入前（应该） | 顶层键、四层必要字段、semantics ∈ {train,eval,inference} |
| `validate_result(mapping)` | index 定稿前 | Result 必选块（与 CONCEPT §6.1 对齐：control、metrics、meta、paths 等项目约定键） |

契约失败抛 `ContractError`。不在 `artifact/` 内实现业务契约。

### 3.5 测试镜像

`tests/rpipe/structure/control/`

| 文件 | 倾向 |
|------|------|
| `test_control.py` | unit + content |
| `test_codec.py` | unit + content（往返保真） |
| `test_contract.py` | unit + content（接受 / 拒绝） |
| `test_control_location.py` | unit + location |

---

## 4. `structure/data/`

把研究所需输入组织为可消费数据流（CONCEPT §4.1）。

### 4.1 模块表

| 文件 | 等级 | 职责 | 主要符号 |
|------|------|------|----------|
| `prepare.py` | 必须 | prepare 入口 | `prepare_data(fields, assets_dir) → DataHandle` |
| `handle.py` | 必须 | 运行期句柄协议 / 实现 | `DataHandle`、`iter_batches`、`split_view` |
| `registry.py` | 必须 | name → builder | `register_dataset`、`build_dataset`、`list_datasets` |
| `dataset.py` | 应该 | 样本字段与 DatasetSpec | `DatasetSpec`、`SampleSchema` |
| `transforms.py` | 应该 | 进 model 前变换 | `build_transforms`、`compose` |
| `sampling.py` | 应该 | batch / shuffle / workers | `BatchSpec`、`build_loader` |
| `errors.py` | 可以 | | `DataError`、`UnknownDatasetError` |
| `backends/native.py` | 应该 | 本地 / torchvision 式小数据集 | MNIST、CIFAR 等注册 |
| `backends/torchvision_datasets.py` | 可以 | 显式 torchvision 数据集 | |
| `backends/hf_datasets.py` | 可以 | HF `datasets` | GSM8K 等 |

### 4.2 `DataFields`（与 CONCEPT 字段表对齐）

| 键（例） | 含义 | 例子 |
|----------|------|------|
| `name` | 数据集注册名 | `mnist`、`cifar10`、`gsm8k` |
| `split` / `splits` | 划分与用途 | `train`、`test`；或 mapping |
| `batch_size` | 批次 | `64` |
| `transforms` | 预处理 / 增强声明 | `normalize`、`resize` |
| `num_workers` | 加载并行 | `0`、`4` |
| `pin_memory` | 主机→设备拷贝优化 | `true` / `false` |
| `cache` | 是否写入 Asset 缓存 | 路径意图由 prepare 解释 |
| `backend` | 可选：强制后端名 | `native`、`hf_datasets` |

未列键可以扩展；`contract` / Experiment 文档应登记稳定键。

### 4.3 `DataHandle` 协议（execute 依赖）

最小能力：

| 方法 / 属性 | 含义 |
|-------------|------|
| `split_view(name)` | 取 train/val/test 等视图 |
| `iter_batches(split, **kw)` | 产生 batch（已应用 transforms） |
| `sample_schema` | 样本字段说明（可供 debug / contract） |
| `metadata` | name、大小、版本等只读信息 |

prepare 职责：解析 fields → registry 构建 → 可达性检查 → 可选缓存到 `assets/cache/` → 返回 Handle。  
execute **不**再 prepare；只消费 Handle。

### 4.4 Asset 触点

| 操作 | 阶段 | 路径意图（经 artifact.asset.kinds） |
|------|------|-------------------------------------|
| 写数据缓存 | prepare | `assets/cache/…` |
| 读缓存 | prepare / execute | 同上 |

### 4.5 测试镜像

`tests/rpipe/structure/data/` → `test_prepare.py`、`test_handle.py`、`test_registry.py`、`test_transforms.py`；backend 标 `external` 若拉网络。

---

## 5. `structure/model/`

构建可调用模型句柄（CONCEPT §4.2）。

### 5.1 模块表

| 文件 | 等级 | 职责 | 主要符号 |
|------|------|------|----------|
| `prepare.py` | 必须 | prepare 入口 | `prepare_model(fields, assets_dir) → ModelHandle` |
| `handle.py` | 必须 | 句柄协议 | `ModelHandle`、`forward`、`train_mode` / `eval_mode`、`parameters` |
| `registry.py` | 必须 | name → builder | `register_model`、`build_model`、`list_models` |
| `build.py` | 应该 | 无权重构建编排 | `build_architecture(fields)` |
| `weights.py` | 应该 | 加载 / 保存 | `load_weights`、`save_weights` |
| `errors.py` | 可以 | | `ModelError`、`UnknownModelError` |
| `backends/native/linear.py` 等 | 应该/可以 | 库内结构定义 | `build_*` |
| `backends/torchvision.py` 等 | 可以 | 外部家族 | |
| `backends/peft.py` | 可以 | LoRA 等包装 **已有** Handle | `wrap_peft(handle, fields)` |

### 5.2 `ModelFields`

| 键（例） | 含义 | 例子 |
|----------|------|------|
| `name` | 注册名 / 结构 | `linear`、`resnet18`、`gpt2` |
| `weights` | 初始化或路径 | `random`、Asset 相对路径、HF id |
| `variant` | 结构变体 | 宽度、层数、分类头维 |
| `freeze` | 冻结策略 | 骨干冻结 |
| `adapter` | LoRA 等 | peft 配置片段 |
| `forward` | 前向相关 | `context_length`、`input_size` |
| `backend` | 可选强制后端 | `native`、`transformers` |

### 5.3 `ModelHandle` 协议

| 能力 | 含义 |
|------|------|
| `forward(batch, **kw)` | 与范式相关的前向（分类 logits / 隐状态等） |
| `train_mode()` / `eval_mode()` | 切换 |
| `parameters()` / `trainable_parameters()` | 供 optim |
| `to(device)` 或经 SystemHandle 放置 | 设备一致 |
| `state_dict` / `load_state_dict` | checkpoint 协作 |

prepare：`build` → `load_weights`（可从 `assets/weights` 或 `assets/checkpoints`）→ Handle。  
train 语义下 execute 经 system/algorithm 写回 checkpoint，不在 model 层写 Result。

### 5.4 native 小模型拆分原则

| 文件 | 何时需要 |
|------|----------|
| `linear.py` | 最小 Experiment（如 mnist_linear）必须可注册 |
| `mlp.py` / `cnn.py` / `resnet.py` | 示例或评测需要时再加 |
| 大模型 | 进 `transformers` / `gguf` 等 backend，不塞进 `native/` |

### 5.5 测试镜像

`tests/rpipe/structure/model/` → prepare / handle / registry / weights；`backends/native/test_linear.py` 等跟源文件镜像。

---

## 6. `structure/algorithm/`

任务范式下「怎么算」；计算语义仅 **train / eval / inference**（CONCEPT §4.3）。

### 6.1 语义 vs 范式

| 维度 | 含义 | 落点 |
|------|------|------|
| **语义** | 本次执行哪些过程 | `algorithm.semantics: ["train","eval"]` → 目录 `train/` `eval/` `inference/` |
| **范式** | 分类 / AR / Diffusion / TTS / 检测… | fields 内 `paradigm` + 各语义 backend 分支；**不**为范式再建与 train/eval/inference 平级的顶层目录 |

### 6.2 包级模块

| 文件 | 等级 | 职责 | 主要符号 |
|------|------|------|----------|
| `dispatch.py` | 必须 | 按序调用语义 | `run_semantics(names, control_algorithm, state) → list[observations]` |
| `state.py` | 必须 | 与 FlowContext.state 对齐的键约定 | 常量键名、`get_handle`、`append_observation` |
| `errors.py` | 应该 | | `AlgorithmError`、`UnknownSemanticError` |

`dispatch` 只接受 `train` / `eval` / `inference`；未知名抛错。

### 6.3 `state` 键约定（必须稳定）

| 键 | 写入方 | 含义 |
|----|--------|------|
| `data` | flow.prepare ← data.prepare | `DataHandle` |
| `model` | flow.prepare ← model.prepare | `ModelHandle` |
| `system` | flow.prepare ← system.prepare | `SystemHandle` |
| `observations` | algorithm.*.run | list[dict]，每语义一段 |
| `metrics` | eval（及 train 可选） | 聚合前/后的 metric 结构 |
| `control` | 可选冗余 | 指向 Control 或 to_dict；通常读 `ctx.control` |

algorithm 模块经 `state.py` 读写上述键，避免魔法字符串散落。

### 6.4 `AlgorithmFields`

| 键（例） | 含义 | 例子 |
|----------|------|------|
| `semantics` | 有序列表 | `["train","eval"]` |
| `paradigm` | 任务范式 | `classification`、`ar`、`diffusion` |
| `train` | train 专用参数 | `lr`、`optimizer`、`num_steps` / `epochs` |
| `eval` | eval 专用 | `metrics`、`split`、`benchmark` |
| `inference` | inference 专用 | `decode`、`temperature`、`max_new_tokens` |

`run(control_algorithm, state)` 中各语义读取自己的子 dict（如 `control_algorithm["train"]`）。

### 6.5 `algorithm/train/`

更新参数；backward 主导；可写 checkpoint / 日志 Asset。

| 文件 | 等级 | 职责 | 主要符号 |
|------|------|------|----------|
| `run.py` | 必须 | 语义入口 | `run(control_algorithm, state) → observations` |
| `loop.py` | 应该 | step/epoch 循环 | `train_loop(…)` |
| `optim.py` | 应该 | 优化器 / 调度器 | `build_optimizer`、`build_scheduler` |
| `checkpoint.py` | 应该 | 周期保存触发 | `maybe_checkpoint(step, state)` |
| `loss.py` | 可以 | 损失构建 | `build_loss` |
| `backends/native.py` | 应该 | 单机循环实现 | |
| `backends/accelerate.py` | 可以 | Accelerate 训练 | |

`observations` 例：`{semantic, steps, loss, …}`（标量可 JSON 化）。

### 6.6 `algorithm/eval/`

度量质量；forward 为主；聚合进 `state["metrics"]`，供 collect。

| 文件 | 等级 | 职责 | 主要符号 |
|------|------|------|----------|
| `run.py` | 必须 | 语义入口 | `run(…)` |
| `metrics.py` | 应该 | 单步 / 可更新 metric | `Metric`、`update`、`compute` |
| `aggregate.py` | 应该 | 跑完聚合 | `aggregate` |
| `backends/native.py` | 应该 | accuracy 等 | |
| `backends/torchmetrics.py` 等 | 可以 | 外部 harness | |

注意：旧代码若存在顶层 `algorithm/metric`，目标布局合并进 **`eval/`**，不恢复平行 `metric/` 语义目录。

### 6.7 `algorithm/inference/`

推理 / 生成；forward 为主；样本写 Asset。

| 文件 | 等级 | 职责 | 主要符号 |
|------|------|------|----------|
| `run.py` | 必须 | 语义入口 | `run(…)` |
| `decode.py` | 应该 | 解码 / 采样 | temperature、top_p、greedy |
| `sample_io.py` | 应该 | 写 `assets/samples/` | `write_samples` |
| `backends/native.py` | 应该 | 分类 predict / 简单生成 | |
| `backends/diffusers.py` | 可以 | 文生图等 | |
| `backends/llama_cpp.py` | 可以 | 本地 LLM | |

旧 `algorithm/generate/` 目标并入 **`inference/`**。

### 6.8 统一语义入口签名

三个 `run.py` **必须**同签名：

```text
run(control_algorithm: mapping, state: mutable mapping) -> mapping  # observations
```

- 从 `state` 取 handles；不 new 全局模型
- 返回值同时 `append` 到 `state["observations"]`（由 run 或 dispatch 统一，二选一写清并固定）
- 需要写文件时经 `system` + `artifact.asset` 路径约定

### 6.9 测试镜像

| 路径 | 覆盖 |
|------|------|
| `tests/rpipe/structure/algorithm/test_dispatch.py` | 顺序、未知语义 |
| `tests/rpipe/structure/algorithm/test_state.py` | 键约定 |
| `tests/rpipe/structure/algorithm/train/test_run.py` 等 | 各语义 |
| backend 测试 | 跟 `backends/` 镜像；`external` / `slow` 按需 |

---

## 7. `structure/system/`

硬件与执行环境（CONCEPT §4.4）。

### 7.1 模块表

| 文件 | 等级 | 职责 | 主要符号 |
|------|------|------|----------|
| `prepare.py` | 必须 | prepare 入口 | `prepare_system(fields, assets_dir) → SystemHandle` |
| `handle.py` | 必须 | 运行期句柄 | `SystemHandle` |
| `device.py` | 必须 | 设备解析与放置 | `resolve_device`、`place_module`、`place_batch` |
| `precision.py` | 应该 | 数值精度策略 | `PrecisionPolicy`、`autocast_context` |
| `parallel.py` | 应该 | 单卡 / DDP 等 | `ParallelPolicy`、`wrap_parallel` |
| `io.py` | 应该 | checkpoint / 日志节奏与路径 | `OutputPaths`、`save_checkpoint`、`log` |
| `resume.py` | 应该 | 恢复 | `load_resume_state` |
| `profile.py` | 可以 | 性能分析钩子 | |
| `errors.py` | 可以 | | `SystemError` |
| `backends/native.py` | 应该 | 默认实现 | |
| `backends/accelerate.py` | 可以 | 设备 + 分布式 | |

### 7.2 `SystemFields`

| 键（例） | 含义 | 例子 |
|----------|------|------|
| `device` | 执行设备 | `cpu`、`cuda:0`、`auto` |
| `precision` | 数值策略 | `fp32`、`fp16`、`bf16`、`mixed` |
| `parallel` | 并行 | `none`、`ddp` |
| `checkpoint_every` | 保存周期 | step / epoch |
| `log_every` | 日志间隔 | |
| `resume` | 是否 / 从哪恢复 | path 或 `true` |
| `backend` | 可选 | `native`、`accelerate` |

### 7.3 `SystemHandle` 协议

| 能力 | 含义 |
|------|------|
| `device` | 当前设备对象 / 字符串 |
| `place(model_or_batch)` | 统一放置 |
| `autocast()` | 上下文管理器（可 no-op） |
| `output_paths` | checkpoints / logs 根（相对 assets） |
| `save_checkpoint(...)` / `load_checkpoint(...)` | 与 algorithm.train.checkpoint 协作 |
| `should_resume` / `resume_payload` | prepare 已解析的恢复态 |

prepare：解析 device → 确认可写 assets → 读 resume Asset → 返回 Handle。

### 7.4 测试镜像

`tests/rpipe/structure/system/` → prepare、device、precision、resume；accelerate 标 `external`。

---

## 8. 跨层协作时序

### 8.1 prepare（由 Flow 调用，逻辑属 Structure 落地）

```
load Config（artifact）
  → validate_config（control.contract）
  → control_from_config（control.codec）
  → prepare_system(control.system, assets_dir)  → state["system"]
  → prepare_data(control.data, assets_dir)      → state["data"]
  → prepare_model(control.model, assets_dir)    → state["model"]
      （model 放置可委托 state["system"]）
```

顺序：**先 system，再 data / model**（设备与输出根先就绪）。data 与 model 可互换，但不得在 system 之前。

### 8.2 execute（Flow → dispatch）

```
run_semantics(control.algorithm["semantics"], control.algorithm, state)
  → train.run / eval.run / inference.run
  → 观测进入 state["observations"] / state["metrics"]
  → Asset：checkpoints、logs、samples
```

### 8.3 禁止事项（再强调）

- Structure 代码创建 Result
- algorithm 绕过 Handle 直接依赖某 backend 模块（应经 registry / run 选择的 backend）
- examples 依赖 `structure.*.backends.*` 私有路径

---

## 9. 与 Artifact / Flow 的触点（本柱视角）

| 触点 | Structure 侧 | 对端 |
|------|--------------|------|
| Config | codec + contract | `artifact.config` IO；grid 写、prepare 读 |
| Asset | data/model/system/algorithm 经路径约定 | `artifact.asset`（kinds：cache、weights、checkpoints、logs、samples） |
| Result | `Control.to_dict`、`validate_result`、metrics 观测 | flow summarize / index 写盘 |
| Context.state | `algorithm.state` 键 | `flow.context.FlowContext.state` |

Asset 逻辑子目录名以 artifact 分册 `kinds.py` 为准；Structure 只引用常量，不复制字面量。

---

## 10. Experiment 最小闭环（Structure 需满足的能力）

以 `examples/experiments/mnist_linear` 为参照，Structure **应该**至少具备：

| 层 | 最小注册 |
|----|----------|
| data | `mnist`（`backends/native`） |
| model | `linear`（`backends/native/linear.py`） |
| algorithm | `train` + `eval` 的 native backend |
| system | `device=cpu` 的 native |
| control | codec 往返 + 基础 contract |

其它数据集 / 大模型 / 分布式均为 **可以** 扩展，不阻塞三柱骨架。

---

## 11. 测试覆盖总表

| 镜像位置 | 层级倾向 | 覆盖 |
|----------|----------|------|
| `tests/rpipe/structure/control/` | unit | Control、codec、contract、location |
| `tests/rpipe/structure/data/` | unit | prepare、handle、registry、transforms |
| `tests/rpipe/structure/model/` | unit | prepare、handle、registry、weights、linear |
| `tests/rpipe/structure/system/` | unit | prepare、device、precision、resume |
| `tests/rpipe/structure/algorithm/` | unit | dispatch、state |
| `tests/rpipe/structure/algorithm/train|eval|inference/` | unit | 各 `run` 与关键子模块 |
| `tests/rpipe/structure/…/backends/…` | unit | 适配器；常加 `external` / `slow` |
| 跨 data+model+train 子图 | integration | 落在**路径起点**（多为 algorithm 或 flow.prepare），不建 `integration/` 目录 |

强制标签三维见 [TESTING.md](../TESTING.md)。

---

## 12. 演进规则

1. 新增第三方能力：只加对应层 `backends/` 叶文件 + registry 注册，不改三柱顶层。
2. 新增算法语义：原则上不新增；若 CONCEPT 变更，先改 CONCEPT 再改本分册目录。
3. 字段键更名：同步 `fields.py`、`contract.py`、Experiment grid、测试；记入变更说明以便 Result 可比对。
4. 与 flow / artifact 分册冲突时：目录以 LAYOUT 为准，阶段读写以 CONCEPT 为准，叶文件以各分册为准并回写对齐。
