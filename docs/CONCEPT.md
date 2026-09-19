# Concept

RPipe 是**可重复、可编排、可序列化的研究执行底座**。

研究者用 **Study** 声明要比什么。structure 的 **make** 写出各 **Experiment** 与 **Run** 的 config，以及调度脚本。**Flow** 服务整个 Study：同一套执行，用参数选择阶段、顺序或按 `round` 并行。命令行入口是 flow 的 **cli**。产物落在该 Study 的 **artifact** 下。

目录见 [LAYOUT.md](LAYOUT.md)；操作见 [STUDY_GUIDE.md](STUDY_GUIDE.md)；structure 细节见 [code_structure/structure.md](code_structure/structure.md)。

---

## 1. 边界

对照相邻系统，而不是只谈职责。DeepScientist：[ResearAI/DeepScientist](https://github.com/ResearAI/DeepScientist)。

| | **RPipe** | **DeepScientist** | **Hugging Face** | **autoresearch** |
|--|-----------|-------------------|------------------|------------------|
| **定位** | 研究**执行底座**：可重复、可编排、可序列化 | local-first 研究 OS / 长程工作室 | 模型 / 数据 / 训练与推理**生态与库** | 包外**自动研究环**：选题、改实验、读产物、再决策 |
| **编排对象** | Study → Experiment → Run | Quest | 无研究编排（repo、pipeline、Trainer） | 调用 RPipe 或同类的 Study / Run |
| **库内组成** | **structure** + **flow** | Agent + Memory + UI + 执行器 | transformers / datasets / hub 等 | 决策与搜索逻辑，在本库之外 |
| **执行** | Flow 服务 Study；每个 Run 走 prepare → execute → collect → summarize → write → process | `bash_exec` 等通用 shell | Trainer / pipeline / 用户脚本 | 触发执行底座，自己不训模型 |
| **持久化** | Study 下 **artifact**：docs / config / result / asset / index | Quest Git、memory、artifact 账本 | Hub 权重与数据集；本地 output_dir | 消费 artifact，自管记忆 / 假设 |
| **决策 / 下一步** | 不内置 | Findings / Bayesian / Map | 无 | **负责**下一步实验 |
| **研究者怎么用** | 写声明与基底；经 flow cli 跑 Study；读 result；写报告 | 在 OS 里提 Quest、看 Canvas | 选模型 / 数据、写训练或推理代码 | 设目标与约束，审阅自动循环 |
| **训练 / 模型** | 经 structure 的 `data` / `model` 适配接入 | Agent 调外部命令接入 | 提供模型卡、权重、Trainer、推理 API | 不提供模型实现 |
| **数据** | 运行时是 structure **data**；落盘是 artifact **asset** | 由 Quest / 外部脚本处理 | Hub datasets、本地缓存 | 指定用哪份数据 |
| **产物契约** | 稳定的 Run 入口、config、result | Quest 账本 + 论文/实验产物 | 权重、metrics 日志 | 依赖 RPipe 的 result / index |
| **边界** | 执行经 Flow；make 脚本调度多次 Runner。不做 OS / UI / 决策器 | 不是薄执行库 | 不是研究编排底座 | 不替代 Flow / structure |

**借力原则**：相对 DeepScientist，学 durable 契约与编排纪律。相对 HF，只适配。相对 autoresearch，提供可消费的执行契约。

包外研究根：`studies/<name>/`。

---

## 2. 主干：Study → Experiment → Run

```text
Study            一轮研究：编排壳 + artifact 根
 └── Experiment  实验变量的一个取值点，不含 seed
      └── Run    该点下的一次实测，含 seed
```

| | **Study** | **Experiment** | **Run** |
|--|-----------|----------------|---------|
| **是什么** | 编排 + 落盘根 | 比较轴上的一个格子 | 该格子的一次抽样 |
| **seed** | 声明 `seeds` | 不含 | 必须有 |
| **例子** | `mnist_train_size` | `train_size=500` | `train_size=500, seed=0` |
| **磁盘** | `studies/<name>/` | **无目录**；index 分组 + 跨 seed 摘要 | `runs/<id>/` |
| **这一级的产物** | 声明、共享、编排、信封、报告 | 该格子下各 seed 的 **mean / std / min / max** | 这一次实测的 config / result / asset |

- `axes` → 多个 Experiment；每个 × `seeds` → 多次 Run
- 对照与结论按 **Experiment**，不按扁平 Run 列表
- `experiment_config.yaml` 是 Study 的基底默认，不是某一个 Experiment 实例

### 2.1 什么落在哪一级

**Experiment 没有自己的文件夹。** 它在磁盘上的存在是两件事：index 里一组 `factors` + 指向各 Run 的清单；以及跨这些 Run（不同 seed）算出的统计摘要。那份摘要才是 Experiment 级产物。

| 落点 | **Study** | **Experiment** | **Run** |
|------|-----------|----------------|---------|
| **声明** | `study.yaml`（`axes` / `seeds`）；基底 `experiment_config.yaml` | `axes` 的一个取值组合（`factors`） | 合并后的 `runs/<id>/config.yaml`（含 seed） |
| **编排** | `index.json`（整棵树的清单）；`scripts/` | index 里的一组：`factors` + 其下各 Run 的 `id` / `config` / `log` | index 里的一条 Run |
| **共享文件** | `shared/data/`、`shared/model/`（make 先准备，再 spawn） | 无 | 不各自下一份数据 |
| **执行摘要** | — | — | `result.json`（`status` / 本次 metrics / paths） |
| **文本日志** | 无总 log | 无 | `assets/logs/run.log`（index.`log` 指向它） |
| **数字曲线** | `docs/figures/`（按 Experiment 画 mean±std） | 跨 seed 对齐后的 history **mean / std / min / max** | `assets/tracker/`（这一次的 history / jsonl） |
| **process** | 根 `process.json`：`scope: study` 的信封（是否 complete、图路径） | 信封里的 `experiments[]`：**跨 seed** 的 metrics 与 history 的 mean / std / min / max（及 Δ baseline） | `runs/<id>/process.json`：只这一次 |
| **人文** | `docs/PLAN.md`、`docs/STUDY_REPORT.md`（按 Experiment 写结论） | 报告里的一格：该取值点的表与图 | 不单独写报告 |

读数顺序：单次数字看 **Run** `result`；该格子稳不稳、该报哪个数，看 **Experiment** 的 mean / std / min / max；整轮是否跑完、图在哪，看 **Study** 信封。

```mermaid
flowchart TB
  Study --> E1[Experiment train_size=500]
  Study --> E2[Experiment train_size=2000]
  E1 --> R10[Run seed=0]
  E1 --> R11[Run seed=1]
  E2 --> R20[Run seed=0]
```

---

## 3. 全局概念图

库内两柱：**structure** 与 **flow**。structure 含 `api`、`control`、四层、**artifact**、**make**。Flow 服务 Study；cli 是 Flow 的命令行入口。Study 下的 **config / result / asset / index / docs** 是落盘成员。

```mermaid
flowchart TB
  Study --> Experiment
  Experiment --> Run
  Study --> Structure
  Study --> Flow
  Structure --> api
  Structure --> control
  Structure --> data
  Structure --> model
  Structure --> algorithm
  Structure --> system
  Structure --> artifact_mod[artifact]
  Structure --> make
  Flow --> entry[cli]
  Flow --> prepare
  Flow --> execute
  Flow --> collect
  Flow --> summarize
  Flow --> write
  Flow --> process
  subgraph art [artifact]
    docs
    config
    result
    asset
    index
  end
  prepare --> config
  prepare <--> art
  execute <--> art
  collect --> result
  summarize --> result
  write --> result
  process --> result
```

| 概念 | 定义 |
|------|------|
| **Study** | 编排壳与 artifact 根 |
| **Experiment** | 比较轴上一个点，不含 seed；无目录；产物是该点下各 Run 的 **mean / std / min / max** |
| **Run** | 一次实测；含 seed；`id` 由内容导出 |
| **structure** | `api` + `control` + 四层 + **artifact** + **make** |
| **api** | 四层对外门面 |
| **control** | 本 Run 对四层的取值指派；一次合并 |
| **make** | 声明 → N 份 config 与 index；按 GPU 与 `round` 写出 `&` / **`wait`** 脚本（一组结束才开下一组，避免显存叠加） |
| **config** | 一次 Run 的 declarative 配置；prepare 只读 |
| **flow** | 服务 Study 的执行。同一套阶段链，用参数选择行为。每个 Run：prepare → execute → collect → summarize → write → process |
| **cli** | Flow 的命令行入口 |
| **artifact** | Study 下的持久化整体；IO 在 structure 的 artifact |
| **result** | 可序列化摘要：`status`、最终 metrics、路径。逐步曲线另见 asset |
| **asset** | 文件通道：数据集、权重、checkpoint、AlgorithmTracker 曲线、Logger 文本 |
| **AlgorithmTracker** | algorithm 层，只记数 |
| **Logger** | system 层，只打字；stdout 与 `assets/logs/` 同一套；行首带 Run `id`；`report` 拼 epoch / `elapsed` / `eta` / Loss |
| **index** | Study 编排清单；make 写入；按 Experiment 列 Run |
| **process** | 定稿后派生：Run 一份旁路；Study 信封里按 Experiment 做跨 seed 统计 |

读写：config 由 **make** 写入；result 由 Flow 的 **write** 写入；文件走 **asset**。index 由 make 写入。

---

## 4. Study

路径：`studies/<name>/`。持有声明、index、文档、共享数据、Study process 信封，以及各次 Run 产物。Experiment 的摘要嵌在信封里，不另开目录。

| 轴 | 含义 | 结果 |
|----|------|------|
| **`axes`** | 有意比较的研究因素 | → Experiment |
| **`seeds`** | 随机复测 | → 每个 Experiment 下的 Run |

入口是 flow 的 cli：`python -m rpipe`。格子由 **make** 从本目录的声明生成。

---

## 5. Experiment 与 Run

**Experiment**：含研究因素，如 `train_size`、`lr`；不含 seed；无顶层目录。科学上它就是「这个格子重复几次 seed 之后的总结」：metrics 与曲线的 **mean / std / min / max**。这些摘要写在 Study `process.json` 的 `experiments[]` 里，不另开目录。

**Run**：Experiment × seed；一 Run 对应一份 config 与 `runs/<id>/`。  
`id` 由 config 内容导出，含 tags、seed，不含 `id` 与 `description`。`baseline` 等是 **tags**。Run 只对自己负责，不算跨 seed。

---

## 6. Structure

Structure 是静态柱。四层与 **control** 描述一次 Run；**artifact** 与 **make** 覆盖整个 Study 树。能力由 Study 基底声明；每个 Run 的取值来自该次 config。

```mermaid
flowchart TB
  Structure --> api
  Structure --> control
  Structure --> data
  Structure --> model
  Structure --> algorithm
  Structure --> system
  Structure --> artifact
  Structure --> make
```

| 成员 | 职责 |
|------|------|
| **api** | 对外门面 |
| **control** | 本 Run 对四层的指派；一次 `RunConfig` |
| **make** | 循环调用 control 与 artifact，写出 N 份 config、index，以及 `&` / **`wait`** 脚本（`wait` 拦住下一组，防止显存叠加） |
| **data** | 运行时组织输入。落盘在 artifact 的 **asset** |
| **model** | 运行时构造网络。权重 / checkpoint 在 **asset** |
| **algorithm** | 怎么算。数字账本是 **AlgorithmTracker**；metric 名（Loss / Accuracy / MSE / RMSE / GLUE）在本层 `evaluate`；循环插入点是 **AlgorithmHook**。优化器、调度器、梯度裁剪、resume 是本层接口 |
| **system** | 设备、精度、并行、执行节奏；prepare 最先落地 seed / deterministic / cudnn。文本日志是 **Logger** |
| **artifact** | IO 与路径：config / result / asset，以及 Study layout |

同一 Study：不同 Experiment 差在实验变量；同一 Experiment 下不同 Run 差在 seed。  
字段与 result 快照见 [structure.md](code_structure/structure.md)。

---

## 7. Flow

Flow 服务 **Study**。同一套执行，用参数选择阶段子集、是否先 make、顺序或按 `round` 并行。

命令行入口是 **cli**。每个已写出的 Run 走：

**prepare → execute → collect → summarize → write → process**

```mermaid
flowchart LR
  prepare --> execute --> collect --> summarize --> write --> process
```

| Phase | 做什么 |
|-------|--------|
| **prepare** | 读 **config**，落地 structure；经 **artifact** 取用文件；保持 config 不变 |
| **execute** | 按 structure 计算；每个 batch 更新 AlgorithmTracker；Logger 按间隔打终端（行首 Run `id`，含 `elapsed` / `eta`）并 flush `run.log` |
| **collect** | 从 AlgorithmTracker 收最终 metrics 摘要 |
| **summarize** | 整理可序列化的 result 草稿，含 `status` |
| **write** | 把 result 写入 artifact |
| **process** | 定稿后派生。Run 阶段只写自己的 `process.json`。`rpipe process` 写 Study 信封，正文按 Experiment 做跨 seed 的 mean / std / min / max |

```mermaid
flowchart TB
  prepare --> config
  prepare <--> artifact
  execute <--> artifact
  collect --> result
  summarize --> result
  write --> result
  process --> result
  subgraph exec [execute]
    data --> algorithm
    model --> algorithm
    system --> algorithm
  end
  prepare --> exec
  exec --> collect
```

说明：

- 跑某一个 Run 时，阶段链认该 Run 的 **config** 与 **artifact**
- 真数据须显式 `data.source`
- 失败时写 `status: failed` 与 `error`
- **write** 写该 Run 的 result；**index** 由 make 写在 Study 根
- `process`：Run 阶段不写别人的文件。Study 根 `process.json` 是信封；**Experiment 级**才是跨 seed 的 mean / std / min / max

---

## 8. Artifact

**artifact** 是该 Study 的持久化整体。路径见 [LAYOUT.md](LAYOUT.md)。

| 成员 | 说明 |
|------|------|
| **docs** | Study 级：计划、报告、figures |
| **config** | Run 级；make 写、prepare 读 |
| **result** | Run 级；可序列化；含 `status` |
| **asset** | Study 级共享在 `shared/`；Run 级 tracker / `run.log` / checkpoint 在 `runs/<id>/assets/` |
| **index** | Study 级编排清单；按 Experiment 列 Run（含每条 `log`） |
| **process** | Run 一份旁路；Study 根一份信封，内嵌各 Experiment 的跨 seed 摘要 |

---

## 9. 编排生命周期

1. 写基底配置与 study 声明：`axes` 与 `seeds`
2. **make**：展开 Experiment × seed → 各 Run config 与 index；按 STUDY_GUIDE §3 同类装箱写出 `&` / `wait` 脚本。一组 `wait` 完才开下一组，免得下一波挤进还占着的显存。有独立 eval 时先并行全部 train，再跑 eval。
3. **Flow**：经 cli，按参数对 Study 下各 Run 跑阶段链；全部 wait 完后跑 Study 级 `process`
4. 读 Experiment 的 mean / std / min / max 与图，写 Study 报告（按格子下结论，不要按单条 Run）

---

## 10. 相关文档

| 文档 | 内容 |
|------|------|
| [LAYOUT.md](LAYOUT.md) | 仓库目录 |
| [CODE_STRUCTURE.md](CODE_STRUCTURE.md) | 库内树与依赖 |
| [STUDY_GUIDE.md](STUDY_GUIDE.md) | 怎么开一轮 Study |
| [code_structure/structure.md](code_structure/structure.md) | 四层 / control / make / AlgorithmTracker / Logger / artifact |
| [code_structure/flow.md](code_structure/flow.md) | Flow：cli 与阶段链 |
| [TESTING.md](TESTING.md) | 测试目录与标签 |
| [BUGS.md](BUGS.md) | 已知缺陷与跟进项 |
| [BRAINSTORM_DEEPSCIENTIST.md](BRAINSTORM_DEEPSCIENTIST.md) | 对照笔记 |
