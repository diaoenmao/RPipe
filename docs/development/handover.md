# Flow 与 Study 重构交接

**日期**：2026-10-10  
**分支**：交接分支 `refactor/flow` 的 PR [#21](https://github.com/diaoenmao/RPipe/pull/21) 已合入 `dev`（`2917e3e`）；阶段接续在 `refactor/study-phases`
**维护者决定**：**本轮不再安排重跑 `main_exp` / `main_probe` 实验**；旧实测结论以提交 [`18cd76c`](https://github.com/diaoenmao/RPipe/tree/18cd76c/studies) 的 `main_historical`、`main_reproduction` 为准。

---

## 一、摘要

### 当前接续状态（2026-10-10）

| **范围** | **接续前** | **接续后** |
| --- | --- | --- |
| 集成 | PR #21 待合并 | 三个必需检查通过，已合入 dev |
| 阶段链 | Study 阶段目录尚未实现 | `flow.study_phases: true` 显式启用；库先、Study 后；阶段源码进入 freeze；Run/Study process 分开 |
| main_exp | 根 compare.py 做终验 | prepare 做来源检查，process 包在 Study 聚合后做终验；根脚本保留手动 partial 读取 |
| main_probe | 只有同进程专用数值对照 | 保留受控计算，补两侧观测 artifact 投影和库 compare；投影不是新训练结果 |
| 本机目录 | main_historical 仍有未跟踪残留 | 按维护者要求删除；现行目录是 main_exp / main_probe |

本地验证：新增回归 16 passed；完整 core 282 passed / 35 deselected，CPU integration/e2e 32 passed / 285 deselected。完整门首次因 kornia 缺失失败，复用已有隔离依赖后通过，原失败保留。**没有重跑任何正式实验。** 详情见 [record.md](record.md) §5，现行阶段合同见 [flow.md §14.0](../code/flow.md#140-study-阶段目录)。

以下内容是 `0e3fec5` 的交接前快照，用于解释接续背景，其中“尚未实现”“待合并”均是当时状态；当前状态以上表为准。

### 接续前摘要（0e3fec5）

1. **已完成**：库侧补上 Study 扩展点（recipe、provenance、`rpipe compare`）；删除已合并的 `refactor/cleanup` 分支；两个正式 Study 改名为 `main_exp`、`main_probe` 并清掉可重跑前的旧证据与归档脚本。
2. **未完成**：Flow 未实现「Study 与库同名的阶段目录（prepare / execute / …）」链式调用；`main_probe/probe.py` 仍是单文件同进程对照，未改为 Run 目录 + `rpipe compare`。
3. **不要做的事**：不要在本交接之后自动 `launch` 长矩阵或探针矩阵；不要恢复已删的 `studies/main_reproduction/code/` 或 `main_historical/run.py` 当作现行入口。

---

## 二、代码分层（宏观）

| **层** | **路径** | **职责** | **规模（约）** |
| --- | --- | --- | --- |
| structure | `src/rpipe/structure/` | Data / Model / Algorithm、artifact、make、checkpoint | ~6400 行 Python |
| flow | `src/rpipe/flow/` | 单条 Run 阶段链 + CLI（make / launch / run-one / compare 等） | ~1400 行 |
| Study | `studies/<name>/` | 声明（yaml）+ 本实验多出来的逻辑 | 因实验而异 |

单条 Run 的库内阶段链（`FlowRunner`，见 [flow.md](../code/flow.md)）：

```text
prepare → execute → collect → summarize → write → process
```

- **prepare～write**：薄封装，主要调用 `structure` 的 API 与 artifact IO。
- **process**：Study 级聚合（`process.json`、曲线、NUMBERS），在 `launch` 全部 wait 之后由 CLI 调一次，不是每个子进程都跑。

Study **不应**再自写 `run.py`、`launch`、子进程 `run-one`、源码哈希清单脚本；这些已由库承担（§14）。

---

## 三、PR #21 库侧变更

设计正文：[flow.md §14](../code/flow.md)（recipe、provenance、compare）。

| **能力** | **入口** | **说明** |
| --- | --- | --- |
| recipe | `study.yaml` 的 `recipe: <file>.py` | prepare 在 `apply_runtime` 之后、`data_api.build` 之前 `register(ctx)`；每个 `run-one` / launch 子进程都会执行；**不进 Run id** |
| provenance | make → `provenance.json`（Study 根，gitignore） | 库 `rpipe/**/*.py`、声明、recipe、`provenance.include`、index 与各 Run `config.yaml` 的哈希 + 环境 + git |
| freeze | `study.yaml` `freeze: true` | launch 前与 prepare 时核对；变化则拒绝（退出 2 / 抛 `ProvenanceChangedError`） |
| compare | `python -m rpipe compare <run_a> <run_b>` | metrics、tracker history、checkpoint（model / optimizer / scheduler）；`--atol` / `--rtol` |

实现位置摘要：

- `src/rpipe/structure/make/recipe.py`、`structure/artifact/provenance.py`
- `flow/prepare`、`flow/summarize`、`flow/cli.py`
- `structure/artifact/readout/compare.py`
- 测试：`tests/rpipe/flow/test_study_hooks.py`、`tests/rpipe/structure/artifact/readout/test_compare.py`

本地验证（2026-10-10，`MKL_THREADING_LAYER=SEQUENTIAL`）：

- `tests/run.py --core`：272 passed / 29 deselected
- CPU integration + e2e（c1/c2，排除 external / gpu / slow）：26 passed / 275 deselected

---

## 四、两个 Study 当前树内状态

### （一）`studies/main_exp`（原 `main_historical`）

| **保留** | **作用** |
| --- | --- |
| `study.yaml`、`experiment_config.yaml` | 64 Run 声明；`recipe`、`freeze`、`provenance.include` |
| `recipe.py` | `register(ctx)`：历史 `historical_4ccb28d`；CUBLAS / 线程；拒绝中断 train 续跑；检查 `docs/PREFLIGHT.json`（需先 `recipe.py preflight`） |
| `prepare_data.py` | 数据下载与 `EXPECTED_DATA.json` 核对 |
| `compare.py` | **Study 终验**（四 seed 均值对原图估读门），不是通用 `rpipe compare` |
| `docs/TARGET.md`、`REFERENCE_CURVES.json`、`docs/reference/*.png` | 固定门与原图 |

**已删除**：`run.py`、各类 verify / publish、全部旧 `docs/*` 报告与 JSON 证据。  
**未运行**：当前树无本轮 `runs/` / 新报告；旧完整结果见 [`18cd76c` 的 main_historical](https://github.com/diaoenmao/RPipe/tree/18cd76c/studies/main_historical)。

调度命令（将来若要跑时再执行，**非本交接授权**）见 [main_exp/README.md](../../studies/main_exp/README.md)。

### （二）`studies/main_probe`（原 `main_reproduction`）

| **保留** | **作用** |
| --- | --- |
| `study.yaml`、`experiment_config.yaml` | 16 Run（8 train + 8 eval）；deterministic / 无 benchmark |
| `probe.py` | 原 main `98648f3` 与当前 RPipe **同进程** 60-step 对照；`prepare` + `run --device cuda` |

**已删除**：整个 `code/`（24 个脚本）、全部旧 `docs/` 证据。  
**未运行**：旧探针与对照见 [`18cd76c` 的 main_reproduction](https://github.com/diaoenmao/RPipe/tree/18cd76c/studies/main_reproduction)。

---

## 五、与「Study 目录镜像 Flow 阶段」的差距

当前 Study 仍把逻辑集中在少数文件（`recipe.py`、`compare.py`、`probe.py`），**没有**：

```text
studies/<name>/
  prepare/__init__.py   # run(ctx)，在库 prepare 之后可选执行
  execute/
  collect/
  summarize/
  write/
  process/
```

**目标行为**（产品/设计意向，尚未实现）：

1. `FlowRunner` 每个阶段先执行 `rpipe.flow.<phase>.run(ctx)`，再若存在 `studies/<name>/<phase>/` 且定义 `run(ctx)`，则加载并执行。
2. Study 阶段只做实验特有步骤（注册、门限、终验），不重写训练循环。
3. `prepare_data.py` 等 **make 之前** 的准备仍放在 Study 根目录，不放进六阶段目录。

实现前需先在 [flow.md](../code/flow.md) 写清加载约定、`study.yaml` 是否显式开关、与 `recipe` 的分工（recipe 偏注册，Study prepare 偏门限）。

---

## 六、Git 与 CI

| **项** | **状态** |
| --- | --- |
| `dev` 基线 | PR #19 已合并（`ea8eb39` 一带） |
| 工作分支 | `refactor/flow`（3 commits：库扩展 + historical 改接 + 改名清理） |
| 已删分支 | `refactor/cleanup`（本地与 `origin`，已进 `dev`） |
| 合并前 | 2026-10-09 查询：PR **OPEN**、**MERGEABLE**；Branch flow、Unit tests（聚合 + 四矩阵）、Build package（Ubuntu/Windows + 聚合）均为 **SUCCESS**。合入 `dev` 需维护者自行在 GitHub 点 merge |

合并后建议：在 `dev` 上继续实现 Study 阶段目录或 `probe.py` 拆分，**不要**在未授权下 push 实验产物。

---

## 七、文档入口

| **内容** | **路径** |
| --- | --- |
| Flow 阶段与 §14 扩展 | [docs/code/flow.md](../code/flow.md) |
| 目录约定 | [docs/code/layout.md](../code/layout.md) |
| Study 用法 | [studies/README.md](../../studies/README.md) |
| CI / 分支 | [docs/development/cicd.md](cicd.md) |
| 开发流水账 | [docs/development/record.md](record.md) §（一）2026-10-10 |

---

## 八、建议接续顺序（不含实验）

1. 合并 PR #21（检查 CI）。
2. 在 `flow.md` 定稿 Study 阶段链，再改 `FlowRunner` + 测试。
3. 将 `main_exp` 的 `recipe.register` 门限与 `compare.py` 按阶段拆到 `studies/main_exp/<phase>/`（或先只拆 `prepare` + `process`）。
4. 将 `main_probe/probe.py` 拆为阶段目录，或一侧输出标准 Run 树后用 `rpipe compare`（仍不跑 GPU，可先单测加载逻辑）。

**明确排除**：本列表不包含 `rpipe launch studies/main_exp` 或 `probe.py run --device cuda` 的全矩阵执行。
