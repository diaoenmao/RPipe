# Brainstorm: 借鉴 DeepScientist，不改变 RPipe 定位

> **状态**：头脑风暴草稿，**不是** CONCEPT / LAYOUT 权威。  
> **参照**：[ResearAI/DeepScientist](https://github.com/ResearAI/DeepScientist)（README、[13 Core Architecture](https://github.com/ResearAI/DeepScientist/blob/main/docs/en/13_CORE_ARCHITECTURE_GUIDE.md)、[90 Architecture](https://github.com/ResearAI/DeepScientist/blob/main/docs/en/90_ARCHITECTURE.md)、[论文](https://arxiv.org/abs/2509.26603)）  
> **目的**：从相邻系统抽可借的机制，标清边界，给 RPipe 后续演进选项。

---

## 1. 一句话对照

| | DeepScientist | RPipe（当前） |
|--|---------------|---------------|
| **定位** | local-first **研究操作系统 / 长程 AI 研究工作室** | **可重复、可编排、可序列化的研究执行底座** |
| **主角** | Agent + Quest + Memory + UI | Study → Experiment → Run + Structure / Flow / Artifact |
| **持久化** | Quest Git 仓 + memory cards + artifact 记录 + events | Run 目录下 Config / Result / Asset（`id` 哈希） |
| **决策循环** | Findings Memory + Bayesian / Research Map + skill 驱动 | 包外 Study / autoresearch **消费** Result；库内不替你想下一步 |
| **执行** | `bash_exec` 长会话 + runner（Codex / Claude…） | Flow：`prepare → execute → collect → …`，读 Config、写 Result |

**借力原则**：DeepScientist 擅长「怎么持续研究」；RPipe 应继续擅长「一次 Run 怎么被诚实、可复现地执行与落盘」。借鉴时优先 **契约与可消费状态**，少碰 **OS / UI / agent 调度**。

---

## 2. DeepScientist 里真正硬的几件事

### 2.1 状态比对话更长寿

- **One quest = one Git repo**：分支 / worktree 表达研究路线，而不是 chat 历史。
- 失败路径保留、摘要、复用，而不是覆盖。
- Web / TUI / connector 都只是 **同一 durable state** 的表面。

### 2.2 刻意收窄的 MCP 面

内置只公开三个命名空间：

| MCP | 角色 |
|-----|------|
| `memory` | 可复用知识（笔记、失败教训、稳定 caveat） |
| `artifact` | 研究控制面（baseline、experiment 记录、里程碑、交互） |
| `bash_exec` | 可停、可回看的长 shell 会话 |

Git / connector / runtime 工具不直接暴露成公共 MCP——行为收进 `artifact` 或 daemon。

### 2.3 Prompt / Skill 驱动，而不是巨型硬编码阶段机

Daemon 负责路由与持久化；阶段纪律主要在 `system.md` + `SKILL.md`。运行时保持薄。

### 2.4 Baseline 门禁 + 分层保真度

论文侧：想法 → 廉价验证 → 昂贵验证；只有有希望的路线升级。Baseline 带 **MetricContract**，后续实验对照同一合同。

### 2.5 Human takeover 一等公民

随时暂停、改计划、改代码、再交回；`artifact.interact` 把协作线程与 checkpoint 绑在长跑过程上。

### 2.6 启动合同（Startup contract）

`Start Research` 不只建目录，还固化：目标、参考文献 / baseline、约束、决策策略——比「随便一句 prompt」更可执行。

---

## 3. 概念映射（便于讨论，不改词表）

| DeepScientist | 靠近的 RPipe 概念 | 差在哪 |
|---------------|-------------------|--------|
| Quest | Study（或 Study + 工作区） | Quest 还含 agent 线程、Git 史、UI；Study 当前偏编排轴 |
| Baseline | 某次「权威」Run / 固定 Config 的 Result | RPipe 尚无正式 baseline / MetricContract |
| Experiment round | Experiment 下的一组 Run | DS 的 experiment 更偏「一次尝试记录」；RPipe Experiment 是类型 + Flow |
| Artifact（控制面） | Artifact（Config/Result/Asset）+ 未来的索引 | 名字撞车：DS 的 artifact ≈ 研究账本；RPipe ≈ 单次 Run 落盘根 |
| Findings Memory | Result 之上的「跨 Run 记忆」层 | RPipe 故意未做；留给 autoresearch |
| Canvas / Research Map | Study 对比视图 / 索引 | RPipe 无 UI；可先做文件级 index |
| `bash_exec` | Flow.execute / 外部 runner | RPipe 要的是结构化 Flow，不是通用 PTY 底座 |
| Skill / prompt | 包外编排脚本、未来 agent adapter | 不宜塞进 `src/rpipe` 核心 |

---

## 4. 建议借 / 不借

### 4.1 值得借（贴合底座）

1. **Result 作为稳定消费契约**  
   让 autoresearch / agent 只依赖 Result schema + 路径索引，而不是解析日志。对齐 DS「workspace 从 durable state 重建」的思路。

2. **Baseline + MetricContract（轻量）**  
   Study 声明：哪次 Run 是 baseline；比较哪些 metric、更高更好还是更低更好、允许的噪声。不必上 Bayesian。

3. **失败也是一等产物**  
   失败 Run 保留 `id` 目录、Result 写 `status` / `error` / 简短原因；禁止「失败就删」。对齐 DS「failed paths are assets」。

4. **Study 级 brief / 启动合同（文件即可）**  
   例如 `examples/studies/<name>/brief.yaml`：目标、baseline 指针、变量轴、预算（最多几次 Run）、禁止改动的层。不引入 daemon。

5. **跨 Run 的 Findings 摘要（包外或薄库）**  
   从多个 `result.json` 汇总成 `findings.md` / `findings.json`（试了什么、赢了什么、别再试什么）。这是 DS Findings Memory 的最小切片。

6. **窄 MCP / API 面给 agent（可选远期）**  
   若要给 Cursor / Codex 接 RPipe：只暴露类似  
   - `run.read_config` / `run.read_result`  
   - `study.list_runs` / `study.compare`  
   - `flow.launch(run_dir)`  
   不要一次暴露整个文件系统与任意 shell。

7. **Registry-first 扩展**  
   DS 用小 registry 扩 runner / channel；RPipe 已有层 Registry/Factory——保持「注册表扩展，少写巨型 if」。

### 4.2 明确不借（避免定位漂移）

| 不借 | 原因 |
|------|------|
| Quest = 独立 Git 仓 + daemon + Web/TUI | RPipe 是库 / 底座，不是 OS |
| 内置 Bayesian / Research Map 决策器 | 属于 autoresearch；底座只提供诚实反馈 |
| Prompt/Skill 当 Flow 调度器 | Flow 阶段应稳定、可测；skill 应在包外 |
| 通用 `bash_exec` 取代 Structure/Flow | 会毁掉可重复与 Config↔Result 契约 |
| 论文写作 / PDF / connector 全家桶 | 超出执行底座边界 |
| 把 RPipe Artifact 改名去对齐 DS | 词表已锁定；文档里区分即可 |

---

## 5. 可落地的 brainstorm 方向（按优先级）

### P0 — 让「下一轮决策者」更好用（仍属底座）

- **Result schema 硬化**：`control` 快照、`metrics`、`status`、`error`、`paths`、可选 `baseline_id`。  
- **Run 生命周期**：`pending | running | succeeded | failed | aborted`，写进 Result。  
- **Study 清单**：`artifact/*/result.json` → 一张 `index.json`（id、seed、metrics、status），对应 DS Canvas 的「文件重建」极简版。

### P1 — Baseline 与比较

- Study 配置：`baseline: <run_id>` + `metrics: [{name, direction}]`。  
- 包外一行工具：对比当前 Run vs baseline，输出 `delta` 表。  
- e2e：mnist_seeds 声明 seed_0 对应 id 为 baseline（或单独 baseline Run）。

### P2 — Findings 薄层（包外优先）

- `summarize_study(exp_dir) -> findings.json`：聚合 metrics、标出 Pareto / 最佳、列出失败原因。  
- 文档约定：autoresearch **只读** findings + Result，不直接改 Flow。

### P3 — Agent 接缝（可选）

- 文档化「RPipe 作为 MCP 工具集」草图（只读 Result + 启动已存在 Config 的 Flow）。  
- 不实现 daemon；用现有 `launch` / Study 脚本当工具后端即可验证。

### P4 — 显式「人类接管」钩子（很轻）

- Config 或 Study brief 里 `require_approval: true` → launch 前停住打印路径，等人改 Config 再继续。  
- 对齐 DS human takeover，但零 UI。

---

## 6. 与现有 mnist_linear 路径的关系

当前已通：

```text
Study (mnist_seeds)
  → grid: experiment_config ⊕ {seed} → artifact/<id>/config.yaml
  → launch: Flow(prepare→execute→collect) → result.json
```

若按本 brainstorm 演进，下一小步可以是：

```text
同上
  → collect 写 status / metrics（已有 loss stub）
  → study 结束写 index.json + 可选 findings.json
  → brief.yaml 声明 baseline_id + 比较哪些 metric
```

不需要引入 quest / Canvas / MCP 才能验证价值。

---

## 7. 开放问题（下次讨论用）

1. **Findings 放哪？** 纯包外 `examples/studies/...`，还是将来 `rpipe.study` 薄模块？  
2. **Baseline 是 Run 指针还是独立 Artifact 角色？**  
3. **id 目录与「研究路线分支」**：RPipe 用 content-hash id；DS 用 Git branch。要不要可选 `label`（不进 hash）仅供人读？  
4. **与 DeepScientist 共存时**：RPipe 是否定位为「可被 DS / 任意 agent 调用的执行引擎」？若是，Result schema 稳定性优先于功能面。  
5. **失败重跑**：同 content 已有成功 Result 时，是跳过、写 `id_timestamp`，还是强制失败覆盖策略写进 Study brief？

---

## 8. 建议结论（供拍板）

- **学机制，不学产品形态**：durable 契约、baseline、失败保留、窄 API、人类可介入。  
- **决策与 UI 继续外包**：Findings / Map / Agent 属于消费方；RPipe 把 Result 做硬。  
- **下一刀最小**：Result `status` + Study `index.json` + brief 里的 baseline 指针——用现有 mnist_seeds 验证，再决定要不要 Findings / MCP。

若采纳某一条，再开单独设计短文写进 CONCEPT / LAYOUT；本文保持 brainstorm，避免污染权威词表。
