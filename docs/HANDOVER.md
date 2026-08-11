# RPipe Handover（压缩）

> **当前设计以 [CONCEPT.md](CONCEPT.md) 为准**；本文档仅保留 v0.2 暂停线时的历史上下文。  
> REPO_LAYOUT / CODE_STRUCTURE / PROVIDERS 等派生文档待 CONCEPT 定稿后再写。

> 用途：暂停当前实现线，**重新梳理项目**前的上下文交接。  
> 日期：2026-08-09  
> 分支：`feat/rpipe-v02-provider-foundation`（已 push；相对 `main`）  
> HEAD：`4331cbe`

---

## 1. 项目目标（最初意图）

把旧研究模板 **RPipe** 演进为 **autoresearch 可用的研究底座**（不是贝叶斯优化本身）：

1. 全栈可联调：data → model → algorithm → system → run → process →（以后）AI 报告  
2. 现代可安装包布局：`src/rpipe` + 包外 `experiments/`  
3. 去掉全局 `cfg`，显式 `RuntimeConfig`  
4. 结果/manifest schema + 可测  
5. 各层可插拔第三方（按真实 PyPI package 对齐）  
6. `lm-eval` 等是 **metric 第三方**，不是独立 eval 层  

---

## 2. 演进简史（为何会乱）

| 阶段 | 做了什么 | 问题 |
|------|----------|------|
| A | 旧 `src/` 拆成 `rpipe` 四层 + `experiments/` 编排 | 合理起点 |
| B | 引入 `plugins/` 堆第三方 provider | **结构错**：第三方成 dump |
| C | 广注册 WebDataset/Fabric/vLLM/Inspect… | **过宽**：用户只要主流 |
| D | 按 package 名收窄；lm-eval∈algorithm | 方向对，旋钮命名仍错 |
| E | 改 `metric_provider`/`trainer_backend` → algorithm/system | 仍把 accelerate 放错层 |
| F | train/metric/generate + pytorch/ggml 绑定；删 plugins；改成 `data/native` 路径 | **更接近意图，但整体仍过复杂**；用户认为问题大 |

核心教训：先把 **概念模型** 钉死，再接线；不要边接边改 taxonomy。

---

## 3. 当前代码状态（事实）

### 3.1 仓库布局

```
RPipe/
  src/rpipe/           # installable library
    provider/          # 薄 registry + bindings（仅此）
    data/              # native datasets 实现 + data/native, data/datasets
    model/             # CV 实现 + model/<provider>/
    algorithm/         # metrics 实现 + train|metric|generate/<name>/
    system/            # io/stats + backend trainers + system/pytorch|ggml
    config/ schema/
  experiments/         # 包外：suites, runner, prepare, process, artifacts, cli
  configs/ docs/ tests/
  pyproject.toml       # rpipe + entry: rpipe-run
```

### 3.2 当前旋钮（实现里的）

```yaml
data_provider: native | datasets
model_provider: native | timm | transformers | modelscope | peft | ollama | gguf
train_algorithm: native | accelerate
metric_algorithm: native | torchmetrics | evaluate | lm_eval | opencompass
generate_algorithm: null | llama_cpp | diffusers
system_provider: pytorch | ggml   # 常由 bindings 自动填
```

绑定（代码在 `rpipe/provider/api.py`）：

- `accelerate`(train) → `pytorch`
- `llama_cpp`(generate) → `ggml` + `gguf`
- `gguf`(model) → `llama_cpp` + `ggml`
- `diffusers`(generate) → `pytorch`

### 3.3 仍可用的“好东西”

- 无全局 cfg：`ExperimentConfig` → `build_runtime_cfg` → `RuntimeConfig`
- `experiments` 在包外；CLI / suites / prepare→train→test→process→artifacts
- schema：`result_blob.v1` / `run_manifest.v1`
- smoke：`python -m experiments --suite smoke`（MNIST+linear）曾跑通
- 测试规范落地：`tests/unit|integration|e2e` + markers + `docs/TEST_REPORT.md`（26 passed 量级；本机多为 CPU torch）

### 3.4 Git

- 分支：`feat/rpipe-v02-provider-foundation`
- 关键 commits：`e40879b` foundation → `e3f0b11` layer folders → `4331cbe` 删 plugins
- `main` 仍是旧模板布局（未合入）

---

## 4. 已知大问题（用户反馈 + 自评）

1. **概念仍乱**：train/metric/generate/system/bindings 叠太多旋钮；一次训练要同时想清 4–6 个字段。  
2. **accelerate / llama.cpp / gguf / diffusers 归属** 改过多次，文档与直觉仍可能不一致。  
3. **native 实现与 provider 包混层**：例如 `data/mnist.py` 与 `data/native/`、`algorithm/metrics/` 与 `algorithm/metric/` 并存，易混淆。  
4. **bindings 隐式改配置**（自动改 system/model）不好调试，也不好讲清楚。  
5. **第三方覆盖面摇摆**：曾广注册又砍掉；调研文档 `BACKEND_SURVEY.md` 与代码可能不同步。  
6. **legacy 字段太多**：`metric_provider` / `trainer_backend` / `algorithm_provider` / `pytorch_accelerator` 仍残留兼容，加重噪音。  
7. **尚未真正服务 autoresearch**：缺稳定“一次实验 → 结构化结果 → 可被 AI 消费”的最小闭环定义。

---

## 5. 建议的重启问题清单（下一轮先答这些）

重新梳理时建议**先不写代码**，只定答案：

### A. 产品最小闭环是什么？

- 只做 CV smoke（MNIST/CIFAR + native train）？  
- 还是必须同时覆盖 LLM metric（lm-eval）与本地 GGUF generate？  
- autoresearch 的输入/输出契约是什么（一份 JSON？）？

### B. 四层各自“只负责一件事”怎么定义？

建议候选（待你拍板）：

| 层 | 只做什么 | 不做什么 |
|----|----------|----------|
| data | 给出可迭代样本 | 不训、不评 |
| model | 给出可调用模型对象/句柄 | 不负责分布式 |
| algorithm | **怎么跑**（train / evaluate / generate） | 不选张量库品牌当并列旋钮？ |
| system | 设备/分布式/精度等运行时能力 | 不塞业务算法名 |

### C. 旋钮要几个？

选项 1（少）：`data` / `model` / `algorithm` / `system` 各一个名字，algorithm 自带 type。  
选项 2（现实现）：train + metric + generate 三个 algorithm 槽 + system。  
选项 3（你更倾向？）：用 **profile/stack**（如 `pytorch-train`、`ggml-llama`）代替多旋钮绑定。

### D. 目录约定

- 坚持 `data/datasets`、`algorithm/train/accelerate` 这种路径？  
- 还是 `integrations/datasets`、核心只留 native？  
- `provider/` 薄注册表要不要改名（`registry`）？

### E. 与旧代码关系

- 在本 branch 上大修？  
- 还是从 `main`/干净树按新 taxonomy 重铺，旧代码只当参考？

---

## 6. 关键设计立场（写给下一轮的“我”）

**用户已明确表达过的偏好（应保留）：**

- data 第三方只要 **`datasets`**（HF），不要流式全家桶  
- 不要 vLLM 当 model zoo；要 **modelscope**、**ollama/gguf** 相关能力  
- algorithm 只保留主流：`torchmetrics`、`lm_eval`、（可选）`opencompass`  
- **accelerate 是训练算法，不是 system**  
- **llama.cpp generate 是算法**；底层张量是 **GGML**；与 **GGUF model** 有绑定  
- **不要**把第三方全塞进 `plugins/`；路径应按层/名字组织  
- 测试要按研讨纪要：unit/integration/e2e × location/content/physical × p1–p3  

**用户最新判断：**

- 当前做法问题大 → **先 handover，再重新梳理**（本文件）

---

## 7. 下一会话建议开场

1. 确认 §5 A–E 的答案（尤其最小闭环与旋钮个数）。  
2. 画一版 **一页纸架构**（最多 4 个旋钮 + 1 张绑定表）。  
3. 再决定：修当前 branch，还是干净重铺。  
4. 代码只实现该一页纸；调研 MD / 测试规范后补。

---

## 8. 关键文件索引

| 路径 | 内容 |
|------|------|
| `docs/CONCEPT.md` | **当前设计主文档**（概念层） |
| `README.md` | 安装与快速运行 |
| `docs/TEST_REPORT.md` | 测试报告快照 |
| `src/rpipe/provider/api.py` | registry + bindings（实现，待概念对齐后重构） |
| `experiments/runner.py` | 当前 Run 编排（阶段名待与 CONCEPT 对齐） |
| `tests/` | 结构化测试 |

---

*End of handover. 下一步：重新梳理概念，不要急着加 provider。*
