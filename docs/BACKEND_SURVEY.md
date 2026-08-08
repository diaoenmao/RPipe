# 各层第三方 Backend 调研审阅（按 PyPI package 名）

> **审阅日期**：2026-08-08（修订：按 package 名对齐；收窄 data/algorithm；补齐本地推理）  
> **原则**：provider 名 = 主 package 名（或公认 import 名）；每条给够 reference；先审阅再接线。  
> 接线清单见 [`PROVIDERS.md`](PROVIDERS.md)。

## 重要级

| 级 | 含义 |
|----|------|
| **P0** | 必选，深接 |
| **P1** | 高价值，应接 |
| **P2** | 可选 / 按需 |
| **Skip** | 不接 |

---

## 0. 修订结论

| 层 | Provider / 旋钮 | 说明 |
|----|-----------------|------|
| data | `native`, `datasets` | 仅此 |
| model | `native`, `timm`, `transformers`, `modelscope`, `peft`, `ollama` | 无 vLLM |
| **algorithm** | **typed**：`algorithm_type=metric\|generate` | metric≠独立层；generate=推理算法 |
| **system** | **`pytorch` \| `ggml`** | 张量库基底，不是 trainer 品牌名 |

归属纠正：

- `accelerate` → `system_provider=pytorch` + `pytorch_accelerator=accelerate`
- `diffusers` → **algorithm** `generate`，要求 `system_provider=pytorch`
- `llama_cpp` generate → **algorithm** `generate`，要求 **`system_provider=ggml`**
- 废弃错位名：`metric_provider`、`trainer_backend`

---

## 1. data

| 重要级 | Provider | PyPI package | 决策 | 说明 |
|--------|----------|--------------|------|------|
| P0 | `native` | （内置） | Adopt | MNIST/CIFAR smoke |
| P0 | **`datasets`** | [`datasets`](https://pypi.org/project/datasets/) | Adopt | HF Datasets；Hub / GSM8K / GLUE |

**References**

- PyPI: https://pypi.org/project/datasets/
- Docs: https://huggingface.co/docs/datasets
- GitHub: https://github.com/huggingface/datasets

**为何不接 WebDataset / Mosaic Streaming / LitData**  
大规模分片流式是 infra 优化，不是当前研究底座的数据面刚需；统一走 **`datasets`** 即可覆盖 Hub 与常见评测集。

---

## 2. model

| 重要级 | Provider | PyPI package | 决策 | 说明 |
|--------|----------|--------------|------|------|
| P0 | `native` | （内置） | Adopt | 研究小模型 |
| P0 | `timm` | [`timm`](https://pypi.org/project/timm/) | Adopt | CV zoo |
| P0 | `transformers` | [`transformers`](https://pypi.org/project/transformers/) | Adopt | HF 权重面 |
| P0 | **`modelscope`** | [`modelscope`](https://pypi.org/project/modelscope/) | Adopt | ModelScope 国内/多模态模型面 |
| P1 | `peft` | [`peft`](https://pypi.org/project/peft/) | Adopt | LoRA/QLoRA |
| P1 | **`ollama`** | [`ollama`](https://pypi.org/project/ollama/) | Adopt | 本地拉起量化模型（常为 GGUF）；返回 client handle |
| Skip | `vllm` | vllm | Skip | **推理/serving 引擎**，不是 model provider |

**References**

- transformers: https://pypi.org/project/transformers/ · https://huggingface.co/docs/transformers
- modelscope: https://pypi.org/project/modelscope/ · https://modelscope.cn · https://github.com/modelscope/modelscope
- ollama: https://pypi.org/project/ollama/ · https://ollama.com · https://github.com/ollama/ollama
- peft: https://pypi.org/project/peft/ · https://huggingface.co/docs/peft
- timm: https://pypi.org/project/timm/
- GGUF ↔ transformers（反量化微调，非主推理路径）: https://huggingface.co/docs/transformers/en/gguf

**GGUF 怎么摆**

| 需求 | 用哪个 package |
|------|----------------|
| 本地聊天 / 拉模型 | **`ollama`**（model 层） |
| 直接加载 `.gguf` 推理 | **`llama-cpp-python`**（system 层 `llama_cpp`） |
| 把 GGUF 反量化进 PyTorch 再训 | `transformers` + `gguf_file=`（高级，非默认） |

---

## 3. algorithm（metrics only）

| 重要级 | Provider | PyPI package | mode | 决策 | 说明 |
|--------|----------|--------------|------|------|------|
| P0 | `native` | （内置） | online | Adopt | Loss/Acc 壳 |
| P0 | `torchmetrics` | [`torchmetrics`](https://pypi.org/project/torchmetrics/) | online | Adopt | 训练态指标主流 |
| P1 | `evaluate` | [`evaluate`](https://pypi.org/project/evaluate/) | online | Adopt | Glue/SQuAD 等 |
| P0 | **`lm_eval`** | [`lm_eval`](https://pypi.org/project/lm_eval/) | benchmark | Adopt | GSM8K/MMLU 等事实标准 |
| P1 | **`opencompass`** | [`opencompass`](https://pypi.org/project/opencompass/) | benchmark | Adopt/Wire | 广覆盖 + 中英；配置驱动 |

**不接（当前）**：`inspect-ai`、`evalplus`、HELM、DeepEval、RAGAS —— 非「主流能力榜单」刚需，避免 registry 膨胀。

**References**

- lm_eval: https://pypi.org/project/lm_eval/ · https://github.com/EleutherAI/lm-evaluation-harness
- opencompass: https://pypi.org/project/opencompass/ · https://github.com/open-compass/opencompass · https://opencompass.readthedocs.io/
- torchmetrics: https://pypi.org/project/torchmetrics/
- evaluate: https://pypi.org/project/evaluate/ · https://huggingface.co/docs/evaluate

---

## 4. system（train + 本地大模型/扩散推理）

| 重要级 | Provider | PyPI package | 决策 | 说明 |
|--------|----------|--------------|------|------|
| P0 | `native` | （内置） | Adopt | 显式 step 训练循环 |
| P0 | `accelerate` | [`accelerate`](https://pypi.org/project/accelerate/) | Adopt | DDP/FSDP/DeepSpeed 门面 |
| P1 | **`llama_cpp`** | [`llama-cpp-python`](https://pypi.org/project/llama-cpp-python/) | Adopt | **GGUF 本地 LLM 推理**（import: `llama_cpp`） |
| P1 | **`diffusers`** | [`diffusers`](https://pypi.org/project/diffusers/) | Adopt | **扩散模型推理**（`DiffusionPipeline`） |
| Skip | Fabric / lightning.fabric | lightning | Skip | 非默认大众路径；训练统一 Accelerate |
| Skip | 完整 Lightning Trainer | lightning | Skip | 与自定义 step 冲突 |

**References**

- accelerate: https://pypi.org/project/accelerate/ · https://huggingface.co/docs/accelerate
- llama-cpp-python: https://pypi.org/project/llama-cpp-python/ · https://github.com/abetlen/llama-cpp-python
- llama.cpp / GGUF: https://github.com/ggml-org/llama.cpp
- diffusers: https://pypi.org/project/diffusers/ · https://huggingface.co/docs/diffusers
- FSDP vs DeepSpeed（经 Accelerate 配置，不另开 provider）: 见 HF Accelerate 文档 distributed 指南

**训练 vs 推理**

- 训练 / 微调循环 → `native` | `accelerate`
- 本地 GGUF 文本生成 → `trainer_backend: llama_cpp` + `model.arch.gguf_path`
- 扩散生成 → `trainer_backend: diffusers` + `model_name` / `hf_name`
- Ollama 拉模型聊天 → `model_provider: ollama`（daemon + `ollama` package）

---

## 5. 跨层优先序（落地队列）

1. **P0** `datasets` · `transformers` · `modelscope` · `timm` · `torchmetrics` · `lm_eval` · `accelerate` · native×4  
2. **P1** `peft` · `ollama` · `evaluate` · `opencompass` · `llama_cpp` · `diffusers`  
3. **Skip** vllm（model）· Fabric · 流式 data 包 · 长尾 eval harness  

---

## 6. Runtime 旋钮（与 package / 张量库对齐）

```yaml
data_provider: native | datasets
model_provider: native | timm | transformers | modelscope | peft | ollama
algorithm_provider: native | torchmetrics | evaluate | lm_eval | opencompass | llama_cpp | diffusers
system_provider: pytorch | ggml
pytorch_accelerator: native | accelerate   # only if system_provider=pytorch
```

| 组合 | algorithm | system |
|------|-----------|--------|
| 训练 + online metric | `native` / `torchmetrics` | `pytorch` |
| LLM benchmark metric | `lm_eval` | `pytorch` |
| GGUF generate | `llama_cpp` | **`ggml`** |
| 扩散 generate | `diffusers` | `pytorch` |

---

## 7. 变更记录

| 日期 | 变更 |
|------|------|
| 2026-08-08 | 初版广注册 |
| 2026-08-08 | **修订**：按 package 名；data 仅 `datasets`；去 vllm/Fabric/流式；加 modelscope/ollama/llama_cpp/diffusers；algorithm 主流化 |
