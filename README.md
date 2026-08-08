# RPipe

Research Pipeline **v0.2** — 可安装研究底座（`rpipe`）+ 包外实验编排（`experiments/`），四层均可插拔第三方 Provider。

> 约定：每次代码改动后同步更新本 README。  
> **第三方调研（按 PyPI package + references + 重要级）**：[`docs/BACKEND_SURVEY.md`](docs/BACKEND_SURVEY.md)  
> 接线索引：[`docs/PROVIDERS.md`](docs/PROVIDERS.md)

---

## 1. 当前状态

| 项 | 状态 |
|----|------|
| 四层 Provider（名 = package） | 已落地 |
| data：`native` \| **`datasets`** | 仅此二者 |
| model：`native` \| `timm` \| `transformers` \| **`modelscope`** \| `peft` \| **`ollama`** | 已注册（无 vLLM） |
| algorithm：`native` \| `torchmetrics` \| `evaluate` \| **`lm_eval`** \| **`opencompass`** | 主流 metric |
| system：`native` \| `accelerate` \| **`llama_cpp`** \| **`diffusers`** | 训练 + 本地推理（无 Fabric） |
| schema + pytest | 已落地 |
| smoke（native） | 已跑通 |

---

## 2. 分层

```
data → model → algorithm(metrics) → system(train | local inference)
                  ├─ online:    torchmetrics / evaluate
                  └─ benchmark: lm_eval / opencompass
```

`lm_eval` 是 **metric 第三方**，不是独立 eval 层。

---

## 3. 重要级摘要

详见 [`docs/BACKEND_SURVEY.md`](docs/BACKEND_SURVEY.md)。

| 级 | 代表 package |
|----|----------------|
| **P0** | `datasets`, `transformers`, `modelscope`, `timm`, `torchmetrics`, `lm_eval`, `accelerate` |
| **P1** | `peft`, `ollama`, `evaluate`, `opencompass`, `llama-cpp-python`, `diffusers` |
| **Skip** | `vllm`（不当 model）, Fabric, WebDataset/Mosaic/LitData, 长尾 eval harness |

---

## 4. Provider API

```python
from rpipe.plugins import get_provider, list_providers

list_providers()
get_provider('data', 'datasets')
get_provider('model', 'modelscope')
get_provider('algorithm', 'lm_eval').evaluate(
    model_args='pretrained=gpt2', tasks=['gsm8k'], device='cpu'
)
```

```yaml
data_provider: native        # | datasets
model_provider: native       # | timm | transformers | modelscope | peft | ollama
metric_provider: native      # | torchmetrics | evaluate | lm_eval | opencompass
trainer_backend: native      # | accelerate | llama_cpp | diffusers
```

GGUF 本地推理：`trainer_backend: llama_cpp` + `model.arch.gguf_path`。  
Ollama：`model_provider: ollama`（需本机 Ollama daemon）。

---

## 5. 结构

```
src/rpipe/plugins/
  data/         # native, datasets
  model/        # native, timm, transformers, modelscope, peft, ollama
  algorithm/    # native, torchmetrics, evaluate, lm_eval, opencompass
  system/       # native, accelerate, llama_cpp, diffusers
docs/BACKEND_SURVEY.md
```

可选依赖：`nlp` / `vision` / `train` / `metrics` / `local_llm` / `all` / `dev`。

---

## 6. 安装与使用

```bash
pip install -e ".[dev]"
pytest
python -m experiments --suite smoke --device cpu
```

---

## 7. 已知后续

1. `opencompass.evaluate` 深接真实 runner  
2. Accelerate 透传 FSDP/DeepSpeed  
3. `lm_eval` 结果写入统一 schema  
4. AI 报告消费 artifacts  

---

## Acknowledgements

*Enmao Diao*
