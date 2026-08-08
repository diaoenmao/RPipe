# RPipe

Research Pipeline **v0.2** — 可安装研究底座（`rpipe`）+ 包外实验编排（`experiments/`）。

> 约定：每次代码改动后同步更新本 README。  
> 调研：[`docs/BACKEND_SURVEY.md`](docs/BACKEND_SURVEY.md) · 接线：[`docs/PROVIDERS.md`](docs/PROVIDERS.md)

---

## 1. 分层与目录（按 package / 实现名）

第三方**不**堆在 `plugins/`。实现就在层目录下，路径 = 旋钮名：

```
src/rpipe/
  provider/                 # 仅 registry + bindings（薄）
  data/native/              data/datasets/
  model/native/ timm/ transformers/ modelscope/ peft/ ollama/ gguf/
  algorithm/
    train/native/  train/accelerate/
    metric/native/ torchmetrics/ evaluate/ lm_eval/ opencompass/
    generate/llama_cpp/  generate/diffusers/
  system/pytorch/  system/ggml/
```

```
data → model → algorithm(train | metric | generate) → system(pytorch | ggml)
```

| 旋钮 | 取值 |
|------|------|
| `data_provider` | `native` \| `datasets` |
| `model_provider` | `native` \| `timm` \| `transformers` \| `modelscope` \| `peft` \| `ollama` \| **`gguf`** |
| `train_algorithm` | `native` \| **`accelerate`**（训练算法，不是 system） |
| `metric_algorithm` | `native` \| `torchmetrics` \| `evaluate` \| `lm_eval` \| `opencompass` |
| `generate_algorithm` | `llama_cpp` \| `diffusers` |
| `system_provider` | `pytorch` \| `ggml`（张量基底；常由绑定自动填） |

### 绑定（选一边会带上另一边）

| 选择 | 自动绑定 |
|------|----------|
| `train_algorithm: accelerate` | `system_provider: pytorch` |
| `generate_algorithm: llama_cpp` | `system: ggml` + `model: gguf`（llama.cpp 栈固定） |
| `model_provider: gguf` | `generate: llama_cpp` + `system: ggml` |
| `generate_algorithm: diffusers` | `system: pytorch` |

---

## 2. API

```python
from rpipe.provider import get_provider, get_algorithm, list_providers

list_providers()
# {'data': [...], 'model': [...],
#  'algorithm': {'train': [...], 'metric': [...], 'generate': [...]},
#  'system': ['pytorch', 'ggml']}

get_provider('data', 'datasets')
get_provider('model', 'gguf')
get_algorithm('train', 'accelerate')
get_algorithm('metric', 'lm_eval').evaluate(
    model_args='pretrained=gpt2', tasks=['gsm8k'], device='cpu'
)
get_algorithm('generate', 'llama_cpp')  # implies ggml + gguf
```

```yaml
data_provider: native
model_provider: native
train_algorithm: native          # | accelerate
metric_algorithm: native         # | torchmetrics | lm_eval | ...
# generate_algorithm: llama_cpp  # then system→ggml, model→gguf
system_provider: pytorch         # | ggml
```

---

## 3. 安装与测试

```bash
pip install -e ".[dev]"
pytest
python tests/generate_report.py
python -m experiments --suite smoke --device cpu
```

---

## Acknowledgements

*Enmao Diao*
