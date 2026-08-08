# Provider 接线索引

调研与重要级（**按 PyPI package 名 + references**）见 [`BACKEND_SURVEY.md`](BACKEND_SURVEY.md)。

| 层 | Providers（= package / 约定名） |
|----|--------------------------------|
| data | `native`, `datasets` |
| model | `native`, `timm`, `transformers`, `modelscope`, `peft`, `ollama` |
| algorithm | `native`, `torchmetrics`, `evaluate`, `lm_eval`, `opencompass` |
| system | `native`, `accelerate`, `llama_cpp`（pkg: `llama-cpp-python`）, `diffusers` |

```yaml
data_provider / model_provider / metric_provider / trainer_backend
```
