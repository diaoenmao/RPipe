# Provider 接线索引

路径即名字：`data/native`、`algorithm/train/accelerate`、`system/ggml`…  
Registry 仅在薄包 `rpipe.provider`；**不要**再往 `plugins/` 堆第三方。

## 旋钮

| 旋钮 | 目录 | 取值 |
|------|------|------|
| `data_provider` | `data/<name>/` | `native`, `datasets` |
| `model_provider` | `model/<name>/` | `native`, `timm`, `transformers`, `modelscope`, `peft`, `ollama`, `gguf` |
| `train_algorithm` | `algorithm/train/<name>/` | `native`, `accelerate` |
| `metric_algorithm` | `algorithm/metric/<name>/` | `native`, `torchmetrics`, `evaluate`, `lm_eval`, `opencompass` |
| `generate_algorithm` | `algorithm/generate/<name>/` | `llama_cpp`, `diffusers` |
| `system_provider` | `system/<name>/` | `pytorch`, `ggml` |

## 绑定

- `accelerate`（train）→ `pytorch`
- `llama_cpp`（generate）→ `ggml` + `gguf`（整栈绑定）
- `gguf`（model）→ `llama_cpp` + `ggml`
- `diffusers`（generate）→ `pytorch`

详见 [`BACKEND_SURVEY.md`](BACKEND_SURVEY.md)。
