from __future__ import annotations

from typing import Any

from rpipe.plugins.api import register


@register('model', 'ollama')
class OllamaModelProvider:
    """PyPI package: ``ollama`` — local GGUF-backed models via Ollama daemon.

    Ollama pulls/runs quantized (often GGUF) models locally; this provider returns
    a thin client handle, not a ``nn.Module``.

    Refs: https://pypi.org/project/ollama/ · https://ollama.com · https://github.com/ollama/ollama
    """

    name = 'ollama'
    package = 'ollama'

    def available(self) -> bool:
        try:
            import ollama  # noqa: F401
            return True
        except ImportError:
            return False

    def list_models(self) -> list[str]:
        return ['llama3.2', 'qwen2.5:0.5b', 'mistral', 'phi3']

    def build(self, model_cfg: dict[str, Any], **kwargs):
        if not self.available():
            raise ImportError('Install package: pip install ollama (and run the Ollama app/daemon)')
        import ollama

        name = model_cfg.get('ollama_name') or model_cfg.get('model_name') or 'llama3.2'
        host = model_cfg.get('ollama_host')
        client = ollama.Client(host=host) if host else ollama.Client()
        # Optional pull so first use is deterministic when requested
        if model_cfg.get('pull', False):
            client.pull(name)
        return _OllamaHandle(client=client, model=name)


class _OllamaHandle:
    """Minimal chat/generate facade for Trainer/eval glue."""

    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def generate(self, prompt: str, **kwargs) -> str:
        r = self.client.generate(model=self.model, prompt=prompt, **kwargs)
        return r.get('response', '') if isinstance(r, dict) else getattr(r, 'response', str(r))

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        r = self.client.chat(model=self.model, messages=messages, **kwargs)
        if isinstance(r, dict):
            return r.get('message', {}).get('content', '')
        return getattr(getattr(r, 'message', None), 'content', str(r))
