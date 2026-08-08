"""model/ollama — PyPI ``ollama`` (local daemon models, often GGUF-backed)."""

from __future__ import annotations

from typing import Any

from rpipe.provider import register


@register('model', 'ollama')
class OllamaModelProvider:
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
            raise ImportError('Install package: pip install ollama (and run Ollama daemon)')
        import ollama
        name = model_cfg.get('ollama_name') or model_cfg.get('model_name') or 'llama3.2'
        host = model_cfg.get('ollama_host')
        client = ollama.Client(host=host) if host else ollama.Client()
        if model_cfg.get('pull', False):
            client.pull(name)
        return _OllamaHandle(client=client, model=name)


class _OllamaHandle:
    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def generate(self, prompt: str, **kwargs) -> str:
        r = self.client.generate(model=self.model, prompt=prompt, **kwargs)
        return r.get('response', '') if isinstance(r, dict) else getattr(r, 'response', str(r))
