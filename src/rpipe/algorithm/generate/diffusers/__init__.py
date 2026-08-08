"""algorithm/generate/diffusers — diffusion generate on system/pytorch."""

from __future__ import annotations

from typing import Any

from rpipe.provider import register_algorithm
from rpipe.provider.util import cfg_get


@register_algorithm('generate', 'diffusers')
class DiffusersGenerateAlgorithm:
    name = 'diffusers'
    package = 'diffusers'
    algorithm_type = 'generate'
    requires_system = 'pytorch'

    def available(self) -> bool:
        try:
            import diffusers  # noqa: F401
            return True
        except ImportError:
            return False

    def generate(self, runtime: Any, **kwargs) -> dict[str, Any]:
        from diffusers import DiffusionPipeline
        import torch
        name = kwargs.get('hf_name') or cfg_get(runtime, 'hf_name', 'ms_name') or getattr(runtime, 'model_name', None)
        if not name:
            raise ValueError('diffusers requires model_name / hf_name')
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        pipe = DiffusionPipeline.from_pretrained(str(name), torch_dtype=dtype)
        if torch.cuda.is_available():
            pipe = pipe.to('cuda')
        prompt = str(kwargs.get('prompt', cfg_get(runtime, 'prompt', default='a photo of a cat')))
        steps = int(kwargs.get('num_inference_steps', cfg_get(runtime, 'num_inference_steps', default=20)))
        result = pipe(prompt, num_inference_steps=steps)
        images = getattr(result, 'images', None)
        return {
            'schema': 'rpipe.inference.v1',
            'algorithm': 'diffusers',
            'algorithm_type': 'generate',
            'system': 'pytorch',
            'prompt': prompt,
            'num_images': len(images) if images is not None else 0,
            'pipeline': type(pipe).__name__,
        }
