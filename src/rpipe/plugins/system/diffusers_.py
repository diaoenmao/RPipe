from __future__ import annotations

from rpipe.plugins.api import register
from rpipe.plugins.system.llama_cpp_ import _cfg_get


@register('system', 'diffusers')
class DiffusersSystemProvider:
    """PyPI package: ``diffusers`` — diffusion model inference backend.

    Loads ``DiffusionPipeline`` for image (and related) generation.
    Not a substitute for Accelerate training loops.

    Refs: https://pypi.org/project/diffusers/ · https://huggingface.co/docs/diffusers
    """

    name = 'diffusers'
    package = 'diffusers'

    def available(self) -> bool:
        try:
            import diffusers  # noqa: F401
            return True
        except ImportError:
            return False

    def build_trainer(self, runtime):
        return DiffusersInferenceRunner(runtime)


class DiffusersInferenceRunner:
    def __init__(self, runtime):
        self.runtime = runtime

    def run(self):
        from diffusers import DiffusionPipeline
        import torch

        r = self.runtime
        name = _cfg_get(r, 'hf_name', 'ms_name') or getattr(r, 'model_name', None)
        if not name:
            raise ValueError('diffusers requires model_name or model.arch.hf_name')
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        pipe = DiffusionPipeline.from_pretrained(str(name), torch_dtype=dtype)
        if torch.cuda.is_available():
            pipe = pipe.to('cuda')
        prompt = str(_cfg_get(r, 'prompt', default='a photo of a cat'))
        steps = int(_cfg_get(r, 'num_inference_steps', default=20))
        result = pipe(prompt, num_inference_steps=steps)
        images = getattr(result, 'images', None)
        return {
            'schema': 'rpipe.inference.v1',
            'provider': 'diffusers',
            'package': 'diffusers',
            'prompt': prompt,
            'num_images': len(images) if images is not None else 0,
            'pipeline': type(pipe).__name__,
        }
