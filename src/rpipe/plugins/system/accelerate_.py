from rpipe.plugins.api import register


@register('system', 'accelerate')
class AccelerateSystemProvider:
    """PyPI package: ``accelerate`` — distributed / mixed-precision training backend.

    Refs: https://pypi.org/project/accelerate/ · https://huggingface.co/docs/accelerate
    """

    name = 'accelerate'
    package = 'accelerate'

    def available(self) -> bool:
        try:
            import accelerate  # noqa: F401
            return True
        except ImportError:
            return False

    def build_trainer(self, runtime):
        from rpipe.system.backend.accelerate_ import AccelerateTrainer
        return AccelerateTrainer(runtime)
