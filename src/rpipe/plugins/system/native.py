from rpipe.plugins.api import register


@register('system', 'native')
class NativeSystemProvider:
    name = 'native'

    def available(self) -> bool:
        return True

    def build_trainer(self, runtime):
        from rpipe.system.backend.native import NativeTrainer
        return NativeTrainer(runtime)
