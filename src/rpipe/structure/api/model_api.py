"""model_api: Model, ModelFactory.build, ModelConfig."""

from rpipe.structure.model import Model, ModelConfig, ModelFactory, ModelRegistry


def build(model_config: ModelConfig, assets_dir, data_meta=None):
    return ModelFactory.build(model_config, assets_dir, data_meta=data_meta)
