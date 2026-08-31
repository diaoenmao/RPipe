"""prepare: read Config, build Control, land Structure; do not modify Config."""

from __future__ import annotations

from rpipe.flow.context import FlowContext
from rpipe.structure.api import algorithm_api, data_api, model_api, system_api
from rpipe.structure.artifact.asset import ensure_assets
from rpipe.structure.artifact.config import load_config
from rpipe.structure.control import control_from_config, validate_control
from rpipe.structure.control.layers import AlgorithmConfig, DataConfig, ModelConfig, SystemConfig


def run(ctx: FlowContext) -> None:
    cfg = load_config(ctx.layout.config_path)
    ctx.config = cfg
    ctx.control = control_from_config(cfg)
    validate_control(ctx.control)
    ensure_assets(ctx.layout)
    if ctx.control.seed is not None:
        _seed_everything(ctx.control.seed)
    ctx.state['seed'] = ctx.control.seed

    system = system_api.build(
        SystemConfig.from_mapping(ctx.control.system),
        ctx.layout.assets_dir,
    )
    data = data_api.build(
        DataConfig.from_mapping(ctx.control.data),
        ctx.layout.shared_data_dir,
    )
    model = model_api.build(
        ModelConfig.from_mapping(ctx.control.model),
        ctx.layout.shared_model_dir,
    )
    if model.module is not None:
        model.module = system.place_module(model.module)
    algorithm = algorithm_api.build(AlgorithmConfig.from_mapping(ctx.control.algorithm))
    tracker = algorithm_api.make_tracker(ctx.layout.assets_dir)

    ctx.state['system'] = system
    ctx.state['data'] = data
    ctx.state['model'] = model
    ctx.state['algorithm'] = algorithm
    ctx.state['tracker'] = tracker
    ctx.state['logger'] = system.logger
    ctx.state.setdefault('observations', [])
    system.logger.info(f'prepared control={ctx.control.id}')


def _seed_everything(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
