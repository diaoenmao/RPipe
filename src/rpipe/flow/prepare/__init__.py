"""prepare: read Config, build Control, land Structure; do not modify Config."""

from __future__ import annotations

from rpipe.artifact.asset import ensure_assets, write_text_asset
from rpipe.artifact.config import load_config
from rpipe.flow.context import FlowContext
from rpipe.structure.control import control_from_config
from rpipe.structure.data import prepare_data
from rpipe.structure.model import prepare_model
from rpipe.structure.system import prepare_system


def run(ctx: FlowContext) -> None:
    cfg = load_config(ctx.layout.config_path)
    ctx.config = cfg
    ctx.control = control_from_config(cfg)
    ensure_assets(ctx.layout)
    if ctx.control.seed is not None:
        _seed_everything(ctx.control.seed)
    ctx.state['seed'] = ctx.control.seed
    ctx.state['data'] = prepare_data(ctx.control.data, ctx.layout.shared_data_dir)
    ctx.state['model'] = prepare_model(ctx.control.model, ctx.layout.shared_model_dir)
    ctx.state['system'] = prepare_system(ctx.control.system, ctx.layout.assets_dir)
    write_text_asset(ctx.layout, 'prepare.log', f'prepared control={ctx.control.id}\n')
    ctx.state.setdefault('observations', [])


def _seed_everything(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
