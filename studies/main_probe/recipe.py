"""Validate the pinned probe and register its Study-specific source."""
from __future__ import annotations

import importlib.util
import sys
import uuid


def _load(path):
    spec = importlib.util.spec_from_file_location('_probe_' + uuid.uuid4().hex, path,
                                                  submodule_search_locations=[str(path.parent)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate(config, seed, engine):
    """Reject a changed numerical recipe before preparing or constructing data."""
    if seed != 0 or config['algorithm'].get('resume_from'):
        raise ValueError('main_probe requires seed0 and a fresh Run without checkpoint resume')
    pair = (config['data']['name'], config['model']['name'])
    if pair[0] not in engine.DATA or pair[1] not in engine.MODELS:
        raise ValueError('unsupported probe pair')
    expected = dict(engine.mapping(), source='main_probe')
    if any(config['algorithm'].get(key) != value for key, value in expected.items()):
        raise ValueError('main_probe algorithm must match the pinned 60-step recipe')
    if config['model'].get('source') != 'custom_torch':
        raise ValueError('main_probe requires the native custom_torch model')
    if config['data'].get('config') != {'batch_size': 250, 'test_batch_ratio': 4,
            'pin_memory': True, 'num_workers': 0, 'augment': True}:
        raise ValueError('main_probe requires the pinned data recipe')
    if not config['system'].get('deterministic') or config['system'].get('cudnn_benchmark'):
        raise ValueError('main_probe requires deterministic execution without benchmark')
    if any(config['system'].get(flag) is not False for flag in ('cudnn_allow_tf32', 'matmul_allow_tf32')):
        raise ValueError('main_probe requires explicit cudnn_allow_tf32=false and matmul_allow_tf32=false')
    return pair


def register(ctx):
    engine = _load(ctx.study_dir / 'structure' / 'algorithm' / '__init__.py')
    pair = validate(ctx.config, ctx.seed, engine.paired)
    engine.register(ctx, pair)
