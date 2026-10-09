"""Register the paired Algorithm; never launch an independent probe script."""
from __future__ import annotations

import importlib.util
import uuid


def _load(path):
    spec = importlib.util.spec_from_file_location('_probe_' + uuid.uuid4().hex, path)
    module = importlib.util.module_from_spec(spec)
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
    return pair


def register(ctx):
    from rpipe.structure.algorithm.base import Algorithm
    from rpipe.structure.algorithm.factory import AlgorithmRegistry
    from rpipe.structure.data.factory import DataRegistry

    engine = _load(ctx.study_dir / 'execute' / 'paired.py')
    pair = validate(ctx.config, ctx.seed, engine)
    workspace = ctx.assets_dir / 'probe'
    import json
    prepared_path = workspace / 'PREPARED.json'
    if not prepared_path.is_file():
        raise RuntimeError('Study prepare.before must prepare the probe before registration')
    prepared = json.loads(prepared_path.read_text(encoding='utf-8'))
    if prepared.get('passed') is not True:
        raise RuntimeError(f'CPU preparation failed; evidence: {workspace}')
    device = str(ctx.config['system'].get('device', 'cpu'))
    engine.seed_runtime(device)

    def build_data(config, root, *, seed=None, origin=None, **kwargs):
        return engine.native_data(workspace, pair[0], ctx.assets_dir)

    class PairedAlgorithm(Algorithm):
        def __init__(self, config):
            super().__init__(config)
            self.workspace = workspace
            self.report = None
            self.initial_rng = engine.rng(device)

        def run(self, data, model, system, tracker):
            self.report = engine.run_pair(workspace, device, pair,
                (data, model, system, tracker, self.initial_rng))
            row = self.report['runs'][0]
            return dict(row['summary'], elapsed_seconds=row['elapsed_seconds'])

    DataRegistry.register(pair[0], 'main_probe', build_data)
    AlgorithmRegistry.register('train', 'main_probe', PairedAlgorithm)
