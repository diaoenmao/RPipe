"""Register the paired Algorithm; never launch an independent probe script."""
from __future__ import annotations

import importlib.util
import uuid


def _load(path):
    spec = importlib.util.spec_from_file_location('_probe_' + uuid.uuid4().hex, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def register(ctx):
    from rpipe.structure.algorithm.base import Algorithm
    from rpipe.structure.algorithm.factory import AlgorithmRegistry
    from rpipe.structure.data.factory import DataRegistry

    if ctx.seed != 0 or ctx.config['algorithm'].get('resume_from'):
        raise ValueError('main_probe requires seed0 and a fresh Run without checkpoint resume')
    pair = (ctx.config['data']['name'], ctx.config['model']['name'])
    engine = _load(ctx.study_dir / 'execute' / 'paired.py')
    if pair[0] not in engine.DATA or pair[1] not in engine.MODELS:
        raise ValueError('unsupported probe pair')
    expected = dict(engine.mapping(), source='main_probe')
    if any(ctx.config['algorithm'].get(key) != value for key, value in expected.items()):
        raise ValueError('main_probe algorithm must match the pinned 60-step recipe')
    if ctx.config['model'].get('source') != 'custom_torch':
        raise ValueError('main_probe requires the native custom_torch model')
    if ctx.config['data'].get('config') != {'batch_size': 250, 'test_batch_ratio': 4,
            'pin_memory': True, 'num_workers': 0, 'augment': True}:
        raise ValueError('main_probe requires the pinned data recipe')
    if not ctx.config['system'].get('deterministic') or ctx.config['system'].get('cudnn_benchmark'):
        raise ValueError('main_probe requires deterministic execution without benchmark')
    workspace = ctx.assets_dir / 'probe'
    source = ctx.study_dir.parent / 'main_exp' / 'shared' / 'data'
    raw = source / pair[0] / 'raw'
    if not raw.is_dir() or not any(raw.rglob('*')):
        raise FileNotFoundError(f'prepare the raw dataset before launch: {raw}')
    prepared = engine.prepare(workspace, source, pair)
    if prepared.get('passed') is not True:
        raise RuntimeError(f'CPU preparation failed; evidence: {workspace}')
    projector = _load(ctx.study_dir / 'write' / '__init__.py').write_observed_run
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
                (data, model, system, tracker, self.initial_rng), projector)
            if self.report.get('selected_passed') is not True:
                raise RuntimeError(f'paired numerical gate failed; evidence: {workspace / "COMPARISON.json"}')
            row = self.report['runs'][0]
            return dict(row['summary'], elapsed_seconds=row['elapsed_seconds'])

    DataRegistry.register(pair[0], 'main_probe', build_data)
    AlgorithmRegistry.register('train', 'main_probe', PairedAlgorithm)
