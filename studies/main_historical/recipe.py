"""Archived 2024 vision recipe, carried by the current RPipe training loop.

The archive remains byte-for-byte equal to Git. Only live object adapters are
used: the old dataset dict becomes a tuple, and ``model.f`` provides logits.
Import this module before RPipe prepare in every worker and call ``register``.
``preflight`` explicitly runs an original/current prefix comparison; importing
or registering never starts a training job.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[2]
COMMIT = '4ccb28d0496110253e9f8e3f3df658853f07996b'
SOURCE = 'historical_4ccb28d'
HORIZON = 80000
DATA_NAMES = ('MNIST', 'CIFAR10')
MODEL_NAMES = ('linear', 'mlp', 'cnn', 'resnet18')
ARCHIVE_ROOT = REPO / '.tmp' / 'historical-source-4ccb28d'
_ARCHIVE: Any = None


def bootstrap() -> None:
    """Load the repository and temporary dependencies without shell env setup."""
    for path in (REPO / 'src', REPO / '.tmp' / 'runtime'):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    # This import order is required by this Windows host's native runtimes.
    import numpy  # noqa: F401


def export_source() -> dict[str, Any]:
    """Export tracked blobs, refusing to overwrite a different existing file."""
    tree = subprocess.check_output(
        ['git', 'ls-tree', '-r', '-z', COMMIT], cwd=REPO
    )
    rows = []
    for entry in tree.split(b'\0'):
        if not entry:
            continue
        attributes, raw_name = entry.split(b'\t', 1)
        _, kind, oid = attributes.decode('ascii').split()
        if kind != 'blob':
            continue
        name = raw_name.decode('utf-8')
        target = ARCHIVE_ROOT / name
        payload = subprocess.check_output(['git', 'cat-file', 'blob', oid], cwd=REPO)
        if target.exists():
            if target.read_bytes() != payload:
                raise ValueError(f'archived source differs from Git: {target}')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        rows.append({'path': name, 'git_blob': oid,
                     'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)})
    manifest = {'commit': COMMIT, 'root': str(ARCHIVE_ROOT), 'files': rows}
    (ARCHIVE_ROOT / 'SOURCE_MANIFEST.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    return manifest


@contextlib.contextmanager
def _working_directory(directory: Path):
    previous = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)


class Archive:
    """Keep old top-level imports out of the process's normal module namespace."""

    def __init__(self) -> None:
        bootstrap()
        self.manifest = export_source()
        roots = ('config', 'module', 'dataset', 'model', 'metric', 'train_model')
        is_old = lambda name: any(name == root or name.startswith(root + '.') for root in roots)
        saved = {name: value for name, value in sys.modules.items() if is_old(name)}
        for name in saved:
            del sys.modules[name]
        old_argv, old_path = sys.argv, list(sys.path)
        try:
            sys.path.insert(0, str(ARCHIVE_ROOT / 'src'))
            sys.argv = ['train_model.py']
            with _working_directory(ARCHIVE_ROOT / 'src'):
                self.config = importlib.import_module('config')
                self.module = importlib.import_module('module')
                self.dataset = importlib.import_module('dataset')
                self.model = importlib.import_module('model')
                self.metric = importlib.import_module('metric')
                self.train_model = importlib.import_module('train_model')
            self.modules = {name: value for name, value in sys.modules.items() if is_old(name)}
            self.defaults = copy.deepcopy(self.config.cfg)
            for name, value in self.modules.items():
                sys.modules['_rpipe_archive_4ccb28d.' + name] = value
        finally:
            for name in list(sys.modules):
                if is_old(name):
                    del sys.modules[name]
            sys.modules.update(saved)
            sys.path[:] = old_path
            sys.argv = old_argv

    def configure(self, data_name: str, model_name: str, seed: int, device: str = 'cpu') -> dict[str, Any]:
        if data_name not in DATA_NAMES or model_name not in MODEL_NAMES:
            raise ValueError(f'unsupported historical combination: {data_name}/{model_name}')
        cfg = self.config.cfg
        cfg.clear()
        cfg.update(copy.deepcopy(self.defaults))
        cfg['control'] = {'data_name': data_name, 'model_name': model_name}
        cfg['control_name'] = f'{data_name}_{model_name}'
        cfg['model_tag'] = f'{seed}_{data_name}_{model_name}'
        self.module.process_control()
        cfg.update(seed=int(seed), iteration=0, device=device, num_samples=250 * HORIZON)
        return cfg

    def datasets(self, data_root: Path, data_name: str) -> dict[str, Any]:
        # make_dataset has a literal data/<name> root. Construction eagerly
        # loads arrays; afterwards all sample reads use those in-memory arrays.
        if data_root.name != 'data':
            raise ValueError('historical data_root must end in "data"')
        with _working_directory(data_root.parent):
            return self.dataset.make_dataset(data_name, verbose=False)


def load_archive() -> Archive:
    global _ARCHIVE
    if _ARCHIVE is None:
        _ARCHIVE = Archive()
    return _ARCHIVE


class Observation:
    """Hash actual sampled and transformed inputs; used only in preflight."""

    def __init__(self) -> None:
        self.hashes = {key: hashlib.sha256() for key in ('indices', 'images', 'targets')}
        self.samples = 0
        self.batches = 0

    def add(self, batch: dict[str, Any]) -> None:
        for key, field in (('indices', 'id'), ('images', 'data'), ('targets', 'target')):
            tensor = batch[field].detach().cpu().contiguous()
            self.hashes[key].update(tensor.numpy().tobytes())
        self.samples += len(batch['target'])
        self.batches += 1

    def result(self) -> dict[str, Any]:
        return {'samples': self.samples, 'batches': self.batches,
                **{key + '_sha256': value.hexdigest() for key, value in self.hashes.items()}}


def _observed_dicts(loader: Iterable[Any], observation: Observation):
    for batch in loader:
        observation.add(batch)
        yield batch


def register(data_root: Path | str, seed: int = 0, sampler_steps: int = HORIZON) -> Archive:
    """Register the immutable archive as source ``historical_4ccb28d``.

    Builders accept each Run's seed; the argument is a fallback for manual
    callers. The sampler keeps the full 80k budget even in a prefix probe.
    Exact resume is deliberately refused because RPipe does not persist the
    original iterator and full RNG state.
    """
    bootstrap()
    import numpy as np
    import torch
    from rpipe.structure.data.factory import Data, DataRegistry
    from rpipe.structure.model.factory import Model, ModelRegistry

    archive = load_archive()
    data_root = Path(data_root).resolve()
    if sampler_steps != HORIZON:
        raise ValueError('historical sampler horizon must remain 80000')

    class HistoricalData(Data):
        def __init__(self, *, name: str, datasets: dict[str, Any], config: dict[str, Any], run_seed: int,
                     assets_dir: Path) -> None:
            self.datasets = datasets
            self.observations: dict[str, Observation] = {}
            self._run_seed = run_seed
            super().__init__(name=name, source=SOURCE, loaders={}, meta={
                'ready': True, 'assets_dir': str(assets_dir), 'config': config,
                'seed': run_seed, 'train_size': len(datasets['train']), 'test_size': len(datasets['test']),
                'batch_size': 250, 'test_batch_size': 250,
                'data_size': [1, 28, 28] if name == 'MNIST' else [3, 32, 32], 'target_size': 10,
                'historical_commit': COMMIT, 'sampler_steps': sampler_steps,
                # Normalization is already in the original dataset transform.
                'normalization_in_dataset': True,
            })
            self._loaders['test'] = torch.utils.data.DataLoader(
                datasets['test'], batch_size=250, shuffle=False, pin_memory=True,
                num_workers=0, collate_fn=archive.dataset.input_collate,
                worker_init_fn=np.random.seed(run_seed),
            )
            self.rebind_train_steps(step=0, num_steps=sampler_steps)

        def rebind_train_steps(self, *, step: int, num_steps: int, step_period: int = 1) -> None:
            del num_steps
            if step != 0 or step_period != 1:
                raise ValueError('exact historical runs require an uninterrupted start at step 0, step_period 1')
            generator = torch.Generator().manual_seed(self._run_seed)
            sampler = torch.utils.data.RandomSampler(
                self.datasets['train'], replacement=False, num_samples=250 * sampler_steps,
                generator=generator,
            )
            self._loaders['train'] = torch.utils.data.DataLoader(
                self.datasets['train'], batch_size=250, sampler=sampler, pin_memory=True,
                num_workers=0, collate_fn=archive.dataset.input_collate,
                worker_init_fn=np.random.seed(self._run_seed),
            )

        def iter_batches(self, split: str):
            for batch in self._loaders[split]:
                if split in self.observations:
                    self.observations[split].add(batch)
                yield batch['data'], batch['target']

    def build_data(data_config: Any, assets_dir: Path, seed: int | None = None):
        run_seed = int(seed if seed is not None else register_seed)
        cfg = dict(data_config.config)
        if int(cfg.get('batch_size', 250)) != 250 or float(cfg.get('test_batch_ratio', 1)) != 1:
            raise ValueError('historical batch sizes must be train250/test250')
        if cfg.get('train_size') is not None or int(cfg.get('num_workers', 0)) != 0:
            raise ValueError('historical recipe uses full datasets and num_workers=0')
        if cfg.get('pin_memory', True) is not True:
            raise ValueError('historical recipe requires pin_memory=true')
        archive.configure(data_config.name, 'linear', run_seed)
        datasets = archive.datasets(data_root, data_config.name)
        return HistoricalData(name=data_config.name, datasets=datasets, config=cfg,
                              run_seed=run_seed, assets_dir=assets_dir)

    def build_model(model_config: Any, assets_dir: Path, data_meta: dict[str, Any] | None = None):
        if not data_meta:
            raise ValueError('historical model requires dataset metadata')
        shape = list(data_meta['data_size'])
        data_name = 'MNIST' if shape == [1, 28, 28] else 'CIFAR10'
        run_seed = int(data_meta.get('seed', register_seed))
        archive.configure(data_name, model_config.name, run_seed)
        module = archive.model.make_model(model_config.name)
        # The original instance, weights, buffers and state_dict keys survive.
        # Forward adapts the current loop's tensor input to its original f(x).
        module.forward = module.f
        return Model(name=model_config.name, source=SOURCE, module=module,
                     meta={'ready': True, 'assets_dir': str(assets_dir), 'historical_commit': COMMIT,
                           'data_size': shape, 'target_size': 10,
                           'config': dict(model_config.config), 'logits_method': 'f'})

    register_seed = int(seed)
    for name in DATA_NAMES:
        DataRegistry.register(name, SOURCE, build_data)
    for name in MODEL_NAMES:
        ModelRegistry.register(name, SOURCE, build_model)
    return archive


def _rng_state() -> dict[str, Any]:
    import torch
    return {'cpu': torch.get_rng_state().clone(),
            'cuda': [value.clone() for value in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


def _rng_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    import torch
    return torch.equal(left['cpu'], right['cpu']) and len(left['cuda']) == len(right['cuda']) and all(
        torch.equal(a, b) for a, b in zip(left['cuda'], right['cuda'])
    )


def _state_cpu(module: Any) -> dict[str, Any]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _compare_states(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    import torch
    keys = set(left) == set(right)
    max_diff = 0.0
    integers_equal = True
    close = keys
    exact = keys
    for name in set(left) & set(right):
        a, b = left[name], right[name]
        if a.shape != b.shape or a.dtype != b.dtype:
            close = exact = False
            continue
        equal = torch.equal(a, b)
        exact &= equal
        if a.is_floating_point():
            max_diff = max(max_diff, float((a - b).abs().max()))
            close &= bool(torch.allclose(a, b, atol=1e-6, rtol=1e-5))
        else:
            integers_equal &= equal
            close &= equal
    return {'keys_equal': keys, 'exact': bool(exact), 'integers_equal': bool(integers_equal),
            'max_absolute_difference': max_diff, 'within_gate': bool(close)}


def algorithm_config(steps: int = HORIZON) -> dict[str, Any]:
    if not 0 < steps <= HORIZON or steps % 200:
        raise ValueError('historical budget must be a positive multiple of 200, at most 80000')
    return {'source': 'custom_torch', 'mode': 'train', 'num_steps': steps, 'progress_unit': 'step',
            'eval_period': 200, 'eval_num_steps': -1, 'log_period': 200,
            'checkpoint': 'latest', 'checkpoint_period': 200, 'save_best': True,
            'best_metric': 'Accuracy', 'best_mode': 'max', 'best_split': 'test',
            'optimizer': 'SGD', 'lr': .01, 'momentum': .9, 'nesterov': True,
            'weight_decay': .0005, 'max_grad_norm': 1., 'scheduler': 'cosine',
            'T_max': HORIZON, 'eta_min': 0., 'resume': False}


def preflight(data_root: Path | str, output_dir: Path | str, *, steps: int = 200, seed: int = 0,
              device: str = 'cuda', combinations: Iterable[tuple[str, str]] | None = None) -> dict[str, Any]:
    """Explicit real-data probe using untouched original train/test functions.

    Uniform deterministic controls allow a cross-implementation numerical gate.
    A new output directory is required so prior evidence cannot be overwritten.
    This probe verifies a prefix, never claims historical long-curve reproduction.
    """
    bootstrap()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from rpipe.structure.algorithm.config import AlgorithmConfig
    from rpipe.structure.algorithm.train import TrainAlgorithm
    from rpipe.structure.algorithm.tracker import AlgorithmTracker
    from rpipe.structure.data.config import DataConfig
    from rpipe.structure.data.factory import DataFactory
    from rpipe.structure.model.config import ModelConfig
    from rpipe.structure.model.factory import ModelFactory
    from rpipe.structure.system.config import SystemConfig
    from rpipe.structure.system.factory import SystemFactory
    from rpipe.structure.system.runtime import apply_runtime

    mapping = algorithm_config(steps)
    archive = register(data_root, seed)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    system_config = SystemConfig.from_mapping({'device': device, 'deterministic': True,
                                              'cudnn_benchmark': False, 'cudnn_deterministic': True})
    pairs = list(combinations or [(d, m) for d in DATA_NAMES for m in MODEL_NAMES])
    result: dict[str, Any] = {'historical_commit': COMMIT, 'current_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'steps': steps, 'sampler_steps': HORIZON, 'seed': seed, 'device': device,
        'torch': torch.__version__, 'controls': {'deterministic': True, 'benchmark': False,
        'CUBLAS_WORKSPACE_CONFIG': os.environ['CUBLAS_WORKSPACE_CONFIG']},
        'recipe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'runs': []}
    start = time.perf_counter()

    class CapturedTrain(TrainAlgorithm):
        def __init__(self, config: Any) -> None:
            super().__init__(config)
            self.snapshots: dict[int, dict[str, Any]] = {}

        def make_optimizer(self, module: Any):
            self.optimizer = super().make_optimizer(module)
            return self.optimizer

        def make_scheduler(self, optimizer: Any, t_max: int):
            self.scheduler = super().make_scheduler(optimizer, t_max)
            return self.scheduler

        def on_checkpoint(self, tracker: Any, logger: Any, data: Any, model: Any,
                          system: Any, extra: dict[str, Any] | None = None) -> None:
            super().on_checkpoint(tracker, logger, data, model, system, extra)
            extra = extra or {}
            step = int(extra.get('step', 0))
            if step > 0 and step % 200 == 0:
                self.snapshots[step] = {'model': _state_cpu(model.module), 'rng': _rng_state(),
                                        'scheduler': copy.deepcopy(self.scheduler.state_dict())}

    for data_name, model_name in pairs:
        cell = output / f'{data_name}_{model_name}'
        cell.mkdir()
        archive.configure(data_name, model_name, seed, device)
        apply_runtime(seed, system_config)
        datasets = archive.datasets(Path(data_root).resolve(), data_name)
        original = archive.model.make_model(model_name).to(device)
        original_init = _state_cpu(original)
        original_init_rng = _rng_state()
        optimizer = archive.model.make_optimizer(original.parameters(), model_name)
        scheduler = archive.model.make_scheduler(optimizer, model_name)
        archive.dataset.process_dataset(datasets)
        loaders = archive.dataset.make_data_loader(datasets, archive.config.cfg[model_name]['batch_size'])
        original_obs = {'train': Observation(), 'test': Observation()}
        train_iterator = enumerate(_observed_dicts(loaders['train'], original_obs['train']))
        logger = archive.metric.make_logger(str(cell / 'original_tensorboard'))
        original_segments = []
        original_snapshots: dict[int, dict[str, Any]] = {}
        while archive.config.cfg['iteration'] < steps:
            archive.train_model.train(train_iterator, original, optimizer, scheduler, logger)
            archive.train_model.test(_observed_dicts(loaders['test'], original_obs['test']), original, logger)
            original_segments.append({'step': archive.config.cfg['iteration'],
                                      'train': {k.split('/')[1]: v for k, v in logger.mean.items() if k.startswith('train/')},
                                      'test': {k.split('/')[1]: v for k, v in logger.mean.items() if k.startswith('test/')}})
            step = int(archive.config.cfg['iteration'])
            snapshot = {'model': _state_cpu(original), 'rng': _rng_state(),
                        'scheduler': copy.deepcopy(scheduler.state_dict())}
            original_snapshots[step] = snapshot
            torch.save(snapshot, cell / f'original_step_{step}.pt')
            logger.reset()
        original_final = _state_cpu(original)
        original_rng = _rng_state()
        original_scheduler = scheduler.state_dict()
        torch.save({'model_state_dict': original_final, 'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': original_scheduler, 'logger_state_dict': logger.state_dict(),
                    'step': steps, 'rng': original_rng}, cell / 'original.pt')
        logger.writer.close()

        archive.configure(data_name, model_name, seed, device)
        apply_runtime(seed, system_config)
        assets = cell / 'current' / 'assets'
        data = DataFactory.build(DataConfig.from_mapping({'name': data_name, 'source': SOURCE,
                                 'config': {'batch_size': 250, 'test_batch_ratio': 1,
                                            'pin_memory': True, 'num_workers': 0}}), assets, seed=seed)
        model = ModelFactory.build(ModelConfig.from_mapping({'name': model_name, 'source': SOURCE}),
                                   assets, data_meta=data.meta)
        initial = _compare_states(original_init, _state_cpu(model.module))
        init_rng = _rng_equal(original_init_rng, _rng_state())
        data.observations = {'train': Observation(), 'test': Observation()}
        system = SystemFactory.build(system_config, assets)
        tracker = AlgorithmTracker(assets)
        algorithm = CapturedTrain(AlgorithmConfig.from_mapping(mapping))
        summary = algorithm.run(data, model, system, tracker)
        final = _compare_states(original_final, _state_cpu(model.module))
        final_rng = _rng_equal(original_rng, _rng_state())
        current_scheduler = algorithm.scheduler.state_dict()
        scheduler_equal = original_scheduler == current_scheduler
        # Compare numerical segments rather than rounded terminal messages.
        current_segments = []
        for index, segment in enumerate(original_segments):
            body = {'step': segment['step']}
            for split in ('train', 'test'):
                body[split] = {name: meter.history[index] for name, meter in tracker._meters[split].items()}
            current_segments.append(body)
        loss_diff = accuracy_diff = 0.
        for old, new in zip(original_segments, current_segments):
            for split in ('train', 'test'):
                loss_diff = max(loss_diff, abs(old[split]['Loss'] - new[split]['Loss']))
                accuracy_diff = max(accuracy_diff, abs(old[split]['Accuracy'] - new[split]['Accuracy']))
        input_old = {key: value.result() for key, value in original_obs.items()}
        input_new = {key: value.result() for key, value in data.observations.items()}
        inputs_equal = input_old == input_new
        segment_comparisons = []
        for old, new in zip(original_segments, current_segments):
            step = old['step']
            old_snapshot, new_snapshot = original_snapshots[step], algorithm.snapshots[step]
            parameters = _compare_states(old_snapshot['model'], new_snapshot['model'])
            rng_equal = _rng_equal(old_snapshot['rng'], new_snapshot['rng'])
            segment_scheduler_equal = old_snapshot['scheduler'] == new_snapshot['scheduler']
            counts = {split: {'original': round(old[split]['Accuracy'] * size / 100),
                             'current': round(new[split]['Accuracy'] * size / 100),
                             'samples': size}
                      for split, size in (('train', 200 * 250), ('test', 10000))}
            counts_equal = all(value['original'] == value['current'] for value in counts.values())
            segment_loss_diff = max(abs(old[split]['Loss'] - new[split]['Loss']) for split in ('train', 'test'))
            segment_comparisons.append({'step': step, 'parameters': parameters, 'rng_equal': rng_equal,
                                        'scheduler_equal': segment_scheduler_equal, 'correct_counts': counts,
                                        'correct_counts_equal': counts_equal,
                                        'maximum_loss_absolute_difference': segment_loss_diff,
                                        'passed': bool(parameters['within_gate'] and rng_equal and
                                                       segment_scheduler_equal and counts_equal and segment_loss_diff <= 1e-6)})
        correct_counts_equal = all(value['correct_counts']['test']['original'] ==
                                   value['correct_counts']['test']['current'] for value in segment_comparisons)
        train_correct_counts_equal = all(value['correct_counts']['train']['original'] ==
                                         value['correct_counts']['train']['current'] for value in segment_comparisons)
        row = {'data': data_name, 'model': model_name, 'initial_state': initial, 'final_state': final,
               'initial_rng_equal': init_rng, 'final_rng_equal': final_rng, 'scheduler_equal': scheduler_equal,
               'original_scheduler': original_scheduler, 'current_scheduler': current_scheduler,
               'inputs_equal': inputs_equal, 'original_inputs': input_old, 'current_inputs': input_new,
               'original_segments': original_segments, 'current_segments': current_segments,
               'maximum_loss_absolute_difference': loss_diff,
               'maximum_accuracy_difference_percentage_points': accuracy_diff,
               'test_correct_counts_equal': correct_counts_equal,
               'train_correct_counts_equal': train_correct_counts_equal,
               'segment_comparisons': segment_comparisons, 'summary': summary,
               'passed': bool(initial['exact'] and final['within_gate'] and init_rng and final_rng and
                              scheduler_equal and inputs_equal and correct_counts_equal and
                              train_correct_counts_equal and loss_diff <= 1e-6 and
                              all(value['passed'] for value in segment_comparisons))}
        result['runs'].append(row)
        result['passed'] = all(value['passed'] for value in result['runs'])
        result['elapsed_seconds'] = time.perf_counter() - start
        (output / 'COMPARISON.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f'preflight {data_name}/{model_name}: passed={row["passed"]} max_parameter_diff={final["max_absolute_difference"]}', flush=True)
    return result
