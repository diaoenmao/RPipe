"""Manual historical/current stage calibration, invoked by recipe.py, not a hook."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid


def main(recipe, argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['cnn', 'resnet18'], required=True)
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--order', choices=['old-first', 'current-first'], default='old-first')
    args = parser.parse_args(argv)
    mapping = recipe.algorithm_config(args.steps)
    if args.steps < 800:
        parser.error('at least 800 steps: warmup plus three measured segments')
    recipe.bootstrap()
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
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

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    controls = SystemConfig.from_mapping({'device': 'cuda', 'deterministic': True,
        'cudnn_deterministic': True, 'cudnn_benchmark': False,
        'cudnn_allow_tf32': False, 'matmul_allow_tf32': False})
    ident = 'speed-' + time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
    output = recipe.REPO / '.tmp' / ident
    output.mkdir()
    archive = recipe.install(recipe.STUDY / 'shared/data', 0)
    report = {'measurement_id': ident, 'historical_commit': recipe.COMMIT,
        'current_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=recipe.REPO, text=True).strip(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'recipe_sha256': hashlib.sha256(Path(recipe.__file__).read_bytes()).hexdigest(),
        'data_manifest_sha256': hashlib.sha256((recipe.STUDY / 'docs/DATA_MANIFEST.json').read_bytes()).hexdigest(),
        'torch': torch.__version__, 'device': torch.cuda.get_device_name(),
        'data': 'CIFAR10', 'model': args.model, 'seed': 0, 'steps': args.steps,
        'sampler_scheduler_steps': 80000, 'batch': 250, 'test_size': 10000,
        'eval_period': 200, 'cpu_threads': 2, 'order': args.order,
        'controls': {'deterministic': True, 'benchmark': False, 'cudnn_tf32': False, 'matmul_tf32': False},
        'scope': 'serial real train/test/checkpoint; first segment warmup; no probe input hashing or snapshots',
        'output_dir_local_only': str(output), 'sides': {}}

    def sync():
        torch.cuda.synchronize()

    def timed(call):
        sync()
        start = time.perf_counter()
        result = call()
        sync()
        return result, time.perf_counter() - start

    def old():
        archive.configure('CIFAR10', args.model, 0, 'cuda')
        apply_runtime(0, controls)
        start = time.perf_counter()
        datasets = archive.datasets(recipe.STUDY / 'shared/data', 'CIFAR10')
        module = archive.model.make_model(args.model).to('cuda')
        initial = recipe._state_cpu(module)
        optimizer = archive.model.make_optimizer(module.parameters(), args.model)
        scheduler = archive.model.make_scheduler(optimizer, args.model)
        archive.dataset.process_dataset(datasets)
        loaders = archive.dataset.make_data_loader(datasets, archive.config.cfg[args.model]['batch_size'])
        iterator = enumerate(loaders['train'])
        logger = archive.metric.make_logger(str(output / 'old/logger'))
        sync()
        setup = time.perf_counter() - start
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        segments = []
        while archive.config.cfg['iteration'] < args.steps:
            _, train = timed(lambda: archive.train_model.train(iterator, module, optimizer, scheduler, logger))
            _, evaluation = timed(lambda: archive.train_model.test(loaders['test'], module, logger))
            metrics = {k: float(v) for k, v in logger.mean.items() if k.startswith('test/')}
            def save():
                payload = {'cfg': archive.config.cfg, 'model_state_dict': module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                    'logger_state_dict': logger.state_dict()}
                archive.module.save(payload, str(output / 'old/checkpoint/model'))
                if logger.compare('test'):
                    (output / 'old/best').mkdir(exist_ok=True)
                    shutil.copy(output / 'old/checkpoint/model', output / 'old/best/model')
                logger.reset()
            _, checkpoint = timed(save)
            segments.append({'step': int(archive.config.cfg['iteration']), 'train_seconds': train,
                'eval_seconds': evaluation, 'checkpoint_seconds': checkpoint, 'test': metrics})
        sync()
        side = {'setup_seconds': setup, 'run_seconds': time.perf_counter() - start,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'segments': segments}
        final = recipe._state_cpu(module)
        logger.writer.close()
        return initial, final, side

    def current():
        archive.configure('CIFAR10', args.model, 0, 'cuda')
        apply_runtime(0, controls)
        start = time.perf_counter()
        assets = output / 'current/assets'
        data = DataFactory.build(DataConfig.from_mapping({'name': 'CIFAR10', 'source': recipe.SOURCE,
            'config': {'batch_size': 250, 'pin_memory': True, 'num_workers': 0}}), assets, seed=0)
        model = ModelFactory.build(ModelConfig.from_mapping({'name': args.model, 'source': recipe.SOURCE}),
            assets, data_meta=data.meta)
        initial = recipe._state_cpu(model.module)
        system = SystemFactory.build(controls, assets)
        tracker = AlgorithmTracker(assets)
        segments = []
        class MeasuredTrain(TrainAlgorithm):
            def on_eval_period(self, *pos, **kw):
                sync()
                before = time.perf_counter()
                train = before - self.segment_start
                result = super().on_eval_period(*pos, **kw)
                sync()
                extra = pos[5] if len(pos) > 5 else kw.get('extra', {})
                segments.append({'step': int(extra['step']), 'train_and_bookkeeping_seconds': train,
                    'eval_seconds': time.perf_counter() - before, 'checkpoint_seconds': 0.})
                return result

            def on_checkpoint(self, *pos, **kw):
                _, elapsed = timed(lambda: super(MeasuredTrain, self).on_checkpoint(*pos, **kw))
                if segments:
                    segments[-1]['checkpoint_seconds'] += elapsed
                self.segment_start = time.perf_counter()
        algorithm = MeasuredTrain(AlgorithmConfig.from_mapping(mapping))
        sync()
        setup = time.perf_counter() - start
        torch.cuda.reset_peak_memory_stats()
        algorithm.segment_start = time.perf_counter()
        summary, elapsed = timed(lambda: algorithm.run(data, model, system, tracker))
        histories = {k: list(v.history) for k, v in tracker._meters['test'].items()}
        for i, row in enumerate(segments):
            row['test'] = {'test/' + k: float(v[i]) for k, v in histories.items()}
        side = {'setup_seconds': setup, 'run_seconds': elapsed,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'segments': segments, 'summary': summary}
        return initial, recipe._state_cpu(model.module), side

    states = {}
    for label in (['old', 'current'] if args.order == 'old-first' else ['current', 'old']):
        initial, final, side = (old if label == 'old' else current)()
        states[label] = (initial, final)
        report['sides'][label] = side
        torch.cuda.empty_cache()
    report['initial_state'] = recipe._compare_states(states['old'][0], states['current'][0])
    report['final_state'] = recipe._compare_states(states['old'][1], states['current'][1])
    diffs = []
    for left, right in zip(report['sides']['old']['segments'], report['sides']['current']['segments']):
        assert left['step'] == right['step']
        diffs.append({'step': left['step'], 'loss_absolute_difference': abs(left['test']['test/Loss'] - right['test']['test/Loss']),
            'accuracy_absolute_difference': abs(left['test']['test/Accuracy'] - right['test']['test/Accuracy'])})
    report['numeric_segments'] = diffs
    report['passed'] = report['initial_state']['exact'] and report['final_state']['within_gate'] and all(
        d['loss_absolute_difference'] <= 1e-6 and d['accuracy_absolute_difference'] == 0 for d in diffs)
    target = recipe.STUDY / 'docs/speed_measurements' / (ident + '.json')
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(target), 'passed': report['passed']}, ensure_ascii=False), flush=True)
    return 0 if report['passed'] else 1
