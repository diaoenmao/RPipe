import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = Path(__file__).resolve().parent
HIST = SCRATCH / 'historical-4ccb28d' / 'src'
WORK = SCRATCH / ('historical-prefix-' + uuid.uuid4().hex[:12])
WORK.mkdir()
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
sys.path.insert(0, str(SCRATCH / 'deps'))
sys.path.insert(0, str(HIST))
os.chdir(HIST)
import numpy as np  # Existing Windows OpenMP import-order workaround.
import torch
import train_model as original
from config import cfg
from module import process_control
from dataset import make_dataset, process_dataset, make_data_loader
from metric import make_logger
from model import make_model, make_optimizer, make_scheduler
from rpipe.structure.data.factory import Data, DataRegistry, DataFactory
from rpipe.structure.data.config import DataConfig
from rpipe.structure.model.factory import Model, ModelRegistry, ModelFactory
from rpipe.structure.model.config import ModelConfig
from rpipe.structure.algorithm.train import TrainAlgorithm
from rpipe.structure.algorithm.config import AlgorithmConfig
from rpipe.structure.algorithm.tracker import AlgorithmTracker
from rpipe.structure.system.config import SystemConfig
from rpipe.structure.system.factory import SystemFactory
from rpipe.structure.system.runtime import apply_runtime

os.chdir(WORK)
torch.set_num_threads(2)
system_config = SystemConfig.from_mapping({'device': 'cuda', 'deterministic': True,
    'cudnn_deterministic': True, 'cudnn_benchmark': False})
start = time.perf_counter()
manifest = {}
source_manifest = json.loads((ROOT / 'studies/main_reproduction/docs/SOURCE_AFTER_B018_MANIFEST.json').read_text(encoding='utf-8'))
def verify_sources():
    assert all(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
               for path, digest in source_manifest.items())
verify_sources()
for source, dest in [
    (ROOT / 'studies/main_reproduction/shared/data/mnist/MNIST/raw', WORK / 'data/MNIST/raw'),
    (ROOT / 'studies/main_reproduction/shared/data/cifar10/cifar-10-batches-py',
     WORK / 'data/CIFAR10/raw/cifar-10-batches-py'),
]:
    shutil.copytree(source, dest)
    for path in source.rglob('*'):
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            target = dest / path.relative_to(source)
            assert digest == hashlib.sha256(target.read_bytes()).hexdigest()
            manifest[str(target.relative_to(WORK))] = digest
(WORK / 'data-copy-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
data_copy_seconds = time.perf_counter() - start


class Seen(torch.utils.data.Dataset):
    def __init__(self, dataset, as_tuple=False):
        self.dataset, self.as_tuple = dataset, as_tuple
        self.indices = []
        self.digest = hashlib.sha256()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        self.indices.append(index)
        self.digest.update(sample['data'].numpy().tobytes())
        self.digest.update(sample['target'].numpy().tobytes())
        return (sample['data'], sample['target']) if self.as_tuple else sample


class LegacyForward(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net.f(x)


def build_data(config, assets_dir, seed=None):
    datasets = make_dataset(config.name)
    train, test = Seen(datasets['train'], True), Seen(datasets['test'], True)
    data = Data(name=config.name, source=config.source, loaders={
        'test': torch.utils.data.DataLoader(test, batch_size=250, pin_memory=True, num_workers=0)},
        meta={'train_size': len(train), 'test_size': len(test), 'batch_size': 250, 'seed': seed,
              'data_size': cfg['data_shape'], 'target_size': 10,
              'config': {'pin_memory': True, 'num_workers': 0}})
    data._train_set = train
    return data


def build_model(config, assets_dir, data_meta=None):
    return Model(name=config.name, source=config.source,
                 module=LegacyForward(make_model(config.name)), meta={'ready': True})


for name in ['MNIST', 'CIFAR10']:
    DataRegistry.register(name, 'historical_2024_probe', build_data)
for name in ['linear', 'mlp', 'cnn', 'resnet18']:
    ModelRegistry.register(name, 'historical_2024_probe', build_model)


def state(module, prefix=''):
    return {key.removeprefix(prefix): value.detach().cpu().clone()
            for key, value in module.state_dict().items()}


def state_check(a, b):
    assert a.keys() == b.keys()
    floats = [key for key in a if a[key].is_floating_point()]
    integers = [key for key in a if not a[key].is_floating_point()]
    return {'max_delta': max(float((a[key] - b[key]).abs().max()) for key in floats),
            'allclose': all(torch.allclose(a[key], b[key], atol=1e-6, rtol=1e-5) for key in floats),
            'integer_buffers_equal': all(torch.equal(a[key], b[key]) for key in integers)}


class TimedTrain(TrainAlgorithm):
    def on_eval_period(self, *args, **kwargs):
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            return super().on_eval_period(*args, **kwargs)
        finally:
            torch.cuda.synchronize()
            self.eval_seconds += time.perf_counter() - started


results = []
metadata = {'historical_commit': '4ccb28d0496110253e9f8e3f3df658853f07996b',
    'work_dir': str(WORK), 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'scope': '200 optimizer-step prefix, eval200/full test10000; no long experiment; legacy data/model through temporary registries',
    'device': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'deterministic': True,
    'cudnn_benchmark': False, 'cudnn_deterministic': True, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
    'train_batch': 250, 'test_batch': 250, 'lr': 0.01, 'cosine_T_max': 80000,
    'clip_grad_norm': 1, 'seed': 0, 'raw_copied_files': len(manifest),
    'data_copy_seconds': data_copy_seconds, 'source_files_verified': len(source_manifest),
    'timing_scope': 'CUDA synchronized wall clock; original train/test include dataset hashing and metric/TensorBoard IO. Current eval hook includes tracker IO; checkpoint includes snapshot observer and disk IO. Not an acceleration benchmark.',
    'cases': results}
print(f'work_dir={WORK}', flush=True)
with (WORK / 'execution.log').open('w', encoding='utf-8') as log:
    for dataset_name in ['MNIST', 'CIFAR10']:
        for model_name in ['linear', 'mlp', 'cnn', 'resnet18']:
            case = WORK / f'{dataset_name}-{model_name}'
            case.mkdir()
            print(f'start {dataset_name}/{model_name}', flush=True)
            case_start = time.perf_counter()
            with contextlib.redirect_stdout(log):
                cfg['control'] = {'data_name': dataset_name, 'model_name': model_name}
                process_control()
                cfg.update(seed=0, iteration=0, device='cuda', eval_period=200, log_interval=0.25,
                           model_tag=f'0_{dataset_name}_{model_name}_historical_prefix200')
                assert cfg['num_steps'] == 80000 and cfg['eval_period'] == 200
                apply_runtime(0, system_config)
                torch.use_deterministic_algorithms(True)
                datasets = make_dataset(dataset_name)
                old = make_model(model_name).cuda()
                initial = state(old)
                cpu_rng = torch.get_rng_state().clone()
                cuda_rng = torch.cuda.get_rng_state().clone()
                optimizer = make_optimizer(old.parameters(), model_name)
                scheduler = make_scheduler(optimizer, model_name)
                logger = make_logger(str(case / 'original-logger'))
                datasets = process_dataset(datasets)
                source_train, source_test = Seen(datasets['train']), Seen(datasets['test'])
                loaders = make_data_loader({'train': source_train, 'test': source_test},
                                          cfg[model_name]['batch_size'])
                iterator = enumerate(loaders['train'])
                source_snapshots = []
                original_setup_seconds = time.perf_counter() - case_start
                torch.cuda.synchronize()
                original_train_start = time.perf_counter()
                original.train(iterator, old, optimizer, scheduler, logger)
                torch.cuda.synchronize()
                original_train_seconds = time.perf_counter() - original_train_start
                original_test_start = time.perf_counter()
                original.test(loaders['test'], old, logger)
                torch.cuda.synchronize()
                original_test_seconds = time.perf_counter() - original_test_start
                snapshot = {'step': cfg['iteration'], 'model': state(old),
                    'scheduler': copy.deepcopy(scheduler.state_dict()),
                    'train_loss': logger.mean['train/Loss'], 'train_accuracy': logger.mean['train/Accuracy'],
                    'test_loss': logger.mean['test/Loss'], 'accuracy': logger.mean['test/Accuracy']}
                source_snapshots.append(snapshot)
                torch.save(snapshot, case / f'original-step{cfg["iteration"]}.pt')
                final_cpu_rng = torch.get_rng_state().clone()
                final_cuda_rng = torch.cuda.get_rng_state().clone()
                logger.reset()
                logger.writer.close()
                assert cfg['iteration'] == 200
                assert len(source_train.indices) == 50000 and len(source_test.indices) == 10000
                current_setup_start = time.perf_counter()
                cfg['iteration'] = 0
                apply_runtime(0, system_config)
                torch.use_deterministic_algorithms(True)
                data = DataFactory.build(DataConfig(name=dataset_name, source='historical_2024_probe'), case, seed=0)
                model = ModelFactory.build(ModelConfig(name=model_name, source='historical_2024_probe'),
                                           case, data_meta=data.meta)
                init = state_check(initial, state(model.module, 'net.'))
                init_rng_equal = torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state())
                tracker = AlgorithmTracker(case / 'current-assets')
                system = SystemFactory.build(system_config, case / 'current-assets')
                snapshots = []
                current_checkpoint_seconds = 0.0
                save = system.save_checkpoint

                def observe(payload, name):
                    global current_checkpoint_seconds
                    torch.cuda.synchronize()
                    checkpoint_start = time.perf_counter()
                    if name == 'latest':
                        snapshots.append({'step': payload['step'], 'model': {
                            k.removeprefix('net.'): v.detach().cpu().clone() for k, v in payload['model'].items()},
                            'scheduler': copy.deepcopy(payload['scheduler'])})
                    result = save(payload, name)
                    torch.cuda.synchronize()
                    current_checkpoint_seconds += time.perf_counter() - checkpoint_start
                    return result

                system.save_checkpoint = observe
                timed_algorithm = TimedTrain(AlgorithmConfig.from_mapping({
                    'mode': 'train', 'num_steps': 200, 'eval_period': 200, 'checkpoint_period': 200,
                    'optimizer': 'SGD', 'lr': 0.01, 'momentum': 0.9, 'nesterov': True,
                    'weight_decay': 0.0005, 'scheduler': 'cosine', 'T_max': 80000,
                    'max_grad_norm': 1, 'best_metric': 'Accuracy', 'eval_num_steps': -1,
                }))
                timed_algorithm.eval_seconds = 0.0
                torch.cuda.synchronize()
                current_setup_seconds = time.perf_counter() - current_setup_start
                current_start = time.perf_counter()
                out = timed_algorithm.run(data, model, system, tracker)
                torch.cuda.synchronize()
                current_run_seconds = time.perf_counter() - current_start
                history = tracker.state_dict()['splits']
                assert len(history['test']['Accuracy']['history']) == 1
                assert len(history['train']['Accuracy']['history']) == 1
                comparisons = []
                for i, source in enumerate(source_snapshots):
                    current = next(item for item in snapshots if item['step'] == source['step'])
                    checks = state_check(source['model'], current['model'])
                    checks.update(step=source['step'], scheduler_equal=source['scheduler'] == current['scheduler'],
                        train_loss_delta=abs(source['train_loss'] - history['train']['Loss']['history'][i]),
                        train_accuracy_delta=abs(source['train_accuracy'] - history['train']['Accuracy']['history'][i]),
                        test_loss_delta=abs(source['test_loss'] - history['test']['Loss']['history'][i]),
                        accuracy_delta=abs(source['accuracy'] - history['test']['Accuracy']['history'][i]),
                        original_test_accuracy=source['accuracy'], current_test_accuracy=history['test']['Accuracy']['history'][i],
                        original_train_accuracy=source['train_accuracy'], current_train_accuracy=history['train']['Accuracy']['history'][i],
                        original_test_loss=source['test_loss'], current_test_loss=history['test']['Loss']['history'][i],
                        original_train_loss=source['train_loss'], current_train_loss=history['train']['Loss']['history'][i])
                    comparisons.append(checks)
                current_train, current_test = data._train_set, data._loaders['test'].dataset
                row = {'data': dataset_name, 'model': model_name, 'initial': init,
                    'initial_rng_equal': init_rng_equal, 'sample_order_equal': source_train.indices == current_train.indices,
                    'final_rng_equal': torch.equal(final_cpu_rng, torch.get_rng_state()) and torch.equal(final_cuda_rng, torch.cuda.get_rng_state()),
                    'augmented_train_input_equal': source_train.digest.hexdigest() == current_train.digest.hexdigest(),
                    'test_input_equal': source_test.digest.hexdigest() == current_test.digest.hexdigest(),
                    'original_train_input_sha256': source_train.digest.hexdigest(),
                    'current_train_input_sha256': current_train.digest.hexdigest(),
                    'original_test_input_sha256': source_test.digest.hexdigest(),
                    'current_test_input_sha256': current_test.digest.hexdigest(),
                    'original_train_indices_sha256': hashlib.sha256(np.asarray(source_train.indices, dtype='<i8').tobytes()).hexdigest(),
                    'current_train_indices_sha256': hashlib.sha256(np.asarray(current_train.indices, dtype='<i8').tobytes()).hexdigest(),
                    'train_samples_each': len(current_train.indices), 'test_samples_each': len(current_test.indices),
                    'comparisons': comparisons, 'current_steps': out['steps'],
                    'timing_seconds': {'original_setup': original_setup_seconds,
                        'original_train': original_train_seconds, 'original_test': original_test_seconds,
                        'current_setup': current_setup_seconds, 'current_run': current_run_seconds,
                        'current_eval_hook': timed_algorithm.eval_seconds, 'current_checkpoint': current_checkpoint_seconds,
                        'current_train_and_other': current_run_seconds - timed_algorithm.eval_seconds - current_checkpoint_seconds},
                    'actual_eval_steps': [200], 'scheduler_last_epoch': scheduler.last_epoch,
                    'next_lr': optimizer.param_groups[0]['lr']}
                row['passed'] = (init['max_delta'] == 0 and init['integer_buffers_equal'] and init_rng_equal
                    and row['final_rng_equal'] and row['sample_order_equal'] and row['augmented_train_input_equal'] and row['test_input_equal']
                    and row['train_samples_each'] == 50000 and row['test_samples_each'] == 10000 and out['steps'] == 200
                    and all(c['allclose'] and c['integer_buffers_equal'] and c['scheduler_equal']
                        and c['train_loss_delta'] <= 1e-6 and c['test_loss_delta'] <= 1e-6
                        and round(c['original_train_accuracy'] * 500) == round(c['current_train_accuracy'] * 500)
                        and round(c['original_test_accuracy'] * 100) == round(c['current_test_accuracy'] * 100)
                        for c in comparisons))
                results.append(row)
                metadata['elapsed_seconds'] = time.perf_counter() - start
                (WORK / 'comparison.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
                log.flush()
            print(f"{dataset_name}/{model_name}: passed={row['passed']} original_train={original_train_seconds:.3f}s original_test={original_test_seconds:.3f}s current_run={current_run_seconds:.3f}s", flush=True)
            assert row['passed'], row
verify_sources()
metadata['passed'] = sum(row['passed'] for row in results)
metadata['total'] = len(results)
metadata['source_files_unchanged'] = True
metadata['elapsed_seconds'] = time.perf_counter() - start
(WORK / 'comparison.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
print(f'{metadata["passed"]}/8 historical 200-step prefix passed; elapsed={metadata["elapsed_seconds"]:.3f}s', flush=True)
