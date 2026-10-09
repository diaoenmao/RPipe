import contextlib
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRATCH / 'deps'))
sys.path.insert(0, str(SCRATCH / 'reference' / 'src'))
os.chdir(SCRATCH / 'reference' / 'src')
import numpy  # Import order avoids the existing Windows Conda/pip OpenMP collision.
import torch
import yaml
from config import cfg
from module import process_control
from dataset import make_data_loader
from metric import make_logger
from model import make_model, make_optimizer, make_scheduler
import train_model as original
from rpipe.structure.model.factory import ModelFactory
from rpipe.structure.model.config import ModelConfig
from rpipe.structure.data.factory import Data
from rpipe.structure.algorithm.train import TrainAlgorithm
from rpipe.structure.algorithm.config import AlgorithmConfig
from rpipe.structure.algorithm.tracker import AlgorithmTracker
from rpipe.structure.system.config import SystemConfig
from rpipe.structure.system.factory import SystemFactory

torch.set_num_threads(2)

class OriginalData(torch.utils.data.Dataset):
    def __init__(self, images, targets):
        self.images, self.targets, self.seen = images, targets, []
    def __len__(self):
        return len(self.images)
    def __getitem__(self, i):
        self.seen.append(i)
        return {'data': self.images[i], 'target': self.targets[i], 'id': torch.tensor(i)}

class NewData(torch.utils.data.TensorDataset):
    def __init__(self, *tensors):
        super().__init__(*tensors)
        self.seen = []
    def __getitem__(self, i):
        self.seen.append(i)
        return super().__getitem__(i)

def core_state(model, prefix):
    return {name.removeprefix(prefix): value.detach().clone() for name, value in model.state_dict().items()}

def delta(a, b):
    assert a.keys() == b.keys()
    return max(float((a[key] - b[key]).abs().max()) for key in a)

results = []
with (SCRATCH / 'cpu-parity-after-B018.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
    for dataset_name, shape, folder in [('MNIST', (1, 28, 28), 'mnist'), ('CIFAR10', (3, 32, 32), 'cifar10')]:
        stats = yaml.safe_load((ROOT / 'studies/main_base/shared/data' / folder / 'stats.yaml').read_text(encoding='utf-8'))
        images = torch.rand((12, *shape), generator=torch.Generator().manual_seed(17))
        targets = torch.arange(12) % 10
        for model_name in ['linear', 'mlp', 'cnn', 'resnet18']:
            case = SCRATCH / 'cpu-after-B018' / f'{dataset_name}-{model_name}'
            cfg.update(control={'data_name': dataset_name, 'model_name': model_name}, tag='cpu-parity', device='cpu')
            process_control()
            cfg.update(step=0, num_steps=4, eval_period=2, step_period=1, log_interval=0.25)
            cfg['log']['tensorboard'] = False
            cfg['model'].update(data_size=shape, target_size=10, stats=SimpleNamespace(
                mean=torch.tensor(stats['mean']), std=torch.tensor(stats['std']),
            ))
            optimizer_cfg = cfg[cfg['tag']]['optimizer']
            optimizer_cfg.update(num_steps=4, batch_size={'train': 2, 'test': 3})
            torch.manual_seed(0)
            old = make_model(cfg['model'])
            old_initial = core_state(old, 'model.')
            optimizer = make_optimizer(old.parameters(), **optimizer_cfg)
            scheduler = make_scheduler(optimizer, optimizer_cfg)
            original_train, original_test = OriginalData(images, targets), OriginalData(images[:6], targets[:6])
            loaders = make_data_loader({'train': original_train, 'test': original_test},
                {'train': 2, 'test': 3}, num_steps=4, seed=0, pin_memory=False)
            iterator = enumerate(loaders['train'])
            logger = make_logger(None, tensorboard=False, metric=cfg['metric'])
            original_snapshots = []
            for _ in range(2):
                original.train(iterator, old, optimizer, scheduler, logger)
                original.test(loaders['test'], old, logger)
                original_snapshots.append({'step': cfg['step'], 'train_loss': logger.mean['train/Loss'],
                    'test_loss': logger.mean['test/Loss'], 'accuracy': logger.mean['test/Accuracy'],
                    'model': core_state(old, 'model.'), 'scheduler': copy.deepcopy(scheduler.state_dict())})
                logger.reset()
            torch.manual_seed(0)
            model = ModelFactory.build(ModelConfig(name=model_name), case, data_meta={
                'data_size': shape, 'target_size': 10, 'mean': stats['mean'], 'std': stats['std'],
                'augment': True, 'train_aug': 'cifar' if dataset_name == 'CIFAR10' else None,
            })
            init_delta = delta(old_initial, core_state(model.module, 'net.'))
            new_train, new_test = NewData(images, targets), NewData(images[:6], targets[:6])
            data = Data(name=dataset_name, source='torch', loaders={
                'test': torch.utils.data.DataLoader(new_test, batch_size=3),
            }, meta={'train_size': 12, 'batch_size': 2, 'seed': 0})
            data._train_set = new_train
            tracker = AlgorithmTracker(case)
            system = SystemFactory.build(SystemConfig.from_mapping({'device': 'cpu'}), case)
            snapshots = []
            save = system.save_checkpoint
            def observe(payload, name):
                if name == 'latest':
                    snapshots.append(copy.deepcopy(payload))
                return save(payload, name)
            system.save_checkpoint = observe
            out = TrainAlgorithm(AlgorithmConfig.from_mapping({
                'mode': 'train', 'num_steps': 4, 'eval_period': 2, 'checkpoint_period': 2,
                'optimizer': 'SGD', 'lr': 0.1, 'momentum': 0.9, 'nesterov': True,
                'weight_decay': 0.0005, 'scheduler': 'cosine',
            })).run(data, model, system, tracker)
            new_history = tracker.state_dict()['splits']
            comparisons = []
            for i, old_snapshot in enumerate(original_snapshots):
                snapshot = next(s for s in snapshots if s['step'] == old_snapshot['step'])
                new_state = {k.removeprefix('net.'): v for k, v in snapshot['model'].items()}
                comparisons.append({'step': old_snapshot['step'], 'parameter_max_delta': delta(old_snapshot['model'], new_state),
                    'train_loss_delta': abs(old_snapshot['train_loss'] - new_history['train']['Loss']['history'][i]),
                    'test_loss_delta': abs(old_snapshot['test_loss'] - new_history['test']['Loss']['history'][i]),
                    'accuracy_delta': abs(old_snapshot['accuracy'] - new_history['test']['Accuracy']['history'][i]),
                    'scheduler_equal': snapshot['scheduler'] == old_snapshot['scheduler']})
            row = {'data': dataset_name, 'model': model_name, 'init_delta': init_delta,
                'sample_order_equal': original_train.seen == new_train.seen, 'comparisons': comparisons}
            results.append(row)
            (SCRATCH / 'cpu-parity-after-B018.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
            print(json.dumps(row), flush=True)
            assert init_delta == 0 and row['sample_order_equal'], row
            assert all(c['parameter_max_delta'] < 1e-6 and c['test_loss_delta'] < 1e-6 and
                       c['train_loss_delta'] < 1e-6 and c['accuracy_delta'] < 1e-4 and c['scheduler_equal']
                       for c in comparisons), row
print(f'{len(results)}/8 CPU original-code parity cases passed')
