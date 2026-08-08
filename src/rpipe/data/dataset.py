import numpy as np
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from rpipe.config.registry import DATASET_REGISTRY
from rpipe.config.runtime import RuntimeConfig
from rpipe.system import apply_recursively
from .utils import Compose


def make_dataset(data_name, process=False, verbose=True, data_root='data'):
    dataset_ = {}
    if verbose:
        print('fetching data {}...'.format(data_name))
    root = os.path.join(data_root, data_name)
    dataset_cls = DATASET_REGISTRY.get(data_name)
    base_transform = Compose([transforms.ToTensor()])
    dataset_['train'] = dataset_cls(root=root, split='train', process=process, transform=base_transform)
    dataset_['test'] = dataset_cls(root=root, split='test', process=process, transform=base_transform)
    if verbose:
        print('data ready')
    return dataset_


def input_collate(input):
    def add_(input_, key=None):
        split_names = key.split('.')
        current = batch
        for split_name in split_names[:-1]:
            if split_name not in current:
                current[split_name] = {}
            current = current[split_name]
        if split_names[-1] not in current:
            current[split_names[-1]] = input_.unsqueeze(0)
        else:
            current[split_names[-1]] = torch.cat([current[split_names[-1]], input_.unsqueeze(0)], dim=0)
        return

    batch = {}
    apply_condition = lambda x: isinstance(x, torch.Tensor)
    identity_condition = lambda x: isinstance(x, (str, type(None)))
    for i in range(len(input)):
        input_i = input[i]
        apply_recursively(add_, input_i, apply_condition=apply_condition, identity_condition=identity_condition)
    return batch


def make_data_collate(collate_mode):
    if collate_mode == 'dict':
        return input_collate
    if collate_mode == 'default':
        return default_collate
    raise ValueError('Not valid collate mode')


def make_data_loader(dataset, batch_size, num_steps=None, step=0, step_period=1, pin_memory=True,
                     num_workers=0, collate_mode='dict', seed=0, shuffle=True):
    data_loader = {}
    for k in dataset:
        if k == 'train' and num_steps is not None:
            num_samples = batch_size[k] * (num_steps - step) * step_period
            if num_samples > 0:
                generator = torch.Generator()
                generator.manual_seed(seed)
                sampler = torch.utils.data.RandomSampler(
                    dataset[k], replacement=False, num_samples=num_samples, generator=generator)
                data_loader[k] = DataLoader(
                    dataset=dataset[k], batch_size=batch_size[k], sampler=sampler,
                    pin_memory=pin_memory, num_workers=num_workers,
                    collate_fn=make_data_collate(collate_mode),
                    worker_init_fn=np.random.seed(seed))
        else:
            data_loader[k] = DataLoader(
                dataset=dataset[k], batch_size=batch_size[k],
                shuffle=(shuffle if k == 'train' else False),
                pin_memory=pin_memory, num_workers=num_workers,
                collate_fn=make_data_collate(collate_mode),
                worker_init_fn=np.random.seed(seed))
    return data_loader


def process_dataset(dataset, runtime: RuntimeConfig):
    """Fill ``runtime`` with dataset meta (sizes, optional epoch→step conversion)."""
    runtime.num_samples = {k: len(dataset[k]) for k in dataset}
    if runtime.model is None:
        raise ValueError('RuntimeConfig.model must be set before process_dataset')
    if hasattr(dataset['train'], 'data_size'):
        runtime.model.data_size = dataset['train'].data_size
        runtime.model.target_size = dataset['train'].target_size
    if runtime.num_epochs is not None:
        n_train = len(dataset['train'])
        if runtime.batch_size > n_train:
            runtime.batch_size = n_train
            runtime.optimizer.batch_size = {
                'train': runtime.batch_size,
                'test': runtime.optimizer.test_batch_ratio * runtime.batch_size,
            }
        runtime.num_steps = int(np.ceil(n_train / runtime.batch_size)) * runtime.num_epochs
        runtime.eval_period = int(np.ceil(n_train / runtime.batch_size))
        runtime.optimizer.num_steps = runtime.num_steps
    return dataset
