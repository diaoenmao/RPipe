from __future__ import annotations

import os

import torch

from rpipe.config import ModelRuntime, OptimizerRuntime, RuntimeConfig
from rpipe.data import make_dataset, make_data_loader, process_dataset
from rpipe.system import Stats, makedir_exist_ok, save


def prepare_datasets(data_names, output_root='output', dim=1, force=False, device='cpu'):
    """Download/process datasets and write mean/std under output/stats/."""
    stats_path = os.path.join(output_root, 'stats')
    makedir_exist_ok(stats_path)
    prepared = []
    with torch.no_grad():
        for data_name in data_names:
            stats_file = os.path.join(stats_path, data_name)
            if os.path.exists(stats_file) and not force:
                print('[prepare] skip stats (exists): {}'.format(data_name))
                prepared.append(data_name)
                continue
            print('[prepare] dataset + stats: {}'.format(data_name))
            runtime = RuntimeConfig(
                control_name=f'{data_name}_stats',
                data_name=data_name,
                model_name='linear',
                tag='make_dataset',
                seed=0,
                device=device,
                output_root=output_root,
                pin_memory=False,
                model=ModelRuntime(data_name=data_name, model_name='linear'),
                optimizer=OptimizerRuntime(batch_size={'train': 250, 'test': 1000}),
            )
            dataset = make_dataset(data_name, process=True)
            process_dataset(dataset, runtime)
            data_loader = make_data_loader(
                dataset, runtime.optimizer.batch_size, shuffle=False,
                pin_memory=False, num_workers=0)
            stats = Stats(dim=dim)
            for _, input in enumerate(data_loader['train']):
                stats.update(input['data'])
            print(data_name, stats)
            save(stats, stats_file, 'torch')
            prepared.append(data_name)
    return prepared
