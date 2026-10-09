"""Project observed probe snapshots into inspectable Run artifacts.

No model execution or checkpoint selection occurs here. The probe keeps its
stricter RNG, input, count, and per-segment numerical gates.
"""

from pathlib import Path


def write_observed_run(root, *, snapshots, best, data, model, implementation):
    import json
    import torch
    from rpipe.structure.artifact import artifact_layout, write_config, write_result
    from rpipe.structure.artifact._atomic import atomic_write_text

    root = Path(root)
    if root.parent.name != 'runs':
        raise ValueError('observed Run must be under a runs directory')
    if root.exists():
        raise FileExistsError(f'preserve existing probe Run: {root}')
    layout = artifact_layout(root.parent.parent, root.name)
    layout.ensure()
    config = {'id': root.name, 'seed': 0, 'data': {'name': data}, 'model': {'name': model},
              'algorithm': {'mode': 'train', 'num_steps': 60},
              'description': f'observed {implementation} probe; normalized state key prefixes'}
    write_config(layout.config_path, config)
    ordered = [snapshots[step] for step in sorted(snapshots)]
    final = ordered[-1]
    metrics = {f'{split}_{key}': value for split in ('train', 'test') for key, value in final[split].items()}
    history = {'splits': {split: {key: {'history': [row[split][key] for row in ordered]}
                                 for key in final[split]} for split in ('train', 'test')}}
    atomic_write_text(layout.assets_dir / 'tracker' / 'tracker_state.json', json.dumps(history, indent=2))
    folder = layout.assets_dir / 'checkpoints'
    folder.mkdir(parents=True, exist_ok=True)
    for name, snapshot in [('latest', final), ('best', best)]:
        # Normalize only the known wrapper prefixes; preserve every buffer.
        state = {(key[6:] if key.startswith('model.') else key[4:] if key.startswith('net.') else key): value
                 for key, value in snapshot['model'].items()}
        if len(state) != len(snapshot['model']):
            raise ValueError('model wrapper prefix normalization would discard a state key')
        torch.save({**snapshot, 'model': state}, folder / f'{name}.pt')
    write_result(layout.result_path, {'status': 'succeeded', 'control': config, 'metrics': metrics,
                                     'paths': {'artifact': str(root), 'assets': str(layout.assets_dir)},
                                     'projection': {'implementation': implementation,
                                                    'model_prefixes': ['model.', 'net.'],
                                                    'source': 'observed snapshots; no computation replay'}})
    return root
