"""Issue #37: make-only fixed eval must be visible in native status filtering."""

import json
import os
import subprocess
import sys

import pytest

from rpipe.structure.artifact import load_config, write_config

pytestmark = [
    pytest.mark.integration, pytest.mark.content, pytest.mark.p1,
    pytest.mark.flow_layer, pytest.mark.module_cli,
    pytest.mark.cost(cost_class='c2'),
    pytest.mark.result_type('categorical', detail='summary'),
]


@pytest.mark.parametrize('origin', ['base', 'fixed'])
@pytest.mark.parametrize('mode', ['eval', 'inference'])
def test_native_status_finds_pending_fixed_mode_without_writes(tmp_path, origin, mode):
    """Native make -> status filters classify a fixed non-training mode without changing any bytes."""
    study = tmp_path / 'toy'
    study.mkdir()
    declaration = {'study': 'pending_eval', 'seeds': [0]}
    if origin == 'fixed':
        declaration['fixed'] = {'algorithm': {'mode': mode}}
    write_config(study / 'study.yaml', declaration)
    write_config(study / 'experiment_config.yaml', {
        'version': 'pending-eval-v1', 'data': {'name': 'Toy', 'source': 'stub'}, 'model': {'name': 'linear'},
        'algorithm': {'mode': mode if origin == 'base' else 'train', 'source': 'custom_torch',
                      'num_steps': 2, 'progress_unit': 'step'}, 'system': {'device': 'cpu'},
    })
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', MPLBACKEND='Agg')
    def command(name, args):
        done = subprocess.run([sys.executable, '-m', 'rpipe', *args], env=env, capture_output=True, timeout=30)
        (tmp_path / (name + '.stdout')).write_bytes(done.stdout)
        (tmp_path / (name + '.stderr')).write_bytes(done.stderr)
        assert done.returncode == 0, done.stderr
        return done.stdout.decode('utf-8')
    command('make', ['make', str(study), '--num-gpus', '0'])
    index = json.loads((study / 'index.json').read_text(encoding='utf-8'))
    experiment = index['experiments'][0]
    assert 'algorithm.mode' not in experiment['factors']
    run_id = experiment['runs'][0]['id']
    assert load_config(study / 'runs' / run_id / 'config.yaml')['algorithm']['mode'] == mode
    assert not (study / 'runs' / run_id / 'result.json').exists()
    before = {p.relative_to(study): p.read_bytes() for p in study.rglob('*') if p.is_file()}
    all_rows = command('status-all', ['status', str(study)])
    selected_rows = command('status-selected', ['status', str(study), '--mode', mode])
    train_rows = command('status-train', ['status', str(study), '--mode', 'train'])
    assert f'pending\t{mode}\t' in all_rows and 'planned=1' in selected_rows
    assert f'pending\t{mode}\t' in selected_rows and 'planned=0' in train_rows
    assert {p.relative_to(study): p.read_bytes() for p in study.rglob('*') if p.is_file()} == before
