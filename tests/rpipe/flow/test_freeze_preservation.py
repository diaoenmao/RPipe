"""Issue #36: preserve a successful Run on initial freeze refusal."""

from importlib import import_module
import os
import subprocess
import sys

import pytest

from rpipe.flow import FlowContext, FlowRunner
from rpipe.structure.artifact import artifact_layout, load_config, load_result, write_config, write_result
from rpipe.structure.artifact.provenance import ProvenanceChangedError, write_provenance
from rpipe.structure.make import expand_study

pytestmark = [
    pytest.mark.integration, pytest.mark.content, pytest.mark.p1,
    pytest.mark.flow_layer, pytest.mark.module_runner,
    pytest.mark.cost(cost_class='c2'),
    pytest.mark.result_type('categorical', detail='summary'),
]


def _success(layout):
    write_result(layout.result_path, {'status': 'succeeded', 'control': {}, 'paths': {}, 'metrics': {'accuracy': 42}})
    return layout.result_path.read_bytes()


@pytest.mark.parametrize('phases', [None, ['execute']])
@pytest.mark.parametrize('prior_success', [False, True])
def test_freeze_refusal_preserves_success_or_creates_failure(tmp_path, monkeypatch, phases, prior_success):
    """Changed frozen source -> no phase execution, old success bytes preserved, first refusal written failed."""
    study = {'freeze': True, 'flow': {'study_phases': True}}
    write_config(tmp_path / 'study.yaml', study)
    layout = artifact_layout(tmp_path, 'prior')
    write_config(layout.config_path, {'seed': 0})
    prior = _success(layout) if prior_success else None
    write_provenance(tmp_path, study)
    with (tmp_path / 'study.yaml').open('a', encoding='utf-8') as stream:
        stream.write('# source changed\n')
    called = []
    for phase in ['prepare', 'execute']:
        monkeypatch.setattr(import_module(f'rpipe.flow.{phase}'), 'run', lambda _ctx: called.append(True))
    ctx = FlowContext(tmp_path, layout, {})
    with pytest.raises(ProvenanceChangedError):
        FlowRunner(phases).run(ctx)
    assert not called
    if prior_success:
        assert layout.result_path.read_bytes() == prior
    else:
        result = load_result(layout.result_path)
        assert result['status'] == 'failed' and 'ProvenanceChangedError' in result['error']
    log = (layout.assets_dir / 'logs/run.log').read_text(encoding='utf-8')
    assert 'ProvenanceChangedError' in log and '[flow] succeeded' not in log


@pytest.mark.parametrize('phase', ['prepare', 'execute', 'summarize', 'write', 'process'])
def test_actual_phase_failure_keeps_failure_write_contract(tmp_path, monkeypatch, phase):
    """Actual phase failure still fails a prior success; process retains its existing preservation rule."""
    layout = artifact_layout(tmp_path, 'prior')
    prior = _success(layout)
    def boom(_ctx):
        raise RuntimeError('actual phase failed')
    monkeypatch.setattr(import_module(f'rpipe.flow.{phase}'), 'run', boom)
    with pytest.raises(RuntimeError, match='actual phase failed'):
        FlowRunner([phase]).run(FlowContext(tmp_path, layout, {}))
    if phase == 'process':
        assert layout.result_path.read_bytes() == prior
    else:
        assert load_result(layout.result_path)['status'] == 'failed'
        assert 'actual phase failed' in load_result(layout.result_path)['error']


@pytest.mark.parametrize('study_phases', [False, True])
def test_native_run_one_preserves_real_cpu_success_on_freeze_refusal(tmp_path, study_phases):
    """CPU 2-step Toy success -> changed source -> native run-one refuses without replacing the result."""
    write_config(tmp_path / 'study.yaml', {'study': 'freeze_preservation', 'seeds': [0], 'freeze': True,
                                         'flow': {'study_phases': study_phases}})
    write_config(tmp_path / 'experiment_config.yaml', {
        'version': 'issue36-v1', 'data': {'name': 'Toy', 'source': 'stub'}, 'model': {'name': 'linear'},
        'algorithm': {'source': 'custom_torch', 'mode': 'train', 'num_steps': 2, 'progress_unit': 'step'},
        'system': {'device': 'cpu', 'deterministic': True, 'cudnn_benchmark': False},
    })
    config = expand_study(tmp_path)['configs'][0]
    layout = artifact_layout(tmp_path, config.parent.name)
    FlowRunner().run(FlowContext(tmp_path, layout, load_config(config)))
    assert load_result(layout.result_path)['status'] == 'succeeded'
    prior = layout.result_path.read_bytes()
    (tmp_path / 'prior-success.json').write_bytes(prior)
    assets = {p.relative_to(layout.assets_dir): p.read_bytes() for p in layout.assets_dir.rglob('*')
              if p.is_file() and p.relative_to(layout.assets_dir).parts[0] != 'logs'}
    with (tmp_path / 'study.yaml').open('a', encoding='utf-8') as stream:
        stream.write('# source changed\n')
    for attempt in range(2):
        result = subprocess.run([sys.executable, '-m', 'rpipe', 'run-one', str(tmp_path), config.parent.name],
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', MPLBACKEND='Agg'),
                                capture_output=True, timeout=30)
        (tmp_path / f'refusal-{attempt}.stdout').write_bytes(result.stdout)
        (tmp_path / f'refusal-{attempt}.stderr').write_bytes(result.stderr)
        assert result.returncode == 1 and b'ProvenanceChangedError' in result.stderr
        assert layout.result_path.read_bytes() == prior
        assert {p.relative_to(layout.assets_dir): p.read_bytes() for p in layout.assets_dir.rglob('*')
                if p.is_file() and p.relative_to(layout.assets_dir).parts[0] != 'logs'} == assets
