import json
from pathlib import Path

import pytest

from rpipe.flow import cli
from rpipe.structure.artifact.readout.compare import compare_runs

pytestmark = [
    pytest.mark.unit,
    pytest.mark.content,
    pytest.mark.p1,
    pytest.mark.structure_layer,
    pytest.mark.module_artifact,
    pytest.mark.cost(cost_class='c1'),
    pytest.mark.result_type('categorical', detail='summary'),
]


def _run(root: Path, name: str, *, weight: float = 1.0, loss: float = 0.5, history=(1.0, 0.5)) -> Path:
    import torch

    run = root / name
    (run / 'assets' / 'tracker').mkdir(parents=True)
    (run / 'assets' / 'checkpoints').mkdir(parents=True)
    (run / 'result.json').write_text(
        json.dumps({'status': 'succeeded', 'control': {}, 'paths': {}, 'metrics': {'loss': loss, 'tag': 'x'}}),
        encoding='utf-8',
    )
    (run / 'assets' / 'tracker' / 'tracker_state.json').write_text(
        json.dumps({'splits': {'train': {'Loss': {'history': list(history)}}}}),
        encoding='utf-8',
    )
    torch.save(
        {
            'model': {'w': torch.full((2, 2), weight), 'b': torch.zeros(2)},
            'optimizer': {'state': {0: {'step': torch.tensor(3.0)}}, 'param_groups': [{'lr': 0.1}]},
            'scheduler': None,
        },
        run / 'assets' / 'checkpoints' / 'latest.pt',
    )
    return run


def test_identical_runs_pass_exactly(tmp_path: Path):
    report = compare_runs(_run(tmp_path, 'a'), _run(tmp_path, 'b'))
    assert report['passed'] is True
    assert set(report['items']) == {'metrics', 'history', 'checkpoint.latest.model', 'checkpoint.latest.optimizer'}
    assert report['items']['metrics']['keys'] == ['loss']


def test_differences_fail_and_tolerance_accepts_small_drift(tmp_path: Path):
    a = _run(tmp_path, 'a')
    b = _run(tmp_path, 'b', weight=1.0 + 1e-7, loss=0.5 + 1e-7, history=(1.0, 0.5 + 1e-7))
    exact = compare_runs(a, b)
    assert exact['passed'] is False
    assert exact['items']['checkpoint.latest.model']['problems'] == ['model.w: max abs diff 1.19209e-07']
    assert compare_runs(a, b, atol=1e-6)['passed'] is True


def test_shape_length_and_missing_checkpoint_fail(tmp_path: Path):
    a = _run(tmp_path, 'a')
    b = _run(tmp_path, 'b', history=(1.0,))
    report = compare_runs(a, b, checkpoints=['latest', 'best'])
    assert report['items']['history']['problems'] == ['history.train.Loss: length 2 != 1']
    assert report['items']['checkpoint.best.model']['missing'] == 'a'
    assert report['passed'] is False


def test_cli_compare_exit_codes_and_json(tmp_path: Path, capsys):
    a, b = _run(tmp_path, 'a'), _run(tmp_path, 'b', loss=0.7)
    out = tmp_path / 'cmp.json'
    assert cli.main(['compare', str(a), str(a)]) == 0
    assert cli.main(['compare', str(a), str(b), '--out', str(out)]) == 1
    body = json.loads(out.read_text(encoding='utf-8'))
    assert body['items']['metrics']['passed'] is False
    assert cli.main(['compare', str(a), str(tmp_path / 'missing')]) == 2
    assert 'FAIL  metrics' in capsys.readouterr().out
