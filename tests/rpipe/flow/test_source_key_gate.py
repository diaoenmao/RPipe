"""Collision refusal follows the native freeze gate and preserves prior success."""
import hashlib
import json
from types import SimpleNamespace

import pytest
import rpipe

from rpipe.flow import FlowContext, FlowRunner
from rpipe.flow.cli import _provenance_gate
from rpipe.structure.artifact import artifact_layout, write_config, write_result
from rpipe.structure.artifact.provenance import ProvenanceChangedError

pytestmark = [pytest.mark.integration, pytest.mark.content, pytest.mark.p1,
              pytest.mark.flow_layer, pytest.mark.module_runner,
              pytest.mark.cost(cost_class='c2'),
              pytest.mark.result_type('categorical', detail='summary')]


@pytest.mark.parametrize('route', ['launch-gate', 'runner'])
@pytest.mark.parametrize('prior_success', [False, True])
def test_collision_refusal_executes_no_phase_and_preserves_success(tmp_path, monkeypatch, capsys, route, prior_success):
    """Actual source capture collision -> CLI gate/Runner refusal -> old bytes preserved, no compute."""
    package, study = tmp_path / 'library', tmp_path / 'study'
    package.mkdir()
    (package / '__init__.py').write_bytes(b'library')
    (study / 'rpipe').mkdir(parents=True)
    (study / 'rpipe/__init__.py').write_bytes(b'project')
    monkeypatch.setattr(rpipe, '__file__', str(package / '__init__.py'))
    write_config(study / 'study.yaml', {'freeze': True, 'flow': {'study_phases': True},
                 'provenance': {'include': ['rpipe/__init__.py']}})
    # A legacy ambiguous record produced before this fix, including the declaration.
    files = {name: hashlib.sha256((study / name).read_bytes()).hexdigest()
             for name in ['study.yaml', 'rpipe/__init__.py']}
    saved = json.dumps({'files': files, 'plan': {}}).encode()
    (study / 'provenance.json').write_bytes(saved)
    layout = artifact_layout(study, 'prior')
    if prior_success:
        write_result(layout.result_path, {'status': 'succeeded', 'control': {}, 'paths': {}, 'metrics': {'accuracy': 42}})
    prior = layout.result_path.read_bytes() if prior_success else None
    called = []
    def boom(_ctx):
        called.append(True)
        raise RuntimeError('phase must not execute')
    monkeypatch.setattr('rpipe.flow.runner.import_module', lambda _: SimpleNamespace(run=boom))
    if route == 'launch-gate':
        assert _provenance_gate(study) is False
        assert 'collid' in capsys.readouterr().err
        assert prior_success or not layout.result_path.exists()
    else:
        with pytest.raises(ProvenanceChangedError, match='collid'):
            FlowRunner(['execute']).run(FlowContext(study, layout, {}))
        log = (layout.assets_dir / 'logs/run.log').read_text(encoding='utf-8')
        assert 'ProvenanceChangedError' in log and '[flow] succeeded' not in log
    assert not called
    if prior_success:
        assert layout.result_path.read_bytes() == prior
    assert (study / 'provenance.json').read_bytes() == saved
