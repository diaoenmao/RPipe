"""Actual failed source capture must not overwrite a previously successful Run."""
from types import SimpleNamespace

import pytest
import rpipe

from rpipe.flow import FlowContext, FlowRunner
from rpipe.flow.cli import _provenance_gate
from rpipe.structure.artifact import artifact_layout, load_result, write_config, write_result
from rpipe.structure.artifact import provenance

pytestmark = [pytest.mark.integration, pytest.mark.content, pytest.mark.p1,
              pytest.mark.flow_layer, pytest.mark.module_runner,
              pytest.mark.cost(cost_class='c2'),
              pytest.mark.result_type('categorical', detail='summary')]


@pytest.mark.parametrize('failure', ['missing', 'permission', 'corrupt-record', 'bad-shape'])
@pytest.mark.parametrize('route', ['launch-gate', 'runner'])
@pytest.mark.parametrize('prior_success', [False, True])
def test_capture_refusal_preserves_results_and_executes_no_phase(tmp_path, monkeypatch, capsys, failure, route, prior_success):
    """Real source capture -> CLI gate/Runner -> refusal, old success/record intact, no compute."""
    package, study = tmp_path / 'library', tmp_path / 'study'
    package.mkdir()
    (package / '__init__.py').write_bytes(b'library')
    study.mkdir()
    extra = study / 'input.txt'
    extra.write_bytes(b'input')
    monkeypatch.setattr(rpipe, '__file__', str(package / '__init__.py'))
    monkeypatch.setattr(provenance, 'capture_environment', lambda: {'synthetic': True})
    monkeypatch.setattr(provenance, 'capture_git', lambda _: {})
    config = {'freeze': True, 'flow': {'study_phases': True},
              'provenance': {'include': ['input.txt']}}
    write_config(study / 'study.yaml', config)
    provenance.write_provenance(study, config)
    layout = artifact_layout(study, 'prior')
    if prior_success:
        write_result(layout.result_path, {'status': 'succeeded', 'control': {}, 'paths': {}, 'metrics': {'accuracy': 42}})
    prior = layout.result_path.read_bytes() if prior_success else None
    if failure == 'missing':
        extra.unlink()  # Only this test's synthetic input.
    elif failure == 'permission':
        original = provenance._sha256
        def denied(path):
            if path == extra:
                raise PermissionError('input denied')
            return original(path)
        monkeypatch.setattr(provenance, '_sha256', denied)
    else:
        payload = b'{\"files\": [\"input.txt\"]}' if failure == 'bad-shape' else b'{invalid json'
        (study / 'provenance.json').write_bytes(payload)
    saved = (study / 'provenance.json').read_bytes()
    called = []
    def forbidden(_ctx):
        called.append(True)
        raise RuntimeError('phase must not execute')
    monkeypatch.setattr('rpipe.flow.runner.import_module', lambda _: SimpleNamespace(run=forbidden))
    if route == 'launch-gate':
        assert _provenance_gate(study) is False
        assert 'Cannot verify' in capsys.readouterr().err
        assert prior_success or not layout.result_path.exists()
    else:
        with pytest.raises(provenance.ProvenanceChangedError, match='Cannot verify'):
            FlowRunner(['execute']).run(FlowContext(study, layout, {}))
        log = (layout.assets_dir / 'logs/run.log').read_text(encoding='utf-8')
        assert 'ProvenanceChangedError' in log and '[flow] succeeded' not in log
        if not prior_success:
            result = load_result(layout.result_path)
            assert result['status'] == 'failed' and 'ProvenanceChangedError' in result['error']
    assert not called
    if prior_success:
        assert layout.result_path.read_bytes() == prior
    assert (study / 'provenance.json').read_bytes() == saved


@pytest.mark.parametrize('prior_success', [False, True])
def test_prepare_capture_refusal_with_study_phases_disabled(tmp_path, monkeypatch, prior_success):
    """Real prepare freeze gate without Study hooks -> source refusal before data/model, preserve prior success."""
    import importlib
    from rpipe.structure.system.logger import Logger
    prepare = importlib.import_module('rpipe.flow.prepare.run')
    package, study = tmp_path / 'library', tmp_path / 'study'
    package.mkdir()
    (package / '__init__.py').write_bytes(b'library')
    study.mkdir()
    extra = study / 'input.txt'
    extra.write_bytes(b'input')
    monkeypatch.setattr(rpipe, '__file__', str(package / '__init__.py'))
    monkeypatch.setattr(provenance, 'capture_environment', lambda: {'synthetic': True})
    monkeypatch.setattr(provenance, 'capture_git', lambda _: {})
    declared = {'freeze': True, 'flow': {'study_phases': False},
                'provenance': {'include': ['input.txt']}}
    write_config(study / 'study.yaml', declared)
    layout = artifact_layout(study, 'prior')
    write_config(layout.config_path, {'seed': 0, 'data': {'name': 'Toy', 'source': 'stub'},
                 'model': {'name': 'linear'}, 'algorithm': {'mode': 'train', 'num_steps': 1},
                 'system': {'device': 'cpu'}})
    provenance.write_provenance(study, declared)
    saved = (study / 'provenance.json').read_bytes()
    if prior_success:
        write_result(layout.result_path, {'status': 'succeeded', 'control': {}, 'paths': {}, 'metrics': {'accuracy': 42}})
    prior = layout.result_path.read_bytes() if prior_success else None
    extra.unlink()  # Only this test's synthetic input.
    monkeypatch.setattr(prepare.system_api, 'apply_runtime', lambda *_: {})
    monkeypatch.setattr(prepare.system_api, 'build', lambda *_: SimpleNamespace(meta={}, logger=Logger(layout.assets_dir)))
    def forbidden(*_args, **_kwargs):
        pytest.fail('data/model must not build after freeze refusal')
    monkeypatch.setattr(prepare.data_api, 'build', forbidden)
    monkeypatch.setattr(prepare.model_api, 'build', forbidden)
    with pytest.raises(provenance.ProvenanceChangedError, match='Cannot verify'):
        FlowRunner(['prepare']).run(FlowContext(study, layout, {}))
    if prior_success:
        assert layout.result_path.read_bytes() == prior
    else:
        assert load_result(layout.result_path)['status'] == 'failed'
    log = (layout.assets_dir / 'logs/run.log').read_text(encoding='utf-8')
    assert 'phase=prepare status=failed' in log and 'ProvenanceChangedError' in log
    assert (study / 'provenance.json').read_bytes() == saved
