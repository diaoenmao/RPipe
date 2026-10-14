"""Unverifiable frozen inputs are refusals, without changing make diagnostics."""
import json

import pytest
import rpipe

from rpipe.structure.artifact import provenance

pytestmark = [pytest.mark.unit, pytest.mark.content, pytest.mark.p1,
              pytest.mark.structure_layer, pytest.mark.module_artifact,
              pytest.mark.cost(cost_class='c1'),
              pytest.mark.result_type('categorical', detail='summary')]


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    library, study = tmp_path / 'library', tmp_path / 'study'
    library.mkdir()
    (library / '__init__.py').write_bytes(b'library')
    study.mkdir()
    extra = study / 'input.txt'
    extra.write_bytes(b'input')
    config = {'freeze': True, 'provenance': {'include': ['input.txt']}}
    monkeypatch.setattr(rpipe, '__file__', str(library / '__init__.py'))
    monkeypatch.setattr(provenance, 'capture_environment', lambda: {'synthetic': True})
    monkeypatch.setattr(provenance, 'capture_git', lambda _: {})
    provenance.write_provenance(study, config)
    return study, extra, config


def fail_capture(study, extra, monkeypatch, failure):
    if failure == 'missing':
        extra.unlink()  # Only this test's synthetic input.
        return FileNotFoundError
    if failure == 'permission':
        original = provenance._sha256
        def denied(path):
            if path == extra:
                raise PermissionError('input denied')
            return original(path)
        monkeypatch.setattr(provenance, '_sha256', denied)
        return PermissionError
    payload = {'bad-root': b'[]', 'bad-files': b'{\"files\": [\"input.txt\"]}',
               'bad-plan': b'{\"plan\": [\"index.json\"]}'}.get(failure, b'{invalid json')
    (study / 'provenance.json').write_bytes(payload)
    return ValueError


@pytest.mark.parametrize('failure', ['missing', 'permission', 'corrupt-record', 'bad-root', 'bad-files', 'bad-plan'])
def test_unverifiable_freeze_preserves_cause_and_original_record(frozen, monkeypatch, failure):
    study, extra, config = frozen
    expected = fail_capture(study, extra, monkeypatch, failure)
    path = study / 'provenance.json'
    before = path.read_bytes()
    with pytest.raises(provenance.ProvenanceChangedError, match='Cannot verify') as caught:
        provenance.check_frozen(study, config)
    assert isinstance(caught.value.__cause__, expected)
    assert path.read_bytes() == before


def test_disabled_freeze_does_not_inspect_inputs(frozen, monkeypatch):
    study, extra, config = frozen
    fail_capture(study, extra, monkeypatch, 'missing')
    provenance.check_frozen(study, dict(config, freeze=False))


@pytest.mark.parametrize('failure', ['missing', 'permission'])
def test_make_keeps_original_input_error_and_prior_record(frozen, monkeypatch, failure):
    study, extra, config = frozen
    expected = fail_capture(study, extra, monkeypatch, failure)
    path = study / 'provenance.json'
    before = path.read_bytes()
    with pytest.raises(expected):
        provenance.write_provenance(study, config)
    assert path.read_bytes() == before


def test_unexpected_programming_error_is_not_reclassified(frozen, monkeypatch):
    study, _, config = frozen
    def broken(*_):
        raise RuntimeError('unexpected bug')
    monkeypatch.setattr(provenance, 'provenance_changes', broken)
    with pytest.raises(RuntimeError, match='unexpected bug'):
        provenance.check_frozen(study, config)


@pytest.mark.parametrize('body', [{'digest': 'legacy'}, {'files': None, 'plan': {}}])
def test_load_retains_optional_legacy_mappings(frozen, body):
    study, _, _ = frozen
    (study / 'provenance.json').write_text(json.dumps(body), encoding='utf-8')
    assert provenance.load_provenance(study) == body
