from pathlib import Path

import pytest

from rpipe.structure.artifact.provenance import (
    ProvenanceChangedError, check_frozen, source_files, write_provenance,
)

pytestmark = [pytest.mark.unit, pytest.mark.content, pytest.mark.p1,
              pytest.mark.structure_layer, pytest.mark.module_artifact,
              pytest.mark.cost(cost_class='c1'), pytest.mark.result_type('categorical', detail='summary')]


def _source(root: Path, relative: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('VALUE = 1\n', encoding='utf-8')
    return path


@pytest.mark.parametrize('enabled', [False, True])
def test_two_column_sources_are_frozen_without_include_or_hooks(tmp_path, enabled):
    sources = ('flow/__init__.py', 'flow/prepare_data.py', 'flow/execute/speed.py',
               'structure/__init__.py', 'structure/data/helper.py')
    for name in sources:
        _source(tmp_path, name)
    _source(tmp_path, 'runs/old/assets/source/model.py')
    _source(tmp_path, 'shared/generated.py')
    _source(tmp_path, 'flow/__pycache__/cached.pyc')
    files = source_files(tmp_path, {'flow': {'study_phases': enabled}})
    assert set(sources) <= set(files)
    assert not any(name.startswith(('runs/', 'shared/')) or '__pycache__' in name for name in files)


@pytest.mark.parametrize('folder', ['flow', 'structure'])
@pytest.mark.parametrize('change', ['add', 'edit', 'delete'])
def test_freeze_detects_two_column_helper_changes(tmp_path, monkeypatch, folder, change):
    """Two-column source → make provenance → changed helper blocks freeze without explicit include."""
    monkeypatch.setattr('rpipe.structure.artifact.provenance.capture_environment', lambda: {'check': 'CPU'})
    monkeypatch.setattr('rpipe.structure.artifact.provenance.capture_git', lambda root: {})
    path = tmp_path / folder / 'helper.py'
    if change != 'add':
        _source(tmp_path, f'{folder}/helper.py')
    study = {'freeze': True}
    write_provenance(tmp_path, study)
    check_frozen(tmp_path, study)
    if change == 'delete':
        path.unlink()
    else:
        _source(tmp_path, f'{folder}/helper.py').write_text('VALUE = 2\n', encoding='utf-8')
    with pytest.raises(ProvenanceChangedError, match=f'{folder}/helper.py'):
        check_frozen(tmp_path, study)


@pytest.mark.parametrize('folder', ['flow', 'structure'])
def test_two_column_source_cannot_escape_study(tmp_path, folder):
    outside = tmp_path / 'outside'
    outside.mkdir()
    _source(outside, 'helper.py')
    study = tmp_path / 'study'
    study.mkdir()
    try:
        (study / folder).symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip('host does not allow directory symlinks')
    with pytest.raises(ValueError, match='inside the Study'):
        source_files(study, {})



_BAD_PROVENANCE_DECLARATIONS = [
    {'provenance': False}, {'provenance': 0}, {'provenance': ''}, {'provenance': []},
    {'provenance': {'include': False}}, {'provenance': {'include': 0}},
    {'provenance': {'include': ''}}, {'provenance': {'include': {}}},
    {'provenance': {'include': 'x'}},
    {'provenance': {'include': {'x': False}}},
]


def _declaration_study(tmp_path, monkeypatch):
    import rpipe
    from rpipe.structure.artifact import provenance

    package = tmp_path / 'library'
    _source(package, '__init__.py')
    monkeypatch.setattr(rpipe, '__file__', str(package / '__init__.py'))
    monkeypatch.setattr(provenance, 'capture_environment', lambda: {'test': 'CPU'})
    monkeypatch.setattr(provenance, 'capture_git', lambda root: {})
    study = tmp_path / 'study'
    study.mkdir()
    (study / 'study.yaml').write_bytes(b'name: synthetic\n')
    (study / 'x').write_bytes(b'immutable input\n')
    return study


@pytest.mark.parametrize('config', _BAD_PROVENANCE_DECLARATIONS)
def test_source_manifest_rejects_malformed_declaration(tmp_path, monkeypatch, config):
    study = _declaration_study(tmp_path, monkeypatch)
    before = {p.name: p.read_bytes() for p in study.iterdir()}
    with pytest.raises(ValueError, match='provenance'):
        source_files(study, config)
    assert {p.name: p.read_bytes() for p in study.iterdir()} == before


@pytest.mark.parametrize('config', _BAD_PROVENANCE_DECLARATIONS)
@pytest.mark.parametrize('prior', [False, True])
def test_provenance_refuses_bad_declaration_without_publication(tmp_path, monkeypatch, config, prior):
    study = _declaration_study(tmp_path, monkeypatch)
    target = study / 'provenance.json'
    old = b'{"files":{"old.py":"prior-hash"},"plan":{}}'
    if prior:
        target.write_bytes(old)
    peer = study / 'peer.json'
    peer.write_bytes(b'{"status":"succeeded"}')
    before = {p.name: p.read_bytes() for p in study.iterdir()}
    with pytest.raises(ValueError, match='provenance'):
        write_provenance(study, config)
    assert {p.name: p.read_bytes() for p in study.iterdir()} == before


@pytest.mark.parametrize('config,included', [
    ({}, False), ({'provenance': None}, False), ({'provenance': {}}, False),
    ({'provenance': {'include': None}}, False), ({'provenance': {'include': []}}, False),
    ({'provenance': {'include': ['x']}}, True),
    ({'provenance': {'include': ['x*']}}, True),
    ({'provenance': {'include': [Path('x')]}}, True),
])
def test_source_manifest_keeps_supported_declaration_forms(tmp_path, monkeypatch, config, included):
    import hashlib
    from rpipe.structure.artifact.provenance import load_provenance

    study = _declaration_study(tmp_path, monkeypatch)
    before = {p.name: p.read_bytes() for p in study.iterdir()}
    files = source_files(study, config)
    assert ('x' in files) is included
    if included:
        assert files['x'] == hashlib.sha256(before['x']).hexdigest()
    write_provenance(study, config)
    saved = load_provenance(study)
    assert saved['files'] == files
    assert saved['plan'] == {}
    assert {p.name: p.read_bytes() for p in study.iterdir() if p.name != 'provenance.json'} == before



_BAD_PLAN_INDICES = [
    False, [], {'experiments': False}, {'experiments': 0}, {'experiments': ''},
    {'experiments': {}}, {'experiments': [False]},
    {'experiments': [{'runs': False}]}, {'experiments': [{'runs': {}}]},
    {'experiments': [{'runs': [False]}]},
    {'experiments': [{'factors': [], 'runs': []}]},
    {'experiments': [{'runs': [{'id': '../../external'}]}]},
]
_PLAN_REFUSAL_OPERATIONS = ['plan', 'write-new', 'write-old', 'freeze']


def _assert_plan_refusal(study, monkeypatch, operation, forbidden=None):
    import hashlib
    import json
    from rpipe.structure.artifact import provenance

    target = study / 'provenance.json'
    if operation == 'write-old':
        target.write_bytes(b'{"files":{"old.py":"retained"},"plan":{}}')
    elif operation == 'freeze':
        saved = {'files': source_files(study, {}), 'plan': {
            'index.json': hashlib.sha256((study / 'index.json').read_bytes()).hexdigest(),
        }}
        target.write_text(json.dumps(saved), encoding='utf-8')
    (study / 'peer.json').write_bytes(b'{"status":"succeeded"}')
    before = {p.relative_to(study).as_posix(): p.read_bytes()
              for p in study.rglob('*') if p.is_file()}
    if forbidden is not None:
        original = provenance._sha256

        def checked_hash(path):
            assert Path(path).resolve() != forbidden.resolve(), 'must refuse before external input hashing'
            return original(path)

        monkeypatch.setattr(provenance, '_sha256', checked_hash)
    if operation == 'freeze':
        with pytest.raises(ProvenanceChangedError) as caught:
            check_frozen(study, {'freeze': True})
        assert isinstance(caught.value.__cause__, ValueError)
    else:
        with pytest.raises(ValueError, match='index|Run|Study'):
            if operation == 'plan':
                provenance.plan_files(study)
            else:
                write_provenance(study, {})
    assert {p.relative_to(study).as_posix(): p.read_bytes()
            for p in study.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('body', _BAD_PLAN_INDICES)
@pytest.mark.parametrize('operation', _PLAN_REFUSAL_OPERATIONS)
def test_plan_manifest_reuses_index_refusal_before_publication(tmp_path, monkeypatch, body, operation):
    import json

    study = _declaration_study(tmp_path, monkeypatch)
    (study / 'runs').mkdir()
    (study / 'index.json').write_text(json.dumps(body), encoding='utf-8')
    outside = tmp_path / 'external' / 'config.yaml'
    outside.parent.mkdir()
    outside.write_bytes(b'id: external\n')
    _assert_plan_refusal(study, monkeypatch, operation, forbidden=outside)
    assert outside.read_bytes() == b'id: external\n'


@pytest.mark.parametrize('boundary', ['index', 'config'])
@pytest.mark.parametrize('operation', _PLAN_REFUSAL_OPERATIONS)
def test_plan_manifest_refuses_external_symlink_before_hashing(tmp_path, monkeypatch, boundary, operation):
    import json

    study = _declaration_study(tmp_path, monkeypatch)
    body = {'experiments': [{'runs': [{'id': 'safe'}]}]}
    (study / 'runs/safe').mkdir(parents=True)
    (study / 'index.json').write_text(json.dumps(body), encoding='utf-8')
    (study / 'runs/safe/config.yaml').write_bytes(b'id: safe\n')
    outside = tmp_path / 'external' / ('index.json' if boundary == 'index' else 'config.yaml')
    outside.parent.mkdir()
    raw = json.dumps(body).encode('utf-8') if boundary == 'index' else b'id: external\n'
    outside.write_bytes(raw)
    linked = study / ('index.json' if boundary == 'index' else 'runs/safe/config.yaml')
    linked.unlink()
    try:
        linked.symlink_to(outside)
    except OSError:
        pytest.skip('host does not allow file symlinks')
    _assert_plan_refusal(study, monkeypatch, operation, forbidden=outside)
    assert linked.is_symlink() and outside.read_bytes() == raw


@pytest.mark.parametrize('body', [
    {}, {'experiments': None}, {'experiments': []}, {'experiments': [{}]},
    {'experiments': [{'runs': None}]}, {'experiments': [{'runs': []}]},
    {'experiments': [{'factors': None, 'runs': [{'id': 'safe'}]}]},
    {'experiments': [{'runs': [{'id': 'safe'}, {'id': 'missing'}]}]},
])
def test_plan_manifest_preserves_supported_partial_index(tmp_path, monkeypatch, body):
    import hashlib
    import json
    from rpipe.structure.artifact.provenance import plan_files

    study = _declaration_study(tmp_path, monkeypatch)
    raw = json.dumps(body).encode('utf-8')
    (study / 'index.json').write_bytes(raw)
    (study / 'runs/safe').mkdir(parents=True)
    config = b'id: safe\n'
    (study / 'runs/safe/config.yaml').write_bytes(config)
    expected = {'index.json': hashlib.sha256(raw).hexdigest()}
    if body.get('experiments') and body['experiments'][0].get('runs'):
        expected['runs/safe/config.yaml'] = hashlib.sha256(config).hexdigest()
    assert plan_files(study) == expected
    assert (study / 'index.json').read_bytes() == raw


def test_plan_manifest_keeps_logical_key_for_internal_config_alias(tmp_path, monkeypatch):
    import json
    import hashlib
    from rpipe.structure.artifact.provenance import plan_files

    study = _declaration_study(tmp_path, monkeypatch)
    body = {'experiments': [{'runs': [{'id': 'safe'}]}]}
    (study / 'index.json').write_text(json.dumps(body), encoding='utf-8')
    (study / 'runs/safe').mkdir(parents=True)
    target = study / 'owned.yaml'
    target.write_bytes(b'id: safe\n')
    try:
        (study / 'runs/safe/config.yaml').symlink_to(target)
    except OSError:
        pytest.skip('host does not allow file symlinks')
    files = plan_files(study)
    assert files['runs/safe/config.yaml'] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert 'owned.yaml' not in files


def test_plan_manifest_missing_index_still_empty(tmp_path):
    from rpipe.structure.artifact.provenance import plan_files

    assert plan_files(tmp_path) == {}
    assert list(tmp_path.iterdir()) == []


def test_plan_manifest_unexpected_hash_bug_propagates(tmp_path, monkeypatch):
    from rpipe.structure.artifact import provenance

    (tmp_path / 'index.json').write_bytes(b'{}')
    error = RuntimeError('programming bug')

    def broken_hash(path):
        raise error

    monkeypatch.setattr(provenance, '_sha256', broken_hash)
    with pytest.raises(RuntimeError) as caught:
        provenance.plan_files(tmp_path)
    assert caught.value is error



@pytest.mark.parametrize('run', [{}, {'id': None}, {'id': 'safe', 'run_dir': '../outside'}])
@pytest.mark.parametrize('operation', _PLAN_REFUSAL_OPERATIONS)
def test_plan_manifest_requires_identity_and_checks_paired_directory(tmp_path, monkeypatch, run, operation):
    import json

    study = _declaration_study(tmp_path, monkeypatch)
    (study / 'index.json').write_text(json.dumps({'experiments': [{'runs': [run]}]}), encoding='utf-8')
    _assert_plan_refusal(study, monkeypatch, operation)


@pytest.mark.parametrize('freeze', [False, True])
def test_deep_provenance_is_a_freeze_refusal_only_when_enabled(tmp_path, freeze):
    from rpipe.structure.artifact.provenance import provenance_path

    path = provenance_path(tmp_path)
    raw = b'{"files":{},"plan":{},"metadata":' + b'[' * 12000 + b'0' + b']' * 12000 + b'}'
    path.write_bytes(raw)
    if freeze:
        with pytest.raises(ProvenanceChangedError) as caught:
            check_frozen(tmp_path, {'freeze': True})
        assert isinstance(caught.value.__cause__, RecursionError)
        assert 'Cannot verify frozen provenance' in str(caught.value)
    else:
        assert check_frozen(tmp_path, {'freeze': False}) is None
    assert path.read_bytes() == raw


def test_deep_provenance_strict_loader_retains_parser_error(tmp_path):
    from rpipe.structure.artifact.provenance import load_provenance, provenance_path

    path = provenance_path(tmp_path)
    raw = b'{"metadata":' + b'[' * 12000 + b'0' + b']' * 12000 + b'}'
    path.write_bytes(raw)
    with pytest.raises(RecursionError):
        load_provenance(tmp_path)
    assert path.read_bytes() == raw


def test_freeze_parser_boundary_does_not_normalize_programming_runtime_error(tmp_path, monkeypatch):
    import rpipe.structure.artifact.provenance as provenance

    error = RuntimeError('unexpected provenance programming error')
    def refuse(*args):
        raise error
    monkeypatch.setattr(provenance, 'provenance_changes', refuse)
    with pytest.raises(RuntimeError) as caught:
        check_frozen(tmp_path, {'freeze': True})
    assert caught.value is error


def test_provenance_binary_whole_read_keeps_first_failure(tmp_path, reader_exit_case, reader_close_fault):
    from rpipe.structure.artifact.provenance import _sha256
    import hashlib
    case = reader_exit_case
    path, peer = tmp_path / 'source.py', tmp_path / 'peer'
    payload = b'synthetic immutable source'
    path.write_bytes(payload)
    peer.write_bytes(b'peer unchanged')
    with reader_close_fault(path, case, 'rb') as (handles, closes):
        if case.expected is None:
            assert _sha256(path) == hashlib.sha256(payload).hexdigest()
        else:
            with pytest.raises(type(case.expected)) as caught:
                _sha256(path)
            assert caught.value is case.expected
    assert len(handles) == len(closes) == 1 and handles[0].closed
    assert path.read_bytes() == payload and peer.read_bytes() == b'peer unchanged'
