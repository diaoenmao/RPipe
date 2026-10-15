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
