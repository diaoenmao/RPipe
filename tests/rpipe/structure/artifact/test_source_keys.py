"""Study inputs must not mask a different library source under one manifest key."""
import hashlib
import json
import os

import pytest
import rpipe

from rpipe.structure.artifact import provenance

pytestmark = [pytest.mark.unit, pytest.mark.content, pytest.mark.p1,
              pytest.mark.structure_layer, pytest.mark.module_artifact,
              pytest.mark.cost(cost_class='c1'),
              pytest.mark.result_type('categorical', detail='summary')]


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    package, study = tmp_path / 'library', tmp_path / 'study'
    package.mkdir()
    (package / '__init__.py').write_bytes(b'library')
    (study / 'rpipe').mkdir(parents=True)
    monkeypatch.setattr(rpipe, '__file__', str(package / '__init__.py'))
    monkeypatch.setattr(provenance, 'capture_environment', lambda: {'synthetic': True})
    monkeypatch.setattr(provenance, 'capture_git', lambda _: {})
    return package, study, {'provenance': {'include': ['rpipe/__init__.py']}}


@pytest.mark.parametrize('same_bytes', [False, True])
def test_distinct_sources_cannot_share_library_manifest_key(inputs, same_bytes):
    package, study, config = inputs
    copy = study / 'rpipe/__init__.py'
    copy.write_bytes((package / '__init__.py').read_bytes() if same_bytes else b'project')
    assert not copy.samefile(package / '__init__.py')
    with pytest.raises(provenance.ProvenanceChangedError, match='collid'):
        provenance.source_files(study, config)


def test_existing_shadowed_freeze_is_refused_after_library_edit(inputs):
    package, study, config = inputs
    copy = study / 'rpipe/__init__.py'
    copy.write_bytes(b'project')
    path = provenance.provenance_path(study)
    saved = json.dumps({'files': {'rpipe/__init__.py': hashlib.sha256(copy.read_bytes()).hexdigest()}, 'plan': {}}).encode()
    path.write_bytes(saved)
    (package / '__init__.py').write_bytes(b'changed library')
    with pytest.raises(provenance.ProvenanceChangedError, match='collid'):
        provenance.check_frozen(study, dict(config, freeze=True))
    assert path.read_bytes() == saved


def test_make_refuses_collision_without_overwriting_prior_provenance(inputs):
    _, study, config = inputs
    (study / 'rpipe/__init__.py').write_bytes(b'project')
    path = provenance.provenance_path(study)
    path.write_bytes(b'prior evidence')
    with pytest.raises(provenance.ProvenanceChangedError, match='collid'):
        provenance.write_provenance(study, config)
    assert path.read_bytes() == b'prior evidence'


@pytest.mark.parametrize('mode', ['package-in-study', 'hardlink'])
def test_same_physical_source_keeps_one_key_and_detects_edits(inputs, monkeypatch, mode):
    package, study, config = inputs
    source = package / '__init__.py'
    copy = study / 'rpipe/__init__.py'
    if mode == 'package-in-study':
        copy.write_bytes(b'self hosted')
        monkeypatch.setattr(rpipe, '__file__', str(copy))
        source = copy
    else:
        os.link(source, copy)
    provenance.write_provenance(study, config)
    provenance.check_frozen(study, dict(config, freeze=True))
    assert len(provenance.load_provenance(study)['files']) == 1
    source.write_bytes(b'changed shared inode')
    with pytest.raises(provenance.ProvenanceChangedError, match='rpipe/__init__.py'):
        provenance.check_frozen(study, dict(config, freeze=True))


def test_reserved_namespace_refuses_extra_even_before_library_has_that_name(inputs):
    _, study, _ = inputs
    (study / 'rpipe/extra.py').write_bytes(b'project extra')
    with pytest.raises(provenance.ProvenanceChangedError, match='library'):
        provenance.source_files(study, {'provenance': {'include': ['rpipe/extra.py']}})


def test_hardlink_does_not_mask_deleted_library_path(inputs):
    package, study, config = inputs
    source = package / '__init__.py'
    os.link(source, study / 'rpipe/__init__.py')
    provenance.write_provenance(study, config)
    source.unlink()  # Only this test's synthetic library path; linked bytes remain.
    with pytest.raises(provenance.ProvenanceChangedError, match='library'):
        provenance.check_frozen(study, dict(config, freeze=True))


def test_similar_nonreserved_namespace_is_accepted(inputs):
    _, study, _ = inputs
    target = study / 'rpipe_inputs/extra.py'
    target.parent.mkdir()
    target.write_bytes(b'extra')
    files = provenance.source_files(study, {'provenance': {'include': ['rpipe_inputs/extra.py']}})
    assert set(files) == {'rpipe/__init__.py', 'rpipe_inputs/extra.py'}


def test_unreadable_source_identity_is_freeze_refusal(inputs, monkeypatch):
    package, study, config = inputs
    source, copy = package / '__init__.py', study / 'rpipe/__init__.py'
    os.link(source, copy)
    provenance.write_provenance(study, config)
    original = type(copy).samefile
    def denied(path, other):
        if path == copy:
            raise PermissionError('identity denied')
        return original(path, other)
    monkeypatch.setattr(type(copy), 'samefile', denied)
    with pytest.raises(provenance.ProvenanceChangedError, match='library') as caught:
        provenance.check_frozen(study, dict(config, freeze=True))
    assert isinstance(caught.value.__cause__, PermissionError)
