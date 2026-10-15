from pathlib import Path

import pytest

from rpipe.structure.artifact import (
    artifact_layout,
    build_index,
    experiment_entries,
    load_config,
    load_index,
    validate_result,
    write_config,
    write_index,
    write_result,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.content,
    pytest.mark.p1,
    pytest.mark.structure_layer,
    pytest.mark.module_artifact,
    pytest.mark.cost(cost_class='c1'),
    pytest.mark.result_type('categorical', detail='summary'),
]


def test_artifact_layout_creates_assets_dir(tmp_path: Path):
    layout = artifact_layout(tmp_path, 'seed_0')
    assert layout.assets_dir.is_dir()


@pytest.mark.parametrize('run_id', ['', '.', '..', '../outside', '..\\outside', 'nested/run', 'C:\\outside',
                                    'C:alias', ' name', 'name ', 'name.', 'CON', 'NUL.txt'])
def test_artifact_layout_rejects_unsafe_run_directory_before_writes(tmp_path, monkeypatch, run_id):
    import rpipe.structure.artifact.layout as module

    def no_write(*args):
        raise AssertionError('invalid Run must be rejected before creating any directory')

    monkeypatch.setattr(module, 'ensure_study_layout', no_write)
    study = tmp_path / 'study'
    with pytest.raises(ValueError, match='Run directory'):
        artifact_layout(study, run_id)
    assert not study.exists()


def test_index_rejects_unsafe_run_directory_before_readout(tmp_path):
    write_index(tmp_path, {'experiments': [{'runs': [{'id': 'safe', 'run_dir': '../outside'}]}]})
    with pytest.raises(ValueError, match='Run directory'):
        load_index(tmp_path)


def test_write_config_roundtrip_preserves_slug(tmp_path: Path):
    layout = artifact_layout(tmp_path, 'seed_0')
    write_config(layout.config_path, {'slug': 'seed_0', 'seed': 0})
    loaded = load_config(layout.config_path)
    assert loaded['slug'] == 'seed_0'


def test_validate_result_empty_mapping_returns_errors():
    errors = validate_result({})
    assert errors


def test_write_result_valid_payload_creates_file(tmp_path: Path):
    layout = artifact_layout(tmp_path, 'seed_0')
    data = {
        'status': 'succeeded',
        'control': {'slug': 'seed_0'},
        'metrics': {},
        'paths': {},
    }
    assert validate_result(data) == []
    path = write_result(layout.result_path, data)
    assert path.is_file()


def test_write_result_missing_status_raises_value_error(tmp_path: Path):
    layout = artifact_layout(tmp_path, 'x')
    with pytest.raises(ValueError, match='status'):
        write_result(layout.result_path, {'control': {}, 'metrics': {}, 'paths': {}})


def test_index_groups_runs_by_experiment_factors(tmp_path: Path):
    study_dir = tmp_path / 'studies' / 'demo'
    study_dir.mkdir(parents=True)
    cfgs: list[tuple[Path, dict]] = []
    for size, seed, run_id in ((500, 0, 'a'), (500, 1, 'b'), (2000, 0, 'c')):
        layout = artifact_layout(study_dir, run_id)
        cfg = {
            'id': run_id,
            'description': f'size={size} seed={seed}',
            'seed': seed,
            'data': {'config': {'train_size': size}},
        }
        write_config(layout.config_path, cfg)
        cfgs.append((layout.config_path, cfg))

    experiments = experiment_entries(
        configs=cfgs,
        axis_keys=['data.config.train_size'],
        study_dir=study_dir,
    )
    assert experiments[0]['runs'][0]['config'] == 'runs/a/config.yaml'
    assert experiments[0]['runs'][0]['log'] == 'runs/a/assets/logs/run.log'
    assert len(experiments) == 2
    assert experiments[0]['factors'] == {'data.config.train_size': 500}
    assert [r['seed'] for r in experiments[0]['runs']] == [0, 1]
    assert experiments[1]['factors'] == {'data.config.train_size': 2000}
    assert [r['id'] for r in experiments[1]['runs']] == ['c']

    index = build_index(
        study='demo',
        description='demo study',
        experiments=experiments,
    )
    assert index['id']
    path = write_index(study_dir, index)
    assert path == study_dir / 'index.json'
    loaded = load_index(study_dir)
    assert loaded['experiments'][0]['runs'][0]['id'] == 'a'
    assert not artifact_layout(study_dir, 'a').result_path.exists()


@pytest.mark.parametrize('entry', [{'id': '../outside', 'run_dir': 'safe'}, {'id': 'safe', 'run_dir': '../outside'}])
def test_index_rejects_unsafe_id_even_with_safe_directory(tmp_path, entry):
    write_index(tmp_path, {'experiments': [{'runs': [entry]}]})
    with pytest.raises(ValueError):
        load_index(tmp_path)


@pytest.mark.parametrize('body', [
    {'experiments': {}},
    {'experiments': {'unexpected': []}},
    {'experiments': [7]},
    {'experiments': [{'runs': {}}]},
    {'experiments': [{'runs': {'unexpected': []}}]},
    {'experiments': [{'runs': [7]}]},
    {'experiments': [{'factors': 7}]},
    {'experiments': [{'factors': []}]},
    {'experiments': [{'factors': [['axis', 1]]}]},
    {'experiments': [{'factors': 'bad'}]},
], ids=['experiments-empty-object', 'experiments-object', 'experiment-scalar',
        'runs-empty-object', 'runs-object', 'run-scalar', 'factors-scalar',
        'factors-list', 'factors-pairs', 'factors-text'])
def test_index_rejects_malformed_nested_containers_without_writes(tmp_path, body):
    import json

    path = tmp_path / 'index.json'
    raw = json.dumps(body).encode('utf-8')
    path.write_bytes(raw)
    peer = tmp_path / 'peer.json'
    peer.write_bytes(b'preserve peer')
    with pytest.raises((TypeError, ValueError), match='Study index'):
        load_index(tmp_path)
    assert path.read_bytes() == raw
    assert peer.read_bytes() == b'preserve peer'
    assert set(tmp_path.iterdir()) == {path, peer}


@pytest.mark.parametrize('command', ['status', 'logs'])
@pytest.mark.parametrize('body', [
    {'experiments': [7]},
    {'experiments': [{'runs': [7]}]},
    {'experiments': {}},
    {'experiments': [{'factors': [['axis', 1]], 'runs': []}]},
], ids=['experiment-scalar', 'run-scalar', 'false-empty-plan', 'factor-pairs'])
def test_readouts_reject_malformed_index_at_shared_boundary(tmp_path, capsys, command, body):
    import json
    from rpipe.flow.cli import main

    path = tmp_path / 'index.json'
    raw = json.dumps(body).encode('utf-8')
    path.write_bytes(raw)
    assert main([command, str(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert 'Study index' in captured.err
    assert not captured.out
    assert path.read_bytes() == raw
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('body', [
    {}, {'experiments': None}, {'experiments': []},
    {'experiments': [{}]},
    {'experiments': [{'runs': None, 'factors': None}]},
    {'experiments': [{'runs': [], 'factors': {}}]},
], ids=['missing', 'null', 'empty-list', 'missing-fields', 'null-fields', 'empty-fields'])
def test_index_keeps_legacy_optional_empty_containers(tmp_path, body):
    import json

    raw = json.dumps(body).encode('utf-8')
    path = tmp_path / 'index.json'
    path.write_bytes(raw)
    assert load_index(tmp_path) == body
    assert path.read_bytes() == raw


@pytest.mark.parametrize('text', ['false', '0', '[]', "''", 'true', '1'],
                         ids=['false', 'zero', 'empty-list', 'empty-string', 'true', 'one'])
def test_config_rejects_non_mapping_yaml_root_without_writes(tmp_path, text):
    from rpipe.structure.artifact.errors import CorruptArtifactError

    path = tmp_path / 'config.yaml'
    raw = text.encode('utf-8')
    path.write_bytes(raw)
    with pytest.raises(CorruptArtifactError, match='mapping'):
        load_config(path)
    assert path.read_bytes() == raw
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('text', ['', '# empty document', 'null', '{}'])
def test_config_keeps_legacy_empty_yaml_document_or_mapping(tmp_path, text):
    path = tmp_path / 'config.yaml'
    path.write_text(text, encoding='utf-8')
    assert load_config(path) == {}
    assert path.read_text(encoding='utf-8') == text


@pytest.mark.parametrize('status', [[], {}, ['succeeded'], None, False, 0, 1.0, '', 'queued'],
                         ids=['list', 'object', 'list-label', 'null', 'bool', 'int', 'float', 'empty', 'unknown'])
def test_result_validator_reports_invalid_status_without_type_escape(status):
    errors = validate_result({'status': status})
    assert isinstance(errors, list)
    assert errors and all(isinstance(error, str) for error in errors)
    assert any('status' in error for error in errors)
