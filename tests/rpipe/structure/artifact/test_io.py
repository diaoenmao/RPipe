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
]


def test_artifact_layout_creates_assets_dir(tmp_path: Path):
    layout = artifact_layout(tmp_path, 'seed_0')
    assert layout.assets_dir.is_dir()


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
