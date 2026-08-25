from pathlib import Path

from rpipe.structure.artifact import (
    artifact_layout,
    build_index,
    experiment_entries,
    load_config,
    load_index,
    write_config,
    write_result,
    write_index,
    validate_result,
)


def test_artifact_layout_and_config_roundtrip(tmp_path: Path):
    layout = artifact_layout(tmp_path, 'seed_0')
    assert layout.assets_dir.is_dir()
    write_config(layout.config_path, {'slug': 'seed_0', 'seed': 0})
    loaded = load_config(layout.config_path)
    assert loaded['slug'] == 'seed_0'


def test_result_validation_and_write(tmp_path: Path):
    layout = artifact_layout(tmp_path, 'seed_0')
    assert validate_result({})
    data = {
        'status': 'succeeded',
        'control': {'slug': 'seed_0'},
        'metrics': {},
        'paths': {},
    }
    assert validate_result(data) == []
    path = write_result(layout.result_path, data)
    assert path.is_file()


def test_index_grouped_by_experiment_factors(tmp_path: Path):
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
    )
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
