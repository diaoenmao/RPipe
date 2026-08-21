from pathlib import Path

from rpipe.artifact import (
    artifact_layout,
    build_index,
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


def test_index_before_results(tmp_path: Path):
    study_dir = tmp_path / 'studies' / 'demo'
    study_dir.mkdir(parents=True)
    exp = tmp_path / 'experiments' / 'demo_exp'
    layout = artifact_layout(exp, 'runabc')
    write_config(
        layout.config_path,
        {
            'id': 'runabc',
            'description': 'demo run',
            'seed': 0,
            'data': {'name': 'X'},
            'model': {'name': 'Y'},
            'algorithm': {'mode': 'train'},
            'system': {'device': 'cpu'},
        },
    )

    index = build_index(
        study='demo',
        description='demo study',
        experiments=[
            {
                'name': 'demo_exp',
                'description': 'demo experiment',
                'path': str(exp),
                'runs': [
                    {
                        'id': 'runabc',
                        'description': 'demo run',
                        'run_dir': 'runabc',
                        'config': str(layout.config_path),
                    }
                ],
            }
        ],
    )
    assert index['id']
    assert index['description'] == 'demo study'
    path = write_index(study_dir, index)
    assert path == study_dir / 'index.json'
    loaded = load_index(study_dir)
    assert loaded['id'] == index['id']
    assert loaded['experiments'][0]['runs'][0]['id'] == 'runabc'
    # no result.json required
    assert not layout.result_path.exists()
