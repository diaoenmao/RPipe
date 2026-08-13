from pathlib import Path

from rpipe.artifact import artifact_layout, load_config, write_config, write_result, validate_result


def test_artifact_layout_and_config_roundtrip(tmp_path: Path):
    layout = artifact_layout(tmp_path, 'seed_0')
    assert layout.assets_dir.is_dir()
    write_config(layout.config_path, {'slug': 'seed_0', 'seed': 0})
    loaded = load_config(layout.config_path)
    assert loaded['slug'] == 'seed_0'


def test_result_validation_and_write(tmp_path: Path):
    layout = artifact_layout(tmp_path, 'seed_0')
    assert validate_result({}) 
    data = {'control': {'slug': 'seed_0'}, 'metrics': {}, 'paths': {}}
    assert validate_result(data) == []
    path = write_result(layout.result_path, data)
    assert path.is_file()
