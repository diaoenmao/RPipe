from pathlib import Path

from rpipe.structure.artifact import artifact_layout, list_asset_files
from rpipe.structure.artifact.asset.kinds import RUN_LOG


def test_list_asset_files_is_recursive(tmp_path: Path):
    layout = artifact_layout(tmp_path, 'run')
    path = layout.assets_dir / RUN_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('x\n', encoding='utf-8')
    names = list_asset_files(layout.assets_dir)
    assert RUN_LOG in names
