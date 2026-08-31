from pathlib import Path

from rpipe.structure.api import data_api
from rpipe.structure.data import DataConfig


def test_stub_source_does_not_download(tmp_path: Path):
    data = data_api.build(
        DataConfig.from_mapping({'name': 'MNIST', 'source': 'stub'}),
        tmp_path,
    )
    assert data.source == 'stub'
    assert data.meta.get('stub') is True
    assert not (tmp_path / 'mnist').exists()
