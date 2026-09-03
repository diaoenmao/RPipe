from pathlib import Path

from rpipe.structure.system.config import SystemConfig
from rpipe.structure.system.factory import SystemFactory


def test_system_save_checkpoint_writes_pt(tmp_path: Path):
    system = SystemFactory.build(SystemConfig.from_mapping({'device': 'cpu'}), tmp_path)
    path = system.save_checkpoint({'step': 3, 'model': {'w': 1}}, 'latest')
    assert path == tmp_path / 'checkpoints' / 'latest.pt'
    assert path.is_file()
    import torch

    loaded = torch.load(path, weights_only=False)
    assert loaded['step'] == 3
    assert loaded['model'] == {'w': 1}
    again = system.load_checkpoint('latest')
    assert again['step'] == 3
    assert system.load_checkpoint('missing') is None
