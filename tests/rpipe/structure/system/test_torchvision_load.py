import importlib
import sys

import pytest

from rpipe.structure.system.torchvision_load import import_torchvision

pytestmark = [
    pytest.mark.unit,
    pytest.mark.content,
    pytest.mark.p1,
    pytest.mark.structure_layer,
    pytest.mark.module_system,
]


def test_import_torchvision_retries_after_missing_nms_op(monkeypatch):
    calls = {'n': 0}
    real = importlib.import_module

    def fake(name, package=None):
        if name == 'torchvision':
            calls['n'] += 1
            if calls['n'] == 1:
                sys.modules['torchvision.partial'] = object()
                raise RuntimeError('operator torchvision::nms does not exist')
            return sys.modules.setdefault('torchvision', object())
        return real(name, package)

    monkeypatch.setattr('rpipe.structure.system.torchvision_load.importlib.import_module', fake)
    monkeypatch.setattr('rpipe.structure.system.torchvision_load.time.sleep', lambda _seconds: None)
    module = import_torchvision()
    assert calls['n'] == 2
    assert module is sys.modules['torchvision']
    assert 'torchvision.partial' not in sys.modules


def test_import_torchvision_does_not_retry_other_runtime_errors(monkeypatch):
    def fake(name, package=None):
        raise RuntimeError('unrelated')

    monkeypatch.setattr('rpipe.structure.system.torchvision_load.importlib.import_module', fake)
    with pytest.raises(RuntimeError, match='unrelated'):
        import_torchvision()
