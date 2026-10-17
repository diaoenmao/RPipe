"""Offline hardware discovery and unchanged public exports."""

import pickle
import subprocess
import sys
from types import SimpleNamespace

import pytest

from rpipe.structure import make
from rpipe.structure.api import make_api
from rpipe.structure.make import capacity, hardware

pytestmark = [pytest.mark.unit, pytest.mark.content, pytest.mark.p1,
              pytest.mark.structure_layer, pytest.mark.module_make,
              pytest.mark.cost(cost_class='c1'),
              pytest.mark.result_type('categorical', detail='summary')]


def test_hardware_exports_keep_identity_and_legacy_pickle_resolution():
    assert make_api.GpuInfo is make.GpuInfo is capacity.GpuInfo is hardware.GpuInfo
    assert make_api.probe_gpus is make.probe_gpus is capacity.probe_gpus is hardware.probe_gpus
    assert hardware.GpuInfo.__module__ == 'rpipe.structure.make.hardware'
    # Synthetic protocol-0 GLOBAL through the pre-move public export.
    old = b'crpipe.structure.make.capacity\nGpuInfo\n(I0\nVsynthetic\nI10\nI7\ntR.'
    info = pickle.loads(old)
    assert type(info) is hardware.GpuInfo
    assert repr(info) == "GpuInfo(index=0, name='synthetic', total_bytes=10, free_bytes=7)"
    assert pickle.loads(pickle.dumps(info)) == info
    assert not hasattr(capacity, '_probe_torch') and not hasattr(capacity, '_probe_nvidia_smi')


@pytest.mark.parametrize('count', [0, -1])
def test_no_requested_gpus_does_not_probe(monkeypatch, count):
    def forbidden(*args):
        pytest.fail('empty request must not access hardware')
    monkeypatch.setattr(hardware, '_probe_nvidia_smi', forbidden)
    monkeypatch.setattr(hardware, '_probe_torch', forbidden)
    assert hardware.probe_gpus(2, count) == []


@pytest.mark.parametrize('smi_available', [False, True])
def test_probe_keeps_smi_priority_and_requested_indices(monkeypatch, smi_available):
    gpu = hardware.GpuInfo(2, 'synthetic', 10, 7)
    calls = []
    def smi(wanted):
        calls.append(('smi', wanted))
        return [gpu] if smi_available else []
    def torch(wanted):
        calls.append(('torch', wanted))
        return [gpu]
    monkeypatch.setattr(hardware, '_probe_nvidia_smi', smi)
    monkeypatch.setattr(hardware, '_probe_torch', torch)
    assert hardware.probe_gpus(2, 2) == [gpu]
    assert calls == [('smi', [2, 3])] + ([] if smi_available else [('torch', [2, 3])])


def test_smi_parsing_keeps_requested_order_and_ignores_bad_rows(monkeypatch):
    monkeypatch.setattr(hardware.shutil, 'which', lambda name: '/synthetic/nvidia-smi')
    def output(command, **kwargs):
        assert command[0] == '/synthetic/nvidia-smi'
        assert kwargs == {'text': True, 'timeout': 10}
        return 'bad row\nno,index,total,free\n0, zero, 16, 8\n2, two, 32, 12\n'
    monkeypatch.setattr(hardware.subprocess, 'check_output', output)
    assert hardware._probe_nvidia_smi([2, 0, 9]) == [
        hardware.GpuInfo(2, 'two', 32 * 1024**2, 12 * 1024**2),
        hardware.GpuInfo(0, 'zero', 16 * 1024**2, 8 * 1024**2)]


@pytest.mark.parametrize('error', [OSError('synthetic refusal'), subprocess.TimeoutExpired('synthetic', 10)])
def test_smi_unavailable_keeps_empty_fallback(monkeypatch, error):
    monkeypatch.setattr(hardware.shutil, 'which', lambda name: '/synthetic/nvidia-smi')
    def refused(*args, **kwargs):
        raise error
    monkeypatch.setattr(hardware.subprocess, 'check_output', refused)
    assert hardware._probe_nvidia_smi([0]) == []
    monkeypatch.setattr(hardware.shutil, 'which', lambda name: None)
    assert hardware._probe_nvidia_smi([0]) == []


@pytest.mark.parametrize('available', [False, True])
def test_torch_probe_uses_only_mocked_device_state(monkeypatch, available):
    calls = []
    def memory(index):
        calls.append(index)
        return (7, 10)
    cuda = SimpleNamespace(is_available=lambda: available, device_count=lambda: 3,
                           mem_get_info=memory,
                           get_device_properties=lambda index: SimpleNamespace(name=f'fake-{index}'))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda))
    result = hardware._probe_torch([-1, 2, 3, 0])
    assert result == ([hardware.GpuInfo(2, 'fake-2', 10, 7), hardware.GpuInfo(0, 'fake-0', 10, 7)]
                      if available else [])
    assert calls == ([2, 0] if available else [])


def test_torch_optional_dependency_stays_optional(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', None)
    assert hardware._probe_torch([0]) == []
