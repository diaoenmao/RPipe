import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.content,
    pytest.mark.p2,
    pytest.mark.flow_layer,
    pytest.mark.module_cli,
]

from rpipe.flow.cli import _configure_stdio
from rpipe.structure.make.capacity import pack_label


def test_windows_stdio_accepts_multiplication_sign():
    _configure_stdio()
    assert pack_label(['linear'] * 9) == 'linear×9'
