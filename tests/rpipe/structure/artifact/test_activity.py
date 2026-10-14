"""Optional activity hints cannot hide authoritative Run status."""
from pathlib import Path
import json

import pytest

from rpipe.structure.artifact.activity import read_activity

pytestmark = [pytest.mark.unit, pytest.mark.content, pytest.mark.p1,
              pytest.mark.structure_layer, pytest.mark.module_artifact,
              pytest.mark.cost(cost_class='c1'),
              pytest.mark.result_type('categorical', detail='summary')]


@pytest.mark.parametrize('payload', [b'\xff', b'{invalid', b'[]', b'{}'])
def test_unusable_activity_is_ignored_without_rewriting(tmp_path, payload):
    path = tmp_path / 'activity.json'
    path.write_bytes(payload)
    assert read_activity(tmp_path) is None
    assert path.read_bytes() == payload


@pytest.mark.parametrize('operation', ['is_file', 'read_text'])
def test_unreadable_activity_is_ignored_including_existence_check(tmp_path, monkeypatch, operation):
    path = tmp_path / 'activity.json'
    path.write_bytes(b'{"phase": "make"}')
    before = path.read_bytes()
    original = getattr(Path, operation)
    def denied(candidate, *args, **kwargs):
        if candidate == path:
            raise PermissionError('activity denied')
        return original(candidate, *args, **kwargs)
    monkeypatch.setattr(Path, operation, denied)
    assert read_activity(tmp_path) is None
    assert path.read_bytes() == before


def test_missing_activity_stays_absent(tmp_path):
    assert read_activity(tmp_path) is None
    assert list(tmp_path.iterdir()) == []


def test_valid_unicode_activity_keeps_original_bytes(tmp_path):
    path = tmp_path / 'activity.json'
    body = {'phase': 'make', 'detail': '准备共享输入'}
    payload = json.dumps(body, ensure_ascii=False).encode('utf-8')
    path.write_bytes(payload)
    assert read_activity(tmp_path) == body
    assert path.read_bytes() == payload


def test_unexpected_activity_reader_bug_is_not_suppressed(tmp_path, monkeypatch):
    path = tmp_path / 'activity.json'
    path.write_bytes(b'{"phase": "make"}')
    original = Path.read_text
    def broken(candidate, *args, **kwargs):
        if candidate == path:
            raise RuntimeError('unexpected reader bug')
        return original(candidate, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', broken)
    with pytest.raises(RuntimeError, match='unexpected reader bug'):
        read_activity(tmp_path)
