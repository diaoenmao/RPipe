import re

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.content,
    pytest.mark.p1,
    pytest.mark.structure_layer,
    pytest.mark.module_system,
    pytest.mark.cost(cost_class='c1'),
    pytest.mark.result_type('categorical', detail='summary'),
]

from pathlib import Path

from rpipe.structure.algorithm.tracker import AlgorithmTracker
from rpipe.structure.system.logger import Logger

_HEADER = re.compile(
    r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}[+-]\d{2}:\d{2} '
    r'(INFO |WARN |ERROR) '
)


def test_logger_writes_timestamp_level_and_run_id(tmp_path: Path, capsys):
    assets = tmp_path / 'runs' / 'abc123' / 'assets'
    logger = Logger(assets)
    logger.info('start phases=prepare')
    captured = capsys.readouterr().out
    assert _HEADER.match(captured)
    assert ' INFO  abc123 [flow] start phases=prepare' in captured
    text = (assets / 'logs' / 'run.log').read_text(encoding='utf-8')
    assert text == captured


def test_logger_report_uses_metric_tags(tmp_path: Path, capsys):
    tracker = AlgorithmTracker(tmp_path)
    tracker.append('train', n=1, values={'Loss': 0.5, 'Accuracy': 0.25})
    logger = Logger(tmp_path)
    logger.report(tracker, 'train', extra={'epoch': 1, 'elapsed': '0:00:01', 'eta': '0:00:09'})
    text = (tmp_path / 'logs' / 'run.log').read_text(encoding='utf-8')
    assert _HEADER.match(text)
    assert ' - [epoch] 1 [split] train [metric] Accuracy=0.2500 Loss=0.5000 [time] elapsed=0:00:01 eta=0:00:09' in text
    assert capsys.readouterr().out == text


def test_logger_exception_writes_traceback_on_each_line(tmp_path: Path, capsys):
    assets = tmp_path / 'runs' / 'deadbeef' / 'assets'
    logger = Logger(assets)

    def inner():
        raise ValueError('nope')

    try:
        inner()
    except ValueError as exc:
        logger.exception('phase=execute status=failed', exc)

    text = (assets / 'logs' / 'run.log').read_text(encoding='utf-8')
    out = capsys.readouterr().out
    for blob in (text, out):
        assert ' ERROR deadbeef [error] phase=execute status=failed ValueError: nope' in blob
        assert ' ERROR deadbeef [error] Traceback (most recent call last):' in blob
        assert 'ValueError: nope' in blob
        assert 'inner' in blob


def test_logger_report_after_save_reset_uses_segment(tmp_path: Path, capsys):
    tracker = AlgorithmTracker(tmp_path)
    tracker.append('test', n=10, values={'Loss': 0.5, 'Accuracy': 0.8})
    tracker.save('test')
    tracker.reset('test')
    Logger(tmp_path).report(tracker, 'test')
    text = capsys.readouterr().out
    assert '[split] test [metric] Accuracy=0.8000 Loss=0.5000' in text
