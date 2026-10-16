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


def test_logger_constructor_preserves_evidence_arriving_before_open(tmp_path, monkeypatch):
    from rpipe.structure.artifact.asset import kinds

    target = tmp_path / kinds.RUN_LOG
    native = Path.open
    arrivals = []

    def arriving(path, mode='r', *args, **kwargs):
        if path == target and mode in ('w', 'a') and not arrivals:
            arrivals.append(True)
            with native(path, 'wb') as stream:
                stream.write(b'prior error evidence\n')
        return native(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', arriving)
    Logger(tmp_path)
    assert arrivals == [True]
    assert target.read_bytes() == b'prior error evidence\n'


def test_logger_constructor_still_rejects_directory(tmp_path):
    from rpipe.structure.artifact.asset import kinds

    target = tmp_path / kinds.RUN_LOG
    target.mkdir(parents=True)
    with pytest.raises(OSError):
        Logger(tmp_path)


def test_logger_constructor_does_not_reopen_existing_evidence(tmp_path, monkeypatch):
    from rpipe.structure.artifact.asset import kinds

    target = tmp_path / kinds.RUN_LOG
    target.parent.mkdir(parents=True)
    target.write_bytes(b'existing diagnostic bytes\n')
    native = Path.open

    def no_reopen(path, *args, **kwargs):
        if path == target:
            raise PermissionError('existing evidence must not be reopened by initialization')
        return native(path, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(Path, 'open', no_reopen)
        Logger(tmp_path)
    assert target.read_bytes() == b'existing diagnostic bytes\n'


@pytest.mark.parametrize('phase', ['write', 'flush'])
@pytest.mark.parametrize('outcome', ['error', 'cancel', 'normal', 'programming'])
def test_logger_stream_close_preserves_write_or_flush_failure(tmp_path, stream_close_fault, phase, outcome):
    import errno
    logger = Logger(tmp_path)
    logger.path.write_bytes(b'old log evidence\n')
    peer = tmp_path / 'peer.log'
    peer.write_bytes(b'peer evidence')
    primary = KeyboardInterrupt('cancel log') if outcome == 'cancel' else None if outcome == 'normal' else OSError(errno.ENOSPC, 'log IO')
    secondary = RuntimeError('unexpected close error') if outcome == 'programming' else OSError(errno.EIO, 'close IO')
    expected = secondary if outcome in ('normal', 'programming') else primary
    handles, calls = stream_close_fault(lambda path, mode: path == logger.path and mode == 'a', primary, secondary, phase=phase)
    with pytest.raises(type(expected)) as caught:
        logger.info('new log event')
    assert caught.value is expected
    assert len(handles) == 1 and handles[0].closed and calls.count('close') == 1
    assert logger.path.read_bytes().startswith(b'old log evidence\n')
    assert peer.read_bytes() == b'peer evidence'
    assert calls[0] == 'write' and (phase != 'flush' or 'flush' in calls)


@pytest.mark.parametrize('reopen', [False, True], ids=['same-logger', 'reopened-logger'])
@pytest.mark.parametrize('tail', ['ascii-cut', 'utf8-cut', 'lf', 'crlf', 'cr', 'empty'])
def test_logger_keeps_interrupted_tail_separate_from_next_record(tmp_path, capsys, reopen, tail):
    import os
    logger = Logger(tmp_path)
    header = b'2001-01-01T00:00:00.000+00:00 INFO  - [flow] interrupted '
    prefixes = {'ascii-cut': header, 'utf8-cut': header + b'\xe4',
                'lf': header + b'\n', 'crlf': header + b'\r\n', 'cr': header + b'\r', 'empty': b''}
    before = prefixes[tail]
    logger.path.write_bytes(before)
    if reopen:
        logger = Logger(tmp_path)
    assert logger.path.read_bytes() == before
    peer = tmp_path / 'peer.bin'
    peer.write_bytes(b'peer')
    logger.error('current failure')
    emitted = capsys.readouterr().out
    assert _HEADER.match(emitted)
    separator = os.linesep.encode() if tail.endswith('cut') else b''
    assert logger.path.read_bytes() == before + separator + emitted.replace('\n', os.linesep).encode()
    assert peer.read_bytes() == b'peer'


@pytest.mark.parametrize('error_type', [PermissionError, FileNotFoundError])
def test_optional_log_tail_inspection_keeps_append_only_access(tmp_path, monkeypatch, capsys, error_type):
    import os
    logger = Logger(tmp_path)
    before = b'existing record\n'
    logger.path.write_bytes(before)
    native = Path.open

    def append_only(path, *args, **kwargs):
        if path == logger.path and args and args[0] == 'rb':
            raise error_type('tail unavailable')
        return native(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, 'open', append_only)
        logger.error('still appendable')
    emitted = capsys.readouterr().out
    assert logger.path.read_bytes() == before + emitted.replace('\n', os.linesep).encode()
