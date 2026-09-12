from pathlib import Path

from rpipe.structure.algorithm.tracker import AlgorithmTracker
from rpipe.structure.system.logger import Logger


def test_logger_writes_run_log_and_reads_tracker(tmp_path: Path, capsys):
    tracker = AlgorithmTracker(tmp_path)
    tracker.append('train', n=1, values={'Loss': 0.5, 'Accuracy': 0.25})
    logger = Logger(tmp_path)
    logger.report(tracker, 'train', extra={'epoch': 1, 'elapsed': '0:00:01', 'eta': '0:00:09'})
    captured = capsys.readouterr()
    assert 'Loss' in captured.out
    text = (tmp_path / 'logs' / 'run.log').read_text(encoding='utf-8')
    assert 'Loss' in text
    assert 'epoch 1' in text
    assert 'elapsed=0:00:01' in text
    assert 'eta=0:00:09' in text


def test_logger_report_after_save_reset_uses_segment(tmp_path: Path, capsys):
    tracker = AlgorithmTracker(tmp_path)
    tracker.append('test', n=10, values={'Loss': 0.5, 'Accuracy': 0.8})
    tracker.save('test')
    tracker.reset('test')
    Logger(tmp_path).report(tracker, 'test')
    text = capsys.readouterr().out
    assert 'Accuracy 0.8000' in text
    assert 'Loss 0.5000' in text
