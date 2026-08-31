from pathlib import Path

from rpipe.structure.algorithm.tracker import AlgorithmTracker
from rpipe.structure.system.logger import Logger


def test_logger_writes_run_log_and_reads_tracker(tmp_path: Path, capsys):
    tracker = AlgorithmTracker(tmp_path)
    tracker.append('train', n=1, values={'Loss': 0.5, 'Accuracy': 0.25})
    logger = Logger(tmp_path)
    logger.report(tracker, 'train', extra={'epoch': 1})
    captured = capsys.readouterr()
    assert 'Loss' in captured.out
    text = (tmp_path / 'logs' / 'run.log').read_text(encoding='utf-8')
    assert 'Loss' in text
    assert 'epoch 1' in text
