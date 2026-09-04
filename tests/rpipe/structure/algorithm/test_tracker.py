from pathlib import Path

from rpipe.structure.algorithm.tracker import AlgorithmTracker


def test_tracker_keeps_jsonl_on_init(tmp_path: Path):
    leftover = tmp_path / 'tracker' / 'scalars.jsonl'
    leftover.parent.mkdir(parents=True, exist_ok=True)
    leftover.write_text('{"stale": true}\n', encoding='utf-8')
    tracker = AlgorithmTracker(tmp_path)
    assert 'stale' in tracker.jsonl_path.read_text(encoding='utf-8')
    tracker.append('train', n=1, values={'Loss': 1.0})
    tracker.flush('train')
    text = tracker.jsonl_path.read_text(encoding='utf-8')
    assert 'Loss' in text


def test_tracker_weighted_mean_and_segment(tmp_path: Path):
    tracker = AlgorithmTracker(tmp_path)
    tracker.append('train', n=2, values={'Loss': 1.0})
    tracker.append('train', n=2, values={'Loss': 3.0})
    assert tracker.mean('train')['Loss'] == 2.0
    tracker.save('train')
    tracker.reset('train')
    assert tracker.mean('train')['Loss'] == 0.0
    assert tracker.segment_mean('train')['Loss'] == 2.0
    tracker.flush('train')
    assert (tmp_path / 'tracker' / 'tracker_state.json').is_file()
