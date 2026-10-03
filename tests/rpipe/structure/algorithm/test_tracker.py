import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.content,
    pytest.mark.p2,
    pytest.mark.structure_layer,
    pytest.mark.module_algorithm,
    pytest.mark.cost(cost_class='c1'),
    pytest.mark.result_type('categorical', detail='summary'),
]

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


def test_segment_presence_includes_full_metrics_and_survives_checkpoint(tmp_path):
    import torch
    from rpipe.structure.algorithm.metric import MetricBundle

    tracker = AlgorithmTracker(tmp_path, MetricBundle({'train': ['RMSE']}))
    values = tracker.evaluate('train', 'batch', {'target': torch.tensor([0.0])}, {'pred': torch.tensor([2.0])})
    tracker.append('train', values=values)
    assert values == {} and tracker.has_samples('train')
    state = tracker.state_dict()
    restored = AlgorithmTracker(tmp_path / 'restored')
    restored.load_state_dict(state)
    assert restored.has_samples('train')
    tracker.save('train')
    tracker.reset('train')
    assert not tracker.has_samples('train')
    assert tracker.segment_mean('train') == {'RMSE': 2.0}
