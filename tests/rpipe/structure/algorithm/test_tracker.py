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


def test_checkpoint_preserves_accumulator_without_mean_round_trip(tmp_path):
    tracker = AlgorithmTracker(tmp_path)
    tracker.append('train', n=54, values={'Loss': 7.370161863460789e-09 / 54})
    state = tracker.state_dict()
    state['splits']['train']['Loss']['weighted_sum'] = 7.370161863460789e-09
    state['splits']['train']['Loss']['mean'] = 7.370161863460789e-09 / 54
    assert state['splits']['train']['Loss']['mean'] * 54 != state['splits']['train']['Loss']['weighted_sum']
    restored = AlgorithmTracker(tmp_path / 'restored')
    restored.load_state_dict(state)
    assert restored.state_dict()['splits'] == state['splits']


def test_legacy_tracker_without_accumulator_still_loads(tmp_path):
    tracker = AlgorithmTracker(tmp_path)
    tracker.append('train', n=3, values={'Loss': 2.0})
    state = tracker.state_dict()
    state['splits']['train']['Loss'].pop('weighted_sum')
    restored = AlgorithmTracker(tmp_path / 'restored')
    restored.load_state_dict(state)
    assert restored.mean('train')['Loss'] == 2.0


def test_tracker_constructor_preserves_evidence_arriving_before_open(tmp_path, monkeypatch):
    from rpipe.structure.artifact.asset import kinds

    target = tmp_path / kinds.TRACKER_JSONL
    native = Path.open
    arrivals = []

    def arriving(path, mode='r', *args, **kwargs):
        if path == target and mode in ('w', 'a') and not arrivals:
            arrivals.append(True)
            with native(path, 'wb') as stream:
                stream.write(b'prior error evidence\n')
        return native(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', arriving)
    AlgorithmTracker(tmp_path)
    assert arrivals == [True]
    assert target.read_bytes() == b'prior error evidence\n'


def test_tracker_constructor_still_rejects_directory(tmp_path):
    from rpipe.structure.artifact.asset import kinds

    target = tmp_path / kinds.TRACKER_JSONL
    target.mkdir(parents=True)
    with pytest.raises(OSError):
        AlgorithmTracker(tmp_path)
