from rpipe.structure.algorithm.config import AlgorithmConfig
from rpipe.structure.algorithm.train import TrainAlgorithm
from rpipe.structure.algorithm.tracker import AlgorithmTracker


class _Data:
    name = 'Toy'


def test_two_epoch_stub_train_does_not_raise(tmp_path):
    algo = TrainAlgorithm(AlgorithmConfig.from_mapping({'mode': 'train', 'num_epochs': 2}))
    tracker = AlgorithmTracker(tmp_path)
    out = algo.run(_Data(), None, None, tracker)
    assert out['mode'] == 'train'
    assert tracker.segment_mean('train').get('Loss') == 0.0
