from rpipe.flow.collect import run as collect_run
from rpipe.flow.context import FlowContext
from rpipe.structure.algorithm.tracker import AlgorithmTracker
from rpipe.structure.artifact.layout import ArtifactLayout


class _Tracker:
    def segment_mean(self, split: str):
        if split == 'train':
            return {'Loss': 0.4}
        return {'Accuracy': 0.8}


def test_collect_adds_best_accuracy_from_execute(tmp_path):
    ctx = FlowContext(
        study_dir=tmp_path,
        layout=ArtifactLayout(root=tmp_path / 'runs' / 'x', study_dir=tmp_path),
        config={},
    )
    ctx.state['tracker'] = _Tracker()
    ctx.state['execute'] = {'best_accuracy': 0.91, 'accuracy': 0.8}
    collect_run(ctx)
    assert ctx.state['collected']['metrics']['accuracy'] == 0.8
    assert ctx.state['collected']['metrics']['train_loss'] == 0.4
    assert ctx.state['collected']['metrics']['best_accuracy'] == 0.91


def test_collect_eval_mode_aliases_eval_accuracy(tmp_path):
    ctx = FlowContext(
        study_dir=tmp_path,
        layout=ArtifactLayout(root=tmp_path / 'runs' / 'e', study_dir=tmp_path),
        config={},
    )
    ctx.state['tracker'] = _Tracker()
    ctx.state['execute'] = {'mode': 'eval', 'accuracy': 0.8}
    collect_run(ctx)
    assert ctx.state['collected']['metrics']['accuracy'] == 0.8
    assert ctx.state['collected']['metrics']['eval_accuracy'] == 0.8


def test_collect_without_best_leaves_metric_out(tmp_path):
    ctx = FlowContext(
        study_dir=tmp_path,
        layout=ArtifactLayout(root=tmp_path / 'runs' / 'x', study_dir=tmp_path),
        config={},
    )
    ctx.state['tracker'] = AlgorithmTracker(tmp_path)
    collect_run(ctx)
    assert 'best_accuracy' not in ctx.state['collected']['metrics']
