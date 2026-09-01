import pytest

from rpipe.flow.context import FlowContext
from rpipe.flow.execute import run as execute_run
from rpipe.structure.artifact.layout import ArtifactLayout


class _Boom:
    mode = 'train'

    def run(self, *_args, **_kwargs):
        raise RuntimeError('boom')


class _Log:
    def __init__(self):
        self.lines: list[str] = []

    def info(self, message: str) -> None:
        self.lines.append(f'info {message}')

    def error(self, message: str) -> None:
        self.lines.append(f'error {message}')


class _Tracker:
    def flush_state(self) -> None:
        return None


def test_execute_failure_logs_failed_not_finished(tmp_path):
    ctx = FlowContext(
        study_dir=tmp_path,
        layout=ArtifactLayout(root=tmp_path / 'runs' / 'x', study_dir=tmp_path),
        config={},
    )
    ctx.control = object()
    log = _Log()
    ctx.state['algorithm'] = _Boom()
    ctx.state['tracker'] = _Tracker()
    ctx.state['logger'] = log
    with pytest.raises(RuntimeError, match='boom'):
        execute_run(ctx)
    assert any('execute failed' in line for line in log.lines)
    assert not any('execute finished' in line for line in log.lines)
