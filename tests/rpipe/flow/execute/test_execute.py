import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.content,
    pytest.mark.p1,
    pytest.mark.flow_layer,
    pytest.mark.module_runner,
    pytest.mark.cost(cost_class='c1'),
    pytest.mark.result_type('categorical', detail='summary'),
]


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

    def exception(self, message: str, exc: BaseException) -> None:
        self.lines.append(f'error {message} {type(exc).__name__}: {exc}')


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


@pytest.mark.parametrize('primary_kind', ['success', 'runtime', 'interrupt', 'exit'])
@pytest.mark.parametrize('flush_kind', ['success', 'io', 'encoding'])
def test_execute_final_flush_success_is_strict_and_failure_keeps_primary(tmp_path, primary_kind, flush_kind):
    """Shared execute boundary finalizes once: successful body rejects write failure, existing exceptions/termination remain primary."""
    primary = {'success': None, 'runtime': RuntimeError('body failed'),
               'interrupt': KeyboardInterrupt('body interrupted'), 'exit': SystemExit(7)}[primary_kind]
    final = {'success': None, 'io': PermissionError('final state refused'),
             'encoding': TypeError('state encoding refused')}[flush_kind]
    calls = []
    class Algorithm:
        mode = 'eval'
        def run(self, *_args):
            if primary is not None:
                raise primary
            return {'accuracy': 75.0}
    class Tracker:
        def flush_state(self):
            calls.append('flush')
            if final is not None:
                raise final
    ctx = FlowContext(study_dir=tmp_path,
                      layout=ArtifactLayout(root=tmp_path / 'runs/x', study_dir=tmp_path), config={})
    ctx.control = object()
    log = _Log()
    ctx.state.update(algorithm=Algorithm(), tracker=Tracker(), logger=log)
    expected = primary if primary is not None else final
    if expected is None:
        execute_run(ctx)
        assert ctx.state['execute'] == {'accuracy': 75.0}
    else:
        with pytest.raises(type(expected)) as caught:
            execute_run(ctx)
        assert caught.value is expected
    assert calls == ['flush']
    if isinstance(primary, Exception):
        assert sum('execute failed' in line for line in log.lines) == 1
        assert not any('execute finished' in line for line in log.lines)
