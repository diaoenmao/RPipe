"""Prepare isolated evidence before Factory construction, then check objects."""
from .data import prepare
from ..execute.paired import seed_runtime
from ..execute import paired
from ..recipe import validate


def before(ctx):
    pair = validate(ctx.config, ctx.control.seed, paired)
    workspace = ctx.layout.assets_dir / 'probe'
    source = ctx.study_dir.parent / 'main_exp' / 'shared' / 'data'
    raw = source / pair[0] / 'raw'
    if not raw.is_dir() or not any(path.is_file() for path in raw.rglob('*')):
        raise FileNotFoundError(f'prepare the raw dataset before launch: {raw}')
    prepared = prepare(workspace, source, pair)
    if prepared.get('passed') is not True:
        raise RuntimeError(f'CPU preparation failed; evidence: {workspace}')
    ctx.state['probe_prepared'] = prepared
    seed_runtime(str(ctx.control.system.get('device', 'cpu')))


def run(ctx):
    if ctx.state.get('probe_prepared', {}).get('passed') is not True:
        raise RuntimeError('probe preparation evidence missing')
    if ctx.state['algorithm'].workspace != ctx.layout.assets_dir / 'probe':
        raise RuntimeError('paired Algorithm must use the prepared Run workspace')
