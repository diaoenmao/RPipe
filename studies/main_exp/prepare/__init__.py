"""Check that the built Run uses the declared historical adapters."""

from ..recipe import COMMIT, SOURCE


def run(ctx):
    for layer in ('data', 'model'):
        obj = ctx.state[layer]
        if obj.source != SOURCE or obj.meta.get('historical_commit') != COMMIT:
            raise ValueError(f'main_exp {layer} does not use the declared historical recipe')
