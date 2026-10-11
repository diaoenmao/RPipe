"""Apply the declared curve gates once after Study aggregation."""

def run(ctx):
    if ctx.scope != 'study':
        return
    from .curves import main
    args = ['--study', str(ctx.study_dir), '--reference', str(ctx.study_dir / 'docs' / 'REFERENCE_CURVES.json')]
    if not ctx.state['process'].get('complete'):
        args.append('--partial')
    code = main(args)
    if code:
        raise RuntimeError(f'main_exp curve validation failed (exit {code}); see docs/COMPARISON.json')
