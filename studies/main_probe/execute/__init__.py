"""Flow executes the registered paired Algorithm before this postcondition."""
def run(ctx):
    report = ctx.state['algorithm'].report
    if not report or len(report.get('runs', [])) != 1:
        raise RuntimeError('paired Algorithm did not capture one pair')
    ctx.state['probe_execution'] = report
