"""Flow executes the registered paired Algorithm before this postcondition."""
def run(ctx):
    report = ctx.state['algorithm'].report
    if not report or report.get('selected_passed') is not True:
        raise RuntimeError('paired Algorithm did not produce a passing report')
    ctx.state['probe'] = report
