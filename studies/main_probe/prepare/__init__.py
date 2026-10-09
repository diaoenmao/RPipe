"""CPU preparation runs in recipe registration before Data/Model construction."""
def run(ctx):
    workspace = ctx.state['algorithm'].workspace
    if not (workspace / 'PREPARED.json').is_file():
        raise RuntimeError('probe preparation evidence missing')
