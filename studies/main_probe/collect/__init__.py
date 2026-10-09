def run(ctx):
    ctx.state['collected']['metrics']['probe_passed'] = ctx.state['probe']['selected_passed']
