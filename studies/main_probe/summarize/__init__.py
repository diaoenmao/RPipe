def run(ctx):
    ctx.state['result_draft']['paths']['probe'] = str(ctx.state['algorithm'].workspace)
    ctx.state['result_draft']['probe'] = ctx.state['probe']
