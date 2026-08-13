"""index: finalize Result with Asset paths."""

from __future__ import annotations

from rpipe.artifact.asset import ensure_assets
from rpipe.artifact.result import write_result
from rpipe.flow.context import FlowContext


def run(ctx: FlowContext) -> None:
    draft = dict(ctx.state.get('result_draft') or {})
    assets = ensure_assets(ctx.layout)
    asset_files = sorted(p.name for p in assets.iterdir() if p.is_file())
    paths = dict(draft.get('paths') or {})
    paths['result'] = str(ctx.layout.result_path)
    paths['asset_files'] = asset_files
    draft['paths'] = paths
    write_result(ctx.layout.result_path, draft)
    ctx.state['result'] = draft
