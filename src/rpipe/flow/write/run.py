"""write: serialize Result onto disk."""

from __future__ import annotations

from rpipe.flow.context import FlowContext
from rpipe.structure.artifact.asset import ensure_assets, list_asset_files
from rpipe.structure.artifact.result import write_result


def run(ctx: FlowContext) -> None:
    draft = dict(ctx.state.get('result_draft') or {})
    if not draft:
        raise RuntimeError('summarize must run before write')
    ensure_assets(ctx.layout)
    paths = dict(draft.get('paths') or {})
    paths['result'] = str(ctx.layout.result_path)
    paths['assets'] = str(ctx.layout.assets_dir)
    paths['shared'] = str(ctx.layout.shared_dir)
    paths['asset_files'] = list_asset_files(ctx.layout.assets_dir)
    draft['paths'] = paths
    write_result(ctx.layout.result_path, draft)
    ctx.state['result'] = draft
