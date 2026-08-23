"""process: post-persist derived work (baseline Δ, report stubs).

Empty by design for now — hook reserved so Study runner / autoresearch can
fill compare tables without changing Flow phase order.
"""

from __future__ import annotations

from rpipe.flow.context import FlowContext


def run(ctx: FlowContext) -> None:
    """No-op. Result is already persisted; do not rewrite Config."""
    _ = ctx
    return
