"""Sequential Flow runner."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from rpipe.flow.context import FlowContext

PHASES = ('prepare', 'execute', 'collect', 'summarize', 'index')


class FlowRunner:
    def __init__(self, phases: tuple[str, ...] | list[str] | None = None):
        self.phases = tuple(phases) if phases else PHASES
        unknown = [p for p in self.phases if p not in PHASES]
        if unknown:
            raise ValueError(f'unknown phases: {unknown}; allowed: {PHASES}')

    def run(self, ctx: FlowContext) -> Path:
        for name in self.phases:
            module = import_module(f'rpipe.flow.{name}')
            module.run(ctx)
        return ctx.layout.result_path
