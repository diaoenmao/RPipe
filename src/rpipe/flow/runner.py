"""Sequential Flow runner."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from rpipe.structure.artifact.result import STATUS_FAILED, STATUS_SUCCEEDED, write_result
from rpipe.flow.context import FlowContext

PHASES = ('prepare', 'execute', 'collect', 'summarize', 'write', 'process')

# Old callers may still pass ``index`` / ``persist``; map to ``write``.
_PHASE_ALIASES = {'index': 'write', 'persist': 'write'}


class FlowRunner:
    def __init__(self, phases: tuple[str, ...] | list[str] | None = None):
        raw = tuple(phases) if phases else PHASES
        self.phases = tuple(_PHASE_ALIASES.get(p, p) for p in raw)
        unknown = [p for p in self.phases if p not in PHASES]
        if unknown:
            raise ValueError(f'unknown phases: {unknown}; allowed: {PHASES}')

    def run(self, ctx: FlowContext) -> Path:
        logger = self._ensure_logger(ctx)
        logger.info(f'flow start phases={",".join(self.phases)}')
        current = None
        try:
            for name in self.phases:
                current = name
                module = import_module(f'rpipe.flow.{name}')
                module.run(ctx)
        except Exception as exc:
            self._log_failure(ctx, current, exc)
            self._write_failed_result(ctx, exc)
            raise
        logger.info('flow succeeded')
        return ctx.layout.result_path

    def _ensure_logger(self, ctx: FlowContext):
        existing = ctx.state.get('logger')
        if existing is not None:
            return existing
        from rpipe.structure.system.logger import Logger

        logger = Logger(ctx.layout.assets_dir)
        ctx.state['logger'] = logger
        return logger

    def _log_failure(self, ctx: FlowContext, phase: str | None, exc: BaseException) -> None:
        if ctx.state.get('failure_logged'):
            return
        logger = self._ensure_logger(ctx)
        where = phase or 'flow'
        try:
            logger.exception(f'phase={where} status=failed', exc)
        except Exception:
            try:
                logger.error(f'phase={where} status=failed {type(exc).__name__}: {exc}')
            except Exception:
                return
        ctx.state['failure_logged'] = True

    def _write_failed_result(self, ctx: FlowContext, exc: BaseException) -> None:
        """Best-effort failed Result; must not hide the original error.

        process failure must not overwrite an already-written succeeded result.
        """
        try:
            from rpipe.structure.artifact.result import load_result

            if ctx.layout.result_path.is_file():
                existing = load_result(ctx.layout.result_path)
                if existing.get('status') == STATUS_SUCCEEDED:
                    return
        except Exception:
            pass
        try:
            draft = dict(ctx.state.get('result_draft') or {})
            draft['status'] = STATUS_FAILED
            draft['error'] = f'{type(exc).__name__}: {exc}'
            paths = dict(draft.get('paths') or {})
            paths.setdefault('artifact', str(ctx.layout.root))
            paths.setdefault('config', str(ctx.layout.config_path))
            paths.setdefault('assets', str(ctx.layout.assets_dir))
            paths.setdefault('shared', str(ctx.layout.shared_dir))
            paths['result'] = str(ctx.layout.result_path)
            draft['paths'] = paths
            if ctx.control is not None and 'control' not in draft:
                draft['control'] = ctx.control.to_dict()
            if 'metrics' not in draft:
                collected = ctx.state.get('collected') or {}
                draft['metrics'] = collected.get('metrics') or {}
            draft['study'] = str(ctx.study_dir)
            write_result(ctx.layout.result_path, draft)
            ctx.state['result'] = draft
        except Exception:
            return
