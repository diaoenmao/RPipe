"""System Logger: stdout + assets/logs/run.log (flush on every emit)."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from rpipe.structure.artifact.asset import kinds


class Logger:
    def __init__(self, assets_dir: Path | str) -> None:
        self.assets_dir = Path(assets_dir)
        parts = self.assets_dir.parts
        self.run_id = (
            parts[-2]
            if len(parts) >= 3 and parts[-1] == 'assets' and parts[-3] == 'runs'
            else None
        )
        self.path = self.assets_dir / kinds.RUN_LOG
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.is_file():
            self.path.write_text('', encoding='utf-8')

    def _emit(self, line: str) -> None:
        text = line.rstrip('\n')
        if self.run_id:
            text = f'{self.run_id} {text}'
        print(text, flush=True)
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(text + '\n')
            handle.flush()

    def info(self, message: str) -> None:
        self._emit(message)

    def warning(self, message: str) -> None:
        self._emit(f'WARNING {message}')

    def error(self, message: str) -> None:
        self._emit(f'ERROR {message}')

    def exception(self, message: str, exc: BaseException) -> None:
        """ERROR line plus traceback. Same text on stdout and ``run.log``."""
        self.error(f'{message} {type(exc).__name__}: {exc}')
        formatted = traceback.format_exception(type(exc), exc, exc.__traceback__)
        for line in ''.join(formatted).splitlines():
            if line:
                self._emit(line)

    def report(self, tracker: Any, split: str, extra: dict[str, Any] | None = None) -> None:
        extra = extra or {}
        parts: list[str] = []
        if extra.get('epoch') is not None:
            parts.append(f"epoch {extra['epoch']}")
        if extra.get('elapsed') is not None:
            parts.append(f"elapsed={extra['elapsed']}")
        if extra.get('eta') is not None:
            parts.append(f"eta={extra['eta']}")
        parts.append(str(split))
        means = {}
        lasts = {}
        if tracker is not None:
            display = getattr(tracker, 'display_mean', None)
            means = display(split) if callable(display) else tracker.mean(split)
            lasts = tracker.last(split)
        for name in sorted(set(means) | set(lasts)):
            mean = means.get(name)
            last = lasts.get(name)
            if mean is not None:
                parts.append(f'{name} {mean:.4f}')
            elif last is not None:
                parts.append(f'{name} {last:.4f}')
        skip = {'epoch', 'elapsed', 'eta'}
        for key, value in extra.items():
            if key in skip:
                continue
            parts.append(f'{key}={value}')
        self._emit(' '.join(parts))

    def state_dict(self) -> dict[str, Any]:
        return {'path': str(self.path)}

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        del state
