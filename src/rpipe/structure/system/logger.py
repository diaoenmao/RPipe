"""System Logger: stdout + assets/logs/run.log (flush on every emit)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rpipe.structure.artifact.asset import kinds


class Logger:
    def __init__(self, assets_dir: Path | str) -> None:
        self.assets_dir = Path(assets_dir)
        self.path = self.assets_dir / kinds.RUN_LOG
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.is_file():
            self.path.write_text('', encoding='utf-8')

    def _emit(self, line: str) -> None:
        text = line.rstrip('\n')
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

    def report(self, tracker: Any, split: str, extra: dict[str, Any] | None = None) -> None:
        extra = extra or {}
        parts: list[str] = []
        if extra.get('epoch') is not None:
            parts.append(f"epoch {extra['epoch']}")
        parts.append(str(split))
        means = {}
        lasts = {}
        if tracker is not None:
            means = tracker.mean(split)
            lasts = tracker.last(split)
        for name in sorted(set(means) | set(lasts)):
            mean = means.get(name)
            last = lasts.get(name)
            if mean is not None:
                parts.append(f'{name} {mean:.4f}')
            elif last is not None:
                parts.append(f'{name} {last:.4f}')
        for key, value in extra.items():
            if key == 'epoch':
                continue
            parts.append(f'{key}={value}')
        self._emit(' '.join(parts))

    def state_dict(self) -> dict[str, Any]:
        return {'path': str(self.path)}

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        del state
