"""Runtime System object, factory, and Logger."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rpipe.structure.system.config import SystemConfig
from rpipe.structure.system.logger import Logger


class System:
    def __init__(
        self,
        *,
        device: str,
        logger: Logger,
        assets_dir: Path,
        meta: dict[str, Any],
    ) -> None:
        self.device = device
        self.logger = logger
        self.assets_dir = assets_dir
        self.meta = meta

    def place_module(self, module: Any) -> Any:
        if module is None:
            return None
        import torch

        return module.to(torch.device(self.device))

    def checkpoint_dir(self) -> Path:
        from rpipe.structure.artifact.asset import kinds

        directory = Path(self.assets_dir) / kinds.CHECKPOINTS
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def save_checkpoint(self, payload: dict[str, Any], name: str) -> Path:
        """Publish an atomic bundle after updating its diagnostic piece mirror."""
        import json
        import shutil
        import torch

        from rpipe.structure.artifact._atomic import atomic_replace, atomic_write_text

        stem = str(name).replace('/', '_').replace('\\', '_')
        if not stem or stem in ('.', '..') or Path(stem).name != stem:
            raise ValueError(f'invalid checkpoint name: {name!r}')
        root = self.checkpoint_dir()
        bundle = root / f'{stem}.pt'
        pending_bundle = root / f'.{stem}.pt.writing'
        folder = root / stem
        incomplete = root / f'.{stem}.incomplete'
        if not bundle.is_file() and folder.is_dir() and not incomplete.exists():
            previous = self.load_checkpoint(str(folder))
            if previous is not None:
                torch.save(previous, pending_bundle)
                atomic_replace(pending_bundle, bundle)
        torch.save(payload, pending_bundle)
        tmp = root / f'.{stem}.writing'
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        torch_keys = ('model', 'optimizer', 'scheduler')
        for key in torch_keys:
            value = payload.get(key)
            if value is not None:
                torch.save(value, tmp / f'{key}.pt')
        if payload.get('tracker') is not None:
            atomic_write_text(tmp / 'tracker.json', json.dumps(payload['tracker'], indent=2, ensure_ascii=False))
        if payload.get('logger') is not None:
            atomic_write_text(tmp / 'logger.json', json.dumps(payload['logger'], indent=2, ensure_ascii=False))
        meta = {
            key: value
            for key, value in payload.items()
            if key not in torch_keys and key not in ('tracker', 'logger')
        }
        try:
            atomic_write_text(tmp / 'meta.json', json.dumps(meta, indent=2, ensure_ascii=False))
        except TypeError:
            torch.save(meta, tmp / 'meta.pt')
        # ponytail: one writer per Run/stem; concurrent writers need separate staging and locking.
        incomplete.touch()
        folder.mkdir(exist_ok=True)
        for part in ('model.pt', 'optimizer.pt', 'scheduler.pt', 'tracker.json', 'logger.json', 'meta.json', 'meta.pt', 'payload.pt'):
            source, target = tmp / part, folder / part
            if source.is_file():
                atomic_replace(source, target)
            else:
                target.unlink(missing_ok=True)
        atomic_replace(pending_bundle, bundle)
        try:
            incomplete.unlink()
            tmp.rmdir()
        except OSError:
            pass  # The bundle is committed; leftover diagnostics cannot turn this into a failed save.
        return bundle

    def load_checkpoint(self, name: str) -> dict[str, Any] | None:
        """Read a ``.pt`` bundle, a checkpoint directory, or a path."""
        import json
        import torch

        raw = str(name)
        candidate = Path(raw)
        path = candidate
        if not (candidate.suffix.lower() == '.pt' or candidate.is_file() or candidate.is_dir() or '/' in raw or '\\' in raw):
            stem = raw.replace('/', '_').replace('\\', '_')
            bundle = self.checkpoint_dir() / f'{stem}.pt'
            folder = self.checkpoint_dir() / stem
            if bundle.is_file():
                path = bundle
            elif folder.is_dir():
                path = folder
            else:
                path = bundle
        if not path.is_file():
            folder = path if path.is_dir() or path.suffix.lower() != '.pt' else path.with_suffix('')
            bundle = folder.with_name(f'{folder.name}.pt')
            if bundle.is_file():
                path = bundle
            elif (folder.parent / f'.{folder.name}.incomplete').exists():
                raise OSError(f'incomplete checkpoint without a committed bundle: {folder}')
        if path.is_dir():
            payload_path = path / 'payload.pt'
            if payload_path.is_file():
                loaded = torch.load(payload_path, map_location=self.device, weights_only=False)
                return loaded if isinstance(loaded, dict) else {'model': loaded}
            assembled: dict[str, Any] = {}
            meta_json = path / 'meta.json'
            if meta_json.is_file():
                assembled.update(json.loads(meta_json.read_text(encoding='utf-8')))
            meta_pt = path / 'meta.pt'
            if meta_pt.is_file():
                extra = torch.load(meta_pt, map_location=self.device, weights_only=False)
                if isinstance(extra, dict):
                    assembled.update(extra)
            for key in ('model', 'optimizer', 'scheduler'):
                piece = path / f'{key}.pt'
                if piece.is_file():
                    assembled[key] = torch.load(piece, map_location=self.device, weights_only=False)
            tracker_json = path / 'tracker.json'
            if tracker_json.is_file():
                assembled['tracker'] = json.loads(tracker_json.read_text(encoding='utf-8'))
            logger_json = path / 'logger.json'
            if logger_json.is_file():
                assembled['logger'] = json.loads(logger_json.read_text(encoding='utf-8'))
            return assembled or None
        if not path.is_file():
            return None
        loaded = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(loaded, dict):
            return loaded
        return {'model': loaded}

    def to_result_snapshot(self) -> dict[str, Any]:
        return {'device': self.device, **self.meta}


class SystemFactory:
    @staticmethod
    def build(system_config: SystemConfig, assets_dir: Path | str) -> System:
        return _build_native(system_config, Path(assets_dir))


def _build_native(system_config: SystemConfig, assets_dir: Path) -> System:
    device = str(system_config.setting('device', 'cpu'))
    logger = Logger(assets_dir)
    deterministic = bool(system_config.setting('deterministic', False))
    return System(
        device=device,
        logger=logger,
        assets_dir=assets_dir,
        meta={
            'ready': True,
            'source': system_config.source or 'native',
            'deterministic': deterministic,
            'cudnn_benchmark': system_config.setting('cudnn_benchmark'),
            'cudnn_deterministic': system_config.setting('cudnn_deterministic'),
        },
    )
