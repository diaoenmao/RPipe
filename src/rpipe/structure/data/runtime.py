"""Runtime data consumption and loader state, independent of dataset construction."""

from __future__ import annotations

from typing import Any, Iterator


class Data:
    def __init__(
        self,
        *,
        name: str,
        source: str,
        loaders: dict[str, Any],
        meta: dict[str, Any],
    ) -> None:
        self.name = name
        self.source = source
        self._loaders = loaders
        self.meta = meta
        self._train_set = None
        self._train_start = None
        self._resume_data = None

    def iter_batches(self, split: str) -> Iterator[Any]:
        loader = self._loaders.get(split)
        if loader is None:
            return iter(())
        if split == 'train':
            from rpipe.structure.data.resume import (
                BatchCursor, cursor_loader, generator_states, loader_spec, restore_generators,
            )
            if loader_spec(loader) is not None:
                import torch

                if not isinstance(loader.batch_sampler, BatchCursor):
                    loader = cursor_loader(loader)
                    self._loaders[split] = loader
                cursor = loader.batch_sampler
                pending = self._resume_data
                self._resume_data = None
                inside = pending is not None and not pending['exhausted']
                if pending is not None:
                    restore_generators(loader, pending['after'] if pending['exhausted'] else pending['start'])
                cursor.consumed = pending['consumed'] if inside else 0
                cursor.exhausted = False
                self._train_start = generator_states(loader)
                # A replacement iterator must not consume another global base-seed draw.
                rng = torch.get_rng_state() if inside else None
                iterator = iter(loader)
                if rng is not None:
                    torch.set_rng_state(rng)
                return iterator
        return iter(loader)

    def training_state(self) -> dict[str, Any] | None:
        from rpipe.structure.data.resume import BatchCursor, generator_states, loader_spec

        loader = self._loaders.get('train')
        spec = loader_spec(loader)
        if spec is None or not isinstance(loader.batch_sampler, BatchCursor) or self._train_start is None:
            return None
        cursor = loader.batch_sampler
        return {'version': 1, 'spec': spec, 'seed': self.meta.get('seed'),
                'identity': {'name': self.name, 'source': self.source, 'config': self.meta.get('config')},
                'consumed': cursor.consumed, 'exhausted': cursor.exhausted,
                'start': self._train_start, 'after': generator_states(loader)}

    def restore_training_state(self, state: dict[str, Any]) -> bool:
        from rpipe.structure.data.resume import loader_spec

        loader = self._loaders.get('train')
        if state.get('version') != 1 or state.get('spec') != loader_spec(loader) or state.get('seed') != self.meta.get('seed'):
            raise ValueError('checkpoint data sampler/batch/seed does not match current data')
        if state.get('identity') != {'name': self.name, 'source': self.source, 'config': self.meta.get('config')}:
            raise ValueError('checkpoint data identity/config does not match current data')
        consumed = state.get('consumed')
        if type(consumed) is not int or not 0 <= consumed <= len(loader.batch_sampler) or type(state.get('exhausted')) is not bool:
            raise ValueError('invalid checkpoint data cursor')
        self._resume_data = state
        return not state['exhausted']

    def steps_per_epoch(self, split: str = 'train') -> int | None:
        loader = self._loaders.get(split)
        if loader is not None:
            try:
                value = int(len(loader))
                if value > 0:
                    return value
            except (TypeError, ValueError):
                pass
        if split != 'train':
            return None
        train_size = self.meta.get('train_size')
        batch_size = self.meta.get('batch_size')
        if train_size and batch_size:
            import math

            return max(int(math.ceil(int(train_size) / int(batch_size))), 1)
        return None

    def rebind_train_steps(
        self,
        *,
        step: int,
        num_steps: int,
        step_period: int = 1,
    ) -> None:
        """Rebuild train loader for remaining optimizer steps (main ``make_data_loader``).

        ``num_samples = batch_size * (num_steps - step) * step_period``. Generator is
        re-seeded from ``meta['seed']`` each time, so a mid-run resume sees the
        **prefix** of the same shuffle, not the unseen suffix (same as main).
        Native exact resume instead binds the original full budget at step 0,
        then applies ``restore_training_state`` to select the unseen suffix.
        """
        dataset = self._train_set
        if dataset is None:
            return
        batch_size = int(self.meta.get('batch_size') or getattr(self._loaders.get('train'), 'batch_size', 1) or 1)
        period = max(int(step_period), 1)
        remaining = max(int(num_steps) - int(step), 0)
        num_samples = batch_size * remaining * period
        opts = _loader_options(self.meta.get('config') if isinstance(self.meta.get('config'), dict) else {})
        self._loaders['train'] = _train_loader_for_samples(
            dataset,
            batch_size=batch_size,
            num_samples=num_samples,
            seed=self.meta.get('seed'),
            pin_memory=opts['pin_memory'],
            num_workers=opts['num_workers'],
        )

    def to_result_snapshot(self) -> dict[str, Any]:
        out = {'name': self.name, 'source': self.source, **self.meta}
        out.pop('train_loader', None)
        out.pop('test_loader', None)
        return out


def _loader_options(cfg: dict[str, Any]) -> dict[str, Any]:
    pin = cfg.get('pin_memory')
    workers = cfg.get('num_workers', 0)
    return {
        'pin_memory': bool(pin) if pin is not None else False,
        'num_workers': int(workers or 0),
    }


def _empty_train_loader(dataset: Any, batch_size: int) -> Any:
    class _Empty:
        def __len__(self) -> int:
            return 0

        def __iter__(self):
            return iter(())

    loader = _Empty()
    loader.dataset = dataset
    loader.batch_size = batch_size
    return loader


def _train_loader_for_samples(
    dataset: Any,
    *,
    batch_size: int,
    num_samples: int,
    seed: int | None,
    pin_memory: bool = False,
    num_workers: int = 0,
) -> Any:
    from torch.utils.data import DataLoader, RandomSampler

    from rpipe.structure.api.system_api import make_generator, worker_init_fn

    if num_samples <= 0:
        return _empty_train_loader(dataset, batch_size)
    generator = make_generator(seed)
    sampler = RandomSampler(
        dataset,
        replacement=False,
        num_samples=int(num_samples),
        generator=generator,
    )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        sampler=sampler,
        pin_memory=bool(pin_memory),
        num_workers=int(num_workers),
        worker_init_fn=worker_init_fn if seed is not None else None,
    )
