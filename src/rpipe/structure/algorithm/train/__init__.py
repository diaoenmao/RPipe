"""Train mode."""

from __future__ import annotations

from typing import Any

from rpipe.structure.algorithm.base import Algorithm
from rpipe.structure.algorithm.config import AlgorithmConfig
from rpipe.structure.algorithm.eval_hook import eval_test_split, should_early_stop
from rpipe.structure.algorithm.tracker import AlgorithmTracker


class TrainAlgorithm(Algorithm):
    def __init__(self, config: AlgorithmConfig) -> None:
        super().__init__(config)
        self._best_test: float | None = None
        self._stall = 0

    def on_eval_period(
        self,
        tracker: AlgorithmTracker,
        logger: Any,
        data: Any,
        model: Any,
        system: Any,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        extra = extra or {}
        if getattr(model, 'module', None) is None:
            return False
        metrics = eval_test_split(tracker, logger, data, model, system, extra)
        patience = self.config.setting('early_stop_patience')
        if patience is not None:
            patience = int(patience)
        min_delta = float(self.config.setting('early_stop_min_delta', 0.0) or 0.0)
        stop, self._best_test, self._stall = should_early_stop(
            accuracy=metrics.get('Accuracy'),
            best=self._best_test,
            stall=self._stall,
            patience=patience,
            min_delta=min_delta,
        )
        if stop and logger is not None:
            logger.info(
                f"early stop epoch={extra.get('epoch')} "
                f'best_test_acc={self._best_test} stall={self._stall}'
            )
        return stop

    def run(self, data: Any, model: Any, system: Any, tracker: AlgorithmTracker) -> dict[str, Any]:
        if getattr(data, 'name', None) == 'MNIST' and getattr(model, 'module', None) is not None:
            return _run_mnist_linear(self, data, model, system, tracker)
        return _run_stub(self.config, tracker)


def run(control_algorithm: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Legacy dict entry; prefer TrainAlgorithm.run."""
    algo = TrainAlgorithm(AlgorithmConfig.from_mapping(control_algorithm))
    return algo.run(
        state.get('data'),
        state.get('model'),
        state.get('system'),
        state['tracker'],
    )


def due_eval_period(period: int, epoch: int) -> bool:
    """Call ``on_eval_period`` this epoch. ``period <= 0`` means only after the loop."""
    return period > 0 and epoch % period == 0


def make_scheduler(optimizer: Any, config: AlgorithmConfig, epochs: int) -> Any:
    """Build an LR scheduler from ``algorithm.scheduler``. ``None`` / ``constant`` = fixed lr."""
    name = config.setting('scheduler')
    if name is None or str(name).lower() in ('', 'none', 'constant'):
        return None
    key = str(name).lower().replace('-', '_')
    if key in ('cosine', 'cosine_annealing', 'cosineannealinglr'):
        import torch

        eta_min = float(config.setting('eta_min', 0.0) or 0.0)
        t_max = int(config.setting('T_max', epochs) or epochs)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(t_max, 1), eta_min=eta_min
        )
    raise ValueError(f'unknown scheduler: {name}')


def _current_lr(optimizer: Any, fallback: float) -> float:
    groups = getattr(optimizer, 'param_groups', None)
    if not groups:
        return fallback
    return float(groups[0].get('lr', fallback))


def _hook_extra(*, epoch: int, lr: float, step: int | None = None) -> dict[str, Any]:
    extra: dict[str, Any] = {'epoch': epoch, 'lr': lr}
    if step is not None:
        extra['step'] = step
    return extra


def _run_stub(config: AlgorithmConfig, tracker: AlgorithmTracker) -> dict[str, Any]:
    steps = int(config.setting('num_steps', 1))
    tracker.append('train', n=1, values={'Loss': 0.0})
    tracker.save('train')
    tracker.reset('train')
    tracker.flush('train')
    return {'mode': 'train', 'steps': steps, 'stub': True}


def _run_mnist_linear(
    algo: TrainAlgorithm,
    data: Any,
    model: Any,
    system: Any,
    tracker: AlgorithmTracker,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    config = algo.config
    device = torch.device(getattr(system, 'device', 'cpu'))
    module = system.place_module(model.module)
    model.module = module
    logger = system.logger
    epochs = int(config.setting('num_epochs', config.setting('num_steps', 1)))
    lr = float(config.setting('lr', 0.1))
    log_interval = config.setting('log_interval')
    log_interval = int(log_interval) if log_interval is not None else None
    eval_period = algo.eval_period()

    optimizer = torch.optim.SGD(module.parameters(), lr=lr)
    scheduler = make_scheduler(optimizer, config, epochs)
    module.train()
    steps = 0
    try:
        for epoch in range(1, epochs + 1):
            epoch_lr = _current_lr(optimizer, lr)
            for images, targets in data.iter_batches('train'):
                images = images.view(images.size(0), -1).to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                logits = module(images)
                loss = F.cross_entropy(logits, targets)
                loss.backward()
                optimizer.step()
                values = tracker.evaluate('train', 'batch', (images, targets), logits)
                tracker.append('train', n=int(images.size(0)), values=values)
                steps += 1
                if log_interval and steps % log_interval == 0:
                    logger.report(
                        tracker, 'train', extra={'epoch': epoch, 'lr': epoch_lr, 'step': steps}
                    )
                    tracker.flush('train')
            logger.report(tracker, 'train', extra={'epoch': epoch, 'lr': epoch_lr})
            tracker.flush('train')
            tracker.save('train')
            tracker.reset('train')
            tracker.flush_state()
            extra = _hook_extra(epoch=epoch, lr=epoch_lr)
            if due_eval_period(eval_period, epoch):
                if algo.on_eval_period(tracker, logger, data, model, system, extra):
                    break
            if scheduler is not None:
                scheduler.step()
        if eval_period <= 0:
            algo.on_eval_period(
                tracker,
                logger,
                data,
                model,
                system,
                _hook_extra(epoch=epochs, lr=_current_lr(optimizer, lr)),
            )
    finally:
        tracker.flush_state()

    train_loss = tracker.segment_mean('train').get('Loss', 0.0)
    accuracy = tracker.segment_mean('test').get('Accuracy', 0.0)
    return {
        'mode': 'train',
        'steps': steps,
        'epochs': epochs,
        'train_size': data.meta.get('train_size') if getattr(data, 'meta', None) else None,
        'train_loss': train_loss,
        'accuracy': accuracy,
    }
