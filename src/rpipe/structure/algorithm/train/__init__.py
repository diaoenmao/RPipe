"""Train mode."""

from __future__ import annotations

from typing import Any

from rpipe.structure.algorithm.base import Algorithm
from rpipe.structure.algorithm.config import AlgorithmConfig
from rpipe.structure.algorithm.batch import prepare_tensors
from rpipe.structure.algorithm.eval_hook import (
    _is_better,
    best_spec,
    eval_test_split,
    should_early_stop,
)
from rpipe.structure.algorithm.optim import clip_gradients, make_scheduler
from rpipe.structure.algorithm.progress import (
    UNIT_EPOCH,
    UNIT_STEP,
    ProgressBudget,
    checkpoint_names,
    crossed_percents,
    due_period,
    infer_steps_per_epoch,
    resolve_budget,
)
from rpipe.structure.algorithm.tracker import AlgorithmTracker

due_eval_period = due_period


class TrainAlgorithm(Algorithm):
    def __init__(self, config: AlgorithmConfig) -> None:
        super().__init__(config)
        self._best_test: float | None = None
        self._best_metric = 'Accuracy'
        self._stall = 0
        self.last_improved = False

    def on_eval_period(
        self,
        tracker: AlgorithmTracker,
        logger: Any,
        data: Any,
        model: Any,
        system: Any,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        extra = extra if extra is not None else {}
        self.last_improved = False
        if getattr(model, 'module', None) is None:
            return False
        metrics = eval_test_split(tracker, logger, data, model, system, extra)
        split, metric, mode = best_spec(self.config)
        self._best_metric = metric
        patience = self.config.setting('early_stop_patience')
        if patience is not None:
            patience = int(patience)
        min_delta = float(self.config.setting('early_stop_min_delta', 0.0) or 0.0)
        pool = metrics if split == 'test' else tracker.segment_mean(split)
        score = pool.get(metric)
        previous = self._best_test
        stop, self._best_test, self._stall = should_early_stop(
            value=score,
            best=self._best_test,
            stall=self._stall,
            patience=patience,
            min_delta=min_delta,
            mode=mode,
        )
        self.last_improved = score is not None and _is_better(
            score, previous, min_delta=min_delta, mode=mode
        )
        extra['test_accuracy'] = metrics.get('Accuracy')
        extra['best_metric'] = metric
        extra['best_value'] = self._best_test
        extra['best_accuracy'] = self._best_test if metric == 'Accuracy' else extra.get('best_accuracy')
        extra['improved'] = self.last_improved
        if stop and logger is not None:
            logger.info(
                f"early stop epoch={extra.get('epoch')} step={extra.get('step')} "
                f'best_test_acc={self._best_test} stall={self._stall}'
            )
        return stop

    def on_checkpoint(
        self,
        tracker: AlgorithmTracker,
        logger: Any,
        data: Any,
        model: Any,
        system: Any,
        extra: dict[str, Any] | None = None,
    ) -> None:
        del data
        extra = extra or {}
        names = list(extra.get('checkpoint_names') or [])
        payload = extra.get('payload')
        writer = getattr(system, 'save_checkpoint', None)
        if not names or payload is None or not callable(writer):
            return
        for name in names:
            path = writer(payload, name)
            if logger is not None:
                logger.info(f'checkpoint {name} -> {path}')

    def run(self, data: Any, model: Any, system: Any, tracker: AlgorithmTracker) -> dict[str, Any]:
        if getattr(model, 'module', None) is not None and hasattr(data, 'iter_batches'):
            return _run_supervised(self, data, model, system, tracker)
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


def _build_payload(
    module: Any,
    optimizer: Any,
    scheduler: Any,
    extra: dict[str, Any],
    tracker: AlgorithmTracker | None = None,
    logger: Any | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'epoch': extra.get('epoch'),
        'step': extra.get('step'),
        'model': module.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
    }
    if extra.get('test_accuracy') is not None:
        payload['test_accuracy'] = extra['test_accuracy']
    if extra.get('best_accuracy') is not None:
        payload['best_accuracy'] = extra['best_accuracy']
    if extra.get('best_value') is not None:
        payload['best_value'] = extra['best_value']
        payload['best_metric'] = extra.get('best_metric')
    if tracker is not None:
        payload['tracker'] = tracker.state_dict()
    if logger is not None and hasattr(logger, 'state_dict'):
        payload['logger'] = logger.state_dict()
    return payload


def _run_stub(config: AlgorithmConfig, tracker: AlgorithmTracker) -> dict[str, Any]:
    budget = resolve_budget(config, steps_per_epoch=1)
    tracker.append('train', n=1, values={'Loss': 0.0})
    tracker.save('train')
    tracker.reset('train')
    tracker.flush('train')
    return {
        'mode': 'train',
        'steps': budget.num_steps,
        'epochs': budget.num_epochs or 1,
        'stub': True,
    }


def _run_supervised(
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
    logger = getattr(system, 'logger', None)
    budget = resolve_budget(config, steps_per_epoch=infer_steps_per_epoch(data))
    lr = float(config.setting('lr', 0.1))
    log_interval = config.setting('log_interval')
    log_interval = int(log_interval) if log_interval is not None else None
    eval_period = algo.eval_period()
    ckpt_period = algo.checkpoint_period()
    ckpt_mode = algo.checkpoint_mode()
    percents = algo.checkpoint_percents()
    keep_best = algo.save_best()
    already_percent: set[float] = set()

    resume_extra: dict[str, Any] = {}
    restored = algo.resume(data, model, system, tracker, resume_extra)
    optimizer = algo.make_optimizer(module)
    scheduler = algo.make_scheduler(optimizer, budget.scheduler_t_max())
    if restored:
        opt_state = restored.get('optimizer')
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
        sched_state = restored.get('scheduler')
        if scheduler is not None and sched_state is not None:
            scheduler.load_state_dict(sched_state)
        if restored.get('best_value') is not None:
            algo._best_test = restored['best_value']
        elif restored.get('best_accuracy') is not None:
            algo._best_test = restored['best_accuracy']
    module.train()
    epoch = int(restored.get('epoch') or 0) if restored else 0
    steps = int(restored.get('step') or 0) if restored else 0
    stop = False

    def fire_eval(extra: dict[str, Any]) -> bool:
        return bool(algo.on_eval_period(tracker, logger, data, model, system, extra))

    def fire_ckpt(*, extra: dict[str, Any], improved: bool, is_last: bool) -> None:
        nonlocal already_percent
        current = budget.progress(epoch=epoch, step=steps)
        names = checkpoint_names(
            mode=ckpt_mode,
            save_best=keep_best,
            period=ckpt_period,
            percents=percents,
            current=current,
            total=budget.total,
            unit=budget.unit,
            improved=improved,
            is_last=is_last,
            already_percent=already_percent,
        )
        if not names:
            return
        extra = dict(extra)
        extra['best_value'] = algo._best_test
        extra['best_metric'] = algo._best_metric
        if algo._best_metric == 'Accuracy':
            extra['best_accuracy'] = algo._best_test
        extra['checkpoint_names'] = names
        extra['payload'] = _build_payload(module, optimizer, scheduler, extra, tracker, logger)
        algo.on_checkpoint(tracker, logger, data, model, system, extra)
        if ckpt_mode == 'percent':
            for hit in crossed_percents(current, budget.total, percents, already_percent):
                already_percent.add(hit)

    already_done = steps >= budget.num_steps
    try:
        if already_done:
            extra = _hook_extra(epoch=epoch, lr=_current_lr(optimizer, lr), step=steps)
            extra['best_accuracy'] = algo._best_test
            fire_eval(extra)
        else:
            while True:
                if stop:
                    break
                epoch += 1
                epoch_lr = _current_lr(optimizer, lr)
                batches_this_epoch = 0
                accum = 0
                step_period = max(int(config.setting('step_period', 1) or 1), 1)
                optimizer.zero_grad()
                for batch in data.iter_batches('train'):
                    batches_this_epoch += 1
                    images, targets = prepare_tensors(batch, module, device)
                    logits = module(images)
                    loss = F.cross_entropy(logits, targets) / step_period
                    loss.backward()
                    accum += 1
                    values = tracker.evaluate('train', 'batch', (images, targets), logits)
                    tracker.append('train', n=int(images.size(0)), values=values)
                    if accum % step_period != 0:
                        continue
                    clip_gradients(module, config)
                    optimizer.step()
                    optimizer.zero_grad()
                    steps += 1
                    if log_interval and steps % log_interval == 0:
                        logger.report(
                            tracker, 'train', extra={'epoch': epoch, 'lr': epoch_lr, 'step': steps}
                        )
                        tracker.flush('train')
                    if budget.unit == UNIT_STEP:
                        extra = _hook_extra(epoch=epoch, lr=epoch_lr, step=steps)
                        improved = False
                        if due_period(eval_period, steps):
                            if fire_eval(extra):
                                stop = True
                            improved = algo.last_improved
                        fire_ckpt(extra=extra, improved=improved, is_last=False)
                        if scheduler is not None:
                            scheduler.step()
                        if stop:
                            break
                    if steps >= budget.num_steps:
                        stop = True
                        break
                optimizer.zero_grad()
                if batches_this_epoch == 0:
                    break
                if logger is not None:
                    logger.report(
                        tracker, 'train', extra={'epoch': epoch, 'lr': epoch_lr, 'step': steps}
                    )
                tracker.flush('train')
                tracker.save('train')
                tracker.reset('train')
                tracker.flush_state()
                extra = _hook_extra(epoch=epoch, lr=_current_lr(optimizer, lr), step=steps)
                if budget.unit == UNIT_EPOCH:
                    improved = False
                    if due_period(eval_period, epoch):
                        if fire_eval(extra):
                            stop = True
                        improved = algo.last_improved
                    last_epoch = budget.num_epochs is not None and epoch >= budget.num_epochs
                    last_steps = steps >= budget.num_steps
                    fire_ckpt(extra=extra, improved=improved, is_last=last_epoch or last_steps)
                    if scheduler is not None:
                        scheduler.step()
                if steps >= budget.num_steps:
                    stop = True
        extra = _hook_extra(epoch=epoch, lr=_current_lr(optimizer, lr), step=steps)
        extra['best_accuracy'] = algo._best_test
        if eval_period <= 0 and not already_done:
            fire_eval(extra)
            if keep_best and algo.last_improved:
                fire_ckpt(extra=extra, improved=True, is_last=True)
        if not already_done:
            fire_ckpt(extra=extra, improved=False, is_last=True)
    finally:
        tracker.flush_state()

    train_loss = tracker.segment_mean('train').get('Loss', 0.0)
    accuracy = tracker.segment_mean('test').get('Accuracy', 0.0)
    summary: dict[str, Any] = {
        'mode': 'train',
        'steps': steps,
        'epochs': epoch,
        'progress_unit': budget.unit,
        'train_size': data.meta.get('train_size') if getattr(data, 'meta', None) else None,
        'train_loss': train_loss,
        'accuracy': accuracy,
        'best_value': algo._best_test,
        'best_metric': algo._best_metric,
    }
    if algo._best_metric == 'Accuracy':
        summary['best_accuracy'] = algo._best_test
    return summary


def budget_for(config: AlgorithmConfig) -> ProgressBudget:
    return resolve_budget(config)
