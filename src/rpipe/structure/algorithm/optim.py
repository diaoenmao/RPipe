"""Native optimizer / scheduler (algorithm-layer interface)."""

from __future__ import annotations

from typing import Any

from rpipe.structure.algorithm.config import AlgorithmConfig


def make_optimizer(module: Any, config: AlgorithmConfig) -> Any:
    """Build a torch optimizer from ``algorithm.optimizer`` extras."""
    import torch

    name = str(config.setting('optimizer', 'SGD') or 'SGD')
    key = name.lower().replace('-', '_')
    lr = float(config.setting('lr', 0.1))
    params = module.parameters()
    if key == 'sgd':
        kwargs: dict[str, Any] = {'lr': lr}
        momentum = config.setting('momentum')
        if momentum is not None:
            kwargs['momentum'] = float(momentum)
        weight_decay = config.setting('weight_decay')
        if weight_decay is not None:
            kwargs['weight_decay'] = float(weight_decay)
        nesterov = config.setting('nesterov')
        if nesterov is not None:
            kwargs['nesterov'] = bool(nesterov)
        return torch.optim.SGD(params, **kwargs)
    if key == 'adam':
        kwargs = {'lr': lr}
        betas = config.setting('betas')
        if betas is not None:
            kwargs['betas'] = tuple(betas)
        weight_decay = config.setting('weight_decay')
        if weight_decay is not None:
            kwargs['weight_decay'] = float(weight_decay)
        return torch.optim.Adam(params, **kwargs)
    if key == 'adamw':
        kwargs = {'lr': lr}
        betas = config.setting('betas')
        if betas is not None:
            kwargs['betas'] = tuple(betas)
        weight_decay = config.setting('weight_decay')
        if weight_decay is not None:
            kwargs['weight_decay'] = float(weight_decay)
        return torch.optim.AdamW(params, **kwargs)
    raise ValueError(f'unknown optimizer: {name}')


def make_scheduler(optimizer: Any, config: AlgorithmConfig, t_max: int) -> Any:
    """Build an LR scheduler. ``None`` / ``constant`` = fixed lr."""
    name = config.setting('scheduler')
    if name is None or str(name).lower() in ('', 'none', 'constant'):
        return None
    key = str(name).lower().replace('-', '_')
    import torch

    horizon = int(config.setting('T_max', t_max) or t_max)
    horizon = max(horizon, 1)
    if key in ('cosine', 'cosine_annealing', 'cosineannealinglr'):
        eta_min = float(config.setting('eta_min', 0.0) or 0.0)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=horizon, eta_min=eta_min
        )
    if key in ('linear', 'linear_warmup', 'linearannealinglr'):
        warmup = config.setting('warmup_steps')
        if warmup is None:
            ratio = float(config.setting('warmup_ratio', 0.0) or 0.0)
            warmup = int(horizon * ratio)
        warmup = max(int(warmup or 0), 0)

        def lr_lambda(step: int) -> float:
            if warmup > 0 and step < warmup:
                return float(step + 1) / float(max(warmup, 1))
            denom = max(horizon - warmup, 1)
            return max(0.0, float(horizon - step) / float(denom))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    raise ValueError(f'unknown scheduler: {name}')
