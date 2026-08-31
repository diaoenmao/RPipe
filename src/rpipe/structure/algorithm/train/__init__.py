"""Train mode."""

from __future__ import annotations

from typing import Any

from rpipe.structure.algorithm.config import AlgorithmConfig
from rpipe.structure.algorithm.base import Algorithm
from rpipe.structure.algorithm.tracker import AlgorithmTracker


class TrainAlgorithm(Algorithm):
    def run(self, data: Any, model: Any, system: Any, tracker: AlgorithmTracker) -> dict[str, Any]:
        if getattr(data, 'name', None) == 'MNIST' and getattr(model, 'module', None) is not None:
            return _run_mnist_linear(self.config, data, model, system, tracker)
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


def _run_stub(config: AlgorithmConfig, tracker: AlgorithmTracker) -> dict[str, Any]:
    steps = int(config.setting('num_steps', 1))
    tracker.append('train', n=1, values={'Loss': 0.0})
    tracker.save('train')
    tracker.reset('train')
    tracker.flush('train')
    return {'mode': 'train', 'steps': steps, 'stub': True}


def _run_mnist_linear(
    config: AlgorithmConfig,
    data: Any,
    model: Any,
    system: Any,
    tracker: AlgorithmTracker,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    device = torch.device(getattr(system, 'device', 'cpu'))
    module = system.place_module(model.module)
    model.module = module
    logger = system.logger
    epochs = int(config.setting('num_epochs', config.setting('num_steps', 1)))
    lr = float(config.setting('lr', 0.1))
    log_interval = config.setting('log_interval')
    log_interval = int(log_interval) if log_interval is not None else None

    optimizer = torch.optim.SGD(module.parameters(), lr=lr)
    module.train()
    steps = 0
    try:
        for epoch in range(1, epochs + 1):
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
                    logger.report(tracker, 'train', extra={'epoch': epoch, 'lr': lr, 'step': steps})
                    tracker.flush('train')
            logger.report(tracker, 'train', extra={'epoch': epoch, 'lr': lr})
            tracker.flush('train')
            tracker.save('train')
            tracker.reset('train')
            tracker.flush_state()
            self.on_eval_period(tracker, logger, data, model, system)

        module.eval()
        with torch.no_grad():
            for images, targets in data.iter_batches('test'):
                images = images.view(images.size(0), -1).to(device)
                targets = targets.to(device)
                logits = module(images)
                values = tracker.evaluate('test', 'batch', (images, targets), logits)
                tracker.append('test', n=int(images.size(0)), values=values)
        logger.report(tracker, 'test', extra={'lr': lr})
        tracker.flush('test')
        tracker.save('test')
        tracker.flush_state()
    finally:
        tracker.flush_state()

    train_loss = tracker.segment_mean('train').get('Loss', 0.0)
    accuracy = tracker.segment_mean('test').get('Accuracy', 0.0)
    return {
        'mode': 'train',
        'steps': steps,
        'epochs': epochs,
        'train_size': data.meta.get('train_size'),
        'train_loss': train_loss,
        'accuracy': accuracy,
    }
