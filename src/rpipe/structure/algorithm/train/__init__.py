"""Train mode: real loop when MNIST + linear are prepared; else stub."""

from __future__ import annotations

from typing import Any


def run(control_algorithm: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    data = state.get('data') or {}
    model = state.get('model') or {}
    system = state.get('system') or {}

    if data.get('name') == 'MNIST' and model.get('module') is not None:
        return _run_mnist_linear(control_algorithm, state)

    steps = int(control_algorithm.get('num_steps', 1))
    return {'semantic': 'train', 'steps': steps, 'loss': 0.0}


def _run_mnist_linear(
    control_algorithm: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    data = state.get('data') or {}
    model = state.get('model') or {}
    system = state.get('system') or {}

    device = torch.device(system.get('device') or 'cpu')
    module = model['module'].to(device)
    train_loader = data['train_loader']
    test_loader = data['test_loader']

    epochs = int(control_algorithm.get('num_epochs', control_algorithm.get('num_steps', 1)))
    lr = float(control_algorithm.get('lr', 0.1))
    seed = state.get('seed')
    if seed is not None:
        torch.manual_seed(int(seed))

    optimizer = torch.optim.SGD(module.parameters(), lr=lr)
    module.train()
    last_loss = 0.0
    steps = 0
    for _ in range(epochs):
        for images, targets in train_loader:
            images = images.view(images.size(0), -1).to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = module(images)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())
            steps += 1

    module.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.view(images.size(0), -1).to(device)
            targets = targets.to(device)
            pred = module(images).argmax(dim=1)
            correct += int((pred == targets).sum().item())
            total += int(targets.size(0))
    accuracy = float(correct / total) if total else 0.0

    return {
        'semantic': 'train',
        'steps': steps,
        'epochs': epochs,
        'train_size': data.get('train_size'),
        'loss': last_loss,
        'accuracy': accuracy,
        'metric': {'accuracy': accuracy},
    }
