"""Accelerate-backed trainer: same research loop, HF Accelerate for device/DDP/AMP."""

from __future__ import annotations

from rpipe.system.backend.native import Evaluator, NativeTrainer


class AccelerateTrainer(NativeTrainer):
    def __init__(self, runtime):
        super().__init__(runtime)
        try:
            from accelerate import Accelerator
        except ImportError as e:
            raise ImportError('Install accelerate: pip install accelerate') from e
        mixed = None
        if getattr(runtime, 'mixed_precision', None):
            mixed = runtime.mixed_precision
        self.accelerator = Accelerator(mixed_precision=mixed) if mixed else Accelerator()
        # Prefer accelerator device when available
        if self.runtime.device.startswith('cuda') and self.accelerator.device.type == 'cuda':
            self.runtime.device = str(self.accelerator.device)

    def _prepare(self, model, optimizer, data_loader):
        train_loader = data_loader['train']
        model, optimizer, train_loader = self.accelerator.prepare(model, optimizer, train_loader)
        data_loader = dict(data_loader)
        data_loader['train'] = train_loader
        return model, optimizer, data_loader

    def _backward(self, loss):
        self.accelerator.backward(loss)


# Re-export Evaluator for convenience
__all__ = ['AccelerateTrainer', 'Evaluator']
