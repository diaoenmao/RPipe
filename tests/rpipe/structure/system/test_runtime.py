from rpipe.structure.system.config import SystemConfig
from rpipe.structure.system.runtime import apply_runtime, make_generator


def test_apply_runtime_defaults_match_main_benchmark():
    applied = apply_runtime(0, SystemConfig())
    assert applied['seed'] == 0
    assert applied['deterministic'] is False
    assert applied['cudnn_deterministic'] is False
    assert applied['cudnn_benchmark'] is True


def test_apply_runtime_deterministic_turns_benchmark_off():
    cfg = SystemConfig.from_mapping({'deterministic': True})
    applied = apply_runtime(1, cfg)
    assert applied['deterministic'] is True
    assert applied['cudnn_deterministic'] is True
    assert applied['cudnn_benchmark'] is False


def test_make_generator_same_seed_same_draw():
    import torch

    a = make_generator(7)
    b = make_generator(7)
    c = make_generator(8)
    assert a is not None and b is not None and c is not None
    assert torch.rand(4, generator=a).tolist() == torch.rand(4, generator=b).tolist()
    assert torch.rand(4, generator=c).tolist() != torch.rand(4, generator=make_generator(7)).tolist()


def test_dataloader_shuffle_bound_to_seed():
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    data = torch.arange(32)
    dataset = TensorDataset(data)

    def first_batch(seed: int) -> list[int]:
        loader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=True,
            generator=make_generator(seed),
        )
        batch, = next(iter(loader))
        return batch.tolist()

    assert first_batch(3) == first_batch(3)
    assert first_batch(3) != first_batch(4)
