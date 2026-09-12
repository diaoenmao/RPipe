from rpipe.structure.algorithm.metric import MetricBundle, accuracy_value, mse_value, pack_io, resolve_metric_names


def test_resolve_metric_names_aliases():
    names = resolve_metric_names({'train': ['loss', 'accuracy'], 'test': ['MSE']})
    assert names['train'] == ['Loss', 'Accuracy']
    assert names['test'] == ['MSE']


def test_pack_io_tuple_and_dict():
    import torch

    logits = torch.tensor([[0.1, 0.9]])
    target = torch.tensor([1])
    packed = pack_io((torch.zeros(1, 1), target), logits)
    assert packed['target'] is target
    assert packed['logits'] is logits
    packed = pack_io({'target': target}, {'logits': logits, 'loss': torch.tensor(0.2)})
    assert abs(float(packed['loss'].item()) - 0.2) < 1e-6


def test_accuracy_and_mse_values():
    import torch

    logits = torch.tensor([[0.1, 4.0], [4.0, 0.1]])
    target = torch.tensor([1, 0])
    packed = pack_io((None, target), logits)
    assert accuracy_value(packed) == 1.0
    pred = torch.tensor([1.0, 2.0])
    y = torch.tensor([1.0, 3.0])
    assert abs(mse_value(pack_io({'target': y}, {'pred': pred})) - 0.5) < 1e-6


def test_metric_bundle_batch_defaults():
    import torch

    bundle = MetricBundle()
    logits = torch.tensor([[0.1, 4.0]])
    target = torch.tensor([1])
    values = bundle.evaluate('train', 'batch', (None, target), logits)
    assert 'Loss' in values
    assert 'Accuracy' in values
    assert values['Accuracy'] == 1.0
