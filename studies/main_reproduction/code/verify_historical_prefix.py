"""Audit saved artifacts from the authorized 200-step prefix, without training."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np  # Windows OpenMP import order.
import torch

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent / 'historical-prefix-9f51acfe2a55'
record = json.loads((WORK / 'comparison.json').read_text(encoding='utf-8'))
assert record['passed'] == record['total'] == 8
assert record['script_sha256'] == hashlib.sha256(
    (WORK.parent / 'historical_prefix_probe.py').read_bytes()).hexdigest()
assert record['source_files_unchanged'] and record['source_files_verified'] == 91
assert record['cosine_T_max'] == 80000 and record['seed'] == 0
assert record['train_batch'] == record['test_batch'] == 250
assert record['lr'] == 0.01 and record['clip_grad_norm'] == 1
assert {(row['data'], row['model']) for row in record['cases']} == {
    (data, model) for data in ('MNIST', 'CIFAR10')
    for model in ('linear', 'mlp', 'cnn', 'resnet18')}
manifest = json.loads((WORK / 'data-copy-manifest.json').read_text(encoding='utf-8'))
assert len(manifest) == 16
for relative, digest in manifest.items():
    assert hashlib.sha256((WORK / relative).read_bytes()).hexdigest() == digest
for relative, digest in json.loads((ROOT / 'studies/main_reproduction/docs/SOURCE_AFTER_B018_MANIFEST.json').read_text(encoding='utf-8')).items():
    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest

for row in record['cases']:
    folder = WORK / f"{row['data']}-{row['model']}"
    original = torch.load(folder / 'original-step200.pt', map_location='cpu', weights_only=True)
    current = torch.load(folder / 'current-assets/checkpoints/latest.pt', map_location='cpu', weights_only=True)
    current_state = {key.removeprefix('net.'): value for key, value in current['model'].items()}
    assert original['step'] == current['step'] == 200
    assert original['model'].keys() == current_state.keys()
    assert all(torch.equal(value, current_state[key]) for key, value in original['model'].items())
    assert original['scheduler'] == current['scheduler']
    assert current['scheduler']['T_max'] == 80000 and current['scheduler']['last_epoch'] == 200
    for group in current['optimizer']['param_groups']:
        assert group['momentum'] == 0.9 and group['nesterov'] is True
        assert group['weight_decay'] == 0.0005
        assert math.isclose(group['lr'], 0.01 * (1 + math.cos(math.pi * 200 / 80000)) / 2,
                            rel_tol=0, abs_tol=1e-12)
    tracker = json.loads((folder / 'current-assets/tracker/tracker_state.json').read_text(encoding='utf-8'))
    assert tracker['progress']['step'] == 200
    events = [json.loads(line) for line in (folder / 'current-assets/tracker/scalars.jsonl').read_text(encoding='utf-8').splitlines()]
    test_events = [event for event in events if event.get('split') == 'test']
    assert len(test_events) == 2
    assert {event['name'] for event in test_events} == {'Loss', 'Accuracy'}
    assert all(event['optimizer_step'] == 200 for event in test_events)
    assert row['train_samples_each'] == 50000 and row['test_samples_each'] == 10000
    assert row['initial_rng_equal'] and row['final_rng_equal'] and row['passed']
    for split, n, source_loss, source_accuracy in (
        ('train', 50000, original['train_loss'], original['train_accuracy']),
        ('test', 10000, original['test_loss'], original['accuracy']),
    ):
        actual_loss = tracker['splits'][split]['Loss']['history']
        actual_accuracy = tracker['splits'][split]['Accuracy']['history']
        assert len(actual_loss) == len(actual_accuracy) == 1
        assert abs(source_loss - actual_loss[0]) <= 1e-6
        assert round(source_accuracy * n / 100) == round(actual_accuracy[0] * n / 100)
    print(f"Saved evidence verified: {row['data']}/{row['model']}", flush=True)
print('8/8 stored checkpoint, hyperparameter, sample, metric and eval-coordinate audits passed.')
