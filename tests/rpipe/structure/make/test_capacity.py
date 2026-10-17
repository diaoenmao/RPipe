import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.content,
    pytest.mark.p2,
    pytest.mark.structure_layer,
    pytest.mark.module_make,
    pytest.mark.cost(cost_class='c1'),
    pytest.mark.result_type('categorical', detail='summary'),
]

from rpipe.structure.make.capacity import (
    GpuInfo,
    batch_summaries,
    capacity_report,
    estimate_job_bytes,
    estimate_job_seconds,
    estimate_wall_seconds,
    format_duration,
    pack_jobs,
    pack_label,
    requires_gpu,
    round_fits,
    suggest_round,
    summarize_capacity,
    usable_bytes,
)


def test_pack_same_model_together():
    gpu = GpuInfo(index=0, name='fake', total_bytes=20 * 1024**3, free_bytes=20 * 1024**3)
    jobs = [
        {'run_id': 'r0', 'mode': 'train', 'vram_bytes': 7 * 1024**3, 'estimate_model': 'resnet18'},
        {'run_id': 'r1', 'mode': 'train', 'vram_bytes': 7 * 1024**3, 'estimate_model': 'resnet18'},
        {'run_id': 'l0', 'mode': 'train', 'vram_bytes': 1 * 1024**3, 'estimate_model': 'linear'},
        {'run_id': 'l1', 'mode': 'train', 'vram_bytes': 1 * 1024**3, 'estimate_model': 'linear'},
        {'run_id': 'l2', 'mode': 'train', 'vram_bytes': 1 * 1024**3, 'estimate_model': 'linear'},
    ]
    batches = pack_jobs(jobs, [gpu])
    labels = [{j['estimate_model'] for j in batch} for batch in batches]
    assert all(len(models) == 1 for models in labels)
    linear = next(batch for batch in batches if batch[0]['estimate_model'] == 'linear')
    assert {j['run_id'] for j in linear} == {'l0', 'l1', 'l2'}
    resnet_ids = [j['run_id'] for batch in batches if batch[0]['estimate_model'] == 'resnet18' for j in batch]
    assert set(resnet_ids) == {'r0', 'r1'}


def test_pack_label_collapses_repeats():
    assert pack_label(['linear'] * 9) == 'linear×9'
    assert pack_label(['linear', 'linear', 'resnet18']) == 'linear×2+resnet18'
    assert pack_label([]) == ''
    rows = batch_summaries(
        [[{'estimate_model': 'linear', 'seconds': 1, 'vram_bytes': 1}] * 3]
    )
    assert rows[0]['label'] == 'linear×3'


def test_pack_eval_after_train():
    gpu = GpuInfo(index=0, name='fake', total_bytes=32 * 1024**3, free_bytes=32 * 1024**3)
    jobs = [
        {'run_id': 'e0', 'mode': 'eval', 'vram_bytes': 1 * 1024**3, 'estimate_model': 'linear'},
        {'run_id': 't0', 'mode': 'train', 'vram_bytes': 1 * 1024**3, 'estimate_model': 'linear'},
    ]
    batches = pack_jobs(jobs, [gpu])
    assert [j['run_id'] for j in batches[0]] == ['t0']
    assert [j['run_id'] for j in batches[1]] == ['e0']






def test_estimate_wall_seconds_is_sum_of_group_maxima():
    batches = [
        [{'seconds': 10}, {'seconds': 40}],
        [{'seconds': 5}],
    ]
    assert estimate_wall_seconds(batches) == 45
    assert format_duration(90) == '1m30s'


def test_suggest_round_follows_vram_and_safety():
    gpu = GpuInfo(index=0, name='fake', total_bytes=32 * 1024**3, free_bytes=32 * 1024**3)
    usable = usable_bytes(gpu)
    heavy = usable // 2 + 1
    jobs = [
        {'run_id': 'a', 'gpu': '0', 'mode': 'train', 'vram_bytes': heavy},
        {'run_id': 'b', 'gpu': '0', 'mode': 'train', 'vram_bytes': heavy},
        {'run_id': 'c', 'gpu': '0', 'mode': 'train', 'vram_bytes': heavy},
        {'run_id': 'd', 'gpu': '0', 'mode': 'train', 'vram_bytes': heavy},
    ]
    assert suggest_round(jobs, [gpu]) == 1
    light = usable // 5
    for job in jobs:
        job['vram_bytes'] = light
    assert suggest_round(jobs, [gpu]) == 4


def test_round_fits_same_gpu_sum():
    jobs = [
        {'run_id': 'a', 'gpu': '0', 'mode': 'train', 'vram_bytes': 7},
        {'run_id': 'b', 'gpu': '0', 'mode': 'train', 'vram_bytes': 7},
    ]
    assert round_fits(jobs, 1, {'0': 10})
    assert not round_fits(jobs, 2, {'0': 10})
    assert round_fits(jobs, 2, {'0': 14})


def test_cpu_jobs_use_small_process_cap():
    jobs = [
        {
            'run_id': f'r{i}',
            'mode': 'train',
            'vram_bytes': 1,
            'estimate_device': 'cpu',
        }
        for i in range(6)
    ]
    assert suggest_round(jobs, []) == 4


def test_cpu_and_gpu_jobs_pack_separately():
    gpu = GpuInfo(index=0, name='fake', total_bytes=8 * 1024**3, free_bytes=8 * 1024**3)
    cpu = {
        'run_id': 'cpu',
        'mode': 'train',
        'device': 'cpu',
        'gpu': '0',
        'vram_bytes': 1,
        'estimate_model': 'linear',
    }
    cuda = {
        'run_id': 'cuda',
        'mode': 'train',
        'device': 'cuda',
        'gpu': '0',
        'vram_bytes': 1,
        'estimate_model': 'linear',
    }
    batches = pack_jobs([cpu, cuda], [gpu])
    assert [[job['run_id'] for job in batch] for batch in batches] == [['cpu'], ['cuda']]
    assert 'gpu' not in cpu
    assert cuda['gpu'] == '0'
    assert requires_gpu([cpu, cuda])
    assert not requires_gpu([cpu])


def test_cpu_capacity_summary_does_not_mention_gpu():
    jobs = [{'run_id': 'cpu', 'device': 'cpu', 'vram_bytes': 1}]
    report = capacity_report(jobs, [], round_size=1, round_source='pack')
    report['batches'] = batch_summaries([jobs])
    summary = summarize_capacity(report)
    assert summary.endswith('| CPU')
    assert 'GPU' not in summary


def test_tiny_jobs_round_equals_pending_count():
    gpu = GpuInfo(index=0, name='fake', total_bytes=32 * 1024**3, free_bytes=32 * 1024**3)
    jobs = [
        {'run_id': f'r{i}', 'gpu': '0', 'mode': 'train', 'vram_bytes': 64 * 1024 * 1024}
        for i in range(9)
    ]
    assert suggest_round(jobs, [gpu]) == 9




@pytest.mark.parametrize('name,train_size', [('MNIST', 60000), ('FashionMNIST', 60000), ('SVHN', 73257)])
def test_epoch_estimate_uses_training_not_test_cardinality(name, train_size):
    cfg = {'data': {'name': name, 'config': {'batch_size': 250}}, 'model': {'name': 'linear'},
           'algorithm': {'num_epochs': 1, 'progress_unit': 'epoch', 'eval_period': 0}}
    explicit = {**cfg, 'data': {'name': name, 'config': {'batch_size': 250, 'train_size': train_size}}}
    assert estimate_job_seconds(cfg) == estimate_job_seconds(explicit)


def test_accumulation_counts_forward_batches_not_only_optimizer_steps():
    cfg = {'data': {'name': 'MNIST', 'config': {'batch_size': 64}}, 'model': {'name': 'linear'},
           'algorithm': {'num_steps': 1000, 'eval_period': 0}}
    accumulated = {**cfg, 'algorithm': {**cfg['algorithm'], 'config': {'step_period': 2}}}
    assert estimate_job_seconds(accumulated) - estimate_job_seconds(cfg) == 6


def test_epoch_derived_step_budget_includes_full_accumulation_batches():
    cfg = {'data': {'name': 'MNIST', 'config': {'batch_size': 64, 'train_size': 257}},
           'model': {'name': 'linear'}, 'algorithm': {'num_epochs': 100, 'step_period': 2, 'eval_period': 0}}
    steps = {**cfg, 'algorithm': {'num_steps': 300, 'step_period': 2, 'eval_period': 0}}
    assert estimate_job_seconds(cfg) == estimate_job_seconds(steps)


def test_eval_and_train_peaks_account_for_actual_test_batch():
    cfg = {'data': {'name': 'CIFAR10', 'config': {'batch_size': 64, 'test_batch_ratio': 16}},
           'model': {'name': 'resnet18'}, 'algorithm': {'mode': 'train'}}
    train_small_test = {**cfg, 'data': {'name': 'CIFAR10', 'config': {'batch_size': 64, 'test_batch_ratio': 1}}}
    evaluation = {**cfg, 'algorithm': {'mode': 'eval'}}
    evaluation_small_test = {**train_small_test, 'algorithm': {'mode': 'eval'}}
    assert estimate_job_bytes(evaluation) > estimate_job_bytes(evaluation_small_test)
    assert estimate_job_bytes(cfg) == max(estimate_job_bytes(train_small_test), estimate_job_bytes(evaluation))


def test_calibrated_budget_replaces_guess_but_keeps_safety():
    budget = 4 * 1024**3
    cfg = {'system': {'vram_bytes': 8 * 1024**3, 'config': {'vram_bytes': budget}}}
    assert estimate_job_bytes(cfg) == budget
    gpu = GpuInfo(index=0, name='measured-device', total_bytes=24 * 1024**3, free_bytes=20 * 1024**3)
    jobs = [{'run_id': str(i), 'mode': 'train', 'device': 'cuda', 'vram_bytes': estimate_job_bytes(cfg)}
            for i in range(4)]
    assert [len(batch) for batch in pack_jobs(jobs, [gpu])] == [2, 2]


@pytest.mark.parametrize('invalid', [True, '4294967296', 4294967296.0, -1, 0, 512 * 1024**2 - 1])
def test_calibrated_budget_rejects_invalid_or_missing_overhead(invalid):
    with pytest.raises(ValueError, match='system.vram_bytes'):
        estimate_job_bytes({'system': {'config': {'vram_bytes': invalid}}})


def test_attach_estimates_reads_calibrated_budget_from_run_config(tmp_path):
    from rpipe.structure.artifact import write_config
    from rpipe.structure.make.capacity import attach_estimates

    path = tmp_path / 'config.json'
    write_config(path, {'system': {'device': 'cuda', 'config': {'vram_bytes': 4 * 1024**3}}})
    jobs = attach_estimates([{'config': str(path)}])
    assert jobs[0]['vram_bytes'] == 4 * 1024**3
    assert jobs[0]['estimate_device'] == 'cuda'


@pytest.mark.parametrize('models, expected', [
    ([], ''), (['a'], 'a'), (['a', 'b', 'a'], 'a+b+a'),
    (['a', 'a', 'b', 'a', 'a'], 'a×2+b+a×2'), (['', ''], '×2'),
])
def test_pack_label_keeps_separated_groups(models, expected):
    assert pack_label(models) == expected


def test_pack_interleaved_classes_keeps_first_seen_order():
    jobs = [{'run_id': str(i), 'estimate_model': name, 'device': 'cpu', 'vram_bytes': 1}
            for i, name in enumerate(['b', 'a', 'b', 'c', 'a'])]
    assert [[job['run_id'] for job in batch] for batch in pack_jobs(jobs, [])] == [
        ['0', '2'], ['1', '4'], ['3']]


@pytest.mark.parametrize('seconds,text', [
    (-1, '0s'), (0, '0s'), (59, '59s'), (60, '1m'), (61, '1m1s'),
    (3599, '59m59s'), (3600, '1h'), (3601, '1h'), (3661, '1h1m'),
])
def test_capacity_duration_preserves_display_precision(seconds, text):
    assert format_duration(seconds) == text


@pytest.mark.parametrize('size,text', [
    (0, '0B'), (1023, '1023B'), (1024, '1.0KiB'),
    (1024**4, '1.0TiB'), (1024**5, '1024.0TiB'),
])
def test_capacity_bytes_uses_final_unit_without_fallback(size, text):
    from rpipe.structure.make.capacity import format_bytes
    assert format_bytes(size) == text


def test_capacity_wall_sum_preserves_empty_and_nonnegative_groups():
    assert estimate_wall_seconds([]) == 0
    assert estimate_wall_seconds([[], [{'seconds': -5}], [{'seconds': '3'}, {'seconds': 2}]]) == 3


@pytest.mark.parametrize('gpu_count', [0, 1, 2])
def test_pack_jobs_keeps_wave_resource_order_and_original_jobs(gpu_count):
    jobs = [{'run_id': 'eval', 'mode': 'eval', 'device': 'cpu', 'estimate_model': 'linear'},
            {'run_id': 'cpu-a', 'device': 'cpu', 'estimate_model': 'linear', 'gpu': 'old'},
            {'run_id': 'cuda', 'device': 'cuda', 'estimate_model': 'linear', 'vram_bytes': 1},
            {'run_id': 'cpu-b', 'device': 'cpu', 'estimate_model': 'linear'}]
    gpus = [GpuInfo(index=i, name='synthetic', total_bytes=10, free_bytes=10)
            for i in range(gpu_count)]
    batches = pack_jobs(jobs, gpus)
    assert [[job['run_id'] for job in group] for group in batches] == [
        ['cpu-a', 'cpu-b'], ['cuda'], ['eval']]
    assert batches[0][0] is jobs[1] and batches[0][1] is jobs[3]
    assert batches[1][0] is jobs[2] and batches[2][0] is jobs[0]
    assert all('gpu' not in jobs[i] for i in [0, 1, 3])
    assert jobs[2].get('gpu') == ('0' if gpu_count else None)


@pytest.mark.parametrize('vram, expected', [(-10, 2), (0, 1), (None, 1), (1, 2)])
def test_suggest_round_keeps_positive_vram_floor_and_default(vram, expected):
    jobs = [{'run_id': name, 'gpu': '0', 'vram_bytes': vram} for name in ['a', 'b']]
    gpu = GpuInfo(0, 'synthetic', total_bytes=4, free_bytes=4)
    assert suggest_round(jobs, [gpu]) == expected
