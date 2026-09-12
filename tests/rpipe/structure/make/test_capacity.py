from rpipe.structure.make.capacity import (
    GpuInfo,
    batch_summaries,
    estimate_job_bytes,
    estimate_job_seconds,
    estimate_wall_seconds,
    format_duration,
    pack_jobs,
    pack_label,
    round_fits,
    suggest_round,
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


def test_estimate_resnet_heavier_than_linear():
    linear = estimate_job_bytes(
        {
            'data': {'name': 'MNIST', 'config': {'batch_size': 64}},
            'model': {'name': 'linear'},
            'algorithm': {'mode': 'train'},
        }
    )
    resnet = estimate_job_bytes(
        {
            'data': {'name': 'CIFAR10', 'config': {'batch_size': 250}},
            'model': {'name': 'resnet18'},
            'algorithm': {'mode': 'train'},
        }
    )
    eval_resnet = estimate_job_bytes(
        {
            'data': {'name': 'CIFAR10', 'config': {'batch_size': 250}},
            'model': {'name': 'resnet18'},
            'algorithm': {'mode': 'eval'},
        }
    )
    assert resnet > linear * 5
    assert eval_resnet < resnet
    assert eval_resnet > linear


def test_estimate_seconds_resnet_heavier_and_eval_lighter():
    linear = estimate_job_seconds(
        {
            'data': {'name': 'MNIST', 'config': {'batch_size': 250, 'train_size': 500}},
            'model': {'name': 'linear'},
            'algorithm': {'mode': 'train', 'num_epochs': 20},
        }
    )
    resnet = estimate_job_seconds(
        {
            'data': {'name': 'CIFAR10', 'config': {'batch_size': 250, 'train_size': 50000}},
            'model': {'name': 'resnet18'},
            'algorithm': {'mode': 'train', 'num_epochs': 20},
        }
    )
    eval_resnet = estimate_job_seconds(
        {
            'data': {'name': 'CIFAR10', 'config': {'batch_size': 250, 'train_size': 50000}},
            'model': {'name': 'resnet18'},
            'algorithm': {'mode': 'eval', 'num_epochs': 20},
        }
    )
    assert resnet > linear
    assert eval_resnet < resnet
    assert eval_resnet >= 1


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


def test_tiny_jobs_round_equals_pending_count():
    gpu = GpuInfo(index=0, name='fake', total_bytes=32 * 1024**3, free_bytes=32 * 1024**3)
    jobs = [
        {'run_id': f'r{i}', 'gpu': '0', 'mode': 'train', 'vram_bytes': 64 * 1024 * 1024}
        for i in range(9)
    ]
    assert suggest_round(jobs, [gpu]) == 9
