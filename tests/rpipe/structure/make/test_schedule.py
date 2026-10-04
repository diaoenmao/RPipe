"""Sibling-dependent launch recovery and stale-result protection."""

import json
import os
from pathlib import Path

import pytest

import rpipe.structure.make.schedule as schedule
from rpipe.structure.artifact import artifact_layout
from rpipe.structure.artifact.index import write_index
from rpipe.structure.artifact.result import load_result, write_result

pytestmark = [
    pytest.mark.unit,
    pytest.mark.content,
    pytest.mark.p1,
    pytest.mark.structure_layer,
    pytest.mark.module_make,
    pytest.mark.cost(cost_class='c1'),
    pytest.mark.result_type('categorical', detail='summary'),
]


def _study(root: Path):
    jobs = []
    experiments = []
    for mode, prefix in [('train', 't'), ('eval', 'e')]:
        runs = []
        for seed in range(2):
            run_id = f'{prefix}{seed}'
            config = artifact_layout(root, run_id).config_path
            config.write_text(json.dumps({'algorithm': {'mode': mode}, 'system': {'device': 'cpu'}}), encoding='utf-8')
            jobs.append({'run_id': run_id, 'mode': mode, 'device': 'cpu', 'config': str(config)})
            runs.append({'id': run_id, 'seed': seed})
        experiments.append({'factors': {'algorithm.mode': mode}, 'runs': runs})
    write_index(root, {'study': 'recovery', 'experiments': experiments})
    return jobs


def _succeed(study: Path, run_id: str):
    write_result(artifact_layout(study, run_id).result_path, {
        'status': 'succeeded', 'metrics': {'accuracy': 42}, 'control': {}, 'paths': {},
    })


def _fake_processes(monkeypatch, study, failures):
    started = []
    counts = {}

    class Proc:
        def __init__(self, args, **_kwargs):
            self.run_id = args[-1]
            started.append(self.run_id)
            counts[self.run_id] = counts.get(self.run_id, 0) + 1

        def wait(self):
            if counts[self.run_id] <= failures.get(self.run_id, 0):
                write_result(artifact_layout(study, self.run_id).result_path,
                             {'status': 'failed', 'error': 'injected train failure'})
                return 1
            _succeed(study, self.run_id)
            return 0

    monkeypatch.setattr(schedule.subprocess, 'Popen', Proc)
    return started


@pytest.mark.parametrize('failures', [1, 2], ids=['retry-recovers', 'retry-exhausted'])
def test_launch_finishes_train_retry_before_dependent_eval(tmp_path, monkeypatch, failures):
    jobs = _study(tmp_path)
    started = _fake_processes(monkeypatch, tmp_path, {'t0': failures})
    # Even a supplied mixed batch must respect the dependency barrier.
    codes = schedule.launch_jobs(tmp_path, jobs, batches=[jobs], console='shared')
    assert started[:3] == ['t0', 't1', 't0']
    assert schedule.run_succeeded(tmp_path, 'e1')
    if failures == 1:
        assert started == ['t0', 't1', 't0', 'e0', 'e1']
        assert schedule.run_succeeded(tmp_path, 'e0')
        assert codes == [1, 0, 0, 0, 0]
    else:
        assert 'e0' not in started
        assert not schedule.run_succeeded(tmp_path, 'e0')
        result = load_result(artifact_layout(tmp_path, 'e0').result_path)
        assert result['status'] == 'failed'
        assert 't0' in result['error']


def test_train_only_rerun_invalidates_only_its_successful_sibling_eval(tmp_path, monkeypatch):
    jobs = _study(tmp_path)
    for job in jobs:
        _succeed(tmp_path, job['run_id'])
    started = _fake_processes(monkeypatch, tmp_path, {})
    schedule.launch_jobs(tmp_path, jobs[:1], console='shared')
    assert started == ['t0']
    stale = load_result(artifact_layout(tmp_path, 'e0').result_path)
    assert stale['status'] == 'failed'
    assert stale['metrics']['accuracy'] == 42  # retain old evidence, exclude it from success
    assert schedule.run_succeeded(tmp_path, 'e1')


@pytest.mark.parametrize('parent_succeeded', [True, False], ids=['newer-parent', 'unfinished-parent'])
def test_relaunch_and_make_include_previously_successful_stale_eval(tmp_path, monkeypatch, parent_succeeded):
    jobs = _study(tmp_path)
    for job in jobs:
        _succeed(tmp_path, job['run_id'])
    eval_result = artifact_layout(tmp_path, 'e0').result_path
    train_result = artifact_layout(tmp_path, 't0').result_path
    if not parent_succeeded:
        write_result(train_result, {'status': 'failed', 'error': 'injected failure'})
    os.utime(eval_result, ns=(1_000_000_000, 1_000_000_000))
    os.utime(train_result, ns=(2_000_000_000, 2_000_000_000))
    before = eval_result.read_bytes()
    schedule.write_jobs_json(tmp_path, jobs, round_size=2, init_gpu=0, num_gpus=0,
                             batches=[jobs[:2], jobs[2:]])
    plan = schedule.load_launch_plan(tmp_path, init_gpu=0, num_gpus=0, round_size=2)
    expected = ['e0'] if parent_succeeded else ['t0', 'e0']
    assert [job['run_id'] for job in plan['job_list']] == expected
    made = schedule.plan_jobs(tmp_path, [Path(job['config']) for job in jobs], num_gpus=0)
    assert [job['run_id'] for job in made] == expected
    assert eval_result.read_bytes() == before  # planning remains read-only for results
    started = _fake_processes(monkeypatch, tmp_path, {})
    schedule.launch_jobs(tmp_path, plan['job_list'], batches=plan['batches'], console='shared')
    assert started == expected
    assert schedule.run_succeeded(tmp_path, 'e0')


def test_eval_only_launch_blocks_unsuccessful_parent_without_spawning(tmp_path, monkeypatch):
    jobs = _study(tmp_path)
    started = _fake_processes(monkeypatch, tmp_path, {})
    assert schedule.launch_jobs(tmp_path, jobs[2:3], console='shared') == [1]
    assert started == []
    assert not schedule.run_succeeded(tmp_path, 'e0')


@pytest.mark.parametrize('independent', ['disabled', 'external', 'own'])
def test_independent_eval_does_not_require_sibling_success(tmp_path, monkeypatch, independent):
    jobs = _study(tmp_path)
    algorithm = {'mode': 'eval'}
    if independent == 'disabled':
        algorithm['resume'] = False
    else:
        checkpoint = (tmp_path / 'external.pt' if independent == 'external' else
                      artifact_layout(tmp_path, 'e0').assets_dir / 'checkpoints' / 'best.pt')
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b'path resolution only')
        if independent == 'external':
            algorithm['resume_from'] = str(checkpoint)
    artifact_layout(tmp_path, 'e0').config_path.write_text(json.dumps({'algorithm': algorithm}), encoding='utf-8')
    started = _fake_processes(monkeypatch, tmp_path, {})
    schedule.launch_jobs(tmp_path, jobs[2:3], console='shared')
    assert started == ['e0']
    assert schedule.run_succeeded(tmp_path, 'e0')


@pytest.mark.parametrize('location', ['own', 'external'])
@pytest.mark.parametrize('material', ['empty', 'unrelated', 'empty-meta', 'model.pt', 'payload.pt', 'bundle'])
def test_checkpoint_directory_requires_material_to_be_independent(tmp_path, monkeypatch, location, material):
    jobs = _study(tmp_path)
    own = artifact_layout(tmp_path, 'e0').assets_dir / 'checkpoints' / 'best'
    folder = own if location == 'own' else tmp_path / 'external' / 'best'
    folder.mkdir(parents=True)
    if material in ('model.pt', 'payload.pt'):
        (folder / material).write_bytes(b'path resolution only')
    elif material == 'bundle':
        folder.with_suffix('.pt').write_bytes(b'path resolution only')
    elif material == 'empty-meta':
        (folder / 'meta.json').write_text('{}', encoding='utf-8')
    elif material == 'unrelated':
        (folder / 'notes.txt').write_text('not a checkpoint', encoding='utf-8')
    if location == 'external':
        artifact_layout(tmp_path, 'e0').config_path.write_text(json.dumps({
            'algorithm': {'mode': 'eval', 'resume_from': str(folder)},
        }), encoding='utf-8')
    # A failed train can have a checkpoint; an empty override must not bypass its guard.
    partial = artifact_layout(tmp_path, 't0').assets_dir / 'checkpoints' / 'best.pt'
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b'partial train checkpoint')
    started = _fake_processes(monkeypatch, tmp_path, {})
    codes = schedule.launch_jobs(tmp_path, jobs[2:3], console='shared')
    independent = material in ('model.pt', 'payload.pt', 'bundle')
    assert started == (['e0'] if independent else [])
    assert codes == ([0] if independent else [1])
    assert schedule.run_succeeded(tmp_path, 'e0') is independent


def test_killed_rerun_cannot_reuse_previous_success(tmp_path, monkeypatch):
    jobs = _study(tmp_path)
    for job in jobs:
        _succeed(tmp_path, job['run_id'])

    class Killed:
        def wait(self):
            return 1

    monkeypatch.setattr(schedule.subprocess, 'Popen', lambda *_args, **_kwargs: Killed())
    schedule.launch_jobs(tmp_path, jobs[:1], retry_failed=False, console='shared')
    assert not schedule.run_succeeded(tmp_path, 't0')
    assert not schedule.run_succeeded(tmp_path, 'e0')


@pytest.mark.parametrize('complete', [True, False], ids=['recovered', 'incomplete'])
def test_bash_chunks_execute_shared_launcher_and_process_after_failure(tmp_path, monkeypatch, complete):
    import rpipe.flow.process as process

    jobs = _study(tmp_path)
    chunks = schedule.render_bash(tmp_path, jobs, python_exe='/opt/python', round_size=2, split_round=1)
    launched = []
    processed = []

    def launch(study, selected, **kwargs):
        assert study == tmp_path
        assert kwargs['batches'] == [selected]
        launched.append([job['run_id'] for job in selected])
        return [1, 0]  # A transient failed execution must not determine final exit.

    monkeypatch.setattr(schedule, 'launch_jobs', launch)
    monkeypatch.setattr(schedule, 'run_succeeded', lambda *_args: complete)
    monkeypatch.setattr(process, 'run_study', lambda study: processed.append(study) or {'complete': complete})
    assert len(chunks) == 2
    for index, chunk in enumerate(chunks):
        body = chunk.split("<<'RPIPE_LAUNCH'\n", 1)[1].rsplit('\nRPIPE_LAUNCH', 1)[0]
        with pytest.raises(SystemExit) as exit_info:
            exec(compile(body, f'launch_{index + 1}.sh', 'exec'), {})
        assert exit_info.value.code == int(not complete)
        assert len(processed) == index  # Only the final chunk processes, even after failure.
    assert launched == [['t0', 't1'], ['e0', 'e1']]
    assert processed == [tmp_path]
