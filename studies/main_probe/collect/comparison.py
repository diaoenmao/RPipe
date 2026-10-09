"""Compare captured observations without replaying model execution."""
from types import SimpleNamespace
from ..execute.paired import state_compare, optimizer_compare, rng_equal


def observation(value):
    return SimpleNamespace(samples=value['samples'], result=lambda: value)


def compare_observations(raw, name, model_name):
    old_snaps, current_snaps = raw['old_snaps'], raw['current_snaps']
    old_best, new_best = raw['old_best'], raw['new_best']
    old_eval, new_eval = raw['old_eval'], raw['new_eval']
    summary, eval_summary = raw['summary'], raw['eval_summary']
    obs_old = {key: observation(value) for key, value in raw['obs_old'].items()}
    obs_new = {key: observation(value) for key, value in raw['obs_new'].items()}
    old_eval_observation = observation(raw['old_eval_observation'])
    eval_observations = {key: observation(value) for key, value in raw['eval_observations'].items()}
    init_check = state_compare(raw['old_init'], raw['current_init'])
    init_rng_check = rng_equal(raw['old_init_rng'], raw['current_init_rng'])
    segment_checks = []
    for step in (30, 60):
        previous, current = old_snaps[step], current_snaps[step]
        parameters = state_compare(previous['model'], current['model'])
        optimizer_check = optimizer_compare(previous['optimizer'], current['optimizer'])
        sched_equal = previous['scheduler'] == current['scheduler']
        rng_matches = rng_equal(previous['rng'], current['rng'])
        loss_diffs = {split: abs(previous[split]['Loss'] - current[split]['Loss']) for split in ('train', 'test')}
        counts = {split: {'original': round(previous[split]['Accuracy'] * size / 100),
                          'current': round(current[split]['Accuracy'] * size / 100), 'samples': size}
                  for split, size in (('train', 7500), ('test', 10000))}
        counts_equal = all(value['original'] == value['current'] for value in counts.values())
        cumulative_counts = {version: {split: {key: snapshot['observations'][split][key]
                                              for key in ('samples', 'batches')}
                                               for split in ('train', 'test')}
                             for version, snapshot in (('original', previous), ('current', current))}
        expected_counts = {'train': {'samples': step * 250, 'batches': step},
                           'test': {'samples': (step // 30) * 10000, 'batches': (step // 30) * 10}}
        observed_full = all(value == expected_counts for value in cumulative_counts.values())
        segment_checks.append({'step': step, 'parameters': parameters, 'scheduler_equal': sched_equal,
            'optimizer': optimizer_check,
            'rng_equal': rng_matches, 'loss_differences': loss_diffs, 'correct_counts': counts,
            'actual_cumulative_counts': cumulative_counts, 'expected_cumulative_counts': expected_counts,
            'actual_sample_count_gate_passed': observed_full,
            'original_metrics': {split: previous[split] for split in ('train','test')},
            'current_metrics': {split: current[split] for split in ('train','test')},
            'passed': bool(parameters['within_gate'] and optimizer_check['within_gate'] and sched_equal and rng_matches and counts_equal and observed_full and max(loss_diffs.values()) <= 1e-6)})
    eval_weights = state_compare(old_best['model'], raw["eval_model_state"])
    eval_diff = abs(old_eval['Loss'] - new_eval['Loss'])
    eval_counts_equal = round(old_eval['Accuracy']*100) == round(new_eval['Accuracy']*100)
    old_self_diff = abs(old_eval['Loss'] - old_best['test']['Loss'])
    new_self_diff = abs(new_eval['Loss'] - current_snaps[int(new_best['step'])]['test']['Loss'])
    old_self_counts = round(old_eval['Accuracy']*100) == round(old_best['test']['Accuracy']*100)
    new_self_counts = round(new_eval['Accuracy']*100) == round(current_snaps[int(new_best['step'])]['test']['Accuracy']*100)
    independent_eval_counts = {version: {key: item.result()[key] for key in ('samples', 'batches')}
                              for version, item in (('original', old_eval_observation), ('current', eval_observations['test']))}
    independent_eval_full = all(value == {'samples': 10000, 'batches': 10} for value in independent_eval_counts.values())
    row = {'data': name, 'model': model_name, 'initial_state': init_check, 'initial_rng_equal': init_rng_check,
        'segments': segment_checks, 'original_inputs': {key: value.result() for key,value in obs_old.items()},
        'current_inputs': {key: value.result() for key,value in obs_new.items()},
        'input_core_train_equal': obs_old['train'].result()==obs_new['train'].result(),
        # torchvision test tuples omit id; compare actual images,
        # labels and normalized core inputs, not an invented id hash.
        'input_core_test_equal': all(obs_old['test'].result()[key]==obs_new['test'].result()[key]
            for key in ('samples','batches','images_sha256','targets_sha256','core_inputs_sha256')),
        'best': {'original_step': old_best['step'], 'current_step': new_best['step'],
                 'selection_equal': old_best['step']==new_best['step'],
                 'weights': state_compare(old_best['model'],new_best['model'])},
        'independent_eval': {'original': old_eval, 'current': new_eval, 'weights': eval_weights,
                             'actual_counts': independent_eval_counts, 'full_sample_count_gate_passed': independent_eval_full,
                             'loss_difference': eval_diff, 'correct_counts_equal': eval_counts_equal,
                             'original_vs_own_best_loss_difference':old_self_diff,
                             'current_vs_own_best_loss_difference':new_self_diff,
                             'original_vs_own_best_correct_counts_equal':old_self_counts,
                             'current_vs_own_best_correct_counts_equal':new_self_counts,
                             'current_summary':eval_summary}, 'summary':summary,
        'elapsed_seconds': raw['elapsed_seconds'],
        'gpu_peak_allocated_bytes': raw['gpu_peak_allocated_bytes'],
        'gpu_peak_reserved_bytes': raw['gpu_peak_reserved_bytes']}
    row['full_sample_counts_equal'] = all(
        observation[split].samples == size for observation in (obs_old, obs_new)
        for split, size in (('train', 15000), ('test', 20000)))
    row['passed'] = bool(init_check['exact'] and init_rng_check and all(x['passed'] for x in segment_checks)
        and row['input_core_train_equal'] and row['input_core_test_equal'] and row['full_sample_counts_equal'] and row['best']['selection_equal']
        and row['best']['weights']['within_gate'] and eval_weights['within_gate'] and eval_diff<=1e-6
        and eval_counts_equal and independent_eval_full and old_self_diff<=1e-6 and new_self_diff<=1e-6 and old_self_counts and new_self_counts)

    return row
