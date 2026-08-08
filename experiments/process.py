"""Aggregate multi-seed results into Excel + learning-curve plots."""

from __future__ import annotations

import os
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rpipe.system import load, makedir_exist_ok, save

matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.weight'] = 'bold'
matplotlib.rcParams['axes.labelweight'] = 'bold'
matplotlib.rcParams['axes.titleweight'] = 'bold'
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['xtick.labelsize'] = 'large'
matplotlib.rcParams['ytick.labelsize'] = 'large'


def tree():
    return defaultdict(tree)


def process_suite_results(controls, seeds, output_root='output', save_format='png', dpi=300):
    result_path = os.path.join(output_root, 'result')
    vis_path = os.path.join(output_root, 'vis', save_format)
    exp = list(seeds)

    result = tree()
    for control in controls:
        tag = '_'.join(control)
        _gather_result(list(control), tag, result, exp, result_path)
    _summarize_result(None, result)
    save(result, os.path.join(result_path, 'processed_result'))
    processed = tree()
    _extract_result(processed, result, [])
    make_df(processed, 'mean', result_path)
    df_history = make_df(processed, 'history', result_path)
    make_vis_history(df_history, vis_path, dpi=dpi, save_format=save_format)
    return processed


def _gather_result(control, tag, processed_result, exp, result_path):
    if len(control) == 1:
        exp_idx = exp.index(control[0])
        base_result_path_i = os.path.join(result_path, '{}'.format(tag))
        if os.path.exists(base_result_path_i):
            base_result = load(base_result_path_i)
            for split in base_result['logger']:
                for metric_name in base_result['logger'][split]['mean']:
                    processed_result[split][metric_name]['mean'][exp_idx] = \
                        base_result['logger'][split]['mean'][metric_name]
                for metric_name in base_result['logger'][split]['history']:
                    processed_result[split][metric_name]['history'][exp_idx] = \
                        base_result['logger'][split]['history'][metric_name]
        else:
            print('Missing {}'.format(base_result_path_i))
    else:
        _gather_result([control[0]] + control[2:], tag, processed_result[control[1]], exp, result_path)


def _summarize_result(key, value):
    if key in ['mean', 'history']:
        stacked = []
        for k, v in value.items():
            if k == 'summary':
                continue
            stacked.append(v)
        value['summary']['value'] = np.stack(stacked, axis=0)
        value['summary']['mean'] = np.mean(value['summary']['value'], axis=0)
        value['summary']['std'] = np.std(value['summary']['value'], axis=0)
        value['summary']['max'] = np.max(value['summary']['value'], axis=0)
        value['summary']['min'] = np.min(value['summary']['value'], axis=0)
        value['summary']['argmax'] = np.argmax(value['summary']['value'], axis=0)
        value['summary']['argmin'] = np.argmin(value['summary']['value'], axis=0)
        value['summary']['value'] = value['summary']['value'].tolist()
    else:
        for k, v in value.items():
            _summarize_result(k, v)


def _extract_result(extracted, processed_result, control):
    def extract(split, metric_name, mode):
        if split == 'train' and metric_name in ['test/Loss', 'test/Accuracy'] and mode == 'history':
            return True
        if split == 'test' and metric_name in ['test/Loss', 'test/Accuracy'] and mode == 'mean':
            return True
        return False

    if 'summary' in processed_result:
        control_name, split, metric_name, mode = control
        if not extract(split, metric_name, mode):
            return
        for stat in ['mean', 'std']:
            exp_name = '_'.join([control_name, metric_name.split('/')[1], stat])
            extracted[mode][exp_name] = processed_result['summary'][stat]
    else:
        for k, v in processed_result.items():
            _extract_result(extracted, v, control + [k])


def make_df(processed_result, mode, result_path):
    df = defaultdict(list)
    for exp_name in processed_result[mode]:
        df[exp_name].append(
            pd.DataFrame(data=processed_result[mode][exp_name].reshape(1, -1), index=[1]))
    startrow = 0
    with pd.ExcelWriter(os.path.join(result_path, 'result_{}.xlsx'.format(mode)), engine='xlsxwriter') as writer:
        for df_name in df:
            df[df_name] = pd.concat(df[df_name])
            df[df_name].to_excel(writer, sheet_name='Sheet1', startrow=startrow + 1, header=False, index=False)
            writer.sheets['Sheet1'].write_string(startrow, 0, df_name)
            startrow = startrow + len(df[df_name].index) + 3
    return df


def make_vis_history(df_history, vis_path, dpi=300, save_format='png'):
    label_dict = {
        'linear': 'Linear', 'mlp': 'MLP', 'cnn': 'CNN',
        'resnet10': 'ResNet10', 'resnet18': 'ResNet18',
        'wresnet28x2': 'WRN-28-2', 'wresnet28x8': 'WRN-28-8',
    }
    color_cycle = ['red', 'orange', 'blue', 'dodgerblue', 'green', 'purple', 'brown']
    linestyle_cycle = ['-', '--', ':', '-.', '-', '--']
    loc_dict = {'Accuracy': 'lower right', 'Loss': 'upper right'}
    fig, ax_dict_1, style_idx = {}, {}, {}
    for df_name in df_history:
        parts = df_name.split('_')
        metric_name, stat = parts[-2], parts[-1]
        if metric_name in ['Loss'] or stat != 'mean':
            continue
        model_name = parts[1]
        df_name_std = '_'.join([*parts[:-1], 'std'])
        fig_name = '_'.join([parts[0], *parts[2:]])
        fig[fig_name] = plt.figure(fig_name, figsize=(6.4, 4.8))
        if fig_name not in ax_dict_1:
            ax_dict_1[fig_name] = fig[fig_name].add_subplot(111)
        ax_1 = ax_dict_1[fig_name]
        y = df_history[df_name].iloc[0].to_numpy()
        y_err = df_history[df_name_std].iloc[0].to_numpy() if df_name_std in df_history else np.zeros_like(y)
        x = np.arange(len(y))
        if model_name not in style_idx:
            style_idx[model_name] = len(style_idx)
        si = style_idx[model_name]
        color = color_cycle[si % len(color_cycle)]
        ax_1.plot(x, y, label=label_dict.get(model_name, model_name), color=color,
                  linestyle=linestyle_cycle[si % len(linestyle_cycle)])
        ax_1.fill_between(x, (y - y_err), (y + y_err), color=color, alpha=.1)
        ax_1.set_xlabel('Epoch', fontsize=16)
        ax_1.set_ylabel(metric_name, fontsize=16)
        ax_1.legend(loc=loc_dict.get(metric_name, 'best'), fontsize=12)
    for fig_name in fig:
        ax_dict_1[fig_name].grid(linestyle='--', linewidth='0.5')
        dir_path = os.path.join(vis_path, 'lc')
        makedir_exist_ok(dir_path)
        plt.figure(fig_name)
        plt.tight_layout()
        plt.savefig(os.path.join(dir_path, '{}.{}'.format(fig_name, save_format)),
                    dpi=dpi, bbox_inches='tight', pad_inches=0.03)
        plt.close(fig_name)
