import os
import itertools
import numpy as np
import pandas as pd
from utils import save, load, makedir_exist_ok
import matplotlib.pyplot as plt
from collections import defaultdict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
result_path = os.path.join('output', 'result')
save_format = 'png'
vis_path = os.path.join('output', 'vis', '{}'.format(save_format))
num_experiments = 4
exp = [str(x) for x in list(range(num_experiments))]
dpi = 300


def make_control(control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    controls = [exp] + [control_names]
    controls = list(itertools.product(*controls))
    return controls


def make_controls(mode):
    if mode == 'base':
        data_names = ['MNIST', 'CIFAR10']
        model_names = ['linear', 'mlp', 'cnn', 'resnet18']
        control_name = [[data_names, model_names]]
        controls = make_control(control_name)
    else:
        raise ValueError('Not valid mode')
    return controls


def main():
    modes = ['base']
    controls = []
    for mode in modes:
        controls += make_controls(mode)
    processed_result = process_result(controls)
    df_mean = make_df(processed_result, 'mean')
    df_history = make_df(processed_result, 'history')
    make_vis_history(df_history)
    return


def tree():
    return defaultdict(tree)


def process_result(controls):
    result = tree()
    for control in controls:
        model_tag = '_'.join(control)
        gather_result(list(control), model_tag, result)
    summarize_result(None, result)
    save(result, os.path.join(result_path, 'processed_result.pt'))
    processed_result = tree()
    extract_result(processed_result, result, [])
    return processed_result


def gather_result(control, model_tag, processed_result):
    if len(control) == 1:
        exp_idx = exp.index(control[0])
        base_result_path_i = os.path.join(result_path, '{}.pt'.format(model_tag))
        if os.path.exists(base_result_path_i):
            base_result = load(base_result_path_i)
            for split in base_result['logger']:
                for metric_name in base_result['logger'][split].mean:
                    processed_result[split][metric_name]['mean'][exp_idx] = base_result['logger'][split].mean[
                        metric_name]
                for metric_name in base_result['logger'][split].history:
                    processed_result[split][metric_name]['history'][exp_idx] = base_result['logger'][split].history[
                        metric_name]
        else:
            print('Missing {}'.format(base_result_path_i))
    else:
        gather_result([control[0]] + control[2:], model_tag, processed_result[control[1]])
    return


def summarize_result(key, value):
    if key in ['mean', 'history']:
        value['summary']['value'] = np.stack(list(value.values()), axis=0)
        value['summary']['mean'] = np.mean(value['summary']['value'], axis=0)
        value['summary']['std'] = np.std(value['summary']['value'], axis=0)
        value['summary']['max'] = np.max(value['summary']['value'], axis=0)
        value['summary']['min'] = np.min(value['summary']['value'], axis=0)
        value['summary']['argmax'] = np.argmax(value['summary']['value'], axis=0)
        value['summary']['argmin'] = np.argmin(value['summary']['value'], axis=0)
        value['summary']['value'] = value['summary']['value'].tolist()
    else:
        for k, v in value.items():
            summarize_result(k, v)
        return
    return


def extract_result(extracted_processed_result, processed_result, control):
    def extract(split, metric_name, mode):
        output = False
        if split == 'train':
            if metric_name in ['test/Loss', 'test/Accuracy']:
                if mode == 'history':
                    output = True
        elif split == 'test':
            if metric_name in ['test/Loss', 'test/Accuracy']:
                if mode == 'mean':
                    output = True
        return output

    if 'summary' in processed_result:
        control_name, split, metric_name, mode = control
        if not extract(split, metric_name, mode):
            return
        stats = ['mean', 'std']
        for stat in stats:
            exp_name = '_'.join([control_name, metric_name.split('/')[1], stat])
            extracted_processed_result[mode][exp_name] = processed_result['summary'][stat]
    else:
        for k, v in processed_result.items():
            extract_result(extracted_processed_result, v, control + [k])
    return


def make_df(processed_result, mode):
    df = defaultdict(list)
    for exp_name in processed_result[mode]:
        exp_name_list = exp_name.split('_')
        df_name = '_'.join([*exp_name_list])
        index_name = [1]
        df[df_name].append(pd.DataFrame(data=processed_result[mode][exp_name].reshape(1, -1), index=index_name))
    startrow = 0
    writer = pd.ExcelWriter('{}/result_{}.xlsx'.format(result_path, mode), engine='xlsxwriter')
    for df_name in df:
        df[df_name] = pd.concat(df[df_name])
        df[df_name].to_excel(writer, sheet_name='Sheet1', startrow=startrow + 1)
        writer.sheets['Sheet1'].write_string(startrow, 0, df_name)
        startrow = startrow + len(df[df_name].index) + 3
    writer.save()
    return df


def make_vis_history(df_history):
    label_dict = {'linear': 'Linear', 'mlp': 'MLP', 'cnn': 'CNN', 'resnet18': 'ResNet18'}
    color_dict = {'linear': 'red', 'mlp': 'orange', 'cnn': 'blue', 'resnet18': 'dodgerblue'}
    linestyle_dict = {'linear': '-', 'mlp': '--', 'cnn': ':', 'resnet18': '-.'}
    marker_dict = {'linear': 'o', 'mlp': 's', 'cnn': 'p', 'resnet18': 'd'}
    loc_dict = {'Accuracy': 'lower right', 'Loss': 'upper right'}
    fontsize = {'legend': 16, 'label': 16, 'ticks': 16}
    figsize = (5, 4)
    fig = {}
    ax_dict_1 = {}
    for df_name in df_history:
        df_name_list = df_name.split('_')
        metric_name, stat = df_name_list[-2], df_name_list[-1]
        mask = metric_name not in ['Loss'] and stat == 'mean'
        if mask:
            model_name = df_name_list[1]
            df_name_std = '_'.join([*df_name_list[:-1], 'std'])
            fig_name = '_'.join([df_name_list[0], *df_name_list[2:]])
            fig[fig_name] = plt.figure(fig_name, figsize=figsize)
            if fig_name not in ax_dict_1:
                ax_dict_1[fig_name] = fig[fig_name].add_subplot(111)
            ax_1 = ax_dict_1[fig_name]
            y = df_history[df_name].iloc[0].to_numpy()
            y_err = df_history[df_name_std].iloc[0].to_numpy()
            x = np.arange(len(y))
            xlabel = 'Epoch'
            pivot = model_name
            ylabel = metric_name
            ax_1.plot(x, y, label=label_dict[pivot], color=color_dict[pivot],
                      linestyle=linestyle_dict[pivot])
            ax_1.fill_between(x, (y - y_err), (y + y_err), color=color_dict[pivot], alpha=.1)
            ax_1.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_1.set_ylabel(ylabel, fontsize=fontsize['label'])
            ax_1.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_1.yaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_1.legend(loc=loc_dict[metric_name], fontsize=fontsize['legend'])
    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        ax_dict_1[fig_name].grid(linestyle='--', linewidth='0.5')
        fig[fig_name].tight_layout()
        dir_name = 'lc'
        dir_path = os.path.join(vis_path, dir_name)
        fig_path = os.path.join(dir_path, '{}.{}'.format(fig_name, save_format))
        makedir_exist_ok(dir_path)
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


if __name__ == '__main__':
    main()
