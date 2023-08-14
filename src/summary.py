import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from torchinfo import summary
from config import cfg, process_args
from dataset import make_dataset, make_data_loader, process_dataset, collate
from model import make_model
from module import save, to_device, process_control

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiment']))
    for i in range(cfg['num_experiment']):
        model_tag_list = [str(seeds[i]), cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    dataset = make_dataset(cfg['data_name'])
    dataset = process_dataset(dataset)
    batch_size = 2
    cfg[cfg['model_name']]['batch_size']['train'] = batch_size
    data_loader = make_data_loader(dataset, cfg['model_name'])
    input = next(iter(data_loader['train']))
    input = collate(input)
    input = to_device(input, cfg['device'])
    model = make_model(cfg['model_name'])
    content = summary(model, input_data=[{'data': input['data']}], depth=50,
                      col_names=['input_size', 'output_size', 'num_params', 'params_percent', 'kernel_size',
                                 'mult_adds', 'trainable'])
    print(content)
    save(content, os.path.join('output', 'summary', '{}'.format(cfg['model_tag'])))
    return


if __name__ == "__main__":
    main()
