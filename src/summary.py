import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from torchinfo import summary
from config import cfg, process_args
from dataset import make_dataset, make_data_loader, process_dataset
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
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        tag_list = [str(seeds[i]), cfg['control_name']]
        cfg['tag'] = '_'.join([x for x in tag_list if x])
        process_control()
        print('Experiment: {}'.format(cfg['tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    cfg['path'] = os.path.join('output', 'exp')
    cfg['tag_path'] = os.path.join(cfg['path'], cfg['tag'])
    dataset = make_dataset(cfg['data_name'])
    cfg['step'] = 0
    dataset = process_dataset(dataset)
    model = make_model(cfg['model'])
    model = model.to(cfg['device'])
    batch_size = {'train': 2, 'test': 2}
    data_loader = make_data_loader(dataset, batch_size)
    input = next(iter(data_loader['train']))
    input = to_device(input, cfg['device'])
    content = summary(model, input_data=[{'data': input['data']}], depth=50,
                      col_names=['input_size', 'output_size', 'num_params', 'params_percent', 'kernel_size',
                                 'mult_adds', 'trainable'])
    save(content, os.path.join(cfg['tag_path'], 'summary'))
    return


if __name__ == "__main__":
    main()
