import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from dataset import make_dataset, make_data_loader, process_dataset
from metric import make_logger
from model import make_model
from module import save, resume, to_device, process_control

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['control_name']]
        cfg['tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    path = os.path.join('output', 'exp')
    tag_path = os.path.join(path, cfg['tag'])
    checkpoint_path = os.path.join(tag_path, 'checkpoint')
    best_path = os.path.join(tag_path, 'best')
    result_path = os.path.join('output', 'result', cfg['tag'])
    dataset = make_dataset(cfg['data_name'])
    model = make_model(cfg['model_name'])
    result = resume(best_path)
    cfg['iteration'] = result['cfg']['iteration']
    model = model.to(cfg['device'])
    model.load_state_dict(result['model'])
    dataset = process_dataset(dataset)
    data_loader = make_data_loader(dataset, cfg[cfg['model_name']]['batch_size'])
    test_logger = make_logger(os.path.join(tag_path, 'logger', 'test', 'runs'))
    test(data_loader['test'], model, test_logger)
    result = resume(checkpoint_path)
    result = {'cfg': cfg, 'logger': {'train': result['logger'],
                                     'test': test_logger.state_dict()}}
    save(result, result_path)
    return


def test(data_loader, model, logger):
    with torch.no_grad():
        model.train(False)
        for i, input in enumerate(data_loader):
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            output = model(input)
            evaluation = logger.evaluate('test', 'batch', input, output)
            logger.append(evaluation, 'test', input_size)
        evaluation = logger.evaluate('test', 'full')
        logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(cfg['tag']),
                         'Test Epoch: {}({:.0f}%)'.format(cfg['iteration'] // cfg['eval_period'], 100.)]}
        logger.append(info, 'test')
        print(logger.write('test'))
        logger.save(True)
    return


if __name__ == "__main__":
    main()
