from config import cfg


def process_control():
    cfg['data_name'] = cfg['control']['data_name']
    cfg['model_name'] = cfg['control']['model_name']

    cfg['batch_size'] = 250
    cfg['step_period'] = 1
    cfg['num_steps'] = 80000
    cfg['eval_period'] = 200
    # cfg['num_epochs'] = 400
    cfg['collate_mode'] = 'dict'

    cfg['model'] = {}
    cfg['model']['model_name'] = cfg['model_name']
    data_shape = {'MNIST': [1, 28, 28], 'FashionMNIST': [1, 28, 28], 'SVHN': [3, 32, 32], 'CIFAR10': [3, 32, 32],
                  'CIFAR100': [3, 32, 32]}
    target_size = {'MNIST': 10, 'FashionMNIST': 10, 'SVHN': 10, 'CIFAR10': 10, 'CIFAR100': 100}
    cfg['model']['data_shape'] = data_shape[cfg['data_name']]
    cfg['model']['target_size'] = target_size[cfg['data_name']]
    cfg['model']['linear'] = {}
    cfg['model']['mlp'] = {'hidden_size': 128, 'scale_factor': 2, 'num_layers': 2, 'activation': 'relu'}
    cfg['model']['cnn'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['model']['resnet10'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['model']['resnet18'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['model']['wresnet28x2'] = {'depth': 28, 'widen_factor': 2, 'drop_rate': 0.0}
    cfg['model']['wresnet28x8'] = {'depth': 28, 'widen_factor': 8, 'drop_rate': 0.0}

    tag = cfg['tag']
    cfg[tag] = {}
    cfg[tag]['optimizer'] = {}
    cfg[tag]['optimizer']['optimizer_name'] = 'SGD'
    cfg[tag]['optimizer']['lr'] = 1e-1
    cfg[tag]['optimizer']['momentum'] = 0.9
    cfg[tag]['optimizer']['betas'] = (0.9, 0.999)
    cfg[tag]['optimizer']['weight_decay'] = 5e-4
    cfg[tag]['optimizer']['nesterov'] = True
    cfg[tag]['optimizer']['batch_size'] = {'train': cfg['batch_size'], 'test': cfg['batch_size']}
    cfg[tag]['optimizer']['step_period'] = cfg['step_period']
    cfg[tag]['optimizer']['num_steps'] = cfg['num_steps']
    cfg[tag]['optimizer']['scheduler_name'] = 'CosineAnnealingLR'
    return
