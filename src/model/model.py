import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import model
from transformers import get_linear_schedule_with_warmup
from module import filter_args


def make_model(cfg):
    core = eval('model.{}(cfg)'.format(cfg['model_name']))
    model_ = model.base(core)
    return model_


def init_param(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        if m.weight is not None:
            m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        if m.bias is not None:
            m.bias.data.zero_()
    return m


def make_optimizer(parameters, **cfg):
    optimizer_class = getattr(optim, cfg['optimizer_name'])
    valid_cfg = filter_args(optimizer_class, cfg)
    optimizer = optimizer_class(parameters, **valid_cfg)
    return optimizer


def make_scheduler(optimizer, cfg):
    if cfg['scheduler_name'] == 'None':
        scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1)
    elif cfg['scheduler_name'] == 'LinearAnnealingLR':
        cfg['num_warmup_steps'] = cfg['num_steps'] * cfg['warmup_ratio']
        cfg['num_training_steps'] = cfg['num_steps']
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(cfg['num_steps'] * cfg['warmup_ratio']),
                                                    num_training_steps=cfg['num_steps'])
    else:
        if cfg['scheduler_name'] == 'CosineAnnealingLR':
            cfg['T_max'] = cfg['num_steps']
        scheduler_class = getattr(optim.lr_scheduler, cfg['scheduler_name'])
        valid_cfg = filter_args(scheduler_class, cfg)
        scheduler = scheduler_class(optimizer, **valid_cfg)
    return scheduler
