import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import model
from transformers import get_linear_schedule_with_warmup


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


def make_optimizer(parameters, cfg):
    if cfg['optimizer_name'] == 'SGD':
        optimizer = optim.SGD(parameters, lr=cfg['lr'], momentum=cfg['momentum'],
                              weight_decay=cfg['weight_decay'], nesterov=cfg['nesterov'])
    elif cfg['optimizer_name'] == 'Adam':
        optimizer = optim.Adam(parameters, lr=cfg['lr'], betas=cfg['betas'],
                               weight_decay=cfg['weight_decay'])
    elif cfg['optimizer_name'] == 'AdamW':
        optimizer = optim.AdamW(parameters, lr=cfg['lr'], betas=cfg['betas'],
                                weight_decay=cfg['weight_decay'])
    elif cfg['optimizer_name'] == 'LBFGS':
        optimizer = optim.LBFGS(parameters, lr=cfg['lr'])
    else:
        raise ValueError('Not valid optimizer name')
    return optimizer


def make_scheduler(optimizer, cfg):
    if cfg['scheduler_name'] == 'None':
        scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1)
    elif cfg['scheduler_name'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'], gamma=cfg['factor'])
    elif cfg['scheduler_name'] == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'],
                                                   gamma=cfg['factor'])
    elif cfg['scheduler_name'] == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif cfg['scheduler_name'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['num_steps'],
                                                         eta_min=0)
    elif cfg['scheduler_name'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg['factor'],
                                                         patience=cfg['patience'], verbose=False,
                                                         threshold=cfg['threshold'], threshold_mode='rel',
                                                         min_lr=cfg['min_lr'])
    elif cfg['scheduler_name'] == 'CyclicLR':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg['lr'], max_lr=10 * cfg['lr'])
    elif cfg['scheduler_name'] == 'LinearAnnealingLR':
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(cfg['num_steps'] * cfg['warmup_ratio']),
                                                    num_training_steps=cfg['num_steps'])
    elif cfg['scheduler_name'] == 'ConstantLR':
        scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=cfg['factor'])
    else:
        raise ValueError('Not valid scheduler name')
    return scheduler
