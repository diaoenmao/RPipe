import torch
import torch.nn as nn
import kornia.augmentation as K
from .loss import make_loss


class Base(nn.Module):
    def __init__(self, model, data_name, stats):
        super().__init__()
        self.model = model
        self.loss = make_loss

        self.augmentation = {}
        if data_name in ['MNIST', 'FashionMNIST']:
            self.augmentation['train'] = K.Normalize(mean=stats.mean, std=stats.std)
            self.augmentation['test'] = K.Normalize(mean=stats.mean, std=stats.std)
        elif data_name in ['CIFAR10', 'CIFAR100']:
            self.augmentation['train'] = nn.Sequential(
                K.RandomHorizontalFlip(p=0.5),
                K.RandomCrop((32, 32), padding=4, padding_mode='reflect'),
                K.Normalize(mean=stats.mean, std=stats.std)
            )
            self.augmentation['test'] = K.Normalize(mean=stats.mean, std=stats.std)
        elif data_name in ['SVHN']:
            self.augmentation['train'] = nn.Sequential(
                K.RandomCrop((32, 32), padding=4, padding_mode='reflect'),
                K.Normalize(mean=stats.mean, std=stats.std)
            )
            self.augmentation['test'] = K.Normalize(mean=stats.mean, std=stats.std)
        else:
            raise ValueError('Not valid data_name')

    def forward(self, **input):
        output = {}
        x = input['data']
        if self.training:
            x = self.augmentation['train'](x)
        else:
            x = self.augmentation['test'](x)
        output['pred'] = self.model(x)
        output['loss'] = self.loss(output, input)
        return output


def base(model, data_name, stats):
    return Base(model, data_name, stats)




# class Base(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#         self.loss = make_loss
#
#     def forward(self, **input):
#         output = {}
#         output['pred'] = self.model(input['data'])
#         output['loss'] = self.loss(output, input)
#         return output
#
#
# def base(model):
#     model = Base(model)
#     return model
