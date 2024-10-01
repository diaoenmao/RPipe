import torch
import torch.nn as nn
from .loss import make_loss


class Base(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = make_loss

    def forward(self, **input):
        output = {}
        output['pred'] = self.model(input['data'])
        output['loss'] = self.loss(output, input)
        return output


def base(model):
    model = Base(model)
    return model
