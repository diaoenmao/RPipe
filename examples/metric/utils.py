import evaluate
import torch
import torch.nn.functional as F


def Accuracy(input, output, topk=1):
    if 'target' in input:
        input_target = input['target']
    elif 'labels' in input:
        input_target = input['labels']
    else:
        input_target = None
        msg = 'Not valid input target'
        raise ValueError(msg)
    if 'pred' in output:
        output_target = output['pred']
    elif 'logits' in output:
        output_target = output['logits']
    else:
        output_target = None
        msg = 'Not valid output target'
        raise ValueError(msg)
    with torch.no_grad():
        if input_target.dtype != torch.int64:
            input_target = (input_target.topk(1, -1, True, True)[1]).view(-1)
        batch_size = torch.numel(input_target)
        pred_k = output_target.topk(topk, -1, True, True)[1]
        correct_k = pred_k.eq(input_target.unsqueeze(-1).expand_as(pred_k)).float().sum()
        acc = (correct_k * (100.0 / batch_size)).item()
    return acc


def MSE(input, output):
    input_target = input['target']
    output_target = output['pred']
    with torch.no_grad():
        mse = F.mse_loss(output_target, input_target).item()
    return mse


class RMSE:
    def __init__(self):
        self.reset()

    def reset(self):
        self.se = torch.zeros((1,))
        self.count = torch.zeros((1,))
        return

    def add(self, input, output):
        self.se += F.mse_loss(output['pred'], input['target'], reduction='sum')
        self.count += output['pred'].numel()
        return

    def __call__(self, input, output):
        rmse = ((self.se / self.count) ** 0.5).item()
        self.reset()
        return rmse


class GLUE:
    def __init__(self, subset_name):
        self.metric = evaluate.load('glue', subset_name)
        self.subset_name = subset_name

    def add(self, input, output):
        if self.subset_name in ['stsb']:
            predictions = output['pred']
        else:
            predictions = output['pred'].argmax(dim=-1)
        references = input['target']
        self.metric.add_batch(predictions=predictions, references=references)
        return

    def __call__(self, *args, **kwargs):
        glue = self.metric.compute()
        metric_name = list(glue.keys())[0]
        glue = glue[metric_name]
        return glue
