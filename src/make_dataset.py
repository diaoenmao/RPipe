import os
import torch
from torchvision import transforms
from config import cfg
from dataset import make_dataset, make_data_loader, process_dataset, collate, Compose
from module import save, Stats, makedir_exist_ok, process_control

if __name__ == "__main__":
    stats_path = os.path.join('output', 'stats')
    dim = 1
    data_names = ['MNIST', 'FashionMNIST', 'SVHN', 'CIFAR10', 'CIFAR100']
    process_control()
    cfg['seed'] = 0
    with torch.no_grad():
        for data_name in data_names:
            dataset = make_dataset(data_name)
            dataset['train'].transform = Compose([transforms.ToTensor()])
            process_dataset(dataset)
            data_loader = make_data_loader(dataset, cfg['model_name'])
            stats = Stats(dim=dim)
            for i, input in enumerate(data_loader['train']):
                input = collate(input)
                stats.update(input['data'])
            stats = (stats.mean.tolist(), stats.std.tolist())
            print(data_name, stats)
            makedir_exist_ok(stats_path)
            save(stats, os.path.join(stats_path, '{}'.format(data_name)))
