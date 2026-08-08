from rpipe.config.registry import DATASET_REGISTRY
from .dataset import make_dataset, make_data_loader, process_dataset, make_data_collate, input_collate
from .utils import Compose, download_url, extract_file
from .mnist import MNIST, FashionMNIST
from .cifar import CIFAR10, CIFAR100
from .svhn import SVHN

DATASET_REGISTRY.register('MNIST')(MNIST)
DATASET_REGISTRY.register('FashionMNIST')(FashionMNIST)
DATASET_REGISTRY.register('CIFAR10')(CIFAR10)
DATASET_REGISTRY.register('CIFAR100')(CIFAR100)
DATASET_REGISTRY.register('SVHN')(SVHN)

__all__ = [
    'make_dataset', 'make_data_loader', 'process_dataset', 'make_data_collate', 'input_collate',
    'Compose', 'download_url', 'extract_file',
    'MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'SVHN',
]
