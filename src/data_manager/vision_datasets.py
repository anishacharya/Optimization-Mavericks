# Copyright (c) Anish Acharya.
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License

from .data_manager import DataManager
from torchvision import datasets, transforms
from typing import Dict
import os
curr_dir = os.path.dirname(__file__)
root = os.path.join(curr_dir, './data/')


class MNIST(DataManager):
    def __init__(self, data_config: Dict):
        DataManager.__init__(self, data_config=data_config)

    def download_data(self):
        _train_dataset = datasets.MNIST(root=root, download=True)

        mean, std = self._get_common_data_trans(_train_dataset)
        train_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        test_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

        _train_dataset = datasets.MNIST(root=root, download=True, transform=train_trans)
        _test_dataset = datasets.MNIST(root=root, download=True, train=False, transform=test_trans)

        return _train_dataset, _test_dataset


class FashionMNIST(DataManager):
    def __init__(self, data_config: Dict):
        DataManager.__init__(self, data_config=data_config)

    def download_data(self):
        _train_dataset = datasets.FashionMNIST(root=root, download=True)
        mean, std = self._get_common_data_trans(_train_dataset)
        train_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        test_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

        _train_dataset = datasets.FashionMNIST(root=root, download=True, transform=train_trans)
        _test_dataset = datasets.FashionMNIST(root=root, download=True, train=False, transform=test_trans)

        return _train_dataset, _test_dataset


class ExtendedMNIST(DataManager):
    def __init__(self, data_config: Dict):
        DataManager.__init__(self, data_config=data_config)

    def download_data(self):
        _train_dataset = datasets.EMNIST(root=root, download=True, split='balanced')
        mean, std = self._get_common_data_trans(_train_dataset)
        train_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        test_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

        _train_dataset = datasets.EMNIST(root=root, download=True, transform=train_trans, split='balanced')
        _test_dataset = datasets.EMNIST(root=root, download=True, train=False, transform=test_trans, split='balanced')

        return _train_dataset, _test_dataset


class CIFAR10(DataManager):
    def __init__(self, data_config: Dict):
        DataManager.__init__(self, data_config=data_config)

    def download_data(self):
        _train_dataset = datasets.CIFAR10(root=root, download=True)
        mean, std = self._get_common_data_trans(_train_dataset)
        train_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        test_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

        _train_dataset = datasets.CIFAR10(root=root, download=True, transform=train_trans)
        _test_dataset = datasets.CIFAR10(root=root, download=True, train=False, transform=test_trans)

        return _train_dataset, _test_dataset


class ImageNet(DataManager):
    def __init__(self, data_config: Dict):
        DataManager.__init__(self, data_config=data_config)

    # noinspection PyTypeChecker
    def download_data(self):
        _train_dataset = datasets.ImageNet(root=root, download=True)
        mean, std = self._get_common_data_trans(_train_dataset)
        train_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        test_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

        _train_dataset = datasets.ImageNet(root=root, download=True, transform=train_trans)
        _test_dataset = datasets.ImageNet(root=root, download=True, train=False, transform=test_trans)

        return _train_dataset, _test_dataset
