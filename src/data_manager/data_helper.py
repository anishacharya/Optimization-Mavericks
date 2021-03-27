# Copyright (c) Anish Acharya.
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License

from typing import Dict
from .vision_datasets import (MNIST,
                              CIFAR10,
                              FashionMNIST,
                              ImageNet,
                              MiniImageNet,
                              ExtendedMNIST)
from .data_manager import DataManager


def process_data(data_config: Dict) -> DataManager:
    data_set = data_config["data_set"]
    if data_set == 'cifar10':
        return CIFAR10(data_config=data_config)
    elif data_set == 'mnist':
        return MNIST(data_config=data_config)
    elif data_set == 'fashion_mnist':
        return FashionMNIST(data_config=data_config)
    elif data_set == 'imagenet':
        return ImageNet(data_config=data_config)
    elif data_set == 'm_imagenet':
        return MiniImageNet(data_config=data_config)
    elif data_set == 'extended_mnist':
        return ExtendedMNIST(data_config=data_config)
    else:
        raise NotImplemented
