# Copyright (c) Anish Acharya.
# Licensed under the MIT License

from .mlp import *
from .cnn import *
import torch
import functools
import numpy as np
from typing import Dict


def zero_grad(learner):
    """Given a model clear all grads; inspired by PyTorch optimizer.zero_grad"""
    for w in learner.parameters():
        if w.grad is not None:
            w.grad.detach_()
            w.grad.zero_()


def flatten_params(learner) -> np.ndarray:
    """ Given a model flatten all params and return as np array """
    flat_param = np.concatenate([w.data.cpu().numpy().flatten() for w in learner.parameters()])
    return flat_param


def flatten_grads(learner) -> np.ndarray:
    """ Given a model flatten all params and return as np array """
    flat_grad = np.concatenate([w.grad.data.cpu().numpy().flatten() for w in learner.parameters()])
    return flat_grad


def dist_weights_to_model(weights, learner):
    """ Given Weights and a model architecture this method updates the model parameters with the supplied weights """
    parameters = learner.to('cpu').parameters()
    offset = 0
    for param in parameters:
        new_size = functools.reduce(lambda x, y: x * y, param.shape)
        current_data = weights[offset:offset + new_size]
        param.data[:] = torch.from_numpy(current_data.reshape(param.shape))
        offset += new_size


def dist_grads_to_model(grads, learner):
    """ Given Gradients and a model architecture this method updates the model gradients (Corresponding to each param)
    with the supplied grads """
    parameters = learner.to('cpu').parameters()
    offset = 0
    for param in parameters:
        new_size = functools.reduce(lambda x, y: x * y, param.shape)
        current_data = grads[offset:offset + new_size]
        param.grad = torch.from_numpy(current_data.reshape(param.shape))
        offset += new_size


def get_model(learner_config: Dict, data_config: Dict):
    """ wrapper to return appropriate model class """
    net = learner_config.get("net", 'lenet')
    print('Loading Model: {}'.format(net))
    print('----------------------------')
    data_set = data_config["data_set"]
    nc = data_config.get("num_channels", 1)
    shape = data_config.get("shape", [28, 28])

    if net == 'lenet':
        model = LeNet(nc=nc, nh=shape[0], hw=shape[1], num_classes=data_config["num_labels"])

    elif net == 'log_reg':
        dim_in = np.prod(data_config["shape"]) * data_config["num_channels"]
        model = LogisticRegression(dim_in=dim_in,
                                   dim_out=data_config["num_labels"])
    elif net == 'mlp':
        dim_in = np.prod(shape) * nc
        model = MLP(dim_in=dim_in, dim_out=data_config["num_labels"])
    else:
        raise NotImplementedError

    print(model)
    return model


def cycle(iterable):
    while True:
        for x in iterable:
            yield x



