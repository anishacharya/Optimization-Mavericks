# Copyright (c) Anish Acharya.
# Licensed under the MIT License

from .mlp import *
from .cnn import *
from .vgg import *
from .resnet import *
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
    flat_param = []
    # for w in learner.parameters():
    flat_param.extend(torch.reshape(w.data, (-1,)).tolist() for w in learner.parameters())
    return np.asarray(flat_param)
    # flat_param = np.concatenate([w.data.cpu().numpy().flatten() for w in learner.parameters()])
    # return flat_param


def flatten_grads(learner) -> np.ndarray:
    """ Given a model flatten all params and return as np array """
    # flat_grad = []
    # for w in learner.parameters():
    # flat_grad.extend(torch.reshape(w.grad.data, (-1,)).numpy() for w in learner.parameters())
    # return np.asarray(flat_grad)
    flat_grad = np.concatenate([w.grad.data.cpu().numpy().flatten() for w in learner.parameters()])
    return flat_grad


def dist_weights_to_model(weights, learner):
    """ Given Weights and a model architecture this method updates the model parameters with the supplied weights """
    parameters = learner.parameters()
    offset = 0
    for param in parameters:
        new_size = functools.reduce(lambda x, y: x * y, param.shape)
        current_data = weights[offset:offset + new_size]
        param.data[:] = torch.from_numpy(current_data.reshape(param.shape))
        offset += new_size


def dist_grads_to_model(grads, learner):
    """ Given Gradients and a model architecture this method updates the model gradients (Corresponding to each param)
    with the supplied grads """
    parameters = learner.parameters()
    # grads.to(learner.device)
    offset = 0
    for param in parameters:
        new_size = functools.reduce(lambda x, y: x * y, param.shape)
        current_data = grads[offset:offset + new_size]
        param.grad = torch.from_numpy(current_data.reshape(param.shape)).to(learner.device)
        offset += new_size


def get_model(learner_config: Dict, data_config: Dict, seed=1):
    """ wrapper to return appropriate model class """
    net = learner_config.get("net", 'lenet')
    print('Loading Model: {}'.format(net))
    print('----------------------------')
    nc = data_config.get("num_channels", 1)
    shape = data_config.get("shape", [28, 28])

    if net == 'lenet':
        model = LeNet(nc=nc, nh=shape[0], hw=shape[1], num_classes=data_config["num_labels"], seed=seed)

    elif net == 'small_cnn':
        model = MnistClassifierCnn()

    elif net == 'log_reg':
        dim_in = np.prod(data_config["shape"]) * data_config["num_channels"]
        model = LogisticRegression(dim_in=dim_in,
                                   dim_out=data_config["num_labels"])
    elif net == 'mlp':
        dim_in = np.prod(shape) * nc
        mlp_config = learner_config.get('mlp_config', {})
        h1 = mlp_config.get('h1', 300)
        h2 = mlp_config.get('h2', 300)
        model = MLP(dim_in=dim_in, dim_out=data_config["num_labels"], hidden1=h1, hidden2=h2,
                    seed=seed)

    elif net in ['VGG11', 'VGG13', 'VGG16', 'VGG19']:
        print('Building {}'.format(net))
        model = VGG(net)
    elif net == 'resnet':
        model = ResNet18()
    else:
        raise NotImplementedError

    print(model)
    return model


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
