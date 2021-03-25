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
        mlp_config = learner_config.get('mlp_config', {})
        h1 = mlp_config.get('h1', 300)
        h2 = mlp_config.get('h2', 300)
        model = MLP(dim_in=dim_in, dim_out=data_config["num_labels"], hidden1=h1, hidden2=h2)

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


def evaluate_classifier(epoch, num_epochs, model, train_loader, test_loader, metrics,
                        criterion=None, device="cpu") -> float:

    test_error, test_acc, _ = _evaluate(model=model, data_loader=test_loader, device=device)
    train_error, train_acc, train_loss = _evaluate(model=model, data_loader=train_loader,
                                                   criterion=criterion, device=device)
    print('Epoch progress: {}/{}, train loss = {}, train acc = {}, test acc = {}'.
          format(epoch, num_epochs, train_loss, train_acc, test_acc))
    metrics["train_error"].append(train_error)
    metrics["train_loss"].append(train_loss)
    metrics["train_acc"].append(train_acc)

    metrics["test_error"].append(test_error)
    metrics["test_acc"].append(test_acc)

    return train_loss


def _evaluate(model, data_loader, verbose=False, criterion=None, device="cpu"):
    model.to(device)
    with torch.no_grad():
        correct = 0
        total = 0
        total_loss = 0
        batches = 0

        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            if criterion is not None:
                total_loss += criterion(outputs, labels).detach().item()
            else:
                raise ValueError

            batches += 1
            _, predicted = torch.max(outputs.detach().data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        total_loss /= batches
        if verbose:
            print('Accuracy: {} %'.format(acc))
        return 100 - acc, acc, total_loss
