# Copyright (c) Anish Acharya.
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License

import torch.optim as opt
from typing import Dict
import torch.nn as nn


def get_loss(loss: str):
    if loss == 'mse':
        return nn.MSELoss()
    elif loss == 'ce':
        return nn.CrossEntropyLoss()
    elif loss == 'bce':
        return nn.BCELoss()
    else:
        raise NotImplementedError


def get_optimizer(params, optimizer_config: Dict = None):
    if optimizer_config is None:
        optimizer_config = {}
    opt_alg = optimizer_config.get('optimizer', 'SGD')

    if opt_alg == 'SGD':
        return opt.SGD(params=params,
                       lr=optimizer_config.get('lr0', 1),
                       momentum=optimizer_config.get('momentum', 0),
                       weight_decay=optimizer_config.get('reg', 0),
                       nesterov=optimizer_config.get('nesterov', False),
                       dampening=optimizer_config.get('damp', 0))
    elif opt_alg == 'Adam':
        return opt.Adam(params=params,
                        lr=optimizer_config.get('lr0', 1),
                        betas=optimizer_config.get('betas', (0.9, 0.999)),
                        eps=optimizer_config.get('eps', 1e-08),
                        weight_decay=optimizer_config.get('reg', 0.05),
                        amsgrad=optimizer_config.get('amsgrad', False))
    else:
        raise NotImplementedError


def get_scheduler(optimizer, lrs_config: Dict = None):
    lrs = lrs_config.get('lrs')

    if lrs == 'step':
        return opt.lr_scheduler.StepLR(optimizer=optimizer,
                                       step_size=lrs_config.get('step_size', 10),
                                       gamma=lrs_config.get('gamma', 0.9))
    elif lrs == 'multi_step':
        return opt.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                            milestones=lrs_config.get('milestones', [100]),
                                            gamma=lrs_config.get('gamma', 0.5),
                                            last_epoch=lrs_config.get('last_epoch', -1))
    elif lrs == 'exp':
        return opt.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                              gamma=lrs_config.get('gamma', 0.5))
    elif lrs == 'cyclic':
        max_lr = lrs_config.get('lr0', 0.001)
        base_lr = 0.1*max_lr
        return opt.lr_scheduler.CyclicLR(optimizer=optimizer,
                                         base_lr=base_lr,
                                         max_lr=max_lr,
                                         step_size_up=lrs_config.get('step_size_up', 100))
    else:
        return None
