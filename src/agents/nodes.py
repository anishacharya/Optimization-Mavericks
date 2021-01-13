# Copyright (c) Anish Acharya.
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License

from src.model_manager import (flatten_params,
                               dist_grads_to_model,
                               get_loss,
                               get_optimizer,
                               get_scheduler)
from src.aggregation_manager import GAR
import numpy as np
import torch
import copy
from typing import List, Dict
from src.compression_manager import C


class Agent:
    def __init__(self):
        pass

    def train_step(self,
                   num_steps=1,
                   device="cpu"):
        pass

    def update_step(self, clients):
        pass


class FedClient(Agent):
    def __init__(self,
                 client_id: int,
                 learner,
                 compression: C):
        """ Implements a Federated Client Node """
        Agent.__init__(self)
        self.client_id = client_id

        self.learner = learner
        self.optimizer = None

        self.criterion = None
        self.lrs = None

        self.C = compression

        self.w_current = None
        self.w_old = None

        self.grad_current = None
        self.grad_stale = None

        self.local_train_data = None
        self.train_iter = None

    def initialize_params(self, w_current, w_old):
        self.w_current = w_current
        self.w_old = w_old

    def train_step(self, num_steps=1, device="cpu"):
        for it in range(num_steps):
            model = self.learner.to(device)
            model.train()
            x, y = next(self.train_iter)
            x, y = x.float(), y
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            self.optimizer.zero_grad()
            loss_val = self.criterion(y_hat, y)
            loss_val.backward()
            self.optimizer.step()
            if self.lrs:
                self.lrs.step()

        # update the estimated gradients
        updated_model_weights = flatten_params(learner=self.learner)
        self.grad_current = self.w_current - updated_model_weights

    def train_step_glomo(self, num_steps=1, device="cpu"):
        pass


class FedServer(Agent):
    """ Implements a Federated Server or Master Node """
    def __init__(self, server_model, gar: GAR):
        Agent.__init__(self)
        self.learner = server_model
        self.gar = gar

        # initialize params
        self.w_current = None
        self.w_old = None
        self.u = None

    def update_step(self, clients: List[FedClient]):
        pass

