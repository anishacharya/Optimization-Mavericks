# Copyright (c) Anish Acharya.
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License

from src.model_manager import (flatten_params,
                               dist_grads_to_model,
                               dist_weights_to_model,
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
        self.training_config = None

        self.learner = learner
        self.learner_stale = None

        self.optimizer = None
        self.optimizer_stale = None

        self.lrs = None
        self.lrs_stale = None

        self.criterion = None

        self.C = compression

        self.w_current = None
        self.w_old = None

        self.grad_current = None
        self.grad_stale = None

        self.local_train_data = None
        self.train_iter = None

    def initialize_params(self, w_current, w_old=None):
        self.w_current = w_current
        self.w_old = w_old

    def train_step(self, num_steps=1, device="cpu"):
        dist_weights_to_model(self.w_current, learner=self.learner)
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
        if self.learner_stale is None:
            self.learner_stale = copy.deepcopy(self.learner)

        if self.optimizer_stale is None:
            self.optimizer_stale = get_optimizer(params=self.learner_stale.parameters(),
                                                 optimizer_config=self.training_config.get("optimizer_config", {}))
        if self.lrs_stale is None:
            self.lrs_stale = get_scheduler(optimizer=self.optimizer_stale,
                                           lrs_config=self.training_config.get('lrs_config', {}))

        dist_weights_to_model(self.w_current, learner=self.learner)
        dist_weights_to_model(self.w_old, learner=self.learner_stale)

        for it in range(num_steps):
            model = self.learner.to(device)
            model_stale = self.learner_stale.to(device)
            model.train()
            model_stale.train()

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

            y_hat = model_stale(x)
            self.optimizer_stale.zero_grad()
            loss_val = self.criterion(y_hat, y)
            loss_val.backward()
            self.optimizer_stale.step()
            if self.lrs_stale:
                self.lrs_stale.step()

        # update the estimated gradients
        updated_current_model_weights = flatten_params(learner=self.learner)
        self.grad_current = self.w_current - updated_current_model_weights

        updated_stale_model_weights = flatten_params(learner=self.learner_stale)
        self.grad_stale = self.w_old - updated_stale_model_weights


class FedServer(Agent):
    """ Implements a Federated Server or Master Node """
    def __init__(self, server_model, gar: GAR):
        Agent.__init__(self)
        self.learner = server_model
        self.gar = gar
        self.G = None

        # initialize params
        self.w_current = None
        self.w_old = None
        self.u = None

    def update_step(self, clients: List[FedClient]):
        # stack grads - compute G
        n = len(clients)
        for ix, client in enumerate(clients):
            g_i = client.grad_current
            d = len(g_i)
            if not self.G:
                self.G = np.ndarray((n, d), dtype=g_i.dtype)
            self.G[ix, :] = g_i

        # invoke gar and get aggregate
        agg_g = self.gar.aggregate(G=self.G)

        # Now update server model

    def update_step_glomo(self, clients: List[FedClient]):
        pass

