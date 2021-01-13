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
import copy
from typing import List
from src.compression_manager import C


class Agent:
    def __init__(self):
        pass

    def train_step(self,
                   num_steps=1,
                   device="cpu"):
        pass

    def update_step(self):
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

    def train_step(self, num_steps=1, device="cpu") -> float:
        dist_weights_to_model(self.w_current, learner=self.learner)
        total_loss = 0
        for it in range(num_steps):
            model = self.learner.to(device)
            model.train()
            x, y = next(self.train_iter)
            x, y = x.float(), y
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            self.optimizer.zero_grad()
            loss_val = self.criterion(y_hat, y)
            total_loss += loss_val.item()

            loss_val.backward()
            self.optimizer.step()
            if self.lrs:
                self.lrs.step()

        total_loss /= num_steps

        # update the estimated gradients
        updated_model_weights = flatten_params(learner=self.learner)
        self.grad_current = self.w_current - updated_model_weights

        return total_loss

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

    def __init__(self,
                 server_model,
                 server_optimizer,
                 server_lrs,
                 gar: GAR):
        Agent.__init__(self)
        self.learner = server_model
        self.optimizer = server_optimizer
        self.lrs = server_lrs
        self.gar = gar

        self.G = None
        self.G_stale = None

        # initialize params
        self.w_current = None
        self.w_old = None
        self.u = None
        self.beta = 0.5

    def update_step(self):
        # update server model
        self.optimizer.zero_grad()
        dist_grads_to_model(grads=np.array(self.u, dtype=np.float32), learner=self.learner)
        self.optimizer.step()
        if self.lrs:
            self.lrs.step()

        # update weights
        self.w_old = self.w_current
        self.w_current = flatten_params(learner=self.learner)

    def compute_agg_grad(self, clients: List[FedClient]):
        # Now update server model
        # stack grads - compute G
        n = len(clients)

        for ix, client in enumerate(clients):
            g_i = client.grad_current
            if not self.G:
                d = len(g_i)
                self.G = np.ndarray((n, d), dtype=g_i.dtype)
            self.G[ix, :] = g_i

        # invoke gar and get aggregate
        self.u = self.gar.aggregate(G=self.G)

    def compute_agg_grad_glomo(self, clients: List[FedClient]):
        """ Implements Das et.al. FedGlomo: server update step with (Glo)bal (Mo)mentum"""
        n = len(clients)
        for ix, client in enumerate(clients):
            g_i = client.grad_current
            g_i_stale = client.grad_stale
            if not self.G or self.G_stale:
                d = len(g_i)
                self.G = np.ndarray((n, d), dtype=g_i.dtype)
                self.G_stale = np.ndarray((n, d), dtype=g_i.dtype)
            self.G[ix, :] = g_i
            self.G_stale[ix, :] = g_i_stale

            # invoke gar and get aggregate
            agg_g = self.gar.aggregate(G=self.G)
            agg_g_stale = self.gar.aggregate(G=self.G_stale)

            # compute new u
            u_new = self.beta * agg_g + \
                    ((1 - self.beta) * self.u) + \
                    ((1 - self.beta) * (agg_g - agg_g_stale))
            self.u = u_new