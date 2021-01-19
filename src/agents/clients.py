# Copyright (c) Anish Acharya.
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License
from .base import Agent
from src.model_manager import (flatten_params,
                               dist_weights_to_model,
                               flatten_grads,
                               get_optimizer,
                               get_scheduler)
import copy
from src.compression_manager import C


class FedClient(Agent):
    def __init__(self,
                 client_id: int,
                 learner,
                 compression: C):
        """ Implements a Federated Client Node """
        Agent.__init__(self)
        self.client_id = client_id
        self.training_config = None

        self.learner = learner  # To store: w_{k, tao}
        self.learner_stale = None  # To store: w_{k-1, tao}

        self.learner_local = None  # To store: w_{k, tao-1}
        self.learner_local_stale = None  # To store: w_{k-1, tao-1}

        self.optimizer = None
        self.optimizer_stale = None

        self.lrs = None
        self.lrs_stale = None

        self.criterion = None

        self.C = compression

        self.w_current = None
        self.w_old = None

        self.grad_current = None  # Needed for all methods
        self.glomo_grad = None

        self.v_current = None  # glomo momentum
        self.v_old = None      # glomo momentum

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
            # print('Client local Loss = {}'.format(loss_val.item()))
            loss_val.backward()
            self.optimizer.step()

        total_loss /= num_steps

        # update the estimated gradients
        updated_model_weights = flatten_params(learner=self.learner)
        grad_current = self.w_current - updated_model_weights
        self.grad_current = self.C.compress(g=grad_current)

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

        total_loss = 0

        for it in range(num_steps):
            model = self.learner.to(device)  # w_k
            model_stale = self.learner_stale.to(device)  # w_k-1

            model.train()
            model_stale.train()

            x, y = next(self.train_iter)
            x, y = x.float(), y
            x, y = x.to(device), y.to(device)

            y_hat = model(x)
            self.optimizer.zero_grad()
            loss_val = self.criterion(y_hat, y)
            loss_val.backward()
            # self.optimizer.step() # commented out to implement own momentum
            g = flatten_grads(learner=self.learner)  # extract grad

            total_loss += loss_val.item()

            y_hat = model_stale(x)
            self.optimizer_stale.zero_grad()
            loss_val = self.criterion(y_hat, y)
            loss_val.backward()
            # self.optimizer_stale.step()
            g_stale = flatten_grads(learner=self.learner_stale) # extract grad

            # if T = 0 initialize
            if self.v_old is None or self.v_current is None:
                self.v_old = g_stale
                self.v_current = g
            else:
                self.v_current = g + (self.v_current - g)
                self.v_old = g_stale + (self.v_old )
            # No optimizer step compute w using our update

        # update the estimated gradients
        updated_current_model_weights = flatten_params(learner=self.learner)
        grad_current = self.w_current - updated_current_model_weights

        updated_stale_model_weights = flatten_params(learner=self.learner_stale)
        grad_stale = self.w_old - updated_stale_model_weights

        glomo_grad = grad_current - grad_stale

        self.grad_current = self.C.compress(g=grad_current)
        self.glomo_grad = self.C.compress(g=glomo_grad)

        total_loss /= num_steps
        return total_loss





