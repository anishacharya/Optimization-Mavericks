# Copyright (c) Anish Acharya.
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License

from typing import Dict
from src.model_manager import (flatten_params,
                               dist_grads_to_model,
                               get_loss,
                               get_optimizer,
                               get_scheduler)
import numpy as np
import torch


class Agent:
    def __init__(self):
        pass

    def train_step(self, num_iter=1, device="cpu"):
        pass

    def update_step(self, agg_grad):
        pass


class Client(Agent):
    def __init__(self,
                 client_id: int,
                 client_config: Dict,
                 learner,
                 mal: bool = False):
        """
        :client_id: int , Specify the Client ID
        :config:  , pass parsed json (Dict) as config ; can have it personalized per client
        :learner: , pass the initialized model; the client does k local steps starting from this model. Usually same
         for each client in the Federated/Decentralized Setting
        :mal:bool, If True, Client behaves as adversary as described in the adversary rule in config
        """
        Agent.__init__(self)
        self.client_id = client_id
        self.config = client_config
        self.learner = learner

        self.error_feedback = client_config.get('error_feedback', False)
        self.mal = mal

        self.opt = get_optimizer(params=self.learner.parameters(), optimizer_config=self.config)
        self.lrs = get_scheduler(optimizer=self.opt, optimizer_config=self.config)
        self.criterion = get_loss(loss=self.config.get('criterion', 'cross_entropy'))

        self.current_w = flatten_params(learner=self.learner)
        self.current_e = np.zeros_like(self.current_w)
        self.grad = None
        self.x_train = None
        self.y_train = None

    def train_step(self, x_train=None, y_train=None, num_iter=1, device="cpu"):
        # local SGD
        iter_losses = []
        self.learner.train()

        if self.x_train is None:
            self.x_train = x_train
            self.y_train = y_train

        for i in range(0, num_iter):
            # noinspection PyArgumentList
            all_ix = self.x_train.shape[0]

            # TODO: Other Sampling strategies
            samples = torch.LongTensor(np.random.choice(a=np.arange(all_ix),
                                                        size=self.config.get('batch_size', 1),
                                                        replace=False))
            x = self.x_train[samples].float().to(device)
            y = self.y_train[samples].to(device)
            self.learner.to(device)

            y_hat = self.learner(x)
            loss = self.criterion(y_hat, y)

            iter_losses.append(loss.item())

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            if self.lrs:
                self.lrs.step()

        return iter_losses


class FedServer(Agent):
    def __init__(self,
                 server_model,
                 server_config: Dict):
        Agent.__init__(self)
        self.learner = server_model
        self.config = server_config

        # initialize current w
        self.w_current = flatten_params(learner=self.learner)
        self.current_e = np.zeros_like(self.w_current)

        self.lr = self.config.get('lr0', 1)
        # dev, test data set
        self.x_dev, self.y_dev = None, None
        self.x_test, self.y_test = None, None

        self.opt = get_optimizer(params=self.learner.parameters(), optimizer_config=self.config)
        self.lrs = get_scheduler(optimizer=self.opt, optimizer_config=self.config)

    def update_step(self, agg_grad):
        # update server model
        dist_grads_to_model(grads=agg_grad, learner=self.learner)
        self.opt.step()
        if self.lrs:
            self.lrs.step()
        self.w_current = flatten_params(learner=self.learner)

    def infer(self, device="cpu"):
        self.learner.to(device)
        self.learner.eval()
        correct = 0
        with torch.no_grad():
            x_test = self.x_test.float().to(device)
            y_test = self.y_test.to(device)
            y_hat = self.learner(x_test)
            # print(y_hat)
            prediction = y_hat.argmax(dim=1, keepdim=True)
            # print(prediction)
            correct += prediction.eq(y_test.view_as(prediction)).sum().item()
        accuracy = 100. * correct / len(y_test)
        return accuracy
