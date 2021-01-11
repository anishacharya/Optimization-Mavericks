# Copyright (c) Anish Acharya.
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License

from typing import Dict
from src.model_manager import (flatten_params,
                               dist_grads_to_model,
                               get_loss,
                               get_optimizer,
                               get_scheduler)
from src.aggregation_manager import GAR
import numpy as np
import torch


class Agent:
    def __init__(self):
        pass

    def train_step(self, num_iter=1, device="cpu"):
        pass

    def update_step(self, agg_grad):
        pass


class FedClient(Agent):
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
        self.lrs = get_scheduler(optimizer=self.opt)
        self.criterion = get_loss(loss=self.config.get('criterion', 'cross_entropy'))

        self.current_w = flatten_params(learner=self.learner)
        self.current_e = np.zeros_like(self.current_w)
        self.grad = None
        self.x_train = None
        self.y_train = None

    def train_step(self, x_train=None, y_train=None, num_iter=1, device="cpu"):
        pass


class FedServer(Agent):
    """ Implements a Federated Server or Master Node """
    def __init__(self, server_model, gar: GAR):
        Agent.__init__(self)
        self.learner = server_model
        self.gar = gar

        # initialize current w
        params = flatten_params(learner=self.learner)
        self.w_current = params
        self.w_old = None
        self.u_old = None

    def update_step(self):
        pass

