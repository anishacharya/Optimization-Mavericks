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
from typing import List, Dict
from src.compression_manager import C


class Agent:
    def __init__(self):
        pass

    def train_step(self, num_iter=1, device="cpu"):
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

        self.C = compression

        self.w_current = None
        self.w_old = None

        self.grad_current = None
        self.grad_stale = None

        self.local_train_data = None

    def initialize_params(self, w_current, w_old):
        self.w_current = w_current
        self.w_old = w_old

    def train_step(self, num_iter=1, device="cpu"):
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

