from .base import Agent
from src.model_manager import (flatten_params,
                               dist_grads_to_model)
from src.aggregation_manager import GAR
import numpy as np
from typing import List
from .clients import FedClient
import copy


class FedServer(Agent):
    """ Implements a Federated Server or Master Node """

    def __init__(self,
                 server_model,
                 server_optimizer,
                 server_lrs,
                 gar: GAR,
                 gar_config):
        Agent.__init__(self)
        self.learner = server_model
        self.optimizer = server_optimizer
        self.lrs = server_lrs
        self.gar = gar
        self.gar_config = gar_config

        self.G = None
        self.G_stale = None

        # initialize params
        self.w_current = flatten_params(self.learner)
        self.w_old = copy.deepcopy(self.w_current)      # For Glomo

        self.client_drift = 0   # For MIME
        self.mime_momentum = 0  # MIME
        self.u = None

        self.beta = 1
        self.c = self.gar_config.get('glomo_server_c', 1)

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
            if self.G is None:
                d = len(g_i)
                self.G = np.ndarray((n, d), dtype=g_i.dtype)
            self.G[ix, :] = g_i

        # invoke gar and get aggregate
        self.u = self.gar.aggregate(G=self.G)

    def compute_agg_grad_mime(self, clients: List[FedClient]):
        n = len(clients)
        self.client_drift = 0
        self.w_current = 0
        for client in clients:
            self.client_drift += client.glomo_grad
            self.w_current += client.w_current

        self.client_drift /= len(clients)
        self.w_current /= len(clients)

        self.mime_momentum = (1 - self.c) * self.client_drift + self.c * self.mime_momentum




    def compute_agg_grad_glomo(self, clients: List[FedClient]):
        """ Implements Das et.al. FedGlomo: server update step with (Glo)bal (Mo)mentum"""
        n = len(clients)

        for ix, client in enumerate(clients):
            g_i = client.grad_current
            g_i_glomo = client.glomo_grad

            if self.G is None or self.G_stale is None:
                d = len(g_i)
                self.G = np.ndarray((n, d), dtype=g_i.dtype)
                self.G_stale = np.ndarray((n, d), dtype=g_i.dtype)
            self.G[ix, :] = g_i
            self.G_stale[ix, :] = g_i_glomo

        # invoke gar and get aggregate
        agg_g = self.gar.aggregate(G=self.G)
        agg_g_glomo = self.gar.aggregate(G=self.G_stale)

        if self.u is None:
            self.u = agg_g
        else:
            # compute new u
            u_new = self.beta * agg_g + \
                    ((1 - self.beta) * self.u) + \
                    ((1 - self.beta) * agg_g_glomo)

            self.u = u_new
