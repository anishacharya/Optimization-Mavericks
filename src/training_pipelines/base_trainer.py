from typing import Dict
import numpy as np
import torch
from src.attack_manager import get_feature_attack, get_grad_attack
from src.model_manager import (get_model,
                               get_optimizer,
                               get_scheduler,
                               get_loss)
from src.aggregation_manager import get_gar
from src.compression_manager import get_compression_operator


class TrainPipeline:
    def __init__(self, config, seed):
        # ------------------------ Fetch configs ----------------------- #
        print('---- Fetching configs and Initializing stuff -----')
        self.config = config

        self.data_config = config["data_config"]
        self.training_config = config["training_config"]

        self.train_batch_size = self.data_config.get('train_batch_size')
        self.num_batches = self.training_config.get('num_clients', 1)
        self.num_epochs = self.training_config.get('global_epochs', 10)
        self.eval_freq = self.training_config.get('eval_freq', 10)

        self.learner_config = self.training_config["learner_config"]
        self.optimizer_config = self.training_config.get("optimizer_config", {})
        self.client_optimizer_config = self.optimizer_config.get("client_optimizer_config", {})
        self.client_lrs_config = self.optimizer_config.get('client_lrs_config')

        self.aggregation_config = self.training_config["aggregation_config"]
        self.compression_config = self.aggregation_config.get("compression_config", {})

        self.grad_attack_config = self.aggregation_config.get("grad_attack_config", {})
        self.feature_attack_config = self.data_config.get("feature_attack_config", {})

        # ------------------------ initializations ----------------------- #
        self.metrics = self.init_metric()
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.model = get_model(learner_config=self.learner_config,
                               data_config=self.data_config,
                               seed=seed)
        self.client_optimizer = get_optimizer(params=self.model.parameters(),
                                              optimizer_config=self.client_optimizer_config)
        self.client_lrs = get_scheduler(optimizer=self.client_optimizer,
                                        lrs_config=self.client_lrs_config)

        self.loss_sampling = self.client_optimizer_config.get('loss_sampling', None)
        self.loss_sampling_beta = self.client_optimizer_config.get('beta', 1)
        self.criterion = get_loss(loss=self.client_optimizer_config.get('loss', 'ce'),
                                  reduction='mean' if
                                  (self.loss_sampling is None or self.loss_sampling_beta == 1) else 'none')

        # sparse approximation of the gradients before aggregating
        # self.sparse_rule = self.sparse_approx_config.get('rule', None)
        # if self.sparse_rule not in ['active_norm', 'random']:
        #     self.sparse_selection = None
        # else:
        #     self.sparse_selection = SparseApproxMatrix(conf=self.sparse_approx_config)
        self.G = None
        # Compression Operator
        self.C = get_compression_operator(compression_config=self.compression_config)

        # for adversarial - get attack model
        self.feature_attack_model = get_feature_attack(attack_config=self.feature_attack_config)
        self.grad_attack_model = get_grad_attack(attack_config=self.grad_attack_config)

        self.gar = get_gar(aggregation_config=self.aggregation_config)

    def loss_wrapper(self, outputs, labels):
        loss = self.criterion(outputs, labels)

        if self.loss_sampling is not None:
            if self.loss_sampling == 'top':
                # Implements : Ordered SGD: A New Stochastic Optimization Framework for Empirical Risk Minimization
                # Kawaguchi, Kenji and Lu, Haihao; AISTATS 2020
                k = min(int(self.loss_sampling_beta * self.train_batch_size), len(outputs))
                loss = torch.mean(torch.topk(loss, k, sorted=False)[0])
            else:
                raise NotImplementedError

        return loss

    def init_metric(self):
        metrics = {"config": self.config,

                   "num_param": 0,
                   # Train and Test Performance
                   "test_error": [],
                   "test_loss": [],
                   "test_acc": [],
                   "train_error": [],
                   "train_loss": [],
                   "train_acc": [],

                   "gradient_residual": [],
                   "jacobian_residual": [],
                   # # Grad Matrix Stats
                   # "frac_mass_retained": [],
                   # "grad_norm_dist": [],
                   # "norm_bins": None,
                   # "mass_bins": None,
                   # "max_norm": 0,
                   # "min_norm": 1e6,

                   # compute Time stats per epoch
                   "epoch_compression_cost": [],
                   "epoch_grad_cost": [],
                   "epoch_agg_cost": [],
                   "epoch_gm_iter": [],

                   # Total Costs
                   "total_cost": 0,
                   "total_grad_cost": 0,
                   "total_agg_cost": 0,
                   "total_compression_cost": 0,

                   "total_gm_iter": 0,
                   "avg_gm_cost": 0,

                   "num_iter": 0,
                   "num_steps": 0,
                   }
        return metrics

    def run_batch_train(self, config: Dict, seed):
        raise NotImplementedError("This method needs to be implemented for each pipeline")

    def run_fed_train(self, config: Dict, seed):
        raise NotImplementedError("This method needs to be implemented for each pipeline")
