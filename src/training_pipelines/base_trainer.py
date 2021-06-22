from typing import Dict
import numpy as np
import torch
from src.attack_manager import get_feature_attack, get_grad_attack
from src.model_manager import (get_model,
                               get_optimizer,
                               get_scheduler,
                               get_loss)
from src.aggregation_manager import get_gar
from src.compression_manager import (SparseApproxMatrix,
                                     get_compression_operator)


class TrainPipeline:
    def __init__(self, config, seed):
        # ------------------------ Fetch configs ----------------------- #
        print('---- Fetching configs and Initializing stuff -----')
        self.config = config
        self.data_config = config["data_config"]
        self.training_config = config["training_config"]

        self.num_batches = self.training_config.get('num_clients', 1)
        self.num_epochs = self.training_config.get('global_epochs', 10)

        self.learner_config = self.training_config["learner_config"]
        self.optimizer_config = self.training_config.get("optimizer_config", {})

        self.client_optimizer_config = self.optimizer_config.get("client_optimizer_config", {})
        self.client_lrs_config = self.optimizer_config.get('client_lrs_config')

        self.aggregation_config = self.training_config["aggregation_config"]
        self.sparse_approx_config = self.aggregation_config.get("sparse_approximation_config", {})
        self.compression_config = self.aggregation_config.get("compression_config", {})

        self.grad_attack_config = self.aggregation_config.get("grad_attack_config", {})
        self.feature_attack_config = self.data_config.get("feature_attack_config", {})

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
        self.criterion = get_loss(loss=self.client_optimizer_config.get('loss', 'ce'))
        self.gar = get_gar(aggregation_config=self.aggregation_config)

        # sparse approximation of the gradients before aggregating
        self.sparse_rule = self.sparse_approx_config.get('rule', None)
        self.sparse_selection = SparseApproxMatrix(conf=self.sparse_approx_config) \
            if self.sparse_rule else None
        self.G = None

        # gradient standard vector compression object
        self.C = get_compression_operator(compression_config=self.compression_config)

        # for adversarial - get attack model
        self.feature_attack_model = get_feature_attack(attack_config=self.feature_attack_config)
        self.grad_attack_model = get_grad_attack(attack_config=self.grad_attack_config)

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

                   "communication_residual": [],
                   "sparse_approx_residual": [],
                   # # Grad Matrix Stats
                   # "frac_mass_retained": [],
                   # "grad_norm_dist": [],
                   # "norm_bins": None,
                   # "mass_bins": None,
                   # "max_norm": 0,
                   # "min_norm": 1e6,

                   # compute Time stats per epoch
                   "epoch_sparse_approx_cost": [],
                   "epoch_grad_cost": [],
                   "epoch_agg_cost": [],
                   "epoch_gm_iter": [],

                   # Total Costs
                   "total_cost": 0,
                   "total_grad_cost": 0,
                   "total_agg_cost": 0,
                   "total_sparse_cost": 0,

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
