from typing import Dict
import numpy as np
import torch
from src.attack_manager import get_feature_attack
from src.model_manager import (get_model,
                               get_optimizer,
                               get_scheduler,
                               get_loss)


class TrainPipeline:
    def __init__(self, config, metrics, seed):
        self.config = config
        self.data_config = config["data_config"]
        self.training_config = config["training_config"]

        self.learner_config = self.training_config["learner_config"]
        self.optimizer_config = self.training_config.get("optimizer_config", {})

        self.client_optimizer_config = self.optimizer_config.get("client_optimizer_config", {})
        self.client_lrs_config = self.optimizer_config.get('client_lrs_config')

        self.aggregation_config = self.training_config["aggregation_config"]
        self.sparse_approx_config = self.aggregation_config.get("sparse_approximation_config", {})
        self.compression_config = self.aggregation_config.get("compression_config", {})

        self.grad_attack_config = self.aggregation_config.get("grad_attack_config", {})
        self.feature_attack_config = self.data_config.get("feature_attack_config", {})

        self.metrics = metrics

        self.seed = seed

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.feature_attack_model = get_feature_attack(attack_config=self.feature_attack_config)
        self.client_model = get_model(learner_config=self.learner_config,
                                      data_config=self.data_config,
                                      seed=seed)
        self.client_optimizer = get_optimizer(params=self.client_model.parameters(),
                                              optimizer_config=self.client_optimizer_config)
        self.client_lrs = get_scheduler(optimizer=self.client_optimizer,
                                        lrs_config=self.client_lrs_config)
        self.criterion = get_loss(loss=self.client_optimizer_config.get('loss', 'ce'))

    def run_batch_train(self, config: Dict, metrics: Dict, seed):
        raise NotImplementedError("This method needs to be implemented for each pipeline")

    def run_fed_train(self, config: Dict, metrics: Dict, seed):
        raise NotImplementedError("This method needs to be implemented for each pipeline")
