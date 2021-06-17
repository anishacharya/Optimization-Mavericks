from typing import Dict
import numpy as np
import torch
from base_trainer import TrainPipeline


class SamplingPipeline(TrainPipeline):
    def run_batch_train(self, config: Dict, metrics: Dict, seed):
        # ------------------------ Fetch configs ----------------------- #
        print('---- Fetching configs -----')
        np.random.seed(seed)
        torch.manual_seed(seed)

        data_config = config["data_config"]
        training_config = config["training_config"]

        learner_config = training_config["learner_config"]
        optimizer_config = training_config.get("optimizer_config", {})

        client_optimizer_config = optimizer_config.get("client_optimizer_config", {})
        client_lrs_config = optimizer_config.get('client_lrs_config')

        aggregation_config = training_config["aggregation_config"]
        sparse_approx_config = aggregation_config.get("sparse_approximation_config", {})
        compression_config = aggregation_config.get("compression_config", {})

        grad_attack_config = aggregation_config.get("grad_attack_config", {})
        feature_attack_config = data_config.get("feature_attack_config", {})

    def run_fed_train(self, config: Dict, metrics: Dict, seed):
        pass
