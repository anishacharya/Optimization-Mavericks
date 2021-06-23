from typing import Dict
import numpy as np
import torch
from src.training_pipelines import TrainPipeline


"""
This file deals with training procedure for different batch sampling strategies
"""


class SamplingPipeline(TrainPipeline):
    def __init__(self, config, seed):
        TrainPipeline.__init__(self, config=config, seed=seed)

    def run_batch_train(self, config: Dict, seed):
        # ------------------------ Fetch configs ----------------------- #
        print('---- Fetching configs -----')
        np.random.seed(seed)
        torch.manual_seed(seed)

    def run_fed_train(self, config: Dict, seed):
        raise NotImplementedError
