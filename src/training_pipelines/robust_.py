import time
from typing import Dict
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.training_pipelines import TrainPipeline
from src.data_manager import process_data
from src.model_manager import flatten_grads, dist_grads_to_model

# Reproducibility Checks
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True


class RobustTrainingPipeline(TrainPipeline):
    def __init__(self, config, seed):
        TrainPipeline.__init__(self, config=config, seed=seed)
        self.epoch = 0

    def run_train(self, config: Dict, seed):
        pass

    def run_batch_train(self, config: Dict, seed):
        raise NotImplementedError

    def run_fed_train(self, config: Dict, seed):
        raise NotImplementedError
