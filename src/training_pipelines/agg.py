import time
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from base_trainer import TrainPipeline
from src.data_manager import process_data
from src.model_manager import flatten_grads

"""
This file deals with training procedure for different batch sampling strategies
"""
# Reproducibility Checks
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.use_deterministic_algorithms(True)


class GARPipeline(TrainPipeline):
    def __init__(self, config, seed):
        TrainPipeline.__init__(self, config=config, seed=seed)
        self.epoch = 0

    def run_batch_train(self, config: Dict, metrics: Dict, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)

        # ------------------------- get data --------------------- #
        batch_size = self.data_config.get('batch_size', 1)
        data_manager = process_data(data_config=self.data_config)
        train_dataset, val_dataset, test_dataset = data_manager.download_data()

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        print('Num of Batches in Train Loader = {}'.format(len(train_loader)))
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

        if self.feature_attack_model is not None:
            #  populate feature attack params
            self.feature_attack_model.num_corrupt = np.ceil(self.feature_attack_model.frac_adv * self.num_batches)
            self.feature_attack_model.curr_corr = self.feature_attack_model.num_corrupt

        while self.epoch < self.num_epochs:
            self.model.to(device)
            self.model.train()

            epoch_grad_cost = 0
            epoch_agg_cost = 0
            epoch_gm_iter = 0
            epoch_sparse_cost = 0

            # ------- Training Phase --------- #
            print('epoch {}/{} || learning rate: {}'.format(self.epoch,
                                                            self.num_epochs,
                                                            self.client_optimizer.param_groups[0]['lr']))
            p_bar = tqdm(total=len(train_loader))
            p_bar.set_description("Training Progress: ")

            for batch_ix, (images, labels) in enumerate(train_loader):
                metrics["num_iter"] += 1
                t_iter = time.time()
                # Apply Feature Attack
                if self.feature_attack_model is not None:
                    images, labels = self.feature_attack_model.attack(X=images, Y=labels)
                    self.feature_attack_model.curr_corr -= 1

                images = images.to(device)
                labels = labels.to(device)
                outputs = self.model(images)
                self.client_optimizer.zero_grad()
                loss = self.criterion(outputs, labels)
                loss.backward()
                # Note: No Optimizer Step yet.

                g_i = flatten_grads(learner=self.model)
                if self.G is None:
                    d = len(g_i)
                    print("Num of Parameters {}".format(d))
                    metrics["num_param"] = d
                    self.G = np.zeros((self.num_batches, d), dtype=g_i.dtype)

                ix = batch_ix % self.num_batches
                agg_ix = (batch_ix + 1) % self.num_batches
                self.G[ix, :] = g_i

                iteration_time = time.time() - t_iter
                epoch_grad_cost += iteration_time
                p_bar.update()

                if agg_ix == 0 and batch_ix is not 0:
                    pass

    def aggregate(self):
        # Adversarial Attack
        if self.grad_attack_model is not None:
            G = grad_attack_model.launch_attack(G=G)
        if feature_attack_model is not None:
            # Reset For next set of batches
            feature_attack_model.curr_corr = feature_attack_model.num_corrupt

    def run_fed_train(self, config: Dict, metrics: Dict, seed):
        raise NotImplementedError
