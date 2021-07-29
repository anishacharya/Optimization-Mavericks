import time
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training_pipelines import TrainPipeline
from src.data_manager import process_data

# Reproducibility Checks
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True


class LossPipeline(TrainPipeline):
    def __init__(self, config, seed):
        TrainPipeline.__init__(self, config=config, seed=seed)
        self.epoch = 0

    def run_train(self, config: Dict, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        # ------------------------- get data --------------------- #
        data_manager = process_data(data_config=self.data_config)
        train_dataset, val_dataset, test_dataset = data_manager.download_data()
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.data_config.get('train_batch_size', 1), shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.data_config.get('test_batch_size', 512))
        print('Num of Batches in Train Loader = {}'.format(len(train_loader)))

        grad_steps = -1

        while self.epoch < self.num_epochs:
            self.model.to(device)
            self.model.train()

            epoch_grad_cost = 0
            epoch_agg_cost = 0
            epoch_gm_iter = 0
            epoch_compression_cost = 0

            # ------- Training Phase --------- #
            print('epoch {}/{} '.format(self.epoch, self.num_epochs))
            p_bar = tqdm(total=len(train_loader))
            p_bar.set_description("Training Progress: ")
            for batch_ix, (images, labels) in enumerate(train_loader):
                self.metrics["num_iter"] += 1
                t_iter = time.time()
                # Forward Pass
                images = images.to(device)
                labels = labels.to(device)
                outputs = self.model(images)
                self.client_optimizer.zero_grad()
                loss = self.loss_wrapper(outputs, labels)
                loss.backward()
                self.metrics["num_grad_steps"] += 1
                iteration_time = time.time() - t_iter
                epoch_grad_cost += iteration_time
                p_bar.update()
                self.client_optimizer.zero_grad()
                # Now Do an optimizer step with x_t+1 = x_t - \eta \tilde(g)
                self.client_optimizer.step()

                self.metrics["num_opt_steps"] += 1

                if grad_steps % self.eval_freq == 0:
                    train_loss = self.evaluate_classifier(model=self.model,
                                                          train_loader=train_loader,
                                                          test_loader=test_loader,
                                                          metrics=self.metrics,
                                                          device=device,
                                                          epoch=self.epoch,
                                                          num_epochs=self.num_epochs)
                    # Stop if diverging
                    if (train_loss > 1e3) | np.isnan(train_loss) | np.isinf(train_loss):
                        self.epoch = self.num_epochs
                        print(" *** Training is Diverging - Stopping !!! *** ")

            self.epoch += 1
            if self.client_lrs is not None:
                self.client_lrs.step()

            # update Epoch Complexity metrics
            # print("Epoch Grad Cost: {}".format(epoch_grad_cost))
            self.metrics["epoch_grad_cost"].append(epoch_grad_cost)
            # print("Epoch Aggregation Cost: {}".format(epoch_agg_cost))
            self.metrics["epoch_agg_cost"].append(epoch_agg_cost)
            # print("Epoch GM iterations: {}".format(epoch_gm_iter))
            if epoch_gm_iter > 0:
                self.metrics["epoch_gm_iter"].append(epoch_gm_iter)
            if epoch_compression_cost >0:
                # print("Epoch Sparse Approx Cost: {}".format(epoch_sparse_cost))
                self.metrics["epoch_compression_cost"].append(epoch_compression_cost)
        # Update Total Complexities
        self.metrics["total_grad_cost"] = sum(self.metrics["epoch_grad_cost"])
        self.metrics["total_agg_cost"] = sum(self.metrics["epoch_agg_cost"])
        self.metrics["total_gm_iter"] = sum(self.metrics["epoch_gm_iter"])
        self.metrics["total_compression_cost"] = sum(self.metrics["epoch_compression_cost"])
        self.metrics["total_cost"] = self.metrics["total_grad_cost"] + self.metrics["total_agg_cost"] + self.metrics[
            "total_compression_cost"]
        if self.metrics["total_gm_iter"] != 0:
            # Handle Non GM GARs
            self.metrics["avg_gm_cost"] = self.metrics["total_agg_cost"] / self.metrics["total_gm_iter"]

    def run_batch_train(self, config: Dict, seed):
        raise NotImplementedError

    def run_fed_train(self, config: Dict, seed):
        raise NotImplementedError
