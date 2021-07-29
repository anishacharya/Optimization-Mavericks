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
# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True


class JacobianCompressPipeline(TrainPipeline):
    def __init__(self, config, seed):
        TrainPipeline.__init__(self, config=config, seed=seed)
        self.epoch = 0

    def run_train(self, config: Dict, seed):
        pass

    def run_batch_train(self, config: Dict, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)

        # ------------------------- get data --------------------- #
        tr_batch_size = self.data_config.get('train_batch_size', 1)
        test_batch_size = self.data_config.get('test_batch_size', 512)
        data_manager = process_data(data_config=self.data_config)
        train_dataset, val_dataset, test_dataset = data_manager.download_data()

        train_loader = DataLoader(dataset=train_dataset, batch_size=tr_batch_size, shuffle=True)
        print('Num of Batches in Train Loader = {}'.format(len(train_loader)))

        test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size)

        grad_steps = -1

        while self.epoch < self.num_epochs:
            self.model.to(device)
            self.model.train()

            epoch_grad_cost = 0
            epoch_agg_cost = 0
            epoch_gm_iter = 0
            epoch_compression_cost = 0

            # ------- Training Phase --------- #
            print('epoch {}/{} || learning rate: {}'.format(self.epoch,
                                                            self.num_epochs,
                                                            self.client_optimizer.param_groups[0]['lr']))
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
                # compute grad
                loss.backward()
                grad_steps += 1
                # Note: No Optimizer Step yet.
                g_i = flatten_grads(learner=self.model)

                # Construct the Jacobian
                if self.G is None:
                    d = len(g_i)
                    print("Num of Parameters {}".format(d))
                    self.metrics["num_param"] = d
                    self.G = np.zeros((self.num_batches, d), dtype=g_i.dtype)

                ix = batch_ix % self.num_batches
                agg_ix = (batch_ix + 1) % self.num_batches
                self.G[ix, :] = g_i

                iteration_time = time.time() - t_iter
                epoch_grad_cost += iteration_time

                p_bar.update()

                # Gradient Aggregation
                if agg_ix == 0 and batch_ix is not 0:
                    # Modify G
                    lr = self.client_optimizer.param_groups[0]['lr']  # Need this for Error Feedback
                    # noinspection PyPep8Naming
                    I_k = None

                    if self.C is not None:
                        t0 = time.time()
                        self.G, I_k = self.C.compress(G=self.G, lr=lr)
                        epoch_compression_cost += time.time() - t0
                        # epoch_sparse_cost += time.time() - t0
                        self.metrics["jacobian_residual"].append(self.C.normalized_residual)

                    # Gradient aggregation - get aggregated gradient vector
                    agg_g = self.gar.aggregate(G=self.G, ix=I_k,
                                               axis=self.C.axis if self.C else 0)
                    epoch_gm_iter += self.gar.num_iter
                    epoch_agg_cost += self.gar.agg_time
                    # Reset GAR stats
                    self.gar.agg_time = 0
                    self.gar.num_iter = 0

                    # Update Model Grads with aggregated g : i.e. compute \tilde(g)
                    self.client_optimizer.zero_grad()
                    dist_grads_to_model(grads=agg_g, learner=self.model)
                    self.model.to(device)
                    # Now Do an optimizer step with x_t+1 = x_t - \eta \tilde(g)
                    self.client_optimizer.step()

                    self.metrics["num_steps"] += 1

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

    def run_fed_train(self, config: Dict, seed):
        raise NotImplementedError
