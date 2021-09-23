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
        np.random.seed(seed)
        torch.manual_seed(seed)
        # ------------------------- get data --------------------- #
        data_manager = process_data(data_config=self.data_config)
        train_dataset, val_dataset, test_dataset = data_manager.download_data()
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.data_config.get('train_batch_size', 1),
                                  shuffle=True)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=self.data_config.get('test_batch_size', 512))
        while self.epoch < self.num_epochs:
            self.model.to(device)
            self.model.train()
            epoch_grad_cost = 0
            # ------- Training Phase --------- #
            print('epoch {}/{} '.format(self.epoch, self.num_epochs))
            p_bar = tqdm(total=len(train_loader))
            p_bar.set_description("Training Progress: ")

            for batch_ix, (images, labels) in enumerate(train_loader):
                # Apply Feature Attack
                if self.feature_attack_model is not None:
                    images, labels = self.feature_attack_model.attack(X=images, Y=labels)
                    self.feature_attack_model.curr_corr -= 1

                # --- Forward Pass ----
                self.metrics["num_iter"] += 1
                t_iter = time.time()

                images = images.to(device)
                labels = labels.to(device)
                outputs = self.model(images)

                # --- calculate Loss ---
                loss = self.loss_wrapper(outputs, labels)

                # --- Backward Pass ----
                self.client_optimizer.zero_grad()
                loss.backward()

                self.metrics["num_grad_steps"] += 1
                iteration_time = time.time() - t_iter
                epoch_grad_cost += iteration_time
                p_bar.update()
                # Note: No Optimizer Step yet.
                self.metrics["num_of_communication"] += 1
                g_i = flatten_grads(learner=self.model)

                # ----- Populate Server Jacobian -----
                if self.G is None:
                    d = len(g_i)
                    print("Num of Parameters {}".format(d))
                    self.metrics["num_param"] = d
                    self.G = np.zeros((self.num_batches, d), dtype=g_i.dtype)
                ix = batch_ix % self.num_batches
                agg_ix = (batch_ix + 1) % self.num_batches
                self.G[ix, :] = g_i

                # -----------  Aggregation step / Central Server  ------------ #
                if (self.num_batches == 1) or (agg_ix == 0 and batch_ix is not 0):
                    # Apply Gradient Attack
                    if self.grad_attack_model is not None:
                        self.G = self.grad_attack_model.launch_attack(G=self.G)
                    if self.feature_attack_model is not None:
                        # Reset For next set of batches
                        self.feature_attack_model.curr_corr = self.feature_attack_model.num_corrupt

                    # ------- G Compression ------- #
                    I_k = None
                    if self.C_J is not None:
                        if self.jac_compression_config["rule"] in ['active_norm_sampling',
                                                                   'random_sampling']:
                            lr = self.client_optimizer.param_groups[0]['lr']  # Need this for Error Feedback

                            t0 = time.time()
                            I_k = self.C_J.compress(G=self.G, lr=lr)  # We need I_k to do aggregation faster
                            self.G = self.C_J.G_sparse
                            self.metrics["sparse_approx_residual"].append(self.C_J.normalized_residual)
                        else:
                            raise NotImplementedError

                    # --- Gradient Aggregation ------ #
                    agg_g = self.gar.aggregate(G=self.G, ix=I_k, axis=self.C_J.axis)

                    # Update Model Grads with aggregated g : i.e. compute \tilde(g)
                    self.client_optimizer.zero_grad()
                    dist_grads_to_model(grads=agg_g, learner=self.model)
                    self.model.to(device)
                    # Now Do an optimizer step with x_t+1 = x_t - \eta \tilde(g)
                    self.client_optimizer.step()
                    self.metrics["num_opt_steps"] += 1

                    if self.metrics["num_grad_steps"] % self.eval_freq == 0:
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
                    self.metrics["epoch_grad_cost"].append(epoch_grad_cost)

    def run_fed_train(self, config: Dict, seed):
        raise NotImplementedError
