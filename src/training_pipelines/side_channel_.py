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


class GradientCodingPipeline(TrainPipeline):
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
        while self.epoch < self.num_epochs:
            self.model.to(device)
            self.model.train()
            epoch_grad_cost = 0
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
                self.client_optimizer.zero_grad()
                loss.backward()
                self.metrics["num_grad_steps"] += 1
                iteration_time = time.time() - t_iter
                epoch_grad_cost += iteration_time
                p_bar.update()

                # Do things with the gradient
                # Note: No Optimizer Step yet.
                g_i = flatten_grads(learner=self.model)

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

        # Training Finished #
        # Update Total Complexities
        self.metrics["total_cost"] = sum(self.metrics["epoch_grad_cost"])

    def run_batch_train(self, config: Dict, seed):
        raise NotImplementedError

    def run_fed_train(self, config: Dict, seed):
        raise NotImplementedError