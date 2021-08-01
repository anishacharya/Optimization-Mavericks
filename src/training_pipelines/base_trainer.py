from typing import Dict
import numpy as np
import torch
import math
from src.attack_manager import get_feature_attack, get_grad_attack
from src.model_manager import (get_model,
                               get_optimizer,
                               get_scheduler,
                               get_loss)
from src.aggregation_manager import get_gar
from src.compression_manager import get_compression_operator, get_sampling_scheduler


class TrainPipeline:
    def __init__(self, config, seed):
        # ------------------------ Fetch configs ----------------------- #
        print('---- Fetching configs and Initializing stuff -----')
        self.config = config

        self.data_config = config["data_config"]
        self.training_config = config["training_config"]

        self.train_batch_size = self.data_config.get('train_batch_size')
        self.num_batches = self.training_config.get('num_clients', 1)
        self.num_epochs = self.training_config.get('global_epochs', 10)
        self.eval_freq = self.training_config.get('eval_freq', 10)

        self.learner_config = self.training_config["learner_config"]
        self.optimizer_config = self.training_config.get("optimizer_config", {})
        self.client_optimizer_config = self.optimizer_config.get("client_optimizer_config", {})
        self.client_lrs_config = self.optimizer_config.get('client_lrs_config')

        self.aggregation_config = self.training_config["aggregation_config"]
        self.compression_config = self.aggregation_config.get("compression_config", {})

        self.grad_attack_config = self.aggregation_config.get("grad_attack_config", {})
        self.feature_attack_config = self.data_config.get("feature_attack_config", {})

        # ------------------------ initializations ----------------------- #
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

        self.loss_sampling = self.client_optimizer_config.get('loss_sampling', None)
        self.initial_loss_sampling_fraction = self.client_optimizer_config.get('initial_loss_sampling_fraction', 1)
        self.criterion = get_loss(loss=self.client_optimizer_config.get('loss', 'ce'))

        # Compression Operator
        self.C = get_compression_operator(compression_config=self.compression_config)
        self.gar = get_gar(aggregation_config=self.aggregation_config)
        # For Jacobian
        self.G = None
        # for adversarial - get attack model
        self.feature_attack_model = get_feature_attack(attack_config=self.feature_attack_config)
        self.grad_attack_model = get_grad_attack(attack_config=self.grad_attack_config)

    def loss_wrapper(self, outputs, labels, evaluate=False):
        """
        Implementation of Different Loss Modifications ~
        See Loss Pipeline for usage example
        """
        loss = self.criterion(outputs, labels)  # per sample loss
        batch_loss = torch.mean(loss)   # loss over all samples
        if self.loss_sampling is None or evaluate:
            return batch_loss
        else:
            # Do Loss Sampling only during Training
            k = math.ceil(self.initial_loss_sampling_fraction * len(loss))
            if self.loss_sampling == 'top_loss':
                # Implements : Ordered SGD: A New Stochastic Optimization Framework for Empirical Risk Minimization
                # Kawaguchi, Kenji and Lu, Haihao; AISTATS 2020
                top_k_loss, top_k_ix = torch.topk(loss, k, sorted=False)
                return torch.mean(top_k_loss)
            elif self.loss_sampling == 'prob_top_loss':
                # Probabilistic Sampling
                prob_loss = (loss / torch.sum(loss)).tolist()
                wts = torch.tensor([1 / pi for pi in prob_loss], dtype=loss.dtype)
                weighted_loss = loss * wts

                top_k_ix = np.random.choice(len(loss), k, p=prob_loss)
                top_k_loss = weighted_loss[top_k_ix]
                return torch.mean(top_k_loss)
            else:
                raise NotImplementedError

    def evaluate_classifier(self,
                            epoch: int,
                            num_epochs: int,
                            model,
                            train_loader,
                            test_loader,
                            metrics,
                            device="cpu") -> float:
        train_error, train_acc, train_loss = self._evaluate(model=model, data_loader=train_loader, device=device)
        test_error, test_acc, _ = self._evaluate(model=model, data_loader=test_loader, device=device)

        if test_acc > metrics["best_test_acc"]:
            metrics["best_test_acc"] = test_acc
        print('Epoch progress: {}/{}, train loss = {}, train acc = {}, test acc = {}, best acc ={}'.
              format(epoch, num_epochs, train_loss, train_acc, test_acc, metrics["best_test_acc"]))

        metrics["test_error"].append(test_error)
        metrics["test_acc"].append(test_acc)
        metrics["train_error"].append(train_error)
        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc)

        return train_loss

    def _evaluate(self, model, data_loader, verbose=False, device="cpu"):
        model.to(device)
        with torch.no_grad():
            correct = 0
            total = 0
            total_loss = 0
            batches = 0

            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)

                # if criterion is not None:
                loss = self.loss_wrapper(outputs, labels, evaluate=True)
                total_loss += loss.item()

                batches += 1
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            acc = 100 * correct / total
            total_loss /= batches
            if verbose:
                print('Accuracy: {} %'.format(acc))
            return 100 - acc, acc, total_loss

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
                   "best_test_acc": 0,

                   "gradient_residual": [],
                   "jacobian_residual": [],
                   # # Grad Matrix Stats
                   # "frac_mass_retained": [],
                   # "grad_norm_dist": [],
                   # "norm_bins": None,
                   # "mass_bins": None,
                   # "max_norm": 0,
                   # "min_norm": 1e6,

                   # compute Time stats per epoch
                   "epoch_compression_cost": [],
                   "epoch_grad_cost": [],
                   "epoch_agg_cost": [],
                   "epoch_gm_iter": [],

                   # Total Costs
                   "total_cost": 0,
                   "total_grad_cost": 0,
                   "total_agg_cost": 0,
                   "total_compression_cost": 0,

                   "total_gm_iter": 0,
                   "avg_gm_cost": 0,

                   "num_iter": 0,
                   "num_opt_steps": 0,
                   "num_grad_steps": -1,
                   }
        return metrics

    def run_train(self, config: Dict, seed):
        raise NotImplementedError("This method needs to be implemented for each pipeline")

    def run_batch_train(self, config: Dict, seed):
        raise NotImplementedError("This method needs to be implemented for each pipeline")

    def run_fed_train(self, config: Dict, seed):
        raise NotImplementedError("This method needs to be implemented for each pipeline")
