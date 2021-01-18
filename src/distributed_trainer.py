from src.model_manager import (get_model,
                               get_optimizer,
                               get_scheduler,
                               dist_grads_to_model,
                               flatten_grads,
                               get_loss)
from src.data_manager import process_data
from src.aggregation_manager import get_gar

import torch
import numpy as n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_and_test_model(model, criterion, optimizer, lrs,
                         gar, attack_config,
                         train_data, test_data,
                         train_config, metrics):
    pass


def evaluate_classifier(model, data_loader, verbose=False):
    model.to(device)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        if verbose:
            print('Test Accuracy: {} %'.format(acc))
        return 100 - acc


def run_batch_train(config, metrics):
    # ------------------------ Fetch configs ----------------------- #
    pipeline = config.get('pipeline', 'default')
    data_config = config["data_config"]
    training_config = config["training_config"]

    learner_config = training_config["learner_config"]
    optimizer_config = training_config.get("optimizer_config", {})
    lrs_config = training_config.get('lrs_config')

    aggregation_config = training_config["aggregation_config"]
    compression_config = training_config["compression_config"]

    # ------------------------- Initializations --------------------- #
    model = get_model(learner_config=learner_config, data_config=data_config)
    optimizer = get_optimizer(params=model.parameters(), optimizer_config=optimizer_config)
    lrs = get_scheduler(optimizer=optimizer, lrs_config=lrs_config)
    criterion = get_loss(loss=optimizer_config.get('loss', 'ce'))
    gar = get_gar(aggregation_config=aggregation_config)

    # ------------------------- get data --------------------- #
    data_manager = process_data(data_config=data_config)
    train_dataset, test_dataset = data_manager.download_data()


