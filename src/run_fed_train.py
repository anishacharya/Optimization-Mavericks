# Copyright (c) Anish Acharya.
# Licensed under the MIT License

from src.model_manager import (get_model,
                               get_optimizer,
                               get_scheduler,
                               dist_grads_to_model,
                               flatten_grads,
                               get_loss)
from src.data_manager import process_data
from src.aggregation_manager import get_gar
from src.agents import FedServer, FedClient
from src.compression_manager import get_compression_operator

import torch
from typing import List, Dict
import copy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_and_test_model(server: FedServer, clients: List[FedClient],
                         training_config: Dict, data_config: Dict,
                         metrics):
    n_sampled = training_config.get('client_fraction', 1)

    # Get Data
    data_manager = process_data(data_config=data_config)
    train_dataset, test_dataset = data_manager.download_data()
    pass


def test(model, test_loader, verbose=False):
    model.to(device)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
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


def run_fed_train(config, metrics):
    data_config = config["data_config"]
    training_config = config["training_config"]

    learner_config = training_config["learner_config"]
    optimizer_config = training_config.get("optimizer_config", {})
    lrs_config = training_config.get('lrs_config')
    aggregation_config = training_config["aggregation_config"]
    compression_config = training_config["compression_config"]
    model = get_model(learner_config=learner_config, data_config=data_config)
    gar = get_gar(aggregation_config=aggregation_config)

    optimizer = get_optimizer(params=model.parameters(), optimizer_config=optimizer_config)
    lrs = get_scheduler(optimizer=optimizer, lrs_config=lrs_config)
    criterion = get_loss(loss=optimizer_config.get('loss', 'ce'))

    data_manager = process_data(data_config=data_config)
    train_dataset, test_dataset = data_manager.download_data()

    print('# ------------------------------------------------- #')
    print('#               Initializing Network                #')
    print('# ------------------------------------------------- #')
    # **** Set up Server (Master Node) ****
    # --------------------------------------
    server = FedServer(server_model=copy.deepcopy(model), gar=gar)
    # *** Set up Client Nodes ****
    # -----------------------------
    clients = []
    n = training_config.get('num_clients', 10)

    for client_id in range(n):
        client = FedClient(client_id=client_id,
                           learner=copy.deepcopy(model),
                           compression=get_compression_operator(compression_config=compression_config))

        clients.append(client)

    train_and_test_model(server=server, clients=clients,
                         data_config=data_config, training_config=training_config,
                         metrics=metrics)
    return metrics

