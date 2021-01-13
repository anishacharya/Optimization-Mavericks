# Copyright (c) Anish Acharya.
# Licensed under the MIT License

from src.model_manager import (get_model,
                               get_optimizer,
                               get_scheduler,
                               get_loss)
from src.data_manager import process_data
from src.aggregation_manager import get_gar
from src.agents import FedServer, FedClient
from src.compression_manager import get_compression_operator

import torch
from typing import List, Dict
import copy
import random
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_and_train_clients(server: FedServer,
                           clients: List[FedClient],
                           metrics,
                           num_local_steps: int = 1,
                           device_participation: float = 1.0):



    w_current = server.w_current
    w_old = server.w_old

    epoch_loss = 0
    for client in clients:
        # initialize with global params
        client.initialize_params(w_current=w_current, w_old=w_old)
        # train step
        epoch_loss += client.train_step(num_steps=num_local_steps, device=device)

    epoch_loss /= len(sampled_clients)
    metrics["epoch_loss"].append(epoch_loss)

    # At this point we have all the g_i computed
    # Apply Attack (Here since we can also apply co-ordinated attack)
    # TODO: Incorporate attacks


def train_and_test_model(server: FedServer,
                         clients: List[FedClient],
                         training_config: Dict,
                         data_config: Dict,
                         metrics):
    print('# ------------------------------------------------- #')
    print('#          Getting and Distributing Data            #')
    print('# ------------------------------------------------- #')
    # Get Data
    data_manager = process_data(data_config=data_config)
    train_dataset, test_dataset = data_manager.download_data()

    # Distribute Data among clients
    data_manager.distribute_data(train_dataset=train_dataset, clients=clients)

    print('# ------------------------------------------------- #')
    print('#            Launching Federated Training           #')
    print('# ------------------------------------------------- #')
    device_participation = training_config.get('client_fraction', 1.0)  # partial device participation
    global_epochs = training_config.get('global_epochs', 10)
    local_epochs = training_config.get('local_epochs', 1)

    for comm_round in range(1, global_epochs + 1):
        print(' ------------------------------------------ ')
        print('         Communication Round {}             '.format(comm_round))
        print(' -------------------------------------------')
        # Sample Participating Devices
        num_devices = math.floor(len(clients) * device_participation)
        sampled_clients = random.sample(population=clients, k=num_devices)

        init_and_train_clients(server=server, clients=sampled_clients,
                               metrics=metrics,
                               num_local_steps=local_epochs,
                               device_participation=device_participation)

        # Aggregate client grads and update server model
        server.update_step(clients=sampled_clients)
        # test(model=server.learner, )

def test(model, test_loader, verbose=False) -> float:
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
        return acc


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

    print('# ------------------------------------------------- #')
    print('#               Initializing Network                #')
    print('# ------------------------------------------------- #')
    # **** Set up Server (Master Node) ****
    # --------------------------------------
    # Implementation of Dual Optimization :
    # CIte: Reddi et.al. Adaptive Federated Optimization
    server_model = copy.deepcopy(model)
    server_opt = get_optimizer(params=server_model.parameters(),
                               optimizer_config=optimizer_config.get("server_optimizer_config", {}))
    server_lrs = get_scheduler(optimizer=server_opt, lrs_config=lrs_config.get("server_optimizer_config", {}))
    server = FedServer(server_model=server_model,
                       server_optimizer=server_opt,
                       server_lrs=server_lrs,
                       gar=gar)
    # *** Set up Client Nodes ****
    # -----------------------------
    clients = []
    n = training_config.get('num_clients', 10)

    for client_id in range(n):
        client = FedClient(client_id=client_id,
                           learner=copy.deepcopy(model),
                           compression=get_compression_operator(compression_config=compression_config))
        client.optimizer = get_optimizer(params=client.learner.parameters(), optimizer_config=optimizer_config)
        client.lrs = get_scheduler(optimizer=client.optimizer, lrs_config=lrs_config)
        client.criterion = get_loss(loss=optimizer_config.get('loss', 'ce'))
        client.training_config = training_config

        clients.append(client)

    train_and_test_model(server=server, clients=clients,
                         data_config=data_config,
                         training_config=training_config,
                         metrics=metrics)
    return metrics
