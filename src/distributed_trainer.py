from src.model_manager import (get_model,
                               get_optimizer,
                               get_scheduler,
                               dist_grads_to_model,
                               flatten_grads,
                               get_loss)
from src.data_manager import process_data
from src.aggregation_manager import get_gar

import torch
from torch.utils.data import DataLoader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_and_test_model(model, criterion, optimizer, lrs, gar,
                         train_loader, test_loader, train_config, metrics):

    num_batches = train_config.get('num_clients', 1)
    num_epochs = train_config.get('global_epochs', 10)

    for epoch in range(num_epochs):
        model.to(device)
        model.train()
        G = None
        epoch_loss = 0
        iter_loss = 0
        total_iter = len(train_loader)

        # # randomly sample byzantine nodes (batches)
        # all_batches = np.arange(total_iter)
        # mal_ix = set(np.random.choice(a=all_batches,
        #                               size=int(len(all_batches) * attack_config.get('frac_adv', 0)),
        #                               replace=False))
        # mal_batches_mask = []
        # attack_model = get_attack(attack_config=attack_config)

        for batch_ix, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            iter_loss += loss.item()

            loss.backward()
            # Note: No Optimizer Step yet.
            g_i = flatten_grads(learner=model)
            if G is None:
                d = len(g_i)
                G = np.zeros((num_batches, d), dtype=g_i.dtype)

            # Aggregation Step
            agg_ix = (batch_ix + 1) % num_batches
            G[agg_ix - 1, :] = g_i

            if agg_ix == 0:
                agg_g = gar.aggregate(G=G)
                optimizer.zero_grad()
                # Update Model Grads with aggregated g :\tilde(g)
                dist_grads_to_model(grads=agg_g, learner=model)
                model.to(device)
                # Now Do an optimizer step with x_t+1 = x_t - \eta \tilde(g)
                optimizer.step()
                iter_loss /= num_batches
                print("Epoch [{}/{}], iteration [{}/{}], Learning rate [{}], current batch Loss [{}]"
                      .format(epoch + 1, num_epochs, batch_ix, total_iter, optimizer.param_groups[0]['lr'], iter_loss))
                iter_loss = 0

        # --------
        epoch_loss /= total_iter
        print('\n --------------------------------------------------- \n')
        print("Epoch [{}/{}], Learning rate [{}], Avg Batch Loss [{}]"
              .format(epoch + 1, num_epochs, optimizer.param_groups[0]['lr'], epoch_loss))
        metrics["epoch_loss"].append(epoch_loss)

        metrics["test_error"].append(evaluate_classifier(model=model, data_loader=test_loader, verbose=True))
        if lrs is not None:
            lrs.step()


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
    batch_size = data_config.get('batch_size', 1)
    data_manager = process_data(data_config=data_config)
    train_dataset, test_dataset = data_manager.download_data()

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(train_dataset))

    train_and_test_model(model=model, criterion=criterion,optimizer=optimizer, lrs=lrs,
                         gar=gar, train_loader=train_loader, test_loader=test_loader, metrics=metrics,
                         train_config=training_config)
    return metrics
