# Copyright (c) Anish Acharya.
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License

from typing import Dict
from src.agents import FedClient
import numpy as np
from typing import List
from torch.utils.data import DataLoader, Subset
from src.model_manager import cycle
import torch

torch.manual_seed(1)


class DataManager:
    """
    Base Class for all Data Readers
    """

    def __init__(self,
                 data_config: Dict):

        self.data_config = data_config
        self.data_distribution_map = {}

    @staticmethod
    def _get_common_data_trans(_train_dataset):
        """ Implements a simple way to compute train and test transform that usually works """
        try:
            mean = [_train_dataset.data.float().mean(axis=(0, 1, 2)) / 255]
            std = [_train_dataset.data.float().std(axis=(0, 1, 2)) / 255]
        except:
            mean = _train_dataset.data.mean(axis=(0, 1, 2)) / 255
            std = _train_dataset.data.std(axis=(0, 1, 2)) / 255

        return mean, std

    def download_data(self):
        """ Downloads Data and Apply appropriate Transformations . returns train, test dataset """
        raise NotImplementedError("This method needs to be implemented")

    @staticmethod
    def _iid_sampling(clients: List[FedClient], num_train: int) -> Dict:
        """ Distribute the data iid into all the clients """
        data_distribution_map = {}
        all_indexes = np.arange(num_train)

        # split rest to clients for train
        num_clients = len(clients)
        num_samples_per_machine = num_train // num_clients

        for machine_ix in range(0, num_clients - 1):
            data_distribution_map[clients[machine_ix].client_id] = \
                set(np.random.choice(a=all_indexes, size=num_samples_per_machine, replace=False))
            all_indexes = list(set(all_indexes) - data_distribution_map[clients[machine_ix].client_id])
        # put the rest in the last machine
        data_distribution_map[clients[-1].client_id] = all_indexes

        return data_distribution_map

    @staticmethod
    def _non_iid_equal_sampling(clients: List[FedClient], labels: np.ndarray,
                                num_train: int, num_shard: int = 100) -> Dict:
        data_distribution_map = {}
        all_indexes = np.arange(num_train)
        num_clients = len(clients)

        num_image_per_shard = int(num_train / num_shard)
        num_shards_per_client = int(num_shard / num_clients)
        ix_shard = np.arange(num_shard)

        ix_labels = np.vstack((all_indexes, labels))
        ix_labels = ix_labels[:, ix_labels[1, :].argsort()]
        all_indexes = ix_labels[0, :]

        for machine_ix in range(num_clients):
            rand_set = set(np.random.choice(a=ix_shard, size=num_shards_per_client, replace=False))
            ix_shard = list(set(ix_shard) - rand_set)
            key = clients[machine_ix].client_id
            data_distribution_map[key] = []
            for shard_ix in rand_set:
                sampled_ix = list(all_indexes[shard_ix*num_image_per_shard: (shard_ix + 1)*num_image_per_shard])
                data_distribution_map[key].append(sampled_ix)

        return data_distribution_map

    def distribute_data(self, train_dataset, clients: List[FedClient]):
        """ Distributes Data among clients """
        # Populate Data Distribution Map
        total_train_samples = train_dataset.data.shape[0]
        sampler = self.data_config.get("data_sampling_strategy", 'iid')
        if sampler == 'iid':
            self.data_distribution_map = self._iid_sampling(clients=clients,
                                                            num_train=total_train_samples)
        elif sampler == 'non_iid':
            labels = train_dataset.train_labels.numpy()
            num_shards = self.data_config.get('num_shards', 100)
            self.data_distribution_map = self._non_iid_equal_sampling(clients=clients,
                                                                      labels=labels,
                                                                      num_train=total_train_samples,
                                                                      num_shard=num_shards)
        else:
            raise NotImplementedError

        # populate client data loader based on the distribution map
        for client in clients:
            local_dataset = Subset(dataset=train_dataset,
                                   indices=self.data_distribution_map[client.client_id])

            client.local_train_data = DataLoader(local_dataset.dataset,
                                                 shuffle=True,
                                                 batch_size=self.data_config.get("train_batch_size", 256),
                                                 pin_memory=True,
                                                 num_workers=self.data_config.get("train_num_workers", 1))
            client.train_iter = iter(cycle(client.local_train_data))
