# Copyright (c) Anish Acharya.
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License

from typing import Dict
from src.agents import FedClient
import numpy as np
from typing import List
from torch.utils.data import DataLoader, Subset
from src.model_manager import cycle


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
    def _iid_dist(clients: List[FedClient], num_train: int) -> Dict:
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

    def distribute_data(self, train_dataset, clients: List[FedClient]):
        """ Distributes Data among clients """
        # Populate Data Distribution Map
        total_train_samples = train_dataset.data.shape[0]
        data_distribution_strategy = self.data_config.get("data_distribution_strategy", 'iid')
        if data_distribution_strategy == 'iid':
            data_dist_map = self._iid_dist(clients=clients,
                                           num_train=total_train_samples)
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


