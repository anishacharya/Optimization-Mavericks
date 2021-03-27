# coding=utf-8
from __future__ import print_function
import torch.utils.data as data
import numpy as np
import os
import torch
import pickle

dir_path = os.path.dirname(os.path.realpath(__file__))

'''
Inspired by https://github.com/pytorch/vision/pull/46
'''


class PrototypicalBatchSampler(object):
    """
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.
    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    """

    def __init__(self, labels, classes_per_it, num_samples, iterations):
        """
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be inferred from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations: number of iterations (episodes) per epoch
        """
        super(PrototypicalBatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_samples
        self.iterations = iterations

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        self.idxs = range(len(self.labels))
        self.label_tens = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.label_tens = torch.Tensor(self.label_tens)
        self.label_lens = torch.zeros_like(self.classes)
        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label)[0, 0]
            self.label_tens[label_idx, np.where(np.isnan(self.label_tens[label_idx]))[0][0]] = idx
            self.label_lens[label_idx] += 1

    def __iter__(self):
        """
        yield a batch of indexes
        """
        spc = self.sample_per_class
        cpi = self.classes_per_it

        for it in range(self.iterations):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            c_idxs = torch.randperm(len(self.classes))[:cpi]
            for i, c in enumerate(self.classes[c_idxs]):
                s = slice(i * spc, (i + 1) * spc)
                label_idx = np.argwhere(self.classes == c)[0, 0]
                sample_idxs = torch.randperm(self.label_lens[label_idx])[:spc]
                batch[s] = self.label_tens[label_idx][sample_idxs]
            batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self):
        """
        returns the number of iterations (episodes) per epoch
        """
        return self.iterations


class MiniImagenetDataset(data.Dataset):
    def __init__(self, mode='train', root=dir_path + '/data/mini_imagenet', transform=None, target_transform=None):
        """
        The items are (filename,category). The index of all the categories can be found in self.idx_classes
        Args:
        - root: the directory where the dataset will be stored
        - transform: how to transform the input
        - target_transform: how to transform the target
        - download: need to download the dataset
        """
        super(MiniImagenetDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if not self._check_exists():
            raise RuntimeError(
                'Dataset not found. Follow instructions to download mini-imagenet.')

        pickle_file = os.path.join(self.root, 'mini-imagenet-cache-' + mode + '.pkl')
        f = open(pickle_file, 'rb')
        self.data = pickle.load(f)

        self.x = [np.transpose(x, (2, 0, 1)) for x in self.data['image_data']]
        self.x = [torch.FloatTensor(x) for x in self.x]
        self.y = [-1 for _ in range(len(self.x))]
        class_idx = index_classes(self.data['class_dict'].keys())
        for class_name, idxs in self.data['class_dict'].items():
            for idx in idxs:
                self.y[idx] = class_idx[class_name]

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return os.path.exists(self.root)


def index_classes(items):
    idx = {}
    for i in items:
        if i not in idx:
            idx[i] = len(idx)
    print("== Dataset: Found %d classes" % len(idx))
    return idx
