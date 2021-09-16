# Copyright (c) Anish Acharya.
# Licensed under the MIT License

"""
Here we explore different co-ordinate sampling ideas.
Essentially, the idea is to instead of taking hem the coordinates sub-sample
pre-defined k subset of co-ordinates and aggregate only along these directions.

--- This can be thought of a column sampling of matrix G where each row
corresponds to g_i i.e. gradient vector computed on batch / client i
"""

import numpy as np
from src.compression_manager.compression_base import JacobianCompression

np.random.seed(1)


class SparseApproxMatrix(JacobianCompression):
    def __init__(self, conf):
        JacobianCompression.__init__(self, conf=conf)
        self.frac = conf.get('sampling_fraction', 1)  # fraction of ix to sample
        self.k = None  # Number of ix ~ to be auto populated

    def compress(self, G: np.ndarray, lr=1) -> np.ndarray:

        self.G_sparse = np.zeros_like(G)
        if self.compression_rule not in ['active_norm_sampling', 'random_sampling']:
            raise NotImplementedError

        G = self.memory_feedback(G=G, lr=lr)

        # for the first run compute k and residual error
        if self.k is None:
            self.n, self.d = G.shape
            if self.frac < 0:
                raise ValueError
            elif self.axis == 0:
                self.k = int(self.frac * self.d) if self.frac > 0 else 1
                print('Sampling {} coordinates out of {}'.format(self.k, self.d))
            elif self.axis == 1:
                self.k = int(self.frac * self.n) if self.frac > 0 else 1
                print('Sampling {} samples out of {}'.format(self.k, self.n))

        # Invoke Sampling algorithm
        if self.compression_rule == 'active_norm_sampling':
            I_k = self._active_norm_sampling(G=G)
        elif self.compression_rule == 'random_sampling':
            I_k = self._random_sampling(d=self.d if self.axis == 0 else self.n)
        else:
            raise NotImplementedError

        if self.axis == 0:
            self.G_sparse[:, I_k] = G[:, I_k]
        elif self.axis == 1:
            self.G_sparse[I_k, :] = G[I_k, :]
        else:
            raise ValueError

        self.memory_update(G=G, lr=lr)

        return I_k

    # Implementation of different "Matrix Sparse Approximation" strategies
    def _random_sampling(self, d) -> np.ndarray:
        """
        Implements Random (Gauss Siedel) subset Selection
        """
        all_ix = np.arange(d)
        I_k = np.random.choice(a=all_ix,
                               size=self.k,
                               replace=False)

        return I_k

    def _active_norm_sampling(self, G: np.ndarray) -> np.ndarray:
        """
        Implements Gaussian Southwell Subset Selection / Active norm sampling
        Ref: Drineas, P., Kannan, R., and Mahoney, M. W.  Fast monte carlo algorithms for matrices:
        Approximating matrix multiplication. SIAM Journal on Computing, 36(1):132–157, 2006
        """
        # Exact Implementation ~ O(d log d)
        # norm_dist = G.sum(axis=self.axis)
        # norm_dist = np.square(norm_dist)
        norm_dist = np.linalg.norm(G, axis=self.axis)
        norm_dist /= norm_dist.sum()
        sorted_ix = np.argsort(norm_dist)[::-1]

        I_k = sorted_ix[:self.k]

        mass_explained = np.sum(norm_dist[I_k])
        self.normalized_residual = mass_explained

        return I_k
