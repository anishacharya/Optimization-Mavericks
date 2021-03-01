# Copyright (c) Anish Acharya.
# Licensed under the MIT License

"""
Here we explore different co-ordinate sampling ideas.
Essentially, the idea is to instead of taking all the coordinates sub-sample
pre-defined k subset of co-ordinates and aggregate only along these directions.

--- This can be thought of a column sampling of matrix G where each row
corresponds to g_i i.e. gradient vector computed on batch / client i
"""

import numpy as np
from scipy import stats


class SparseApproxMatrix:
    def __init__(self, conf):
        self.conf = conf
        self.sampling_rule = self.conf.get('rule', None)  # sampling algo
        axis = self.conf.get('axis', 'column')      # 0: column sampling, 1: row sampling
        self.axis = 0 if axis == 'column' else 1
        self.frac = conf.get('frac_coordinates', 1)  # number of coordinates to sample
        self.k = None

        # error feedback
        self.ef = conf.get('ef_server', False)
        self.residual_error = 0

    def sparse_approx(self, G: np.ndarray, lr=1) -> [np.ndarray, np.ndarray]:
        if self.sampling_rule not in ['active_norm', 'random']:
            return G

        print('Applying Sparse Column Selection')
        #####################################################
        # Otherwise do Block Selection with memory feedback #
        #####################################################
        n, d = G.shape
        G_sparse = np.zeros_like(G)

        # for the first run compute k and residual error
        if self.k is None:
            if self.frac > 0:
                self.k = int(self.frac * d)
                self.residual_error = np.zeros((n, d), dtype=G[0, :].dtype)
            elif self.frac == 0:
                self.k = 1
            else:
                raise ValueError

            print('Sampling {} coordinates'.format(self.k))

        # Error Compensation (if ef is False, residual error = 0 as its not updated
        G = (lr * G) + self.residual_error

        # --------------------------------- #
        # Invoke Sampling algorithm here
        # --------------------------------- #
        if self.sampling_rule == 'active_norm':
            I_k = self._active_norm_sampling(G=G)
        elif self.sampling_rule == 'random':
            I_k = self._random_sampling(d=d if self.axis == 0 else n)
        else:
            raise NotImplementedError

        G_sparse[:, I_k] = G[:, I_k]

        if self.ef is True:
            # print('EF')
            # update residual error
            self.residual_error = G - G_sparse
            G_sparse /= lr

        return G_sparse, I_k

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
        # Exact Implementation ~ O(d log k)
        norm_dist = G.sum(axis=self.axis)
        norm_dist = np.square(norm_dist)
        sorted_ix = np.argsort(norm_dist)[::-1]
        I_k = sorted_ix[:self.k]

        # Probabilistic Implementation ~ O(d)
        # norm_dist = np.linalg.norm(G, axis=self.axis)
        # norm_dist /= norm_dist.sum()
        # all_ix = np.arange(G.shape[1])
        # top_k = np.random.choice(a=all_ix, size=self.k, replace=False, p=norm_dist)
        # G_sparse = G_sparse[indices, :]

        return I_k
