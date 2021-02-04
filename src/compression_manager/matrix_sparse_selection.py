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
        # sampling algo
        self.sampling_rule = self.conf.get('rule', None)
        axis = self.conf.get('axis', 'column')  # 0: column sampling, 1: row sampling
        self.axis = 0 if axis == 'column' else 1
        print('axis = {}'.format(self.axis))
        # number of coordinates to sample
        self.frac = conf.get('frac_coordinates', 1)
        self.k = None

        # error feedback
        self.ef = conf.get('ef', False)
        self.residual_error = None

    def sparse_approx(self, G: np.ndarray, lr=1) -> np.ndarray:
        if self.sampling_rule not in ['active_norm', 'random']:
            return G
        n, d = G.shape
        G_sparse = np.zeros_like(G)
        # for the first run compute k
        if self.k is None:
            self.k = int(self.frac * d)
        # Initialize residual error to zero first time
        if self.residual_error is None:
            # print('Initializing Residual Error')
            self.residual_error = np.zeros((n, d), dtype=G[0, :].dtype)

        # Error Compensation (if ef is False, residual error = 0 as its not updated)
        G = (lr * G) + self.residual_error

        # --------------------------------- #
        # Invoke Sampling algorithm here
        # --------------------------------- #
        if self.sampling_rule == 'active_norm':
            # print('Applying active norm column sampling on gradients')
            I_k = self._active_norm_sampling(G=G)
        elif self.sampling_rule == 'random':
            # print('Applying random column sampling on gradients')
            I_k = self._random_sampling(d=d if self.axis == 0 else n)
        else:
            raise NotImplementedError

        if self.axis == 0:
            # column sampling
            G_sparse[:, I_k] = G[:, I_k]
        elif self.axis == 1:
            # row sampling
            G_sparse[I_k, :] = G[I_k, :]
        else:
            raise NotImplementedError
        # update residual error
        if self.ef is True:
            print('Error Feedback at Server')
            self.residual_error = G - G_sparse

        return G_sparse / lr

    # Implementation of different "Matrix Sparse Approximation" strategies
    def _random_sampling(self, d: int) -> np.ndarray:
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
        norm_dist = np.linalg.norm(G, axis=self.axis)
        norm_dist /= sum(norm_dist)

        # Probabilistic Implementation ~ O(d)
        # all_ix = np.arange(G.shape[1])
        # top_k = np.random.choice(a=all_ix, size=self.k, replace=False, p=norm_dist)

        # Probabilistic Implementation ~ O(d log k)
        sorted_ix = np.argsort(norm_dist)[::-1]
        top_k = sorted_ix[:self.k]

        return top_k
