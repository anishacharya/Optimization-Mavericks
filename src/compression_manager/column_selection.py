"""
Here we explore different co-ordinate sampling ideas.
Essentially, the idea is to instead of taking all the coordinates sub-sample
pre-defined k subset of co-ordinates and aggregate only along these directions.

--- This can be thought of a column sampling of matrix G where each row
corresponds to g_i i.e. gradient vector computed on batch / client i
"""
import numpy as np


class SparseApproxMatrix:
    def __init__(self, conf):
        self.conf = conf
        # sampling algo
        self.sampling_rule = self.conf.get('rule', None)
        self.axis = self.conf.get('axis', 0)    # 0: column sampling, 1: row sampling
        # number of coordinates to sample
        self.frac = conf.get('frac_coordinates', 1)
        self.k = None

        # error feedback
        self.ef = conf.get('ef', False)
        self.residual_error = None

    def sparse_approx(self, G: np.ndarray):
        n, d = G.shape
        G_sparse = np.zeros_like(G)
        # for the first run compute k
        if self.k is None:
            self.k = int(self.frac * d)
        # Initialize residual error to zero first time
        if self.residual_error is None:
            print('Initializing Residual Error')
            self.residual_error = np.zeros((n, d), dtype=G[0, :].dtype)

        # Error Compensation (if ef is False, residual error = 0 as its not updated)
        G += self.residual_error

        # Define Sampling Rule here


    # Implementation of different Sparse Approximation strategies
    def _random_sampling(self, d: int) -> np.ndarray:
        """ Implements Random (Gauss Siedel) subset Selection """
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
        norm_dist = np.square(np.linalg.norm(G, axis=self.axis))
        I_k = np.argsort(np.abs(norm_dist))[::-1][:self.k]
        return I_k
