# Copyright (c) Anish Acharya.
# Licensed under the MIT License
import numpy as np
from .base import GAR
from typing import List, Dict
from scipy.spatial.distance import cdist, euclidean
from src.compression_manager import SparseApproxMatrix


class CoordinateWiseMedian(GAR):
    def __init__(self, aggregation_config):
        GAR.__init__(self, aggregation_config=aggregation_config)

    def aggregate(self, G: np.ndarray, ix: List[int] = None) -> np.ndarray:
        if ix is not None:
            g_agg = np.zeros_like(G[0, :])
            G = G[:, ix]
            low_rank_med = np.median(G, axis=0)
            g_agg[ix] = low_rank_med
            return g_agg
        else:
            return np.median(G, axis=0)


class GeometricMedian(GAR):
    def __init__(self, aggregation_config):
        GAR.__init__(self, aggregation_config=aggregation_config)
        self.geo_med_config = aggregation_config.get('geo_med_config', {})
        self.geo_med_alg = self.geo_med_config.get('geo_med_alg', 'weiszfeld')
        print("GM Algorithm: {}".format(self.geo_med_alg))

    def get_gm(self, X: np.ndarray):
        if self.geo_med_alg == 'vardi':
            return vardi(X=X)
        elif self.geo_med_alg == 'weiszfeld':
            return weiszfeld(X=X)
        else:
            raise NotImplementedError

    def aggregate(self, G: np.ndarray, ix: List[int] = None) -> np.ndarray:
        # if ix given only aggregate along the indexes ignoring the rest of the ix
        if ix is not None:
            g_agg = np.zeros_like(G[0, :])
            G = G[:, ix]
            low_rank_gm = self.get_gm(X=G)
            g_agg[ix] = low_rank_gm
            return g_agg
        else:
            return self.get_gm(X=G)


def weiszfeld(X, eps=1e-5, max_iter=1000):
    # inspired by: https://github.com/mrwojo
    """
    Implements: On the point for which the sum of the distances to n given points is minimum
    E Weiszfeld, F Plastria: Annals of Operations Research
    """
    # initial Guess : centroid / empirical mean
    mu = np.mean(X, 0)
    num_iter = 0
    while num_iter < max_iter:
        # noinspection PyTypeChecker
        distances = cdist(X, [mu]).astype(mu.dtype)
        distances = np.where(distances == 0, 1, distances)
        mu1 = (X / distances).sum(axis=0) / (1. / distances).sum(axis=0)
        guess_movement = np.sqrt(((mu - mu1) ** 2).sum())

        mu = mu1

        if guess_movement <= eps:
            break
        num_iter += 1

    return mu


def vardi(X, eps=1e-5, max_iter=15) -> np.ndarray:
    # Copyright (c) Orson Peters
    # Licensed under zlib License
    # Reference: https://stackoverflow.com/questions/30299267/geometric-median-of-multidimensional-points
    """
    Implementation of "The multivariate L1-median and associated data depth;
    Yehuda Vardi and Cun-Hui Zhang; PNAS'2000"
    """
    # Assume each data point is arranged in a row
    mu = np.mean(X, 0)
    num_iter = 0
    while True:
        # noinspection PyTypeChecker
        D = cdist(X, [mu]).astype(mu.dtype)
        # Handle divide by zero
        D = np.where(D == 0, 1, D)

        non_zeros = (D != 0)[:, 0]
        D_inv = 1 / D[non_zeros]
        W = np.divide(D_inv, sum(D_inv))
        T = np.sum(W * X[non_zeros], 0)
        num_zeros = len(X) - np.sum(non_zeros)
        if num_zeros == 0:
            mu1 = T
        elif num_zeros == len(X):
            # print('Time Taken For GM ={}'.format(time.time() - t0))
            # print('Num iter for GM {}'.format(num_iter))
            return mu
        else:
            r = np.linalg.norm((T - mu) * sum(D_inv))
            r_inv = 0 if r == 0 else num_zeros / r
            mu1 = max(0, 1 - r_inv) * T + min(1, r_inv) * mu

        if euclidean(mu, mu1) < eps or num_iter == max_iter:
            # print('Time Taken For GM {}'.format(time.time() - t0))
            # print('Num iter for GM {}'.format(num_iter))
            return mu1
        mu = mu1
        num_iter += 1


if __name__ == '__main__':
    a = np.array([[2., 3., 8.],
                  [10., 4., 3.],
                  [58., 3., 4.],
                  [34., 2., 43.]])
    print('vardi geo median: {}'.format(vardi(X=a)))
