# Copyright (c) Anish Acharya.
# Licensed under the MIT License

"""
g = mean(g_i) Regular Mini-Batch SGD
"""
from .base import GAR
import numpy as np
from typing import List
import time


class Mean(GAR):

    def __init__(self, aggregation_config):
        GAR.__init__(self, aggregation_config=aggregation_config)

    def aggregate(self, G: np.ndarray, ix: List[int] = None) -> np.ndarray:
        # if ix given only aggregate along the indexes ignoring the rest of the ix
        if ix is not None:
            g_agg = np.zeros_like(G[0, :])
            G = G[:, ix]
            t0 = time.time()
            low_rank_mean = self.weighted_average(stacked_grad=G)
            g_agg[ix] = low_rank_mean
            self.agg_time = time.time() - t0
            return g_agg
        else:
            return self.weighted_average(stacked_grad=G)
