# Copyright (c) Anish Acharya.
# Licensed under the MIT License

"""
g = mean(g_i) Regular Mini-Batch SGD
"""
from .base import GAR
import numpy as np
from typing import List


class Mean(GAR):

    def __init__(self, aggregation_config):
        GAR.__init__(self, aggregation_config=aggregation_config)

    def aggregate(self, G: np.ndarray, ix: List[int] = None) -> np.ndarray:
        # if ix given only aggregate along the indexes ignoring the rest of the ix
        g_agg = np.zeros_like(G[0, :])
        if ix is not None:
            G = G[:, ix]

        g_agg[:, ix] = self.weighted_average(stacked_grad=G)
        return g_agg
