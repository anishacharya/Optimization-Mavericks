# Copyright (c) Anish Acharya.
# Licensed under the MIT License

"""
g = mean(g_i) Regular Mini-Batch SGD
"""
from .base import GAR
import numpy as np


class Mean(GAR):

    def __init__(self, aggregation_config):
        GAR.__init__(self, aggregation_config=aggregation_config)

    def aggregate(self, G: np.ndarray) -> np.ndarray:
        agg_grad = self.weighted_average(stacked_grad=G)
        return agg_grad
