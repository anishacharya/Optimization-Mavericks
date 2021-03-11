# Copyright (c) Anish Acharya.
# Licensed under the MIT License

import numpy as np
import torch
from typing import List


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GAR:
    """
    This is the base class for all the implemented GAR
    """

    def __init__(self, aggregation_config):
        self.aggregation_config = aggregation_config
        self.current_losses = []

    def aggregate(self, G: np.ndarray, ix: List[int] = None) -> np.ndarray:
        """
        G: Gradient Matrix where each row is a gradient vector (g_i)
        ix: Columns specified to be aggregated on (if None done on full dimension)
        """
        pass

    @staticmethod
    def weighted_average(stacked_grad: np.ndarray, alphas=None):
        """
        Implements weighted average of grad vectors stacked along rows of G
        If no weights are supplied then its equivalent to simple average
        """
        n, d = stacked_grad.shape  # n is treated as num grad vectors to aggregate, d is grad dim
        if alphas is None:
            # make alpha uniform
            alphas = [1.0 / n] * n
        else:
            assert len(alphas) == n

        agg_grad = np.zeros_like(stacked_grad[0, :])

        for ix in range(0, n):
            agg_grad += alphas[ix] * stacked_grad[ix, :]
        return agg_grad
