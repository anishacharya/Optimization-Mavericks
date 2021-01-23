# Copyright (c) Anish Acharya.
# Licensed under the MIT License
import numpy as np
from .base import GAR
from scipy import stats
"""
Computes Trimmed mean estimates
Cite: Yin, Chen, Ramchandran, Bartlett : Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates 
"""


class TrimmedMean(GAR):
    def __init__(self, aggregation_config):
        GAR.__init__(self, aggregation_config=aggregation_config)
        self.trimmed_mean_config = aggregation_config.get('trimmed_mean_config', {})
        self.proportion = self.trimmed_mean_config.get('proportion', 0.1)

    def aggregate(self, G: np.ndarray) -> np.ndarray:
        agg_grad = stats.trim_mean(a=G, proportiontocut=self.proportion, axis=0)
        return agg_grad
