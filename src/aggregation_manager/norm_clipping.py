# Copyright (c) Anish Acharya.
# Licensed under the MIT License

import numpy as np
from .base import GAR

""" 
Implements remove points with large norm.
 
Ghosh, Maity, Kadhe, Mazumdar, Ramchandran :
Communication-Efficient and Byzantine-Robust Distributed Learning with Error Feedback 
"""


class NormClipping(GAR):
    def __init__(self, aggregation_config):
        GAR.__init__(self, aggregation_config=aggregation_config)

    def aggregate(self, G: np.ndarray) -> np.ndarray:
        # Compute norms of each gradient vector
        norms = np.sqrt(np.einsum('ij,ij->i', G, G))

        # Find Number of top norms to drop
        norm_conf = self.aggregation_config.get("norm_clip_config", {})
        k = int(G.shape[0] * norm_conf.get("alpha", 0.1))
        print('clipping {} clients'.format(k))
        top_k_indices = np.argsort(np.abs(norms))[::-1][:k]

        # set weights of them to 0 filtering k top ones based on norm
        alphas = np.ones(G.shape[0]) * (1 / (G.shape[0] - k))
        alphas[top_k_indices] = 0
        agg_grad = self.weighted_average(stacked_grad=G, alphas=alphas)
        return agg_grad