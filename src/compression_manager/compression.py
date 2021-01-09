# Copyright (c) Anish Acharya
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License

import numpy as np
from typing import Dict, List


def get_compression_operator(compression_config: Dict):
    compression_function = compression_config.get("compression_operator", 'full')
    if compression_function == 'full':
        return Full()
    elif compression_function == 'top_k':
        k = compression_config.get('frac_coordinates_to_keep', 0.1)
        return Top(k=k)
    elif compression_function == 'rand_k':
        k = compression_config.get('frac_coordinates_to_keep', 0.1)
        return Rand(k=k)
    else:
        raise NotImplementedError


class C:
    def __init__(self):
        pass

    def compress(self, g: np.ndarray) -> np.ndarray:
        pass


class Full(C):
    def __init__(self):
        C.__init__(self)

    def compress(self, g: np.ndarray):
        return g


class Top(C):
    def __init__(self, k: float):
        self.k = k
        C.__init__(self)

    def compress(self, g: np.ndarray) -> np.ndarray:
        compressed_g = np.zeros_like(g)
        num_coordinates_to_keep = round(self.k * len(g))
        indices = np.argsort(np.abs(g))[::-1][:num_coordinates_to_keep]
        compressed_g[indices] = g[indices]

        return compressed_g


class Rand(C):
    def __init__(self, k: float):
        self.k = k
        C.__init__(self)

    def compress(self, g: np.ndarray) -> np.ndarray:
        compressed_g = np.zeros_like(g)
        num_coordinates_to_keep = round(self.k * len(g))
        indices = np.random.choice(a=np.arange(len(g)),
                                   size=num_coordinates_to_keep)
        compressed_g[indices] = g[indices]

        return compressed_g
