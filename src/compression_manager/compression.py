# Copyright (c) Anish Acharya
# Licensed under the MIT License

import numpy as np
from typing import Dict


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
    elif compression_function == 'qsgd':
        b = compression_config.get('bits', 2)
        return QSGD(b=b)
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


class QSGD(C):
    def __init__(self, b: float):
        self.b = b
        C.__init__(self)

    def compress(self, g: np.ndarray) -> np.ndarray:
        q = np.zeros_like(g)
        bits = self.b
        s = 2 ** bits
        tau = 1 + min((np.sqrt(q.shape[0]) / s), (q.shape[0] / (s ** 2)))
        for i in range(0, q.shape[1]):
            unif_i = np.random.rand(q.shape[0], )
            x_i = g[:, i]
            q[:, i] = ((np.sign(x_i) * np.linalg.norm(x_i)) / (s * tau)) * \
                      np.floor((s * np.abs(x_i) / np.linalg.norm(x_i)) + unif_i)
        return q


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
