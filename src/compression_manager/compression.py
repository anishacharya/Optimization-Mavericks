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
        q = compression_config.get('bits', 2)
        return Qsgd(q=q)
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


class Qsgd(C):
    def __init__(self, q: int):
        self.q = q
        C.__init__(self)

    def compress(self, g: np.ndarray) -> np.ndarray:
        s = 2**self.q
        #levels = np.arange(s+1)/s

        #compressed_g = np.zeros_like(g)
        g_norm = np.linalg.norm(g)

        if g_norm == 0:
           return np.zeros_like(g).astype(np.float16) #compressed_g

        #g_sign = np.sign(g)
        #g_val = np.abs(g)
        g_levels = np.floor((np.abs(g)/g_norm)*s) #.astype(np.float16)
        #g_probs = ((g_val/g_norm)*s - g_levels)
        g_probs = (np.abs(g)/g_norm)*s - g_levels #np.floor((np.abs(g)/g_norm)*s)

        zeta = np.random.binomial(1, 1.0 - g_probs, len(g)).astype(np.float16)
        val = (zeta*(g_levels/s) + (1.0-zeta)*((g_levels+1)/s)).astype(np.float16)
        #compressed_g = g_norm*np.sign(g)*val
        return (g_norm*np.sign(g)*val) #.astype(np.float16)


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
