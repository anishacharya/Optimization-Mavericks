# Copyright (c) Anish Acharya.
# Licensed under the MIT License

"""
This is Base Class for hem Jacobian Compression.
--- This can be thought of compression of matrix G where each row
corresponds to g_i i.e. gradient vector computed on batch / client i
"""

import numpy as np


class JacobianCompression:
    def __init__(self, conf):
        self.conf = conf
        self.compression_rule = self.conf.get('rule', None)
        self.memory_algo = self.conf.get('memory_algo', None)
        axis = self.conf.get('axis', 'dim')  # 0: column / dimension , 1: row / samples(clients)
        if axis == 'dim':
            self.axis = 0
        elif axis == 'n':
            self.axis = 1
        else:
            raise ValueError

        self.n= None
        self.d = None
        self.G_sparse = None

        self.residual_error = None
        self.normalized_residual = 0

    def compress(self, G: np.ndarray, lr=1) -> [np.ndarray, np.ndarray]:
        raise NotImplementedError("This method needs to be implemented for each Compression Algorithm")

    def memory_feedback(self, G: np.ndarray, lr=1) -> np.ndarray:
        """ Chosen Form of memory is added to Jacobian as feedback """
        if not self.memory_algo:
            return G
        elif self.memory_algo == 'ef':
            if self.residual_error is None:
                self.residual_error = np.zeros((self.n, self.d), dtype=G[0, :].dtype)
            return (lr * G) + self.residual_error
        else:
            raise NotImplementedError

    def memory_update(self, G: np.ndarray, lr=1):
        """ update the memory vector """
        if not self.memory_algo:
            return
        elif self.memory_algo == 'ef':
            delta = G - self.G_sparse
            memory = np.mean(delta, axis=0)
            self.residual_error = np.tile(memory, (G.shape[0], 1))
        else:
            raise NotImplementedError
        self.G_sparse /= lr


class GradientCompression:
    def __init__(self, conf):
        self.conf = conf
        self.compression_rule = self.conf.get('rule', None)
        self.compressed_g = None

        # Memory
        # self.mg = self.conf.get('mg', False)
        # print('Gradient memory: {}'.format(self.mg))
        self.memory_algo = self.conf.get('memory_algo', None)

        self.residual_error = None
        self.normalized_residual = None

        self.stale_grad = None

    def memory_feedback(self, g: np.ndarray, lr=1) -> np.ndarray:
        """ Chosen Form of memory is added to grads as feedback """
        if not self.memory_algo:
            return g
        elif self.memory_algo == 'ef':
            if self.residual_error is None:
                self.residual_error = np.zeros_like(g)
            return (lr * g) + self.residual_error
        elif self.memory_algo == 'sf':
            if self.stale_grad is None:
                self.stale_grad = np.zeros_like(g)
            return (lr * g) + self.stale_grad
        else:
            raise NotImplementedError

    def memory_update(self, g, lr):
        """ update the memory vector """
        if not self.memory_algo:
            return
        elif self.memory_algo == 'ef':
            self.residual_error = g - self.compressed_g
        elif self.memory_algo == 'sf':
            self.stale_grad = g
        else:
            raise NotImplementedError
        self.compressed_g /= lr

    def compress(self, g: np.ndarray, lr=1) -> np.ndarray:
        pass
