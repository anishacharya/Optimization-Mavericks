# Copyright (c) Anish Acharya.
# Licensed under the MIT License

"""
This is Base Class for all Jacobian Compression.
--- This can be thought of compression of matrix G where each row
corresponds to g_i i.e. gradient vector computed on batch / client i
"""

import numpy as np


class JacobianCompression:
    def __init__(self, conf):
        self.conf = conf
        self.compression_rule = self.conf.get('rule', None)

        # Get Row / Column
        axis = self.conf.get('axis', 'dim')  # 0: column / dimension , 1: row / samples(clients)
        if axis == 'dim':
            self.axis = 0
        elif axis == 'n':
            self.axis = 1
        else:
            raise ValueError

        # Memory
        self.mG = conf.get('mG', False)
        print('Jacobian Error Feedback is: {}'.format(self.mG))
        self.mg = conf.get('mg', False)
        print('Gradient Error Feedback is: {}'.format(self.mg))
        self.residual_error = 0
        self.normalized_residual = 0

    def compress(self, G: np.ndarray, lr=1) -> [np.ndarray, np.ndarray]:
        raise NotImplementedError("This method needs to be implemented for each Compression Algorithm")
