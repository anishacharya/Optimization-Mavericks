from .jacobian_compression import SparseApproxMatrix
from .vector_compression import (Full,
                                 Top,
                                 Rand,
                                 Q)
from typing import Dict


def get_compression_operator(compression_config: Dict):
    compression_function = compression_config.get("rule", 'full')
    if compression_function == 'full':
        return Full(conf=compression_config)
    if compression_function == 'top':
        return Top(conf=compression_config)
    if compression_function == 'rand':
        return Rand(conf=compression_config)
    if compression_function == 'Q':
        return Q(conf=compression_config)
    if compression_function in ['active_norm_sampling',
                                'random_sampling']:
        return SparseApproxMatrix(conf=compression_config)
    return None
