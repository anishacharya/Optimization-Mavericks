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
    elif compression_function == 'top_k':
        return Top(conf=compression_config)
    elif compression_function == 'rand_k':
        return Rand(conf=compression_config)
    elif compression_function == 'quantize':
        return Q(conf=compression_config)
    elif compression_function in ['active_norm_sampling',
                                  'random_sampling']:
        return SparseApproxMatrix(conf=compression_config)
    else:
        print('No compression specified or compression operator Not '
              'implemented ~ running without jacobian compression')
        return None