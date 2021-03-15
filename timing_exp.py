import numpy as np
import time
from src.aggregation_manager import get_gar
import json
from typing import Dict
from numpyencoder import NumpyEncoder
import os


def time_gar(grad_agg_rule, X, repeat: int = 5, sparse_approx_config=None):
    if sparse_approx_config is None:
        sparse_approx_config = {}
    T = 0
    for it in range(repeat):
        if sparse_approx_config is not {}:
            t0 = time.time()
            _ = grad_agg_rule.block_descent_aggregate(G=X, sparse_approx_config=sparse_approx_config)
        else:
            t0 = time.time()
            _ = grad_agg_rule.aggregate(G=X)
        T += time.time() - t0
    T /= repeat
    return T


if __name__ == '__main__':
    # Hyper Params
    d = np.arange(start=1e3, stop=1e4, step=200)
    d = [int(el) for el in d]
    directory = 'result_dumps/timing_exp/'

    algo = 'BD'
    op_file = 'bgmd.0.5'
    n = 10000

    res = {}
    conf = {
        "agg_config":
            {
                "gar": "geo_med",
                "trimmed_mean_config": {"proportion": 0.3},
                "krum_config": {"krum_frac": 0.3},
                "norm_clip_config": {"alpha": 0.3},
            },
        "sparse_approximation_config":
            {
                "rule": 'active_norm',
                "axis": "column",
                "frac_coordinates": 0.5,
                "ef_server": True,
            }
    }
    res["config"] = conf

    gar = get_gar(aggregation_config=conf["agg_config"])

    for dim in d:
        G = np.random.normal(0, 0.3, (n, dim))
        if algo == 'BD':
            res[dim] = time_gar(grad_agg_rule=gar, X=G, sparse_approx_config=conf["sparse_approximation_config"])
        else:
            res[dim] = time_gar(grad_agg_rule=gar, X=G)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(directory + op_file, 'w+') as f:
        json.dump(res, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
