import numpy as np
import time
from src.aggregation_manager import get_gar
import json
from numpyencoder import NumpyEncoder
import os

def time_coordinate_select(G: np.ndarray, k: int):
    t0 = time.time()
    norm_dist = np.linalg.norm(G, axis=0)
    norm_dist /= sum(norm_dist)
    all_ix = np.arange(G.shape[1])
    top_k = np.random.choice(a=all_ix, size=k, replace=False, p=norm_dist)
    G_sparse = G[:, top_k]
    T = time.time() - t0

    return G_sparse, T


def time_gar(gar, G, repeat: int = 1):
    T = 0
    for it in range(repeat):
        t0 = time.time()
        _ = gar.aggregate(G=G)
        T += time.time() - t0
    T /= repeat
    return T


if __name__ == '__main__':
    # Hyper Params
    directory = 'result_dumps/timing_exp/cont/'
    algo = 'mean'
    op_file = 'mean'

    # d = [int(1e3), int(5e3), int(1e4), int(5e4)]
    d = np.arange(start=1e3, stop=1e4, step=250)
    d = [int(el) for el in d]
    n = 5000
    f = 0.01
    k = int(f * n)

    res = {}
    agg_config = \
        {
            "gar": "mean",
            "krum_config": {"krum_frac": 0.3},
        }
    gar = get_gar(aggregation_config=agg_config)

    # Gather Times
    for dim in d:
        t = 0
        X = np.random.normal(0, 0.3, (n, dim))

        if algo == 'BGMD':
            # only for BGMD
            X_sparse, t0 = time_coordinate_select(G=X, k=k)
            t += t0
            t += time_gar(gar=gar, G=X_sparse)
        else:
            t += time_gar(gar=gar, G=X)

        res[dim] = t

        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(directory + op_file, 'w+') as f:
            json.dump(res, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)