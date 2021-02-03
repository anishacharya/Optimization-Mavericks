import numpy as np
import time
from src.aggregation_manager import get_gar
import json
from numpyencoder import NumpyEncoder


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
    op_file = 'result_dumps/timing_exp/gm'

    d = [int(1e3), int(5e3), int(1e4), int(5e4)]
    n = 5000

    k = int(0.01 * n)

    res = {}
    agg_config = \
        {
            "gar": "gm",
            "krum_config": {"krum_frac": 0.3},
        }
    gar = get_gar(aggregation_config=agg_config)

    # Gather Times
    for dim in d:
        t = 0
        X = np.random.normal(0, 0.3, (n, dim))

        # only for BGMD
        # X_sparse, t0 = time_coordinate_select(G=X, k=k)
        # t += t0

        t += time_gar(gar=gar, G=X)
        res[dim] = t

        with open(op_file, 'w+') as f:
            json.dump(res, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)