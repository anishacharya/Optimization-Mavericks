import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker
import json
from typing import List, Dict
from src.aggregation_manager import get_gar
import time
from numpyencoder import NumpyEncoder
from src.compression_manager import SparseApproxMatrix


def plot_driver(label: str, res_file: str, plt_type: str = 'epoch_loss',
                line_width=2, marker=None, line_style=None, optima: float = 0.0,
                sampling_freq: int = 1):
    with open(res_file, 'rb') as f:
        result = json.load(f)

    res = result[plt_type]
    # res = res[0::sampling_freq]
    res -= optima * np.ones(len(res))

    x = np.arange(len(res)) + np.ones(len(res)) * sampling_freq
    plt.plot(x, res, label=label, linewidth=line_width, marker=marker, linestyle=line_style)


def plot_timing(res_file: str, label, line_width=2, marker=None, line_style=None):
    # d = [100, 1000, 10000, 100000]
    with open(res_file, 'rb') as f:
        res = json.load(f)
    d = list(res.keys())
    t = list(res.values())
    ax = plt.gca()

    plt.yscale('log')
    #ax.set_xscale('log')
    plt.scatter(d, t, label=label)  #, linewidth=line_width, marker=marker, linestyle=line_style)
    plt.plot(d, t, linestyle='dashed')



def plot_metrics():
    # -------------------------------
    # ** Usually No Need to Modify **
    # -------------------------------
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # -------------------------------------------------------------------------------------------
    # ------------------------------- Modify Here -----------------------------------------------
    d = 'result_dumps/timing_exp/'
    o = [
        'mean',
        'geo_med',
        'geo_med.norm_0.1'

    ]
    labels = [
        'Mean',
        'GM',
        '10-sGM'
              ]

    plot_type = 'timing'
    sampling_freq = 5

    for op, label in zip(o, labels):
        result_file = d + op
        if plot_type is 'timing':
            plot_timing(label=label, res_file=result_file, line_width=4)
        else:
            plot_driver(label=label, res_file=result_file,
                        plt_type=plot_type, optima=0, line_width=4,
                        sampling_freq=sampling_freq)
    # -------------------------------------------------------------------------------------------
    # -------------------------------
    # ** Usually No Need to Modify **
    # -------------------------------
    plt.grid()
    plt.tick_params(labelsize=10)

    if plot_type is 'test_error':
        plt.ylabel('Test Error', fontsize=10)
        plt.xlabel('Communication Rounds', fontsize=10)
        # plt.ylim(bottom=70, top=100)
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(100))

    elif plot_type is 'test_acc':
        plt.ylabel('Test Accuracy', fontsize=10)
        plt.xlabel('Communication Rounds', fontsize=10)
        plt.ylim(bottom=80, top=90)

    elif plot_type is 'train_acc':
        plt.ylabel('Train Accuracy', fontsize=10)
        plt.xlabel('Communication Rounds', fontsize=10)

    elif plot_type is 'train_loss':
        plt.ylabel('Training Loss', fontsize=10)
        plt.xlabel('Communication Rounds', fontsize=10)
        plt.ylim(bottom=0.25, top=0.5)

    elif plot_type is 'train_error':
        plt.ylabel('Train Error', fontsize=10)
        plt.xlabel('Communication Rounds', fontsize=10)

    elif plot_type is 'timing':
        plt.ylabel('Time', fontsize=10)
        plt.xlabel('Dimension', fontsize=10)

    else:
        raise NotImplementedError

    # plt.title('')
    plt.legend(fontsize=11)
    plt.show()


def get_runtime(gar, X, repeat: int = 10):
    T = 0
    for it in range(repeat):
        t0 = time.time()
        _ = gar.aggregate(G=X)
        T += time.time() - t0
    T /= repeat
    return T


def compare_gar_speed(agg_config: Dict,
                      sparse_approximation_config=None):
    # d = [100, 1000, 10000, 100000]
    d = [100000, 1000000, 10000000, 100000000]
    n = 500
    res = {}
    gar = get_gar(aggregation_config=agg_config)

    for dim in d:
        sparse_approx_op = SparseApproxMatrix(conf=sparse_approximation_config)
        # generate n points in d dimensions
        X = np.random.normal(0, 0.3, (n, dim))
        if sparse_approximation_config["rule"] is not None:
            print('Running with sparse G approx')
            X_sparse = sparse_approx_op.sparse_approx(G=X)
            t = get_runtime(gar=gar, X=X_sparse, repeat=3)
        else:
            t = get_runtime(gar=gar, X=X, repeat=3)
        res[dim] = t

    return res


def runtime_exp():
    op_file = 'result_dumps/timing_exp/geo_med.norm_0.1'
    aggregation_config = \
        {
            "gar": "mean",
            "trimmed_mean_config": {"proportion": 0.1},
            "glomo_config": {"glomo_server_c": 1},
        }

    sparse_approximation_config = {
        "rule": 'active_norm',
        "axis": 'column',
        "frac_coordinates": 0.1,
    }

    results = compare_gar_speed(agg_config=aggregation_config,
                                sparse_approximation_config=sparse_approximation_config)

    with open(op_file, 'w+') as f:
        json.dump(results, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)


if __name__ == '__main__':
    plot_metrics()
    # runtime_exp()
