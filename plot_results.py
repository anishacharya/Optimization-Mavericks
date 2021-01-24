import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker
import json
from typing import List, Dict


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


def plot_metrics():
    # -------------------------------
    # ** Usually No Need to Modify **
    # -------------------------------
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # -------------------------------------------------------------------------------------------
    # ------------------------------- Modify Here -----------------------------------------------
    d = 'result_dumps/Robust/distributed/fashion_mnist/'
    o = [
        'mean',
        # 'geo_med',
        'mean.norm_0.1',
        'mean.norm_0.1.ef',
        # 'geo_med.norm_0.1',
        # 'geo_med.norm_0.1.ef',
        'mean.norm_0.05',
        'mean.norm_0.05.ef',
        # 'geo_med.norm_0.05',
        # 'geo_med.norm_0.05.ef'

    ]
    labels = ['Mini-Batch SGD',
              # 'GM SGD',
              'Sparse SGD (k=10%)',
              'Sparse SGD + Memory (k=10%)',
              # 'Co-Sparse GM SGD (k=10%)',
              # 'Co-Sparse GM EC-SGD (k=10%)',
              'Sparse SGD (k=5%)',
              'Sparse SGD + Memory (k=5%)',
              # 'Co-Sparse GM SGD (k=5%)',
              # 'Co-Sparse GM EC-SGD (k=5%)'
              ]

    plot_type = 'train_loss'
    sampling_freq = 5

    for op, label in zip(o, labels):
        result_file = d + op
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

    else:
        raise NotImplementedError

    # plt.title('')
    plt.legend(fontsize=11)
    plt.show()


def compare_gar_speed(gars: List[str], repeat: int = 10) -> Dict[str, float]:
    d = [100, 1000, 10000, 100000]
    n = 500

    runtimes = {}
    for dim in d:
        # generate n points in d dimensions
        X = np.random.normal(0, 0.3, (n, dim))
        for it in range(repeat):

        pass

    return runtimes


if __name__ == '__main__':
    plot_metrics()
