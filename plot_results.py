import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rc
from matplotlib.pyplot import figure
import json


def plot_(label: str, res_file: str, plt_type: str = 'epoch_loss', x_ax='time',
          line_width=4, marker=None, line_style=None, optima: float = 0.0):

    with open(res_file, 'rb') as f:
        result = json.load(f)

    scores = []
    for run in result:
        res = run[plt_type]
        res -= optima * np.ones(len(res))
        scores += [res]

    scores = np.array(scores)
    mean = np.mean(scores, axis=0)
    UB = mean + 3 * np.std(scores, axis=0)
    LB = mean - 3 * np.std(scores, axis=0)

    if x_ax is 'time':
        x_freq = int(result[0]["total_cost"] / len(result[0][plt_type]))
        x = np.arange(len(result[0][plt_type])) * x_freq
    elif x_ax is 'epoch':
        x = np.arange(len(result[0][plt_type]))
    else:
        raise NotImplementedError

    plt.plot(x, mean, label=label, linewidth=line_width, marker=marker, linestyle=line_style)
    plt.fill_between(x, LB, UB, alpha=0.5, linewidth=3)


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def plot_mass(masses):
    # x_labels = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1']
    # legends = ['epoch 5', 'epoch 10', 'epoch 15', 'epoch 20']

    x_labels = ['$0\%$', '$10\%$', '$20\%$']
    legends = [r"\textsc{SGD}",
               r"\textsc{Gm-SGD}",
               r"\textsc{BGmD}"]

    x = np.arange(len(x_labels))

    fig, ax = plt.subplots()

    ax.set_xticks(x)
    ax.set_yticks(np.arange(start=0, stop=100, step=10))
    ax.set_xticklabels(x_labels)

    # with open(res_file, 'rb') as f:
    #    res = json.load(f)
    # masses = res["frac_mass_retained"]
    width = 0.1
    offset = -3 / 2
    for frac_dist, leg in zip(masses, legends):
        # frac_dist = frac_dist[1:]
        # frac_dist[-1] = 1
        # plt.plot(x, frac_dist)
        plt.bar(height=frac_dist, x=x + offset * width, width=width, label=leg)
        offset += 1
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=3,
              borderaxespad=0, frameon=False, fontsize=11)


if __name__ == '__main__':
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # activate latex text rendering
    rc('text', usetex=True)

    # -------------------------------------------------------------------------------------------
    # ------------------------------- Modify Here -----------------------------------------------
    d = 'result_dumps/fmnist/lenet/ag.10/'

    o = [
        #'mean',
        'gm',
        'bgmd.5',
        'bgmd.10',
        'bgmd.20',
        'bgmd.30',

    ]
    labels = [
        #r"\textsc{SGD}",
        r"\textsc{GM-SGD}",
        r"\textsc{BGmD($\beta$ = 0.05)}",
        r"\textsc{BGmD($\beta$ = 0.1)}",
        r"\textsc{BGmD($\beta$ = 0.2)}",
        r"\textsc{BGmD($\beta$ = 0.3)}",
    ]
    plot_type = 'train_loss'
    x_ax = 'epoch'
    sampling_freq = 1

    for op, label in zip(o, labels):
        result_file = d + op

        plot_(label=label, res_file=result_file, plt_type=plot_type, x_ax=x_ax,
              optima=0, line_width=2)

        # Fix the X Labels
        if x_ax is 'time':
            plt.xlabel(r'$\mathcal{O}$(Time)', fontsize=10)
        else:
            plt.xlabel('Epochs (Full Pass over Data)', fontsize=10)

    if plot_type is 'test_error':
        plt.ylabel('Test Error', fontsize=10)
    elif plot_type is 'test_acc':
        plt.ylabel('Test Accuracy', fontsize=10)
    elif plot_type is 'train_acc':
        plt.ylabel('Train Accuracy', fontsize=10)
    elif plot_type is 'train_loss':
        plt.yscale('log')
        plt.ylabel('Training Loss', fontsize=10)
    elif plot_type is 'train_error':
        plt.ylabel('Train Error', fontsize=10)
    else:
        raise NotImplementedError

    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.tick_params(labelsize=10)

    figure(figsize=(1, 1))
    plt.show()

