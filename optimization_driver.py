import argparse
import json
import os
import yaml
import numpy as np

from numpyencoder import NumpyEncoder

from src import run_fed_train, run_batch_train


def _parse_args():
    parser = argparse.ArgumentParser(description='federated/decentralized/distributed training experiment template')
    parser.add_argument('--conf', type=str, default=None, help='Pass Config file path')
    parser.add_argument('--o', type=str, default='output', help='Pass result file path')
    parser.add_argument('--dir', type=str, default=None, help='Pass result file dir')
    parser.add_argument('--n_repeat', type=int, default=1, help='Specify number of repeat runs')
    args = parser.parse_args()
    return args


def init_metric(config):
    metrics = {"config": config,

               "num_param": 0,
               # Train and Test Performance
               "test_error": [],
               "test_loss": [],
               "test_acc": [],
               "train_error": [],
               "train_loss": [],
               "train_acc": [],

               "communication_residual": [],
               "sparse_approx_residual": [],
               # # Grad Matrix Stats
               # "frac_mass_retained": [],
               # "grad_norm_dist": [],
               # "norm_bins": None,
               # "mass_bins": None,
               # "max_norm": 0,
               # "min_norm": 1e6,

               # compute Time stats per epoch
               "epoch_sparse_approx_cost": [],
               "epoch_grad_cost": [],
               "epoch_agg_cost": [],
               "epoch_gm_iter": [],


               # Total Costs
               "total_cost": 0,
               "total_grad_cost": 0,
               "total_agg_cost": 0,
               "total_sparse_cost": 0,

               "total_gm_iter": 0,
               "avg_gm_cost": 0,

               "num_iter": 0,
               "num_agg": 0,
               }
    return metrics


def run_main():
    args = _parse_args()
    print(args)
    root = os.getcwd()
    config_path = args.conf if args.conf else root + '/default_config.yaml'
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)

    # Train
    results = []
    # for seed in np.random.randint(0, 100*args.n_repeat+1, args.n_repeat):
    for seed in np.arange(args.n_repeat + 1):
        train_mode = config.get("train_mode", 'distributed')
        metrics = init_metric(config=config)
        if train_mode == 'fed':
            run_fed_train(config=config, metrics=metrics)
            results.append(metrics)
        elif train_mode == 'distributed':
            run_batch_train(config=config, metrics=metrics, seed=seed)
            results.append(metrics)
        else:
            raise NotImplementedError

    # Write Results
    directory = args.dir if args.dir else "result_dumps/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(directory + args.o, 'w+') as f:
        json.dump(results, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)


if __name__ == '__main__':
    run_main()
