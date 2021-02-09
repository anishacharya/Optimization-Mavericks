import argparse
import json
import os
import yaml

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


def run_main():
    args = _parse_args()
    print(args)
    root = os.getcwd()
    config_path = args.conf if args.conf else root + '/default_config.yaml'
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)

    metrics = {"config": config,

               "test_error": [],
               "test_loss": [],
               "test_acc": [],
               "train_error": [],
               "train_loss": [],
               "train_acc": [],

               "frac_mass_retained": [],
               "grad_norm_dist": [],
               "norm_bins": None,
               "mass_bins": None,
               "max_norm": 0,
               "min_norm": 1e6,

               "batch_grad_cost": 0,
               "batch_agg_cost": 0,
               "total_run_time": 0,
               }
    # Train
    train_mode = config.get("train_mode", 'fed')
    if train_mode == 'fed':
        run_fed_train(config=config, metrics=metrics)
    elif train_mode == 'distributed':
        run_batch_train(config=config, metrics=metrics)
    else:
        raise NotImplementedError

    # Write Results
    directory = args.dir if args.dir else "result_dumps/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(directory + args.o, 'w+') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)


if __name__ == '__main__':
    run_main()
