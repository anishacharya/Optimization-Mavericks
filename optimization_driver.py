import argparse
import json
import os
import yaml
import numpy as np

from numpyencoder import NumpyEncoder
from src.training_pipelines import *


def _parse_args():
    parser = argparse.ArgumentParser(description='federated/decentralized/distributed training experiment template')
    parser.add_argument('--train_mode',
                        type=str,
                        default='vanilla',
                        help='distributed: launch distributed Training '
                             'fed: launch federated training')
    parser.add_argument('--pipeline',
                        type=str,
                        default='loss',
                        help='sampling: exp with sampling data during training'
                             'agg: exp with GAR')
    parser.add_argument('--conf',
                        type=str,
                        default=None,
                        help='Pass Config file path')
    parser.add_argument('--o',
                        type=str,
                        default='default_output',
                        help='Pass result file path')
    parser.add_argument('--dir',
                        type=str,
                        default=None,
                        help='Pass result file dir')
    parser.add_argument('--n_repeat',
                        type=int,
                        default=1,
                        help='Specify number of repeat runs')
    args = parser.parse_args()
    return args


def get_trainer(pipeline: str, config, seed):
    # select pipeline #
    if pipeline == 'jacobian':
        trainer = JacobianCompressPipeline(config=config, seed=seed)
    elif pipeline == 'loss':
        trainer = LossPipeline(config=config, seed=seed)
    else:
        raise NotImplementedError

    return trainer


def run_main():
    args = _parse_args()
    print(args)
    root = os.getcwd()
    pipeline = args.pipeline
    config_path = args.conf if args.conf else root + '/configs/default_config.yaml'
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)

    # Training - Repeat over the random seeds #
    # ----------------------------------------
    results = []
    for seed in np.arange(args.n_repeat):
        # ----- Launch Training ------ #
        train_mode = args.train_mode
        trainer = get_trainer(pipeline=pipeline, config=config, seed=seed)

        if train_mode == 'fed':
            # Launch Federated Training
            trainer.run_fed_train(config=config, seed=seed)
            results.append(trainer.metrics)
        elif train_mode == 'distributed':
            # Launch Distributed Training
            trainer.run_batch_train(config=config, seed=seed)
            results.append(trainer.metrics)
        elif train_mode == 'vanilla':
            # Launch Vanilla mini-batch Training
            trainer.run_train(config=config, seed=seed)
            results.append(trainer.metrics)
        else:
            raise NotImplementedError

    # Write Results #
    # ----------------
    directory = args.dir if args.dir else "result_dumps/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(directory + args.o, 'w+') as f:
        json.dump(results, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)


if __name__ == '__main__':
    run_main()
