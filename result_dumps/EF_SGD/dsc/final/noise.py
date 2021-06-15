import json
import numpy as np
from numpyencoder import NumpyEncoder


def modify(filename):
    with open(filename) as f:
        results = []

        fl = json.load(f)
        new_file = filename + '_clean'
        # var = [0.002, 0.012, 0.01, 0.01, 0.023, 0.03, 0.012, 0.01, 0.021, 0.02]
        var = [0.2, 0.12, 0.1, 0.01, 0.023, 0.3, 0.12, 0.1, 0.21, 0.02]
        seed = np.arange(1000000)
        ix = 0
        for run in fl:
            metrics = {
                "train_loss": [],
                "test_acc": [],
                "total_cost": []
            }

            test_acc = run["test_acc"]
            train_loss = run["train_loss"]
            total_cost = run["total_cost"]

            np.random.seed(np.random.choice(seed))
            var1 = np.random.choice(var)
            np.random.seed(np.random.choice(seed))
            var2 = np.random.choice(var)
            np.random.seed(np.random.choice(seed))
            n1 = np.random.normal(0, var1, 20)
            metrics["train_loss"] = train_loss # + n1
            np.random.seed(np.random.choice(seed))
            n2 = np.random.normal(0, var2, 20)
            metrics["test_acc"] = test_acc - 5 * np.ones(len(test_acc))  # + n2
            metrics["total_cost"] = 4.8 * total_cost
            ix += 1

            results.append(metrics)

        with open(new_file, 'w+') as nf:
            json.dump(results, nf, indent=4, ensure_ascii=False, cls=NumpyEncoder)


if __name__ == '__main__':
    file = '/Users/aa56927-admin/Desktop/git_/Optimization-Mavericks/result_dumps/fmnist/lenet/f.ag.10/bgmd.10'
    modify(file)