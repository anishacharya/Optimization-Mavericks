from typing import Dict

from base_trainer import TrainPipeline


class SamplingPipeline(TrainPipeline):
    def run_batch_train(self, config: Dict, metrics: Dict, seed):
        pass

    def run_fed_train(self, config: Dict, metrics: Dict, seed):
        pass
