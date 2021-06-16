from typing import Dict


class TrainPipeline:
    def __init__(self):
        pass
    
    def run_batch_train(self, config: Dict, metrics: Dict, seed):
        raise NotImplementedError("This method needs to be implemented for each pipeline")

    def run_fed_train(self, config: Dict, metrics: Dict, seed):
        raise NotImplementedError("This method needs to be implemented for each pipeline")
