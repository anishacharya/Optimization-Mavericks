from typing import Dict
from torch.utils.data import DataLoader


class ImageCorruption:
    """ This is the Base Class for Image Corruptions. """

    def __init__(self, attack_config: Dict):
        self.attack_config = attack_config
        self.noise_model = self.attack_config.get("noise_model", None)
        self.frac_adv = self.attack_config.get('frac_adv', 0)


    def launch_attack(self, data_loader:DataLoader):
        pass
