from typing import Dict
from torch.utils.data import DataLoader
import numpy as np
from skimage.util import random_noise
from skimage import io
import matplotlib.pyplot as plt
import cv2


class ImageCorruption:
    """ This is the Base Class for Image Corruptions. """

    def __init__(self, attack_config: Dict, seed=1):
        self.attack_config = attack_config
        self.seed = seed
        self.noise_model = self.attack_config.get("noise_model", None)
        self.frac_adv = self.attack_config.get('frac_adv', 0)

    def launch_attack(self, data_loader: DataLoader):
        np.random.seed(self.seed)

        # sample ix of corrupt batches
        # TODO: Implement This
        pass


class ImageAdditive(ImageCorruption):
    pass


class ImageSaltPepper(ImageCorruption):
    pass


class ImageGaussianBlur(ImageCorruption):
    pass


if __name__ == '__main__':
    # Test some noise and visualize
    # Download .png cifar10 command >>"cifar2png cifar10 data"
    sample_im = io.imread('/Users/aa56927-admin/Desktop/BGMD/NeuRips/image_sample.jpeg')

    # Test and plot noises
    # noinspection PyArgumentEqualDefault
    # sample_im = random_noise(image=sample_im, mode='gaussian', var=0.2)
    # sample_im = random_noise(image=sample_im, mode='poisson')
    # sample_im = random_noise(image=sample_im, mode='pepper', amount=0.8)
    # sample_im = random_noise(image=sample_im, mode='s&p', amount=0.5)
    sample_im = cv2.GaussianBlur(sample_im, (15, 15), 100)

    plt.imshow(sample_im)
    plt.axis('off')
    plt.xlabel('Clean Image')
    plt.show()
