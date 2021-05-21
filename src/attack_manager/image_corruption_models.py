from typing import Dict
import torch
import numpy as np
from skimage.util import random_noise
from skimage import io
import matplotlib.pyplot as plt
import cv2


class ImageCorruption:
    """ This is the Base Class for Image Corruptions. """
    def __init__(self, attack_config: Dict):
        self.attack_config = attack_config
        self.noise_model = self.attack_config.get("noise_model", None)
        self.frac_adv = self.attack_config.get('frac_adv', 0)

    # def launch_attack(self, data_loader: DataLoader):
    #     # TODO: Can we apply Transforms to Batches directly ? Then we can do this only once after DataLoader
    #     raise NotImplementedError


class ImageAdditive(ImageCorruption):
    """
    'gaussian'  Gaussian-distributed additive noise applied to all images passed
    """
    def __init__(self, attack_config: Dict):
        ImageCorruption.__init__(self, attack_config=attack_config)
        self.var = self.attack_config.get("var", 1)
        print(" Additive Image Noise {}".format(self.attack_config))

    def attack(self, X):
        # Toss a coin
        p = np.random.random()
        if p < self.frac_adv:
            # apply attack
            for ix, sample in enumerate(X):
                noisy_img = self.corrupt(img=sample)
                X[ix] = noisy_img
        return X

    def corrupt(self, img):
        return torch.tensor(random_noise(image=img, var=self.var))


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
    sample_im = random_noise(image=sample_im, mode='pepper', amount=0.8)
    # sample_im = random_noise(image=sample_im, mode='s&p', amount=0.5)
    sample_im = cv2.GaussianBlur(sample_im, (15, 15), 100)

    plt.imshow(sample_im)
    plt.axis('off')
    plt.xlabel('Clean Image')
    plt.show()
