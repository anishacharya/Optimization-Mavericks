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
        self.num_corrupt = 0
        self.curr_corr = 0

    def attack(self, X):
        # Toss a coin
        # p = np.random.random()
        # if p < self.frac_adv:
        if self.curr_corr > 0:
            # apply attack
            for ix, sample in enumerate(X):
                noisy_img = self.corrupt(img=sample)
                X[ix] = noisy_img
        return X

    def corrupt(self, img: torch.tensor) -> torch.tensor:
        raise NotImplementedError
    # def launch_attack(self, data_loader: DataLoader):
    #     # TODO: Can we apply Transforms to Batches directly ? Then we can do this only once after DataLoader
    #     raise NotImplementedError


class ImageAdditive(ImageCorruption):
    """
    'gaussian'  Gaussian-distributed additive noise applied to all images passed
    """
    def __init__(self, attack_config: Dict):
        ImageCorruption.__init__(self, attack_config=attack_config)
        self.var = [.08, .12, 0.18, 0.26, 0.38][4]
        print(" Additive Image Noise {}".format(self.attack_config))

    def corrupt(self, img):
        return torch.tensor(np.clip(random_noise(image=img/255., var=self.var), 0, 1) * 255)


class ImagePepper(ImageCorruption):
    def __init__(self, attack_config: Dict):
        ImageCorruption.__init__(self, attack_config=attack_config)
        self.amount = self.attack_config.get('amount', 0.5)
        print(" Pepper Noise Added {}".format(self.attack_config))

    def corrupt(self, img):
        return torch.tensor(random_noise(image=img/255., mode='pepper', amount=self.amount))


class ImageImpulse(ImageCorruption):
    def __init__(self, attack_config: Dict):
        ImageCorruption.__init__(self, attack_config=attack_config)
        self.amount = [.03, .06, .09, 0.17, 0.27][4]

    def corrupt(self, img: torch.tensor):
        return torch.tensor(np.clip(random_noise(image=img/255., mode='s&p', amount=self.amount), 0, 1) * 255)


class ImageGaussianBlur(ImageCorruption):
    pass


if __name__ == '__main__':
    # Test some noise and visualize
    # Download .png cifar10 command >>"cifar2png cifar10 data"
    sample_im = io.imread('/Users/aa56927-admin/Desktop/BGMD/NeuRips/image_sample.jpeg')
    severity = 5
    # Test and plot noises
    # noinspection PyArgumentEqualDefault
    # sample_im = random_noise(image=sample_im, mode='gaussian', var=0.2)
    # sample_im = random_noise(image=sample_im, mode='poisson')
    # sample_im = random_noise(image=sample_im, mode='pepper', amount=0.8)
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]
    sample_im = random_noise(sample_im / 255., mode='s&p', amount=c)
    sample_im = np.clip(sample_im, 0, 1) * 255
    # sample_im = random_noise(image=sample_im, mode='s&p', amount=0.5)
    # sample_im = cv2.GaussianBlur(sample_im, (15, 15), 100)

    plt.imshow(sample_im)
    plt.axis('off')
    plt.xlabel('Clean Image')
    plt.show()
