"""
FILE:            impulse_noise.py
SW-COMPONENT:    Corruption script of impulse_noise
DESCRIPTION:     Script containing a class for corrupting an image with impulse noise
COPYRIGHT:       (C) TU Braunschweig

06.10.2021, TU Braunschweig, Andreas Bär, Edgard Moreira Minete
Initial release.
"""

import numpy as np
import skimage as sk
import torch

from robustness.corruptions import SEVERITY
from robustness.corruptions.base import BaseCorruption


# Todo:
#  This code is transferable to PyTorch (e.g. using kornia). However, not sure if this is really worth it.

class ImpulseNoise(BaseCorruption):
    """
    This code corrupts an image with impulse noise
    We took the original code from
    https://github.com/hendrycks/robustness/blob/master/ImageNet-C/imagenet_c/imagenet_c/corruptions.py
    and modified it.
    """

    def __init__(
            self,
    ) -> None:
        """
        Creates an ImpulseNoise instance
        """
        super().__init__()

    def __call__(self, image, severity):
        """
        Implementation of the corruption
        :param image: the image to be corrupted
        :param severity: the corruption severity

        :return: corrupted_image: the corrupted image
        """

        # Send image to cpu
        if image.is_cuda:
            image = image.cpu()

        # Set the corruption severity
        assert severity in SEVERITY
        c = [.03, .06, .09, 0.17, 0.27][severity - 1]

        # Transpose channels and remove batch dimension
        assert image.shape[0] == 1, "Only batch size of 1 is supported"
        x = image[0, :, :, :]
        x = np.array(x)  # C, H, W

        # Apply impulse noise corruption and clip it to [0-1] range
        corrupted_image = np.clip(sk.util.random_noise(x, mode='s&p', amount=c, seed=0), 0, 1)

        return torch.from_numpy(corrupted_image).unsqueeze(0)
