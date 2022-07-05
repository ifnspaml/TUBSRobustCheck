"""
FILE:            gaussian_noise.py
SW-COMPONENT:    Corruption script of gaussian_noise
DESCRIPTION:     Script containing a class for corrupting an image with Gaussian noise
COPYRIGHT:       (C) TU Braunschweig

06.10.2021, TU Braunschweig, Andreas Bär, Edgard Moreira Minete
Initial release.
"""

import numpy as np
import torch

from robustness.corruptions import SEVERITY
from robustness.corruptions.base import BaseCorruption


# Todo:
#  This code is transferable to PyTorch.

class GaussianNoise(BaseCorruption):
    """
    This code corrupts an image with Gaussian noise
    We took the original code from
    https://github.com/hendrycks/robustness/blob/master/ImageNet-C/imagenet_c/imagenet_c/corruptions.py
    and modified it.
    """

    def __init__(
            self,
    ) -> None:
        """
        Creates an GaussianNoise instance
        """
        super().__init__()

    def __call__(self, image, severity=1):
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
        c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

        # Transpose channels and remove batch dimension
        assert image.shape[0] == 1, "Only batch size of 1 is supported"
        x = image[0, :, :, :]
        x = np.array(x)  # C, H, W

        # Apply Gaussian corruption and clip it to [0-1] range
        corrupted_image = np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1)

        return torch.from_numpy(corrupted_image).unsqueeze(0)
