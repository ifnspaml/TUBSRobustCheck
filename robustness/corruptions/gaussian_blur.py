"""
FILE:            gaussian_blur.py
SW-COMPONENT:    Corruption script of gaussian_blur
DESCRIPTION:     Script containing a class for corrupting an image with gaussian blur
COPYRIGHT:       (C) TU Braunschweig

06.10.2021, TU Braunschweig, Andreas Bär, Edgard Moreira Minete
Initial release.
"""

import numpy as np
import torch
from skimage.filters import gaussian

from robustness.corruptions import SEVERITY
from robustness.corruptions.base import BaseCorruption


# Todo:
#  This code is transferable to PyTorch (e.g. using kornia). However, not sure if this is really worth it.

class GaussianBlur(BaseCorruption):
    """
    This code corrupts an image with gaussian blur
    We took the original code from
    https://github.com/hendrycks/robustness/blob/master/ImageNet-C/imagenet_c/imagenet_c/corruptions.py
    and modified it.
    """

    def __init__(
            self,
    ) -> None:
        """
        Creates an GaussianBlur instance
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
        c = [1, 2, 3, 4, 6][severity - 1]

        # Transpose channels and remove batch dimension
        assert image.shape[0] == 1, "Only batch size of 1 is supported"
        x = image[0, :, :, :]
        x = np.array(x)  # C, H, W

        # Apply Gaussian blur corruption and clip it to [0-1] range
        corrupted_image = np.clip(gaussian(x, sigma=c, channel_axis=0), 0, 1)

        return torch.from_numpy(corrupted_image).unsqueeze(0)
