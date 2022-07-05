"""
FILE:            glass_blur.py
SW-COMPONENT:    Corruption script of glass_blur
DESCRIPTION:     Script containing a class for corrupting an image with glass blur
COPYRIGHT:       (C) TU Braunschweig

06.10.2021, TU Braunschweig, Andreas Bär, Edgard Moreira Minete
Initial release.
"""

import numpy as np
import torch
from skimage.filters import gaussian
from numba import njit, prange

from robustness.corruptions import SEVERITY
from robustness.corruptions.base import BaseCorruption


# Todo:
#  This code is transferable to PyTorch (e.g. using kornia). However, not sure if this is really worth it.

class GlassBlur(BaseCorruption):
    """
    This code corrupts an image with glass blur
    We took the original code from
    https://github.com/hendrycks/robustness/blob/master/ImageNet-C/imagenet_c/imagenet_c/corruptions.py
    and modified it.
    """

    def __init__(
            self,
            efficient=True
    ) -> None:

        """
        Creates an GlassBlur instance
        """
        super().__init__()
        self.efficient = efficient

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

        # sigma, max_delta, iterations
        c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]

        # Transpose channels and remove batch dimension
        assert image.shape[0] == 1, "Only batch size of 1 is supported"
        x = image[0, :, :, :]
        x = np.array(x).transpose((1, 2, 0))  # H, W, C
        x = np.uint8(gaussian(x, sigma=c[0], channel_axis=2) * 255)

        if self.efficient:
            x = _shuffle_pixels_njit_glass_blur(np.array(x).shape[0], np.array(x).shape[1], x, c)
        else:
            x = _shuffle_pixels_original_glass_blur(np.array(x).shape[0], np.array(x).shape[1], x, c)

        # Apply glass blur corruption and clip it to [0-1] range
        corrupted_image = np.clip(gaussian(x / 255., sigma=c[0], channel_axis=2), 0, 1).transpose((2, 0, 1))

        return torch.from_numpy(corrupted_image).unsqueeze(0)


# original implementation
def _shuffle_pixels_original_glass_blur(d0, d1, x, c):
    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(d0 - c[1], c[1], -1):
            for w in range(d1 - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # Swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]
    return x


# speed up everything (see also: https://github.com/bethgelab/imagecorruptions/pull/18)
@njit()
def _shuffle_pixels_njit_glass_blur(d0, d1, x, c):
    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(d0 - c[1], c[1], -1):
            for w in range(d1 - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]
    return x
