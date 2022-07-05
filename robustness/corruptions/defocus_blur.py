"""
FILE:            defocus_blur.py
SW-COMPONENT:    Corruption script of defocus_blur
DESCRIPTION:     Script containing a class for corrupting an image with defocus blur
COPYRIGHT:       (C) TU Braunschweig

06.10.2021, TU Braunschweig, Andreas Bär, Edgard Moreira Minete
Initial release.
"""

import cv2
import numpy as np
import torch

from robustness.corruptions import SEVERITY
from robustness.corruptions.base import BaseCorruption


# ToDo:
#  Might be possible to transfer to PyTorch (as it is just a convultion in the end).
#  However, not sure if this is really worth it.

class DefocusBlur(BaseCorruption):
    """
    This code corrupts an image with defocus blur.
    We took the original code from
    https://github.com/hendrycks/robustness/blob/master/ImageNet-C/imagenet_c/imagenet_c/corruptions.py
    and modified it.
    """

    def __init__(
            self,
    ) -> None:

        """
        Creates an DefocusBlur instance
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
        c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

        # Transpose channels and remove batch dimension
        assert image.shape[0] == 1, "Only batch size of 1 is supported"
        x = image[0, :, :, :]  # C, H, W
        x = np.array(x).transpose((1, 2, 0))

        # Get the kernel
        kernel = self.disk(radius=c[0], alias_blur=c[1])  # The kernel is checked and it is exactly the same!

        # Apply defocus blur corruption and clip it to [0-1] range
        # channels = np.array([
        #     cv2.filter2D(x[0, :, :], -1, kernel),
        #     cv2.filter2D(x[1, :, :], -1, kernel),
        #     cv2.filter2D(x[2, :, :], -1, kernel)
        # ])
        channels = np.array([
            cv2.filter2D(x[:, :, 0], -1, kernel),
            cv2.filter2D(x[:, :, 1], -1, kernel),
            cv2.filter2D(x[:, :, 2], -1, kernel)
        ])

        corrupted_image = np.clip(channels, 0, 1)

        return torch.from_numpy(corrupted_image).unsqueeze(0)

    @staticmethod
    def disk(radius, alias_blur=0.1, dtype=np.float32):
        if radius <= 8:
            L = np.arange(-8, 8 + 1)
            ksize = (3, 3)
        else:
            L = np.arange(-radius, radius + 1)
            ksize = (5, 5)
        X, Y = np.meshgrid(L, L)
        aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
        aliased_disk /= np.sum(aliased_disk)

        # Supersample disk to antialias
        return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)
