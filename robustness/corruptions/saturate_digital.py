"""
FILE:            saturate_digital.py
SW-COMPONENT:    Corruption script of saturate_digital
DESCRIPTION:     Script containing a class for corrupting an image with saturate digital
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

class SaturateDigital(BaseCorruption):
    """
    This code corrupts an image with saturate digital
    We took the original code from
    https://github.com/hendrycks/robustness/blob/master/ImageNet-C/imagenet_c/imagenet_c/corruptions.py
    and modified it.
    """

    def __init__(
            self,
    ) -> None:
        """
        Creates an SaturateDigital instance
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
        c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]

        # Transpose channels and remove batch dimension
        assert image.shape[0] == 1, "Only batch size of 1 is supported"
        x = image[0, :, :, :]
        x = np.array(x).transpose((1, 2, 0))  # C, H, W

        # Apply saturate digital corruption
        x = sk.color.rgb2hsv(x)
        x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
        x = sk.color.hsv2rgb(x)

        # Clip image an transpose it back to [batch, channels, h, w] format
        corrupted_image = np.clip(x, 0, 1).transpose((2, 0, 1))

        return torch.from_numpy(corrupted_image).unsqueeze(0)
