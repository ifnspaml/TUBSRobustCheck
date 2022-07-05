"""
FILE:            contrast_digital.py
SW-COMPONENT:    Corruption script of contrast_digital
DESCRIPTION:     Script containing a class for corrupting an image with contrast digital
COPYRIGHT:       (C) TU Braunschweig

06.10.2021, TU Braunschweig, Andreas Bär, Edgard Moreira Minete
Initial release.
"""

import numpy as np
import torch

from robustness.corruptions import SEVERITY
from robustness.corruptions.base import BaseCorruption


# ToDo: This can be transferred to PyTorch. Probably faster as no transfer between CPU and GPU necessary.

class ContrastDigital(BaseCorruption):
    """
    This code corrupts an image with contrast digital
    We took the original code from
    https://github.com/hendrycks/robustness/blob/master/ImageNet-C/imagenet_c/imagenet_c/corruptions.py
    and modified it.
    """

    def __init__(
            self,
    ) -> None:
        """
        Creates an ContrastDigital instance
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
        c = [0.4, .3, .2, .1, .05][severity - 1]

        # Apply contrast digital corruption and clip it to [0-1] range
        image = np.array(image)  # B, C, H, W
        means = np.mean(image, axis=(2, 3), keepdims=True)
        corrupted_image = np.clip((image - means) * c + means, 0, 1)

        return torch.from_numpy(corrupted_image)
