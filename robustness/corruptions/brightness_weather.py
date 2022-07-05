"""
FILE:            brightness_weather.py
SW-COMPONENT:    Corruption script of brightness_weather
DESCRIPTION:     Script containing a class for corrupting an image with brightness weather
COPYRIGHT:       (C) TU Braunschweig

06.10.2021, TU Braunschweig, Andreas Bär, Edgard Moreira Minete
Initial release.
"""

import numpy as np
import skimage as sk
import torch

from robustness.corruptions import SEVERITY
from robustness.corruptions.base import BaseCorruption


# ToDo: With the kornia package might be transferable to PyTorch. However depends on the speed gain.
#  https://kornia.readthedocs.io/en/latest/_modules/kornia/color/hsv.html

class BrightnessWeather(BaseCorruption):
    """
    This code corrupts an image with brightness weather
    We took the original code from
    https://github.com/hendrycks/robustness/blob/master/ImageNet-C/imagenet_c/imagenet_c/corruptions.py
    and modified it.
    """

    def __init__(
            self,
    ) -> None:
        """
        Creates an BrightnessWeather instance
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
        c = [.1, .2, .3, .4, .5][severity - 1]

        # Apply brightness weather corruption and clip it to [0-1] range
        x = np.array(image)  # B, C, H, W
        x = sk.color.rgb2hsv(x, channel_axis=1)
        x[:, 2, :, :] = np.clip(x[:, 2, :, :] + c, 0, 1)
        corrupted_image = np.clip(sk.color.hsv2rgb(x, channel_axis=1), 0, 1)

        return torch.from_numpy(corrupted_image)
