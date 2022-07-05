"""
FILE:            spatter_weather.py
SW-COMPONENT:    Corruption script of spatter_weather
DESCRIPTION:     Script containing a class for corrupting an image with spatter weather
COPYRIGHT:       (C) TU Braunschweig

06.10.2021, TU Braunschweig, Andreas Bär, Edgard Moreira Minete
Initial release.
"""

import cv2
import numpy as np
import torch
from skimage.filters import gaussian

from robustness.corruptions import SEVERITY
from robustness.corruptions.base import BaseCorruption


# Todo:
#  This code is transferable to PyTorch (e.g. using kornia). However, not sure if this is really worth it.

class SpatterWeather(BaseCorruption):
    """
    This code corrupts an image with spatter weather
    We took the original code from
    https://github.com/hendrycks/robustness/blob/master/ImageNet-C/imagenet_c/imagenet_c/corruptions.py
    and modified it.
    """

    def __init__(
            self,
    ) -> None:

        """
        Creates an SpatterWeather instance
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
        c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
             (0.65, 0.3, 3, 0.68, 0.6, 0),
             (0.65, 0.3, 2, 0.68, 0.5, 0),
             (0.65, 0.3, 1, 0.65, 1.5, 1),
             (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]

        # Transpose channels and remove batch dimension
        assert image.shape[0] == 1, "Only batch size of 1 is supported"
        x = image[0, :, :, :]
        x = np.array(x).transpose((1, 2, 0))  # C, H, W

        # Apply spatter weather corruption and clip it to [0-1] range
        liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])
        liquid_layer = gaussian(liquid_layer, sigma=c[2])
        liquid_layer[liquid_layer < c[3]] = 0

        if c[5] == 0:
            liquid_layer = (liquid_layer * 255).astype(np.uint8)
            dist = 255 - cv2.Canny(liquid_layer, 50, 150)
            dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
            _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
            dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
            dist = cv2.equalizeHist(dist)
            ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
            dist = cv2.filter2D(dist, cv2.CV_8U, ker)
            dist = cv2.blur(dist, (3, 3)).astype(np.float32)

            m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
            m /= np.max(m, axis=(0, 1))
            m *= c[4]

            # water is pale turqouise
            color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                    238 / 255. * np.ones_like(m[..., :1]),
                                    238 / 255. * np.ones_like(m[..., :1])), axis=2)

            color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

            # Clip image an transpose it back to [batch, channels, h, w] format
            corrupted_image = cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR).transpose((2, 0, 1))
        else:
            m = np.where(liquid_layer > c[3], 1, 0)
            m = gaussian(m.astype(np.float32), sigma=c[4])
            m[m < 0.8] = 0

            # Mud brown
            color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
                                    42 / 255. * np.ones_like(x[..., :1]),
                                    20 / 255. * np.ones_like(x[..., :1])), axis=2)

            color *= m[..., np.newaxis]
            x *= (1 - m[..., np.newaxis])

            # Clip image an transpode it back to [batch, channels, h, w] format
            corrupted_image = np.clip(x + color, 0, 1).transpose((2, 0, 1))
        return torch.from_numpy(corrupted_image).unsqueeze(0)
