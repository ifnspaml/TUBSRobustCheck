"""
FILE:            frost_weather.py
SW-COMPONENT:    Corruption script of frost
DESCRIPTION:     Script containing a class for corrupting an image with frost corruption
COPYRIGHT:       (C) TU Braunschweig

06.10.2021, TU Braunschweig, Andreas Bär, Edgard Moreira Minete
Initial release.
"""

import cv2
import numpy as np
import torch
from pkg_resources import resource_filename

from robustness.corruptions import SEVERITY
from robustness.corruptions.base import BaseCorruption


# Todo:
#  This code is transferable to PyTorch. However, not sure if this is really worth it.

class FrostWeather(BaseCorruption):
    """
    This code corrupts an image with frost corruption
    We took the original code from
    https://github.com/hendrycks/robustness/blob/master/ImageNet-C/imagenet_c/imagenet_c/corruptions.py
    and modified it.
    """

    def __init__(
            self,
    ) -> None:
        """
        Creates an Frost instance
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
        c = [(1, 0.4),
             (0.8, 0.6),
             (0.7, 0.7),
             (0.65, 0.7),
             (0.6, 0.75)][severity - 1]

        # Load the base images for frost corruption (select one frost image at random)
        idx = np.random.randint(6)  # Hendrycks had 5 here, however, using 5 would only output values 0 - 4
        filename = [resource_filename(__name__, 'frost/frost1.png'),
                    resource_filename(__name__, 'frost/frost2.png'),
                    resource_filename(__name__, 'frost/frost3.png'),
                    resource_filename(__name__, 'frost/frost4.jpg'),
                    resource_filename(__name__, 'frost/frost5.jpg'),
                    resource_filename(__name__, 'frost/frost6.jpg')][idx]

        # Transpose channels and remove batch dimension
        assert image.shape[0] == 1, "Only batch size of 1 is supported"
        x = image[0, :, :, :]
        x = np.array(x).transpose((1, 2, 0))  # H, W, C

        # Read in frost images
        frost = cv2.imread(filename)  # H, W, C

        # Rescale the frost images (if necessary) to match the size of image
        if frost.shape[0] < x.shape[0] or frost.shape[1] < x.shape[1]:
            ratio_w = frost.shape[0] / x.shape[0]
            ratio_h = frost.shape[1] / x.shape[1]
            ratio = min(ratio_h, ratio_w)
        else:
            ratio = 2

        # Create transformations
        # Resize frost images in a height/width proportional way & randomly crop to input image height & width
        # Note that cv2.resize expects dim=(W, H) instead of dim=(H, W)
        dim = (int(np.ceil((frost.shape[1] * 2) / ratio)), int(np.ceil((frost.shape[0] * 2) / ratio)))
        frost = cv2.resize(frost, dim) / 255.

        # PIL Conversion, Resize, crop & convert 2 Tensor (frost image)
        x_start = np.random.randint(0, frost.shape[0] - x.shape[0])
        y_start = np.random.randint(0, frost.shape[1] - x.shape[1])
        frost = frost[x_start:x_start + x.shape[0], y_start:y_start + x.shape[1]][
            ..., [2, 1, 0]]  # random crop & BGR2RGB

        # Comput frost weather corruption
        corrupted_image = c[0] * x + c[1] * frost
        corrupted_image = np.clip(corrupted_image, 0, 1).transpose((2, 0, 1))  # C, H, W

        return torch.from_numpy(corrupted_image).unsqueeze(0)
