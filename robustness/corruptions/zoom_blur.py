"""
FILE:            zoom_blur.py
SW-COMPONENT:    Corruption script of zoom_blur
DESCRIPTION:     Script containing a class for corrupting an image with zoom blur
COPYRIGHT:       (C) TU Braunschweig

06.10.2021, TU Braunschweig, Andreas Bär, Edgard Moreira Minete
Initial release.
"""

import numpy as np
import torch
from scipy.ndimage import zoom as scizoom

from robustness.corruptions import SEVERITY
from robustness.corruptions.base import BaseCorruption


# Todo:
#  This code is transferable to PyTorch (e.g. using kornia). However, not sure if this is really worth it.

class ZoomBlur(BaseCorruption):
    """
    This code corrupts an image with zoom blur
    """

    def __init__(
            self,
    ) -> None:

        """
        Creates an ZoomBlur instance
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
        c = [np.arange(1, 1.11, 0.01),
             np.arange(1, 1.16, 0.01),
             np.arange(1, 1.21, 0.02),
             np.arange(1, 1.26, 0.02),
             np.arange(1, 1.31, 0.03)][severity - 1]

        # Transpose channels and remove batch dimension
        assert image.shape[0] == 1, "Only batch size of 1 is supported"
        x = image[0, :, :, :]
        x = np.array(x).transpose((1, 2, 0))  # C, H, W

        out = np.zeros_like(x)
        for zoom_factor in c:
            out += self.clipped_zoom(x, zoom_factor)

        x = (x + out) / (len(c) + 1)

        # Apply zoom blur corruption and clip it to [0-1] range
        corrupted_image = np.clip(x, 0, 1).transpose((2, 0, 1))

        return torch.from_numpy(corrupted_image).unsqueeze(0)

    @staticmethod
    def clipped_zoom(img, zoom_factor):
        # Extended to handle non squared images
        h = img.shape[0]
        w = img.shape[1]  # added

        # Ceil crop height and crop width
        ch = int(np.ceil(h / float(zoom_factor)))
        cw = int(np.ceil(w / float(zoom_factor)))

        top = (h - ch) // 2
        left = (w - cw) // 2  # added
        img = scizoom(img[top:top + ch, left:left + cw], (zoom_factor, zoom_factor, 1), order=1)  # changed to add cw

        # Trim off any extra pixels
        trim_top = (img.shape[0] - h) // 2
        trim_left = (img.shape[1] - w) // 2  # added

        image = img[trim_top:trim_top + h, trim_left:trim_left + w]  # changed

        return image
