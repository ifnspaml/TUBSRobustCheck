"""
FILE:            snow_weather.py
SW-COMPONENT:    Corruption script of snow_weather
DESCRIPTION:     Script containing a class for corrupting an image with snow weather
COPYRIGHT:       (C) TU Braunschweig

06.10.2021, TU Braunschweig, Andreas Bär, Edgard Moreira Minete
Initial release.
"""

from io import BytesIO

import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from scipy.ndimage import zoom as scizoom

from robustness.corruptions import SEVERITY, MotionImage
from robustness.corruptions.base import BaseCorruption


# Todo:
#  This code is transferable to PyTorch (e.g. using kornia). However, not sure if this is really worth it.

class SnowWeather(BaseCorruption):
    """
    This code corrupts an image with snow weather
    We took the original code from
    https://github.com/hendrycks/robustness/blob/master/ImageNet-C/imagenet_c/imagenet_c/corruptions.py
    and modified it.
    """

    def __init__(
            self,
    ) -> None:
        """
        Creates an SnowWeather instance
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
        c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
             (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
             (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
             (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
             (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]

        # Transpose channels and remove batch dimension
        assert image.shape[0] == 1, "Only batch size of 1 is supported"
        x = image[0, :, :, :]
        x = np.array(x).transpose((1, 2, 0))  # C, H, W

        # Apply snow weather corruption and clip it to [0-1] range
        snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])  # [:2] for monochrome
        snow_layer = self.clipped_zoom(snow_layer[..., np.newaxis], c[2])
        snow_layer[snow_layer < c[3]] = 0

        snow_layer = PILImage.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
        output = BytesIO()
        snow_layer.save(output, format='PNG')
        snow_layer = MotionImage(blob=output.getvalue())

        snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

        snow_layer = cv2.imdecode(np.fromstring(snow_layer.make_blob(), np.uint8),
                                  cv2.IMREAD_UNCHANGED) / 255.
        snow_layer = snow_layer[..., np.newaxis]

        x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).
                                               reshape(x.shape[0], x.shape[1], 1) * 1.5 + 0.5)

        corrupted_image = np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1).transpose((2, 0, 1))

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
