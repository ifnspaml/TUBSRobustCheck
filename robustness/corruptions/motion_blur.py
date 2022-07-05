"""
FILE:            motion_blur.py
SW-COMPONENT:    Corruption script of motion_blur
DESCRIPTION:     Script containing a class for corrupting an image with motion blur
COPYRIGHT:       (C) TU Braunschweig

06.10.2021, TU Braunschweig, Andreas Bär, Edgard Moreira Minete
Initial release.
"""

from io import BytesIO

import cv2
import numpy as np
import torch
import torchvision.transforms as transform

from robustness.corruptions import SEVERITY, MotionImage
from robustness.corruptions.base import BaseCorruption


# Todo:
#  This code is transferable to PyTorch (e.g. using kornia). However, not sure if this is really worth it.

class MotionBlur(BaseCorruption):
    """
    This code corrupts an image with motion blur
    We took the original code from
    https://github.com/hendrycks/robustness/blob/master/ImageNet-C/imagenet_c/imagenet_c/corruptions.py
    and modified it.
    """

    def __init__(
            self,
    ) -> None:

        """
        Creates an MotionBlur instance
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
        c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]

        # Remove batch dimension
        assert image.shape[0] == 1, "Only batch size of 1 is supported"
        # Convert tensor to PIL
        image = transform.ToPILImage()(image[0, :, :, :])

        output = BytesIO()
        image.save(output, format='PNG')
        image = MotionImage(blob=output.getvalue())

        image.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))
        x = cv2.imdecode(np.fromstring(image.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)

        # Apply motion blur corruption and clip it to [0-1] range
        if x.shape != (x.shape[0], x.shape[1]):  # BGR to RGB
            corrupted_image = np.clip(x[..., [2, 1, 0]] / 255., 0, 1).transpose((2, 0, 1))

        else:  # greyscale to RGB
            corrupted_image = np.clip(np.array([x, x, x]) / 255., 0, 1)
        return torch.from_numpy(corrupted_image).unsqueeze(0)
