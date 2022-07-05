"""
FILE:            pixelate_digital.py
SW-COMPONENT:    Corruption script of pixelate_digital
DESCRIPTION:     Script containing a class for corrupting an image with pixelate digital
COPYRIGHT:       (C) TU Braunschweig

06.10.2021, TU Braunschweig, Andreas Bär, Edgard Moreira Minete
Initial release.
"""

import torchvision.transforms as transform
from PIL import Image as PILImage

from robustness.corruptions import SEVERITY
from robustness.corruptions.base import BaseCorruption


# Todo:
#  This code might be transferable to PyTorch (e.g. using kornia). However, not sure if this is really worth it.

class PixelateDigital(BaseCorruption):
    """
    This code corrupts an image with pixelate digital
    We took the original code from
    https://github.com/hendrycks/robustness/blob/master/ImageNet-C/imagenet_c/imagenet_c/corruptions.py
    and modified it.
    """

    def __init__(
            self,
    ) -> None:
        """
        Creates an PixelateDigital instance
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
        c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]

        # Transpose channels and remove batch dimension
        assert image.shape[0] == 1, "Only batch size of 1 is supported"
        x = image[0, :, :, :]

        # Convert tensor to PIL
        x = transform.ToPILImage()(x)

        # Apply pixelate digital corruption and clip it to [0-1] range
        x = x.resize((int(x.size[0] * c), int(x.size[1] * c)), PILImage.BOX)
        corrupted_image = x.resize((image.shape[3], image.shape[2]), PILImage.BOX)

        return transform.ToTensor()(corrupted_image).unsqueeze(0)
