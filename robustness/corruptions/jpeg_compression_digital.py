"""
FILE:            jpeg_compression_digital.py
SW-COMPONENT:    Corruption script of jpeg_compression_digital
DESCRIPTION:     Script containing a class for corrupting an image with jpeg compression digital
COPYRIGHT:       (C) TU Braunschweig

06.10.2021, TU Braunschweig, Andreas Bär, Edgard Moreira Minete
Initial release.
"""

from io import BytesIO

import torchvision.transforms as transform
from PIL import Image as PILImage

from robustness.corruptions import SEVERITY
from robustness.corruptions.base import BaseCorruption


# Todo:
#  This code is transferable to PyTorch (e.g. using kornia). However, not sure if this is really worth it.

class JpegCompressionDigital(BaseCorruption):
    """
    This code corrupts an image with jpeg compression digital
    We took the original code from
    https://github.com/hendrycks/robustness/blob/master/ImageNet-C/imagenet_c/imagenet_c/corruptions.py
    and modified it.
    """

    def __init__(
            self,
    ) -> None:
        """
        Creates an JpegCompressionDigital instance
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
        c = [25, 18, 15, 10, 7][severity - 1]

        # Apply jpeg compression digital corruption
        # Convert tensor to PIL
        tr_PIL = transform.ToPILImage()
        image = tr_PIL(image[0, :, :, :])

        # Save image as jpeg with quality given by the severity
        output = BytesIO()
        image.save(output, 'JPEG', quality=c)

        # Open PIL image and transform it again to tensor
        corrupted_image = PILImage.open(output)

        return transform.ToTensor()(corrupted_image).unsqueeze(0)
