"""
FILE:            corruption.py
SW-COMPONENT:    Corruption attack base script
DESCRIPTION:     Script containing base classes for corruption attacks
COPYRIGHT:       (C) TU Braunschweig

13.01.2022, TU Braunschweig, Andreas Bär, Edgard Moreira Minete
Initial release.
"""

import abc
import warnings

import torch


class BaseCorruption(abc.ABC):
    """
    Abstract base class for all corruption abstract base classes.
    """

    def __init__(self):
        """

        """

    def __call__(self, image, severity) -> torch.Tensor:
        """
        Add the corruption to the input image
        :param: image: the input image to be corrupted

        :return: corrupted_image: the corrupted image
        """

        # Use default and print warning if not implemented by children attack class
        corrupted_image = torch.zeros(image.size())
        warnings.warn("Using default corruption (no corruption)", category=UserWarning)

        return corrupted_image
