"""
FILE:            base.py
SW-COMPONENT:    Adversarial attack base class script
DESCRIPTION:     Script containing base classes for adversarial attacks
COPYRIGHT:       (C) TU Braunschweig

06.10.2021, TU Braunschweig, Andreas Bär, Edgard Moreira Minete
Initial release.
"""

import abc
import warnings

import torch


class BaseAttack(abc.ABC):
    """
    Abstract base class for all attack abstract base classes.
    """

    def __init__(self):
        """
        """

    def __call__(self, image, model, label=None) -> torch.Tensor:
        """
        Compute adversarial perturbation and feed it through the model
        :param: image: the input image to be perturbed
        :param: model: the model

        :return: adv_example: the adversarial example
        """

        # Use default and print warning if not implemented by children attack class
        adv_ex = torch.zeros(image.size())
        warnings.warn("Using default perturbation of 0", category=UserWarning)

        return adv_ex

    def cuda(self) -> None:
        """
        Send variables to CUDA
        """
        self.is_cuda = True

    def cpu(self) -> None:
        """
        Send variables to CPU
        """
        self.is_cuda = False

    def show_parameters(self) -> None:
        """
        Print all the parameters of the FGSM method
        """
        print(self.parameters, flush=True)
