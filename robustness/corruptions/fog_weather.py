"""
FILE:            fog_weather.py
SW-COMPONENT:    Corruption script of fog_weather
DESCRIPTION:     Script containing a class for corrupting an image with fog weather
COPYRIGHT:       (C) TU Braunschweig

06.10.2021, TU Braunschweig, Andreas Bär, Edgard Moreira Minete
Initial release.
"""

import numpy as np
import torch

from robustness.corruptions import SEVERITY
from robustness.corruptions.base import BaseCorruption


# Todo:
#  This code might be transferable to PyTorch. However, not sure if this is really worth it.

class FogWeather(BaseCorruption):
    """
    This code corrupts an image with fog weather
    We took the original code from
    https://github.com/hendrycks/robustness/blob/master/ImageNet-C/imagenet_c/imagenet_c/corruptions.py
    and modified it.
    """

    def __init__(
            self,
    ) -> None:

        """
        Creates an FogWeather instance
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
        c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity - 1]

        # Transpose channels and remove batch dimension
        assert image.shape[0] == 1, "Only batch size of 1 is supported"
        x = image[0, :, :, :]
        x = np.array(x).transpose((1, 2, 0))  # C, H, W
        max_val = x.max()

        # Apply fog weather corruption and clip it to [0-1] range
        max_ = max(x.shape)
        binary_max_ = bin(max_)[2:]  # This line will give you binary code of max(x.shape), e.g., 10001001
        if binary_max_.count("1") > 1:  # Check whether max(x.shape) is a power of 2
            mapsize = int('1' + '0' * len(binary_max_), 2)  # Set mapsize to the closest bigger power of 2
        else:
            mapsize = max_  # As max(x.shape) a power of two, initialize mapsize with max(x.shape)
        x += c[0] * self.plasma_fractal(wibbledecay=c[1], mapsize=mapsize)[:x.shape[0], :x.shape[1]][..., np.newaxis]
        corrupted_image = np.clip(x * max_val / (max_val + c[0]), 0, 1).transpose((2, 0, 1))
        return torch.from_numpy(corrupted_image).unsqueeze(0)

    # After thoroughly testing this function and comparing it with Hendrycks implementation, the following was found:
    # If the array is transferred from numpy to torch and back to numpy, we can still achieve np.allclose() being true
    # However, if we multiply the Tensor by 255 (as well as Hendrycks multiplying it by 255) and casting to np.uint8(),
    # it will result in np.allclose() returning False (the exact difference in values is 1 or 1/255 after
    # normalizing back to [0, 1]).
    # A solution to this problem is the following: Before transforming from torch to numpy via torch.from_numpy()
    # one can run torch.set_default_dtype(torch.float64). This will set torch default dtype to float64
    # (the dafault dtype of numpy). Now multiplying by 255, casting to torch.uint8/np.uint8 and dividing by 255 will
    # give the same value as with numpy. And this actually applies for all corruptions! :)

    # modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
    @staticmethod
    def plasma_fractal(mapsize=256, wibbledecay=3):
        """
        Generate a heightmap using diamond-square algorithm.
        Return square 2d array, side length 'mapsize', of floats in range 0-255.
        'mapsize' must be a power of two.
        """
        assert (mapsize & (mapsize - 1) == 0)
        maparray = np.empty((mapsize, mapsize), dtype=np.float_)
        maparray[0, 0] = 0
        stepsize = mapsize
        wibble = 100

        def wibbledmean(array):
            return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

        def fillsquares():
            """For each square of points stepsize apart,
               calculate middle value as mean of points + wibble"""
            cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
            squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
            squareaccum += np.roll(squareaccum, shift=-1, axis=1)
            maparray[stepsize // 2:mapsize:stepsize,
            stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

        def filldiamonds():
            """For each diamond of points stepsize apart,
               calculate middle value as mean of points + wibble"""
            mapsize = maparray.shape[0]
            drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
            ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
            ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
            lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
            ltsum = ldrsum + lulsum
            maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
            tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
            tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
            ttsum = tdrsum + tulsum
            maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

        while stepsize >= 2:
            fillsquares()
            filldiamonds()
            stepsize //= 2
            wibble /= wibbledecay

        maparray -= maparray.min()
        return maparray / maparray.max()
