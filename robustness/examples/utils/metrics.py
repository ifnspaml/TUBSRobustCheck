""""
FILE:            metrics.py
SW-COMPONENT:    Metrics script
DESCRIPTION:     Script for computing different metrics
COPYRIGHT:       (C) TU Braunschweig

This script provides the code to evaluate different metrics relying on a
confusion matrix.

30.11.2020, TU Braunschweig, Andreas Bär, Edgard Moreira
Initial release.
"""

import torch
import numpy as np


class Evaluator(object):
    """
    Evaluator class
    """

    def __init__(self, num_class):
        """
        Init function

        :param num_class: Number of classes
        """
        self.num_class = num_class
        self.conf = torch.zeros((self.num_class,) * 2).cuda()

    def acc(self):
        """
        Compute the pixel accuracy.

        :return: Pixel accuracy
        """
        acc = torch.diag(self.conf).sum() \
              / self.conf.sum()
        return acc

    def acc_class(self):
        """
        Compute the pixel accuracy per class.

        :return: Pixel accuracy per class
        """
        acc = torch.diag(self.conf) \
              / self.conf.sum(axis=1)
        acc = torch.mean(acc)
        return acc

    def miou(self):
        """
        Compute the mean intersection-over-union.

        :return: Mean intersection-over-union, IoU per class
        """

        iou = torch.diag(self.conf) \
              / (self.conf.sum(axis=1)
                 + self.conf.sum(axis=0)
                 - torch.diag(self.conf))
        miou = nanmean(iou)
        return miou

    def iou(self):
        """
        Compute the intersection-over-union.

        :return: Mean intersection-over-union, IoU per class
        """

        iou = torch.diag(self.conf) \
              / (self.conf.sum(axis=1)
                 + self.conf.sum(axis=0)
                 - torch.diag(self.conf))
        return iou

    def fw_iou(self):
        """
        Compute the frequency weighted mean intersection-over-union.

        :return: Frequency weighted mean intersection-over-union
        """

        freq = self.conf.sum(axis=1)\
               / self.conf.sum()
        iou = torch.diag(self.conf) \
              / (self.conf.sum(axis=1)
                 + self.conf.sum(axis=0)
                 - torch.diag(self.conf))

        fwiou = (freq[freq > 0] * iou[freq > 0]).sum()
        return fwiou

    def _generate_matrix(self, gt_image, pre_image):
        """
        Compute a confusion matrix.

        :param gt_image: Ground truth segmentation mask
        :param pre_image: Predicted segmentation mask
        :return: Confusion matrix
        """

        mask_gt = (gt_image >= 0) & (gt_image < self.num_class)
        mask_pred = (pre_image >= 0) & (pre_image < self.num_class)
        mask = mask_pred & mask_gt

        label = self.num_class * gt_image[mask].long() + pre_image[mask]

        # bincount is very slow for sparse on the gpu!
        # shifting to cpu and then back to gpu solved the problem!
        count = torch.bincount(label.cpu(),
                               minlength=self.num_class ** 2).cuda()
        confusion_matrix = count.reshape(self.num_class, self.num_class)

        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        """
        Update the current confusion matrix.

        :param gt_image: Ground truth segmentation mask
        :param pre_image: Predicted segmentation mask
        :return:
        """

        assert gt_image.shape == pre_image.shape
        self.conf += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        """
        Reset the confusion matrix.

        :return:
        """
        self.conf = torch.zeros((self.num_class,) * 2).cuda()

class EvaluatorNumpy(object):
    """
    Evaluator class
    """

    def __init__(self, num_class):
        """
        Init function

        :param num_class: Number of classes
        """
        self.num_class = num_class
        self.conf = np.zeros((self.num_class,) * 2)

    def acc(self):
        """
        Compute the pixel accuracy.

        :return: Pixel accuracy
        """
        acc = np.diag(self.conf).sum() \
              / self.conf.sum()
        return acc

    def acc_class(self):
        """
        Compute the pixel accuracy per class.

        :return: Pixel accuracy per class
        """
        acc = np.diag(self.conf) \
              / self.conf.sum(axis=1)
        acc = np.mean(acc)
        return acc

    def miou(self):
        """
        Compute the mean intersection-over-union.

        :return: Mean intersection-over-union, IoU per class
        """

        iou = np.diag(self.conf) \
              / (self.conf.sum(axis=1)
                 + self.conf.sum(axis=0)
                 - np.diag(self.conf))
        #miou = nanmean(iou)
        miou = np.nanmean(iou)
        return miou

    def iou(self):
        """
        Compute the intersection-over-union.

        :return: Mean intersection-over-union, IoU per class
        """

        iou = np.diag(self.conf) \
              / (self.conf.sum(axis=1)
                 + self.conf.sum(axis=0)
                 - np.diag(self.conf))
        return iou

    def fw_iou(self):
        """
        Compute the frequency weighted mean intersection-over-union.

        :return: Frequency weighted mean intersection-over-union
        """

        freq = self.conf.sum(axis=1)\
               / self.conf.sum()
        iou = np.diag(self.conf) \
              / (self.conf.sum(axis=1)
                 + self.conf.sum(axis=0)
                 - np.diag(self.conf))

        fwiou = (freq[freq > 0] * iou[freq > 0]).sum()
        return fwiou

    def _generate_matrix(self, gt_image, pre_image):
        """
        Compute a confusion matrix.

        :param gt_image: Ground truth segmentation mask
        :param pre_image: Predicted segmentation mask
        :return: Confusion matrix
        """

        mask_gt = (gt_image >= 0) & (gt_image < self.num_class)
        mask_pred = (pre_image >= 0) & (pre_image < self.num_class)
        mask = mask_pred & mask_gt
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)

        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        """
        Update the current confusion matrix.

        :param gt_image: Ground truth segmentation mask
        :param pre_image: Predicted segmentation mask
        :return:
        """

        assert gt_image.shape == pre_image.shape
        self.conf += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        """
        Reset the confusion matrix.

        :return:
        """
        self.conf = np.zeros((self.num_class,) * 2).cuda()

def nanmean(vec, *args, inplace=False, **kwargs):
    """
    Compute the mean, neglecting all nan values.

    :param vec: Vector for which the mean should be computed.
    :param args: Arguments
    :param inplace: Perform inplace operation or not
    :param kwargs: Keyword arguments
    :return: Mean without nan values.
    """
    # This is workaround for np.nanmean behaviour in PyTorch
    if not inplace:
        vec = vec.clone()
    is_nan = torch.isnan(vec)
    vec[is_nan] = 0
    return vec.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)
