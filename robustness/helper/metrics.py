# ToDo: Put the metrics evaluator here and equip him with everything which is out there in world

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

import json
import os

import numpy as np
import torch


class MetricsCollector(object):
    """
    Class to keep track of metrics collected for different corruptions at different severity levels
    """

    def __init__(self, corruptions, severities, metrics, default=None):
        """
        Init function

        :param corruptions: Corruptions that are tracked
        :param severities: Levels of severity that are investigated
        :param metrics: Metrics that are tracked
        :param default: Default value for initializing the dictionary
        """
        assert isinstance(metrics, (list, str)), f"Datatype {type(metrics)} of metrics is not supported!"
        assert isinstance(severities, list) or isinstance(severities, str) or isinstance(severities, int), \
            f"Datatype {type(severities)} of severities is not supported!"
        assert isinstance(corruptions, (list, str)), f"Datatype {type(corruptions)} of corruptions is not supported!"

        # Save the severities and keys for reference when adding values
        self.severities = severities.copy()
        self.corruptions = corruptions.copy()
        self.metrics = metrics.copy()

        # Standardize the dictionary keys
        if isinstance(self.severities, list):
            self.severities = [str(s) for s in self.severities]
        elif isinstance(self.severities, (str, int)):
            self.severities = [str(self.severities)]
        self.severities += ['all']

        if isinstance(self.corruptions, (list)):
            pass
        elif isinstance(self.corruptions, (str)):
            self.corruptions = [self.corruptions]

        if isinstance(self.metrics, (list)):
            pass
        elif isinstance(self.metrics, (str)):
            self.metrics = [self.metrics]

        # Build dictionary
        self.corruptions_dict = {}
        for c in self.corruptions:
            self.corruptions_dict[c] = {}
            for s in self.severities:
                self.corruptions_dict[c][s] = {}
                for m in self.metrics:
                    self.corruptions_dict[c][s][m] = default

    def add(self, keys, value):
        """
        Add a value to self.corruptions_dict at position [c][s][m].

        :param keys: List of three elements with [c = corruption, s = severity, m = metric]
        :param value: Value to the respective dict entry
        :return:
        """
        # keys: c = corruption, s = severity, m = metric
        assert isinstance(keys, list), "keys should be a list."
        assert len(keys) == 3, "keys should have a length of 3."
        assert not isinstance(value, dict), "value should be either a scalar or a list/tuple of scalars."
        if isinstance(value, (list, tuple, np.ndarray, torch.Tensor)):
            tmp = np.asarray(value).tolist()
            value = ','.join([str(v) for v in tmp])

        keys = [str(k) for k in keys]
        c, s, m = keys
        # Here we want to ensure that the keys were given in the correct order
        assert c in self.corruptions, f"{c} does not exist in corruptions. Maybe you chose the wrong order of keys."
        assert s in self.severities, f"{s} does not exist in severities. Maybe you chose the wrong order of keys."
        assert m in self.metrics, f"{m} does not exist in metrics. Maybe you chose the wrong order of keys."
        self.corruptions_dict[c][s][m] = value

    def print(self, c=None, s=None, m=None):
        """
        Print the current dictionary.

        :return:
        """
        assert c in self.corruptions + [
            None], f"{c} does not exist in corruptions. Maybe you chose the wrong order of keys."
        assert s in self.severities + [
            None], f"{s} does not exist in severities. Maybe you chose the wrong order of keys."
        assert m in self.metrics + [None], f"{m} does not exist in metrics. Maybe you chose the wrong order of keys."

        if c is not None:
            if s is not None:
                if m is not None:
                    print(json.dumps(self.corruptions_dict[c][s][m], indent=4))
                else:
                    print(json.dumps(self.corruptions_dict[c][s], indent=4))
            else:
                print(json.dumps(self.corruptions_dict[c], indent=4))
        else:
            print(json.dumps(self.corruptions_dict, indent=4))

    def save(self, path='./', name='metrics', type='.txt'):
        """
        Save the current dictionary to a file

        :return:
        """
        with open(os.path.join(path, name + type), 'w') as fp:
            json.dump(self.corruptions_dict, fp, indent=4)

    def get(self):
        """
        return the current dictionary

        :return:
        """
        return self.corruptions_dict


# ToDo: Reimplement Marvins function in PyTorch
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
        miou = torch.nanmean(iou)
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

        freq = self.conf.sum(axis=1) \
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


if __name__ == "__main__":
    collector = MetricsCollector(["fgsm", "pgd"], [1, 2, 3], ["psnr", "miou"])
    value = torch.Tensor(np.asarray([23.456, 24.123, 2524.12]))  #
    # value = np.asarray([23.456, 24.123, 2524.12])
    print(type(value))
    collector.add(["fgsm", 1, "psnr"], value)
    collector.print()
