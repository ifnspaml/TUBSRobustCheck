"""
FILE:            gd_uap.py
SW-COMPONENT:    Adversarial attack script of generalizable data-free (GD) UAP
DESCRIPTION:     Script containing a class for creating an GD UAP attack
COPYRIGHT:       (C) TU Braunschweig

01.03.2022, TU Braunschweig, Andreas Bär, Edgard Moreira Minete
Initial release.
"""

import time
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

from robustness.attacks.base import BaseAttack
from robustness.helper.metrics import Evaluator

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

# ImageNet mean and std
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Cityscapes mean and std
# MEAN = []
# STD = []

# Layers to optimize on
LAYERS = (nn.Conv2d)  # nn.ReLU

# Saturation threshold
THETA_SATURATION = 10 ** -5


class GDUAP(BaseAttack):
    """
    This class implements the Generalizable Data-free Universal Adversarial Perturbations (GD-UAP) adversarial attack
    Paper link: https://arxiv.org/abs/1801.08092

    Distance norm = L_inf
    """

    def __init__(
            self,
            epsilon,
            prior_mode,
            max_iteration,
            num_classes=19,
            image_size=[1024, 2048]
    ) -> None:

        """
        :args: Arguments for the attack
        """
        super().__init__()

        assert isinstance(epsilon, (float, int)), "'epsilon' (attack strength) has to be of type 'float' or 'int'."
        assert 0 <= epsilon <= 1, "'epsilon' (attack strength) has to be an element of [0, 1]."

        # Check if max_iter_uni is acceptable
        assert isinstance(max_iteration, (int)), "'max_iteration' has to be of type 'int'."
        assert max_iteration > 0, "'max_iteration' has to be positive."

        # Check prior mode
        assert prior_mode in ['off', 'range', 'data'], \
            "Please choose a valid prior mode out of 'off', 'range', and 'data'"

        # Setting GD-UAP initial state
        self.computed = False

        ## Initialize attack hyperparameters

        # Prior mode
        self.prior_mode = prior_mode
        self.max_iteration = max_iteration
        self.img_size = image_size
        self.epsilon = epsilon
        self.num_classes = num_classes

        # Initialize the best UAP
        self.best_uap = torch.zeros(1,
                                    3,
                                    self.img_size[0],
                                    self.img_size[1])

        # Initialize the optimizer, fooling rate and best fooling rate
        self.fooling_rates = [0]
        self.best_fooling_rate = 0

        # Initialize model
        self.model = None

        # Initialize (iterative) dataloader for training
        self.dataloader = None
        self.iter_dataloader = None

        # Initialize (iterative) dataloader for validation
        self.dataloader_val = None
        self.iter_dataloader_val = None

        if self.prior_mode == 'range':
            _channels = []
            # Create random R, G and B channel
            for mean, std in zip(MEAN, STD):
                _channels.append(torch.normal(mean=mean, std=std, size=(1, self.img_size[0] * 2, self.img_size[1] * 2)))

            # Put all channels together
            self.random_x = torchvision.transforms.ToPILImage()(torch.cat(_channels, dim=0))

            # Initialize crop operation
            self.crop = torchvision.transforms.RandomCrop((self.img_size[0],
                                                           self.img_size[1]))
            self.toTensor = torchvision.transforms.ToTensor()

        # Initialize loss
        self.loss = torch.zeros(1).cuda()

        # Initialize saturation with a small number
        self.saturation_rate = 10 ** -8

        # Create iouEval objects
        self.metric = Evaluator(self.num_classes)

        # Add parameters to the print list
        self.parameters = {
            "epsilon": self.epsilon,
            "prior_mode": self.prior_mode,
        }

    def __call__(self, x, model):
        """
        Implementation of GD-UAP method.

        :param x: image(s) in normalized range, i.e., [0, 1]
        :param model: model

        :return: adv_ex (the adversarial example)
        """
        # Check if the adversarial perturbation was already calculated
        assert self.computed, "The adversarial perturbation should be computed first."

        # Assert input image is hosted on CUDA
        assert x.is_cuda, "Input image is not hosted on CUDA"

        # Assert model parameters are on CUDA
        assert next(model.parameters()).is_cuda, "Model is not hosted on CUDA"

        # Clamp adversarial example
        adv_ex = torch.clamp(x + self.best_uap, 0, 1)

        return adv_ex

    def compute_perturbation(self, model, dataset):
        """
        Compute the adversarial perturbation from a dataset

        :param dataset: the dataset to compute the adversarial perturbation
        :param model: the model to compute the adversarial perturbation

        :return: the adversarial perturbation
        """
        start = time.time()

        # Initialize the universal adversarial perturbation UAP randomly at U[-epsilon, epsilon]
        uap = (torch.rand(1, 3, self.img_size[0], self.img_size[1]) - 0.5) * 2 * self.epsilon
        uap = uap.cuda()

        # Create leaf variable and optimizer
        uap = torch.autograd.Variable(uap, requires_grad=True)
        optimizer = optim.Adam([uap], lr=0.1)

        # Assign model
        self.model = model

        # Getting convolutional layers to optimize on
        self.set_hooks(model=model)

        # Configure the dataset
        self.setup_dataloader(dataset)

        # Configure the training loop
        stop_validation = False
        stop_counter = 0
        prev_check = 0
        torch.autograd.set_detect_anomaly(True)

        for step in range(self.max_iteration):
            # Initialize UAP with updated UAP
            uap_ = uap

            # Generate perturbed input for the network
            imgs = self.get_input(uap_)

            # Pass the input through the network. The loss is saved inside self.loss via forward hooks.
            _ = model(imgs)

            # Zero out the gradients, backpropagation and optimizer update
            optimizer.zero_grad()
            self.loss.backward(retain_graph=True)
            optimizer.step()

            # Reset loss
            self.loss = torch.zeros(1).cuda()

            # Clip the value range of the copied uap
            uap_ = self.proj_lp(uap_, self.epsilon)

            # Compute the saturation rate
            current_saturation = self.compute_saturation_rate(uap_, self.epsilon)

            # Check saturation constraint
            if abs(current_saturation - self.saturation_rate) < THETA_SATURATION \
                    and current_saturation > 0.5:
                rescale = True
            else:
                rescale = False

            # Set the new saturation rate
            self.saturation_rate = current_saturation

            check_dif = step - prev_check

            # Do validation if criteria is met
            if not stop_validation and (
                    (rescale and check_dif > self.max_iteration // 200) or check_dif == self.max_iteration // 100):
                print("Validation ...")
                prev_check = step

                # Get the current fooling rate and append to list
                self.fooling_rates.append(self.get_fooling_rate(uap_))

                # Check if the current fooling rate is the best
                if self.fooling_rates[-1] > self.best_fooling_rate:

                    # Set the best fooling rate
                    self.best_fooling_rate = self.fooling_rates[-1]

                    # Set the best uap
                    self.best_uap = uap_
                    stop_counter = 0
                else:
                    stop_counter += 1
            else:
                pass

            if step % self.max_iteration // 400 == 0:
                self.log(step)
            else:
                pass

            # Check stopper
            if stop_counter >= 10:
                print("Stop counter reached limit. Training is stopped!")
                break
            else:
                pass

            if rescale:
                # Rescale the UAP
                uap = torch.div(uap, 2.0)
                prev_check = step
                print(f"UAP is rescaled, saturation: {current_saturation}")

        # Remove all hooks
        self.remove_hooks(self.model)

        # Delete the loss
        del self.loss

        # Update adversarial state
        self.computed = True

        end = time.time() - start
        h = int(end // (60 * 60))
        m = int((end % (60 * 60)) // 60)
        s = int((end % (60 * 60)) % 60)
        print(f"Computation finished and took {h} h {m} min {s} sec.")

    def log(self, step):
        """
        Print results during UAP optimization

        :param step: actual step
        """
        print(f"Step: {step + 1} of {self.max_iteration} |",
              f"Current fooling rate {self.fooling_rates[-1]} |"
              f"Best Fooling rate: {self.best_fooling_rate} |",
              f"Saturation rate: {self.saturation_rate} |")

    def setup_dataloader(self, dataset):
        """
        Prepare the dataset for computing the universal adversarial perturbation

        :param dataset: the dataset to be optimized on
        """

        # Take the first 80% of the dataset as train data and the last 20% of the dataset as val data
        train_indices = range(0, len(dataset) - len(dataset) // 5, 1)
        val_indices = range(len(dataset) - len(dataset) // 5, len(dataset), 1)

        # Create train and val set
        trainset = torch.utils.data.Subset(dataset, train_indices)
        valset = torch.utils.data.Subset(dataset, val_indices)

        # Create train loader
        loader_train = DataLoader(trainset, 1, True, num_workers=1, pin_memory=True, drop_last=True)

        # Create val loader
        loader_val = DataLoader(valset, 1, False, num_workers=1, pin_memory=True, drop_last=True)

        # Add to self and make the data loaders iterable
        self.dataloader = loader_train
        self.dataloader_val = loader_val
        self.iter_dataloader = iter(loader_train)
        self.iter_dataloader_val = iter(loader_val)

    def get_input(self, v):
        """
        Return the network input depending on the prior mode

        :param v: universal adversarial perturbation

        :return: input to the network
        """

        # 'Data' prior mode
        if self.prior_mode == 'data':
            try:
                x = next(self.iter_dataloader)
            except StopIteration:
                self.iter_dataloader = iter(self.dataloader)
                x = next(self.iter_dataloader)
                imgs = x["color", 0, 0].cuda() + v

        # 'Range' prior mode
        elif self.prior_mode == 'range':
            x = self.toTensor(self.crop(self.random_x)).cuda()
            x = x.unsqueeze(0)
            imgs = x + v

        # 'OFF' prior mode
        elif self.prior_mode == 'off':
            imgs = v
        else:
            raise Exception(f"prior mode '{self.prior_mode}' is not defined")

        return imgs

    @staticmethod
    def compute_saturation_rate(v, epsilon):
        """
        Compute the rate of epsilon-saturated pixels

        :param v: universal adversarial perturbation
        :param epsilon: max-norm constraint

        :return: saturation rate
        """
        s = (torch.abs(v) == epsilon).sum().float() \
            / v.numel()
        return s

    def compute_feature_loss(self, module_, input_, output_):
        """
        Penalize the loss activations with an logarithmic L2 Norm.
        Aggregate the loss via sum over all layers that are of interest.

        :param module_: the model to be penalized
        :param input_: input
        :param output_: activations

        :return:
        """
        self.loss += -torch.log(torch.norm(output_, p=2) ** 2)

    def set_hooks(self, model) -> None:
        """
        Collect the layers to optimize on

        :param model: model to collect the labels
        """
        layers = []
        model.hooks = []

        # Refer to https://pytorch.org/docs/stable/nn.html#convolution-layers for nn.Conv layer types
        # Iterate over the model layers
        for name, layer in model.named_modules():
            if isinstance(layer, LAYERS):
                model.hooks.append(layer.register_forward_hook(self.compute_feature_loss))
                layers.append(name)
            else:
                pass

        print('Optimizing at the following layers:')
        print(layers)

    @staticmethod
    def remove_hooks(model) -> None:
        """
        Remove the collected layers

        :param model: the model
        """
        for h in model.hooks:
            h.remove()
        del model.hooks

    @staticmethod
    def proj_lp(v, epsilon=10.0 / 255, p="inf"):
        """
        Project the adversarial perturbation to the L_p ball

        :param v: the perturbed image
        :param epsilon: the attack strength
        :param p: norm

        :return: projection of the perturbed image on the L_p ball
        """
        if p == "inf":
            v = torch.clamp(v, -epsilon, epsilon)
        else:
            v = v * min(1, epsilon / (torch.norm(v, p) + 0.00001))

        return v

    def get_fooling_rate(self, uap):
        """
        Compute the generalized fooling rate as defined by Mopuri et al.
        GFR = (R - M(y_clean, y_adv)) / R
        M : Performance measure of range [0, R]
        y_clean : Clean output
        y_adv : Adversarial output
        R : Maximum values of M(...)

        :param uap: the adversarial perturbation

        :return: fooling rate
        """
        for step, data in enumerate(self.dataloader_val):
            # Clean input
            x = data["color", 0, 0].cuda()

            # Perturbed input
            x_adv = torch.clamp(x + uap, 0, 1)

            # clean output
            with torch.no_grad():
                out = torch.argmax(self.model(x), 1, keepdim=True)
                out_adv = torch.argmax(self.model(x_adv), 1, keepdim=True)

            # Add batch to miou computation
            self.metric.add_batch(out, out_adv)

        # Compute final miou
        miou = self.metric.miou()
        self.metric.reset()
        return 1 - miou
