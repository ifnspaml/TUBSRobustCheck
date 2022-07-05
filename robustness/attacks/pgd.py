"""
FILE:            pgd.py
SW-COMPONENT:    Adversarial attack script of projected gradient descent (PGD)
DESCRIPTION:     Script containing a class for creating a PGD attack
COPYRIGHT:       (C) TU Braunschweig

01.03.2022, TU Braunschweig, Andreas Bär, Edgard Moreira Minete
Initial release.
"""

import torch

from robustness.attacks.iterative_fgsm import IterativeFGSM


class PGD(IterativeFGSM):
    """
    This code implements the projected gradient descent attack
    of the paper "Towards Deep Learning Models Resistant to Adversarial Attacks"
    Paper Link: https://arxiv.org/abs/1706.06083

    Distance norm = L_inf
    """

    def __init__(
            self,
            epsilon,
            alpha,
            iterations,
            random_start=True
    ) -> None:
        """
        :param: epsilon: attack strength
        :param: alpha: size of the attack strength per step
        :param_ iterations: the total number of iterations
        """
        ## Note that we first go through the __init__ of the class IterativeFGSM.
        super().__init__(epsilon, alpha, iterations)

        assert random_start in [0, 1, False, True], \
            "The parameter random_start needs to be either 0, 1, False, or True."

        self.random_start = bool(random_start)

        # Add parameters to the print list
        self.parameters = {
            "epsilon": self.epsilon,  # From parent class
            "alpha": self.alpha,  # From parent class
            "iterations": self.iterations,  # From parent class
            "random_start": self.random_start,
            "loss": self.loss  # From parent class
        }

    def __call__(self, x, model, y, targetted=False):
        """
        Implementation of the attack
        :param x: images
        :param model: model
        :param y: labels/targets
        :param targetted: boolean setting to targetted or untargetted attack behaviour

        :return: adv_ex (the adversarial example)
        """
        assert x.device == y.device == next(model.parameters()).device, \
            f"The devices of x and model should be the same but are x:{x.device}, model:{model.device}"

        assert targetted in [0, 1, False, True], \
            "The parameter targetted needs to be either 0, 1, False, or True."

        # Collect input image, model, target segmentation map and label
        x = x
        y = y
        model = model
        self.targetted = targetted

        return self.attack(x, model, y)

    def attack(self, x, model, y):
        """
        Perform a PGD attack

        :param x: image(s)
        :param y: targets/labels
        :param model: model

        :return: adv_ex (the adversarial example)
        """
        x = x.clone().detach()
        y = y.clone().detach()
        model = model

        x_adv = x.clone().detach()

        if self.random_start:
            # A random perturbation in [-epsilon, epsilon]
            r = (torch.rand(x.shape).cuda() - 0.5) * self.epsilon
        else:
            # This is eventually iterative FGSM
            r = torch.zeros(x.shape, dtype=torch.float).cuda()

        x_adv = torch.clamp(x_adv + r, 0, 1).detach()

        # Compute adversarial example
        for _ in range(self.iterations):
            # Compute forward path
            x_adv.requires_grad = True
            y_ = model(x_adv)

            if self.targetted:
                loss = - self.loss(y_, y)
            else:
                loss = self.loss(y_, y)

            grad = torch.autograd.grad(loss, x_adv,
                                       retain_graph=False,
                                       create_graph=False)[0]

            # Compute adversarial perturbation
            x_adv = x_adv.detach() + self.alpha * grad.sign()
            delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)

            # Bound adversarial example to the range [0-1]
            x_adv = torch.clamp(x + delta, 0, 1).detach()

        return x_adv
