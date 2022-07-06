"""
FILE:            momentum_iterative_fgsm.py
SW-COMPONENT:    Adversarial attack script of momentum-iterative FGSM
DESCRIPTION:     Script containing the class for momentum-iterative FGSM
COPYRIGHT:       (C) TU Braunschweig

01.03.2022, TU Braunschweig, Andreas Bär, Edgard Moreira Minete
Initial release.
"""
import torch

from robustness.attacks.base import BaseAttack


class MomentumIterativeFGSM(BaseAttack):
    """
    This code implements the momentum-iterative FGSM (MI-FGSM)
    of the paper "Boosting Adversarial Attacks with Momentum"
    Paper Link: https://arxiv.org/abs/1710.06081

    Distance norm = L_inf
    """

    def __init__(
            self,
            epsilon,
            alpha,
            momentum,
            iterations=None,
    ) -> None:

        """
        :param: epsilon: attack strength
        :param: alpha: size of the attack strength per step
        :param: momentum: weight factor of the old gradient in each iteration
        :param_ iterations: the total number of iterations
        """
        super().__init__()

        ## Perform additional assertions
        assert isinstance(epsilon, (float, int)) and (0 <= epsilon <= 1), \
            "The perturbation strength `epsilon` has to be of type 'float' or 'int' and be in range [0, 1]"

        assert isinstance(alpha, (float, int)) and (0 <= alpha <= 1), \
            "The step size `alpha` has to be of type 'float' or 'int', be in range [0, 1] and smaller than epsilon."

        assert isinstance(momentum, (float, int)) and (momentum >= 0), \
            "The gradient momentum `momentum` has to be of type 'float' or 'int' & positive."

        assert isinstance(iterations, int) and (iterations >= 0), \
            "The number of iterations 'iterations' has to be of type 'int' & positive."

        self.epsilon = epsilon
        self.alpha = alpha
        self.momentum = momentum
        self.iterations = iterations
        self.loss = torch.nn.CrossEntropyLoss()

        # Add parameters to the print list
        self.parameters = {
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "momentum": self.momentum,
            "iterations": self.iterations,
            "loss": self.loss
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
            f"The devices of x, y, and model should be the same but are x:{x.device}, y:{y.device}, and model:{model.device}"

        assert targetted in [0, 1, False, True], \
            "The parameter targetted needs to be either 0, 1, False, or True."

        # Collect input image, model, target segmentation map and label
        x = x
        y = y
        model = model
        self.targetted = targetted

        return self._attack(x, model, y)

    def _attack(self, x, model, y):
        """
        Perform a targetted/untargetted iterative FGSM attack

        :param x: image(s)
        :param y: targets/labels
        :param model: model

        :return: adv_ex (the adversarial example)
        """
        x = x.clone().detach()
        y = y.clone().detach()
        model = model

        x_adv = x.clone().detach()
        r = torch.zeros(x.shape, dtype=torch.float).cuda()
        grad_old = torch.zeros(x.shape, dtype=torch.float).cuda()

        # Compute adversarial example
        for _ in range(self.iterations):
            # Compute forward path
            x_adv.requires_grad = True
            y_ = model(x_adv)

            if self.targetted:
                loss = - self.loss(y_, y)
            else:
                loss = self.loss(y_, y)

            grad_ = torch.autograd.grad(loss, x_adv,
                                        retain_graph=False,
                                        create_graph=False)[0]

            # Note that self.momentum = 0 implements the basic iterative FGSM
            grad = self.momentum * grad_old + (grad_ / torch.norm(grad_, p=1))
            grad_old = grad_.detach()

            # Compute adversarial perturbation
            r += self.alpha * grad.sign()
            r = torch.clamp(r, -self.epsilon, self.epsilon).detach()

            # Bound adversarial example to the range [0-1]
            x_adv = torch.clamp(x + r, 0, 1).detach()

        return x_adv
