"""
FILE:            fgsm.py
SW-COMPONENT:    Adversarial attack script of FGSM
DESCRIPTION:     Script containing a class for FGSM
COPYRIGHT:       (C) TU Braunschweig

01.03.2022, TU Braunschweig, Andreas Bär, Edgard Moreira Minete
Initial release.
"""
from robustness.attacks.iterative_fgsm import IterativeFGSM


class FGSM(IterativeFGSM):
    """
    This code implements the Fast Gradient Sign Method (FGSM) of
    the paper "Explaining and harnessing adversarial examples"
    Paper link: https://arxiv.org/abs/1412.6572

    Distance norm = L_inf
    """

    def __init__(
            self,
            epsilon,
    ) -> None:
        """
        :param: epsilon: attack strength
        """
        # Note that we first go through the __init__ of the class MomentumIterativeFGSM.
        # alpha = epsilon --> FGSM only applies one step, so the step size needs to be equal to epsilon
        # iterations = 0 --> FGSM only applies one step
        super().__init__(epsilon=epsilon, alpha=epsilon, iterations=1)

        # Add parameters to the print list
        self.parameters = {
            "epsilon": self.epsilon,  # From parent class
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
        return super().__call__(x, model, y, targetted)

    def attack(self, x, model, y):
        """
        Perform a targetted/untargetted BIM attack

        :param x: image(s)
        :param y: targets/labels
        :param model: model

        :return: adv_ex (the adversarial example)
        """

        return super()._attack(x, model, y)
