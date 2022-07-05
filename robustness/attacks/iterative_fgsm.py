"""
FILE:            fgsm.py
SW-COMPONENT:    Adversarial attack script of iterative FGSM
DESCRIPTION:     Script containing a class for iterative FGSM
COPYRIGHT:       (C) TU Braunschweig

01.03.2022, TU Braunschweig, Andreas Bär, Edgard Moreira Minete
Initial release.
"""
from robustness.attacks.momentum_iterative_fgsm import MomentumIterativeFGSM


class IterativeFGSM(MomentumIterativeFGSM):
    """
    This code implements the basic iterative method (BIM) or iterative FGSM
    of the paper "Adversarial Machine Learning at Scale" or "Adversarial Examples in the Physical World"
    Paper Link: https://arxiv.org/abs/1611.01236
    Paper Link: https://arxiv.org/abs/1607.02533

    The termin iterative FGSM was used in: https://arxiv.org/abs/1710.06081. For better understanding of this method
    we stick to this term rather than the term basic iterative method.
    Distance norm = L_inf
    """

    def __init__(
            self,
            epsilon,
            alpha,
            iterations=None,
    ) -> None:

        """
        :param: epsilon: attack strength
        :param: alpha: size of the attack strength per step
        :param_ iterations: the total number of iterations
        """
        if iterations == None:
            iterations = int(min(epsilon * 255 + 4, 1.25 * epsilon * 255))
        else:
            iterations = iterations

        # Note that we first go through the __init__ of the class MomentumIterativeFGSM.
        # momentum = 0 --> No momentum
        super().__init__(epsilon=epsilon, alpha=alpha, momentum=0, iterations=iterations)

        # Add parameters to the print list
        self.parameters = {
            "epsilon": self.epsilon,  # From parent class
            "alpha": self.alpha,  # From parent class
            "iterations": self.iterations,  # From parent class
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
