"""
FILE:            metzen_uap.py
SW-COMPONENT:    Adversarial attack script of METZEN UAP attack
DESCRIPTION:     Script containing a class for creating a metzen uap attack
COPYRIGHT:       (C) TU Braunschweig

06.10.2021, TU Braunschweig, Andreas Bär, Edgard Moreira Minete
Initial release.
"""

from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from robustness.attacks.base import BaseAttack
from dataloader.definitions.labels_file import *


class MetzenUAP(BaseAttack):
    """
    This code implements the Metzen UAP of the paper
    "Universal Adversarial Perturbations Against Semantic Image Segmentation"
    Paper link: https://arxiv.org/pdf/1704.05712.pdf
    """

    def __init__(
            self,
            target_type,
            target=None,
            height=512,
            width=512,
            alpha=1.0/255,
            epsilon=10.0/255,
            target_class=11,
            iterations=60,
            weight=0.9999,
            eval_height=512,
            eval_width=1024,
            ignore_index=-100
    ) -> None:

        """
        Creates a `METZEN UAP` instance.

        :param epsilon: attack strength (in [0-1] space)
        :param alpha: attack alpha (in [0-1] space)
        :param target_type: the type of target
        :param target: the target image
        :param target_class: the target class
        :param width: periodic width tile size
        :param height: periodic height tile size
        :param iterations: the number of iterations
        :param weight: gradient loss weight
        :param eval_height: height of image for eval
        :param eval_width: width of image for eval
        """
        super().__init__()

        # Setting METZEN-UAP initial state
        self.computed = False

        ## Initialize parameters
        # Attack strength
        self.epsilon = epsilon

        # Check if the epsilon is valid
        assert isinstance(self.epsilon, (float, int)), "The perturbation `epsilon` has to be of type 'float' or 'int'."
        assert 0 <= self.epsilon <= 1, "Epsilon should be an element of [0-1]."

        # Attack alpha
        self.alpha = alpha

        # Batch size
        self.batch_size = None

        # Attack target type
        # Check if the target attack type is valid
        assert target_type in ["static", "dynamic"], "The target attack type should be either 'static' or 'dynamic'"

        # Assign attack type to internal variable
        self.target_type = target_type

        # Target image for static attack
        if target_type == 'static':
            assert target is not None, "A target image path 'target' should be given as input for the static attack"

        # Assign target path, if available
        self.target = target

        # Assign the target class
        self.target_class = target_class

        # Number of iterations
        self.iterations = iterations

        # Check if the number of iterations is acceptable
        assert isinstance(self.iterations, int), "'iterations' has to be of type 'int'."
        assert self.iterations > 0, "'iterations' has to be positive."

        # Gradient loss weight
        self.weight = weight

        # Ignore Index
        self.ignore_index = ignore_index

        # Periodic tile size width and heigth
        self.width = width
        self.height = height

        # Check if the periodic tile sizes are valid
        assert self.width > 0, "Periodic tile width should be greater than zero"
        assert isinstance(self.width, int), "Periodic tile 'width' has to be of type 'int'."
        assert self.height > 0, "Periodic tile height should be greater than zero"
        assert isinstance(self.height, int), "Periodic tile 'height' has to be of type 'int'."

        # Image size width and heigth (evaluation)
        self.eval_width = eval_width
        self.eval_height = eval_height

        # Instantiate the target_label variable
        self.target_label = None

        # Instantiate a variable to store the number of tiles/patches in height (r) and width (s) dimensions
        self.r = None
        self.s = None

        # Create the tile/patch
        self.xi = torch.zeros(1, 3, self.height, self.width)

        # Create variable to store a resized tile that matches the input image size (adv perurbation)
        self.xi_resized = None

        # Add parameters to the print list
        self.parameters = {
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "target_type": self.target_type,
            "iterations": self.iterations,
            "periodic tile height": self.height,
            "periodic tile width": self.width,
        }

    def __call__(self, x, model, t=None, y=None):
        """
        Implementation of METZEN-UAP method.

        :param x: image(s) in normalized range, i.e., [0, 1]
        :param model: model

        :return: adv_ex (the adversarial example)
        """

        # Check if the adversarial perturbation was already calculated
        assert self.computed, "The adversarial perturbation should be computed first."

        # Check if the image dimensions match with the periodic tile size width and height
        assert (x.size(2) % self.height == 0 and x.size(3) % self.width == 0), \
            "Error: input image can not be tiled in this size."

        # Assert input image is hosted on CUDA
        assert x.is_cuda, "Input image is not hosted on CUDA"

        # Assert model parameters are on CUDA
        assert next(model.parameters()).is_cuda, "Model is not hosted on CUDA"

        # Clamp adversarial example
        adv_ex = torch.clamp(x + self.xi_resized, 0, 1)

        return adv_ex, self.xi_resized


    def compute_perturbation(self, model, dataset):
        """
        Compute the adversarial perturbation from a dataset

        :param model: the model to compute the adversarial perturbation
        :param dataset: the dataset to compute the adversarial perturbation

        :return: the adversarial perturbation
        """

        # Get the data loader
        data_loader = dataset

        # Do an interation on the dataloader
        data = next(iter(data_loader))
        image = data["color_aug", 0, 0]

        # Collect batch size
        self.batch_size = image.shape[0]

        # Create a resized zero-valued tensor with image size
        self.xi_resized = torch.zeros(image.shape).cuda()

        # Get its size
        img_size = self.xi_resized.size()

        # Load the target label
        if self.target_type == 'static':
            self.target_label = self._load_target_label()
        else:
            self.target_label = None

        # Get number of tiles/patches in height (r) and width (s) dimensions
        self.r = int(img_size[2] / self.height)
        self.s = int(img_size[3] / self.width)

        # Send xi to cuda
        self.xi = self.xi.cuda()

        # Get the length of the dataset
        m = len(data_loader)

        # Training loop
        for epoch in range(self.iterations):
            print("Epoch:", epoch)

            # Initialize epoch loss
            epoch_loss = []

            # Create/Initialize a tile-sized zero pertubation and send it to cuda
            grad_x = torch.zeros(self.height, self.width).cuda()

            # Iterate over the dataset
            for step, data in enumerate(data_loader):

                # Get images and send them to CUDA
                images = data["color_aug", 0, 0].cuda()

                # Clone images
                inputs = images.clone()

                # In the case of dynamic target, we need to compute the target label from the network prediction
                if self.target_type == 'dynamic':
                    with torch.no_grad():
                        self.target_label = model(inputs).max(1)[1].unsqueeze(1)

                # Resize the adversarial tile to match the size of the input image
                self.xi_resized = self.xi.repeat(inputs.size()[0], 1, self.r, self.s)

                # Add the (resized) perturbation to the input image
                pert = torch.add(inputs, self.xi_resized).cuda()

                # Clamp the perturbation to the [0-1] range
                pert = torch.clamp(pert, min=0, max=1)

                ## Compute the gradient
                # delta_x: is the input perturbation forwarded through the model (but modified)
                # output: is the output of the model
                # owc: the static/dynamic target image
                # train_loss: the computed loss
                delta_x, output, owc, train_loss = self._loss_gradient(model, pert, self.target_label)

                # Add the newly compute gradient to the input image
                grad_x = torch.add(grad_x, delta_x)

                # Add the computed train loss to a list
                epoch_loss.append(train_loss)

            # Average the computed gradient according to the dataset length
            grad_d = (grad_x / m).cuda()

            # Add the computed perturbation
            self.xi = self.xi - (self.alpha * torch.sign(grad_d))

            # Clamp the adversarial perturbation
            self.xi = torch.clamp(self.xi, -self.epsilon, self.epsilon).cuda()

            # Resize the (final) adversarial perturbation to match the size of the inputs of the neural networks
            self.xi_resized = self.xi.repeat(self.xi_resized.size()[0], 1, self.r, self.s)

            # Set METZEN-UAP computed flag
            self.computed = True

    def show_parameters(self) -> None:
        """
        Print all the parameters of the Metzen UAP method
        """
        print(self.parameters, flush=True)

    def check_init(self) -> None:
        """
        Check whether input parameters are valid on initialization
        """
        pass

    def check_call(self, img, model) -> None:
        """
        Check whether input parameters are valid on call
        """
        pass

    def _load_target_label(self):
        """
        This function loads the target label for static class

        :return: Returns the target label as tensor
        """

        # Instantiate a torch tensor for the target label
        target_label = torch.LongTensor()

        # Require the gradient
        target_label.requires_grad = False

        # Load the target image/label from the given path
        lbl = Image.open(self.target)

        # Apply some transformations to the label
        # Instantiate the transformations: resize it if necessary
        trans = transforms.Compose([transforms.Resize((self.eval_height, self.eval_width), interpolation=Image.NEAREST)])

        # Apply transformations (resizing) to it
        lbl = trans(lbl)

        # Transform the label tensor to numpy int array
        lbl = np.array(lbl, dtype=np.int32)

        # Get the shape of the images
        w, h = lbl.shape

        # Reshape it to a single dimensional array
        lbl = lbl.reshape(-1)

        # Create a function for replacing labels
        remap = np.vectorize(self._remap)

        # Apply vfunc to replace the labels and reshape the array to the original image shape
        lbl = remap(lbl).reshape(w, h)

        # Transform the label back to tensor format
        class_label = torch.from_numpy(lbl).long()

        # Clone and reshape the tensor
        class_label = class_label.clone().repeat(self.batch_size, 1, 1)

        # 'Cast' class label as a long float tensor
        with torch.no_grad():
            target_label = class_label

        # Send it to cuda
        target_label = target_label.cuda()

        # Constructs an image like target, but replaces all pixels target_class with the nearest neighbor.
        target_label = self._output_without_class(target_label, self.ignore_index)

        # Reshape it
        target_label = target_label.unsqueeze(0).repeat(self.batch_size, 1, 1, 1)

        return target_label


    def _loss_gradient(self, model, pert, target):
        """
        This function computes the loss for the Metzen UAP attack.

        :param model: The model which is attacked
        :param pert: the perturbed input image
        :param target: The original image of the network.

        :return: Returns the computed loss
        """

        # Detach the perturbed image from the original graph
        pert = pert.detach()

        # Requires that the gradients are computed
        pert.requires_grad = True

        # Forward the perturbed input image to the model
        output = model(pert)

        # Get the height and width of the tile/patch
        rh, rw = self.height, self.width
        sh, sw = self.height, self.width

        # Extracts sliding local blocks from a batched input tensor
        # unfold(dimension, size, step)
        patches = output.unfold(2, rh, sh).unfold(3, rw, sw)

        # Returns a contiguous in memory tensor containing the same data as self tensor
        patches = patches.contiguous()

        # View the patch tensor in a different way: [Batch, C, Patch, patch height, patch width]
        patches = patches.view(output.size()[0], output.size()[1], -1, rh, rw)

        # Permute patch dimension with channel dimension
        # [B, Patch, C, patch height, patch width]
        patches = patches.permute(0, 2, 1, 3, 4)

        # Compute the target image
        if self.target_type == "dynamic":
            # Remove the target class from the target (network prediction)
            target = self._output_without_class(target.squeeze(1), self.target_class)
        else:
            target = target.squeeze(1)

        # Clone the 'original' target image
        owc = target.clone()

        # Extracts sliding local blocks from a batched input tensor
        targets = target.unfold(1, rh, sh).unfold(2, rw, sw)

        # Returns a contiguous in memory tensor containing the same data as self tensor
        targets = targets.contiguous()

        # View tensor in a different way: [Batch, Patch, patch height, patch width]
        targets = targets.view(target.size()[0], -1, rh, rw)

        # Permute patch dimensions
        # [B, Patch, patch height, patch width]
        targets = targets.permute(0, 1, 2, 3)

        # Iterate over the patches
        losses = []
        for i in range(patches.size()[1]):
            # model pred for perturbed input,
            # shape = [B, Patch, C, patch height, patch width]
            patch = patches[:, i, :, :, :]

            # model pred for clean input; without target class,
            # shape = [Batch, Patch, patch height, patch width]
            target = targets[:, i, :, :]

            # Compute losses for the static case
            if self.target_type == "static":
                # Creates a weight for each patch
                weight = torch.ones(patch.size()[1], dtype=torch.float).cuda()

                # Set the last entry to 0
                # weight[-1] = 0

                # Create the NLL criterion
                criterion_nll = torch.nn.NLLLoss(weight, reduction='mean')

                # Compute the loss between the patch and the target
                loss = criterion_nll(nn.functional.log_softmax(patch, dim=1), target)

                # Add the computed loss to a list
                losses.append(loss)

            # Compute the losses for the dynamic case
            elif self.target_type == "dynamic":
                # Creates a weight for each patch
                weight = torch.ones(patch.size()[1], dtype=torch.float).cuda()

                # Divides the weight by its sum
                weight = weight / (weight.sum())

                # Collect the target patch
                target_out = target

                # Copy the patch and detach it from the graph
                patch_norm = patch.detach().clone()

                # Subtract the minimum value, divide by the max (normalization)
                patch_norm -= patch_norm.min()
                patch_norm /= patch_norm.max().cuda()

                # Compute the maximum *values* in the patch in each line
                patch_max = patch_norm.max(1)

                # The idea is to change the values such that the class predictions
                # for the perturbed image are different from the clean image
                target_out = torch.where((patch_max[0].cuda() > 0.75) & (target_out == patch_max[1].cuda()),
                                         torch.tensor([self.ignore_index]).cuda(), target_out)

                # Boolean mask: ones where the pedestrian is, zero everywhere else
                mask = torch.where(patch.max(1)[1].cpu() == self.target_class, torch.tensor([1.]),
                                   torch.tensor([0.])).cuda().to(torch.float)

                # Boolean mask: ones where the background is, zero everywhere else (pedestrian)
                mask_vice_versa = (torch.ones(mask.cpu().size()).cuda() - mask).cuda().to(torch.float)

                # Predictions of pedestrians only, the rest is zero
                io_tar = target_out.to(torch.float) * mask

                # Predictions of background only, the rest/pedestrian is zero
                ibg_tar = target_out.to(torch.float) * mask_vice_versa

                # Set ignore class to all but the pedestrian pixels
                io_tar = io_tar + (mask_vice_versa * self.ignore_index)

                # Set ignore class to all but the background pixels
                ibg_tar = ibg_tar + (mask * self.ignore_index)

                # Create NLL loss criterion
                criterion_nll = torch.nn.NLLLoss(weight, reduction='none', ignore_index=self.ignore_index)

                # Losses
                # Pedestrian loss
                io_crit = criterion_nll(nn.functional.log_softmax(patch, dim=1), io_tar.to(torch.long))

                # Background loss
                ibg_crit = criterion_nll(nn.functional.log_softmax(patch, dim=1), ibg_tar.to(torch.long))

                # Compute loss
                loss = (self.weight * io_crit + (1 - self.weight) * ibg_crit).mean()
                losses.append(loss)

        # Average the losses
        losses = sum(losses) / len(losses)

        # Compute the gradients
        losses.backward()

        # Get the perturbation gradient
        grad = pert.grad.data.detach().cuda()

        # Extracts sliding local blocks from a batched input tensor
        grads = grad.unfold(2, rh, sh).unfold(3, rw, sw)

        # Returns a contiguous in memory tensor containing the same data as self tensor
        grads = grads.contiguous()

        # View tensor in a different way
        grads = grads.view(grad.size()[0], grad.size()[1], -1, rh, rw)

        # Permute patch dimensions
        grad = grads.permute(0, 2, 1, 3, 4)

        # 'Reduce' the computed gradient
        grad = torch.sum(grad, dim=1)
        grad = torch.sum(grad, dim=0)

        # grad: the input perturbation forwarded through the model (but modified)
        # output: is the output of the model
        # owc: is the static/dynamic target image
        # losses: is the computed loss
        return grad.detach(), output.detach(), owc.detach(), losses.detach().item()


    def _output_without_class(self, target, target_class):
        """
        This function constructs an image like target, but replaces all pixels target_class with the nearest neighbor.

        :param target: The image where a target_class should be replaced.
        :param target_class: The target class which should be replaced in the image target.

        :return: Returns the image target with all pixels classified as target_class replaced by the nearest neigbhor.
        """
        _, w, h = target.size()
        w = w - 1
        h = h - 1
        target_out = target.cpu().numpy()
        target_out = np.ma.masked_array(target_out, target_out == target_class)
        while np.any(target_out.mask):
            for shift in (-1, 1):
                for axis in (1, 2):
                    shifted = np.roll(target_out, shift=shift, axis=axis)
                    idx = ~shifted.mask * target_out.mask
                    if shift == -1:
                        idx[:, w, :] = False
                        idx[:, :, h] = False
                    target_out[idx] = shifted[idx]
        target_out = torch.from_numpy(target_out).cuda()

        return target_out


    def _remap(self, image):
        """
        Function to map from label IDs to train IDs

        :param image: the image to perform the mapping
        """

        # Define the dictionary to map from label IDs to train IDs.
        idx = self.ignore_index
        ID_to_trainID = {
            -1: idx, 0: idx, 1: idx, 2: idx, 3: idx, 4: idx, 5: idx, 6: idx,
            7: 0, 8: 1, 9: idx, 10: idx, 11: 2, 12: 3, 13: 4, 14: idx,
            15: idx, 16: idx, 17: 5, 18: idx, 19: 6, 20: 7, 21: 8, 22: 9,
            23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: idx,
            30: idx, 31: 16, 32: 17, 33: 18
        }

        # Map the IDs to train IDs
        if image in ID_to_trainID:
            return ID_to_trainID[image]
        else:
            return idx
