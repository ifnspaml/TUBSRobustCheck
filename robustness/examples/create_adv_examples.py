import importlib
import os
from argparse import ArgumentParser

import dataloader.pt_data_loader.mytransforms as mytransforms
from dataloader.pt_data_loader.specialdatasets import StandardDataset

import torch
from torchvision.utils import save_image
import torchvision as tv
from attacks import ATTACKS
from dataloader.definitions.labels_file import *
from torch.utils.data import DataLoader

from helper.model import ModelWrapper
from helper.metrics import Evaluator
from utils.model_configs.model_architectures import load_model_def, load_model_state

# Datasets dictionary structure
Dataset = namedtuple('Dataset', ['ignore_label', 'num_class'])

# Define dictionary of datasets
DATASETS = {
    'cityscapes': Dataset(255, 19)
}

# Define ImageNet mean and var
MEAN = [0.485, 0.456, 0.406]
VAR = [0.229, 0.224, 0.225]


def main(args):
    ########### Step 1: Load the model and weights ###########
    # Number of classes
    num_classes = DATASETS['cityscapes'].num_class

    # Load the model definition
    model = load_model_def(args.model)

    # Load model weights
    model = load_model_state(model, args.model, args.weights)

    # Add model wrapper for normalization of the model
    model = ModelWrapper(model, preprocessing=tv.transforms.Normalize(mean=MEAN, std=VAR))

    # Send model to cuda if available
    if args.CUDA:
        model = model.cuda()

    # Set model to evaluation mode
    if not ATTACKS[args.attack].universal:
        model.eval()

    ########### Step 3: Dataset and data transforms ###########
    # Define keys to load
    keys_to_load = ['color', 'segmentation_trainid']

    # Define transformations
    transforms_list = [mytransforms.CreateScaledImage(),
                       mytransforms.ConvertSegmentation(),
                       mytransforms.CreateColoraug(),
                       mytransforms.ToTensor(),
                       mytransforms.Resize((512, 1024),
                                           image_types=['color', 'segmentation',
                                                        'segmentation_trainid']),
                       mytransforms.Relabel(int(DATASETS['cityscapes'].ignore_label), -100),
                       mytransforms.RemoveOriginals()]

    # Loading the dataset using the transformations for each model
    # Create a *train* dataset for computing the Data-free Universal Adversarial Perturbation
    if ATTACKS[args.attack].universal:
        dataset_train = StandardDataset(dataset=args.dataset,
                                        trainvaltest_split='train',
                                        keys_to_load=["color"],
                                        data_transforms=transforms_list,
                                        n_files=None,
                                        output_filenames=True)
    else:
        dataset_train = None

    # Creating a validation dataloader for the Metzen UAP attack
    data_loader = DataLoader(dataset_train,
                             1,
                             False,
                             num_workers=2,
                             pin_memory=True,
                             drop_last=True)

    # Create a *validation* dataset for inference
    dataset_val = StandardDataset(dataset=args.dataset,
                                  trainvaltest_split='validation',
                                  keys_to_load=keys_to_load,
                                  labels_mode='fromtrainid',
                                  data_transforms=transforms_list,
                                  n_files=10,
                                  # folders_to_load=['leftimg8bit/val/lindau', 'lindau'],
                                  # folders_to_load=['leftimg8bit/val/frankfurt', 'frankfurt',
                                  #                 'leftimg8bit/val/munster', 'munster'],
                                  output_filenames=True
                                  )
    # Using 'leftimg8bit/val/city' & 'city' to ensure compatibility to newest dataloader releases

    # Create the *validation* dataloader for inference
    data_loader = DataLoader(dataset_val,
                             1,
                             False,
                             num_workers=2,
                             pin_memory=True,
                             drop_last=True)

    # Load a target image if the attack is metzen
    # Instantiate the target image
    target_img = None

    # Load the target image in case of metzen attack (only necessary in the static case of metzen attack)
    if "metzen" in args.attack:
        if args.attack_type == "static":
            target_img = args.target_img
        else:
            target_img = None

    ########### Step 4: Setting evaluation ###########
    # Clean segmentation iou with respect to groundtruth labels
    clean2gt = Evaluator(num_classes)

    # Attacked segmentation iou with respect to groundtruth labels
    attacked2gt = Evaluator(num_classes)

    # Attacked segmentation iou with respect to the original prediction
    attacked2pred = Evaluator(num_classes)

    ########### Step 5: Instantiating the attack class ###########
    # Loading the attack
    module = importlib.import_module(ATTACKS[args.attack].classpath)
    attack = getattr(module, ATTACKS[args.attack].classname)

    if args.attack == 'fgsm':
        attack = attack(epsilon=10 / 255)
    elif args.attack == "gd_uap":
        attack = attack(epsilon=10 / 255, prior_mode="off", max_iteration=400, image_size=[512, 1024])
    elif args.attack == 'i_fgsm':
        attack = attack(epsilon=10 / 255, alpha=1 / 255)
    elif args.attack == "metzen_uap":
        attack = attack(epsilon=10 / 255, alpha=1 / 255, iterations=60, target_type='static', target=target_img)
    elif args.attack == "metzen":
        attack = attack(epsilon=10 / 255, alpha=1 / 255, iterations=60, target_type='dynamic')
    elif args.attack == 'mi_fgsm':
        attack = attack(epsilon=10 / 255, momentum=1.0, alpha=1 / 255, iterations=10)
    elif args.attack == 'pgd':
        attack = attack(epsilon=10 / 255, alpha=1 / 255, iterations=20, random_start=True)
    else:
        raise Exception('The attack {} is not implemented'.format(args.attack))

    # Print hyperparameters of the attack
    attack.show_parameters()
    # Send model to cuda
    if args.CUDA:
        attack.cuda()

    # Create adversarial attack (UAP-based methods)
    if ATTACKS[args.attack].universal and dataset_train is not None and args.attack == 'gd_uap':
        attack.compute_perturbation(model, dataset_train)
    elif ATTACKS[args.attack].universal and dataset_train is not None and args.attack == 'metzen_uap':
        attack.compute_perturbation(model, data_loader)

    ########### Step 6 : Fetch data, attack, evaluate ###########
    # Iterate over the *validation* dataset
    for step, data in enumerate(data_loader):
        # Fetch image and labels
        images = data["color_aug", 0, 0]
        labels = data[keys_to_load[1], 0, 0].long()

        # Send to cuda if available
        if args.CUDA:
            images = images.cuda()
            labels = labels.cuda()

        # Clean prediction
        clean = model(images)
        clean = torch.argmax(clean, dim=1, keepdim=True)

        # Perform the adversarial attack
        if args.attack == 'fgsm':
            adv_ex = attack(images, model, y=labels[:, 0, :, :])
        elif args.attack == 'gd_uap':
            adv_ex = attack(images, model)
        elif args.attack == 'i_fgsm':
            adv_ex = attack(images, model, y=labels[:, 0, :, :])
        elif args.attack == 'metzen':
            adv_ex, tgt_image = attack(images, model)
        elif args.attack == 'metzen_uap':
            adv_ex, xi_resized = attack(images, model)
        elif args.attack == 'mi_fgsm':
            adv_ex = attack(images, model, y=labels[:, 0, :, :])
        elif args.attack == 'pgd':
            adv_ex = attack(images, model, y=labels[:, 0, :, :])
        else:
            raise Exception('The attack {} is not implemented'.format(args.attack))

        # Forwarding the adversarial image through the Swiftnet
        adversarial = model(adv_ex)
        adversarial = torch.argmax(adversarial, dim=1, keepdim=True)

        # Add batch for evaluation ==> add_batch(PREDICTION, REFERENCE)
        # Clean prediction vs. groundtruth
        clean2gt.add_batch(clean, labels)

        # Attacked prediction vs. groundtruth
        attacked2gt.add_batch(adversarial, labels)

        # Attacked prediction vs. clean prediction
        attacked2pred.add_batch(adversarial, clean)

        if args.save_images:
            # Select dir path
            directory = './attacks/'

            # If dir does not exist, create a dir
            if not os.path.exists(directory):
                os.makedirs(directory)

            # Save image, adversarial example, and adversarial perturbation
            tv.utils.save_image([images[0], adv_ex[0], ((adv_ex[0] - images[0]) * 10) + 0.5],
                                f"{directory}{args.attack}_inputs_clean_adversarial_perturbation_{step}.png")

            # Map train IDS to color
            trainid2color = dataset_labels['cityscapes'].gettrainid2label()

            # Colorize clean prediction, adversarial prediciton and labels
            clean_save = colorize(clean.squeeze(), trainid2color, NUM_CLASSES=num_classes)
            adversarial_save = colorize(adversarial.squeeze(), trainid2color, NUM_CLASSES=num_classes)
            labels_save = colorize(labels.squeeze(), trainid2color, NUM_CLASSES=num_classes)

            # Save outputs
            tv.utils.save_image([clean_save / 255., adversarial_save / 255., labels_save / 255.],
                                f"{directory}{args.attack}_outputs_clean_adversarial_label_{step}.png")

    ########### Step 7: Calculate job statistics ###########
    # Clean iouVal and iou_classes
    clean2gt_miou, clean2gt_iou_classes = clean2gt.miou() * 100, clean2gt.iou() * 100

    # Primary iouVal and iou_classes
    attacked2gt_miou, attacked2gt_iou_classes = attacked2gt.miou() * 100, attacked2gt.iou() * 100

    # Secondary iouVal and iou_classes
    attacked2pred_miou, attacked2pred_iou_classes = attacked2pred.miou() * 100, attacked2pred.iou() * 100

    # Print results
    print(f"Clean: mIoU={clean2gt_miou:.2f}, class-wise IoU={clean2gt_iou_classes}")
    print(f"Attacked (gt vs. pred): mIoU={attacked2gt_miou:.2f}, class-wise IoU={attacked2gt_iou_classes}")
    print(
        f"Attacked (original pred vs. pred): mIoU={attacked2pred_miou:.2f}, class-wise IoU={attacked2pred_iou_classes}")


def colorize(tensor, trainid2label, NUM_CLASSES):
    tensor = tensor.cpu()
    new_tensor = torch.zeros((tensor.shape[0], tensor.shape[1], 3), dtype=torch.uint8)
    for trainid in range(NUM_CLASSES - 1):
        new_tensor[tensor == trainid] = torch.tensor(trainid2label[trainid].color, dtype=torch.uint8)
    new_tensor[tensor == -100] = torch.tensor(trainid2label[255].color, dtype=torch.uint8)
    return new_tensor.permute((2, 0, 1))


if __name__ == '__main__':
    # Get parsed arguments
    parser = ArgumentParser()

    parser.add_argument('--attack',
                        type=str,
                        help="attack to apply",
                        choices=ATTACKS.keys())

    parser.add_argument('--attack_type',
                        type=str,
                        help="Type of Metzen attack: Static or Dynamic")

    parser.add_argument('--CUDA',
                        action='store_true',
                        help="use or not CUDA")

    parser.add_argument('--dataset',
                        type=str,
                        help="dataset to use",
                        choices=DATASETS)

    parser.add_argument('--model',
                        type=str,
                        help="name of model to load")

    parser.add_argument('--weights',
                        type=str,
                        help="path to weights")

    parser.add_argument('--target_img',
                        type=str,
                        default="/beegfs/work/shared/cityscapes/gtFine/train/monchengladbach/monchengladbach_000000_026602_gtFine_labelIds.png",
                        help="Path to the target image for the static Metzen attack")

    parser.add_argument("--save_images",
                        help="If set, example input and output images are saved",
                        type=int,
                        choices=[0, 1],
                        default=0)

    # Call main and parse arguments
    main(parser.parse_args())
