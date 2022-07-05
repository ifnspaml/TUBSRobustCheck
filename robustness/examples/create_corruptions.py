import importlib
import os
from argparse import ArgumentParser
from collections import namedtuple

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

import torch
from torch.utils.data import DataLoader
import torchvision as tv

from dataloader.pt_data_loader.specialdatasets import StandardDataset
from dataloader.definitions.labels_file import *
import dataloader.pt_data_loader.mytransforms as mytransforms

from corruptions import CORRUPTIONS, SEVERITY
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
    model.eval()

    ########### Step 3: Dataset and data transforms ###########
    # Define keys to load
    keys_to_load = ['color', 'segmentation_trainid']

    # Define transformations
    transforms_list = [mytransforms.CreateScaledImage(),
                       mytransforms.ConvertSegmentation(),
                       mytransforms.CreateColoraug(),
                       mytransforms.ToTensor(),
                       mytransforms.Relabel(int(DATASETS['cityscapes'].ignore_label), -100),
                       mytransforms.RemoveOriginals()]

    # Loading the dataset using the transformations for each model
    # Create a *validation* dataset for inference
    dataset_val = StandardDataset(dataset=args.dataset,
                                  trainvaltest_split='validation',
                                  keys_to_load=keys_to_load,
                                  labels_mode='fromtrainid',
                                  data_transforms=transforms_list,
                                  # folders_to_load=['leftimg8bit/val/lindau', 'lindau'],
                                  # folders_to_load=['leftimg8bit/val/frankfurt', 'frankfurt',
                                  #                 'leftimg8bit/val/munster', 'munster']
                                  )

    # Create the *validation* dataloader for inference
    data_loader = DataLoader(dataset_val,
                             1,
                             False,
                             num_workers=2,
                             pin_memory=True,
                             drop_last=True)

    ########### Step 4: Setting evaluation ###########
    # Instantiate two instances of the evaluator function
    Eval_clean = Evaluator(num_classes)
    Eval = Evaluator(num_classes)

    ########### Step 5: Instantiating the attack class ###########
    # Create a list of corruption objects based on inputs
    corruptions = []

    # Loading the corruptions
    # Check if corruptions were given as argument
    if len(args.corruption) == 0 and len(args.corruption_id) == 0:
        raise Exception("No corruption(s) given as argument.")
    if len(args.corruption) != 0 and len(args.corruption_id) != 0:
        raise Exception("You should pass the corruption(s) via --corruption OR -corruption_id argument.")

    if len(args.corruption) != 0:
        for k in range(len(args.corruption)):
            module = importlib.import_module(CORRUPTIONS[args.corruption[k]].classpath)
            corruption = getattr(module, CORRUPTIONS[args.corruption[k]].classname)

            # Instantiate the corruption:
            try:
                corruption = corruption()
            except ImportError:
                raise Exception('The given corruption {} is not implemented.'.format(args.corruption))

            # Append corrution to the list
            corruptions.append(corruption)

        # Creating list of lists for storing mIoU predictions
        mIoU_predictions_partial_gt = []
        mIoU_predictions_partial_pred = []
        mIoU_predictions_gt = [[] for _ in range(len(args.corruption))]
        mIoU_predictions_pred = [[] for _ in range(len(args.corruption))]

    else:
        for k in range(len(args.corruption_id)):
            # Get corruption class path
            class_path = (CORRUPTIONS[key].classpath for key, value in CORRUPTIONS.items() if
                          value == args.corruption_id[k])
            module = importlib.import_module(class_path)

            # Get corruption class name
            class_name = str((CORRUPTIONS[key].classname for key, value in CORRUPTIONS.items() if
                              value == args.corruption_id[k]))
            corruption = getattr(module, class_name)

            # Instantiate the corruption:
            try:
                corruption = corruption()
            except ImportError:
                raise Exception('The given corruption id {} is not implemented.'.format(args.corruption_id))

            # Append corrution to the list
            corruptions.append(corruption)

        # Creating list of lists for storing mIoU predictions
        mIoU_predictions_partial_gt = []
        mIoU_predictions_partial_pred = []
        mIoU_predictions_gt = [] * len(args.corruption_id)
        mIoU_predictions_pred = [] * len(args.corruption_id)

    ########### Step 6 : Fetch data, attack, evaluate ###########
    # Iterate over the *validation* dataset
    for step, data in enumerate(data_loader):
        # Print corruption computation status here and there
        print(f"Computing corruption(s) on sample number {step + 1} from",
              f"{len(data_loader)}, {100 * (step + 1) / len(data_loader)} % ", flush=True)

        # Fetch image and labels
        images = data["color_aug", 0, 0]
        labels = data[keys_to_load[1], 0, 0].long()

        # Send to cuda if available
        if args.CUDA:
            images = images.cuda()
            labels = labels.cuda()

        # Prediction of clean sample for secondary statistics analysis
        clean = model(images)
        clean = torch.argmax(clean, dim=1, keepdim=True)

        # Add batch to compute clean mIoU
        Eval_clean.add_batch(clean, labels)

        # Corrupt the image, predict and compute IoU
        for i in range(len(args.corruption)):
            for j in range(len(args.severity)):
                # Compute the corrupted image with corruption i and severity j
                img_corrupted = corruptions[i](images, args.severity[j]).float()
                # To be more aligned with Hendrycks implementation, the following operations are necessary
                # img_corrupted = (img_corrupted*255).type(torch.uint8)/255.
                # To be even 1e-8 close to Hendrycks implementation torch.set_default_dtype(torch.float64) needs
                # to be activated. Read more in the README.md.

                # Send it to CUDA if available
                if args.CUDA:
                    img_corrupted = img_corrupted.cuda()
                # Forwarding the corrupted image through the Swiftnet
                corrupted = model(img_corrupted)
                corrupted = torch.argmax(corrupted, dim=1, keepdim=True)

                # Add batch to compute corruption mIoU wrt groundtruth labels
                Eval.add_batch(corrupted, labels)
                mIoU_predictions_partial_gt.append(Eval.miou())
                Eval.reset()

                # Add batch to compute corruption mIoU wrt groundtruth prediction
                Eval.add_batch(corrupted, clean)
                mIoU_predictions_partial_pred.append(Eval.miou())
                Eval.reset()

                if step == 0:

                    directory = './corruptions/'
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    tv.utils.save_image([images[0], img_corrupted[0]],
                                        f"{directory}{args.corruption[i]}-{j + 1}_input_image_corrupted.jpg")

                    trainid2color = dataset_labels['cityscapes'].gettrainid2label()
                    clean_save = colorize(clean.squeeze(), trainid2color, NUM_CLASSES=num_classes)
                    corrupted_save = colorize(corrupted.squeeze(), trainid2color, NUM_CLASSES=num_classes)
                    labels_save = colorize(labels.squeeze(), trainid2color, NUM_CLASSES=num_classes)

                    # The operation that is internally executed in tv.utils.save_image(...), thus renormalization needs to be done
                    # grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                    tv.utils.save_image([clean_save / 255., corrupted_save / 255., labels_save / 255.],
                                        f"{directory}{args.corruption[i]}-{j + 1}_output_clean_corrupted_label.jpg")

            # Compute the average over the severities and reset the confusion matrix
            # wrt groundtruth/labels
            mIoU_predictions_gt[i].append(sum(mIoU_predictions_partial_gt) / len(mIoU_predictions_partial_gt))
            mIoU_predictions_partial_gt.clear()

            # wrt original model prediction
            mIoU_predictions_pred[i].append(sum(mIoU_predictions_partial_pred) / len(mIoU_predictions_partial_pred))
            mIoU_predictions_partial_pred.clear()

    ########### Compute statistics ###########
    # Clean mIoU
    mIoU_clean = Eval_clean.miou().item()

    # mIoU wrt groundtruth mean and wrt clean mIoU
    if args.corruption:
        mIoUVal_to_gt_mean = [(args.corruption[i], torch.mean(torch.stack(mIoU_predictions_gt[i])).item())
                              for i in range(len(corruptions))]
        mIoUVal_to_clean_mean = [(args.corruption[i], torch.mean(torch.stack(mIoU_predictions_pred[i])).item())
                                 for i in range(len(corruptions))]
    else:
        mIoUVal_to_gt_mean = [(args.corruption_id[i], torch.mean(torch.stack(mIoU_predictions_gt[i])).item())
                              for i in range(len(corruptions))]
        mIoUVal_to_clean_mean = [(args.corruption_id[i], torch.mean(torch.stack(mIoU_predictions_pred[i])).item())
                                 for i in range(len(corruptions))]

    # Print results
    print("mIoU, clean predictions", mIoU_clean)
    print("mIoU, wrt groundtruth predictions", mIoUVal_to_gt_mean)
    print("mIoU, wrt clean predictions", mIoUVal_to_clean_mean)


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

    parser.add_argument('--corruption',
                        type=str,
                        nargs='+',
                        default=['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                                 'motion_blur', 'zoom_blur', 'snow_weather', 'frost_weather', 'fog_weather',
                                 'brightness_weather', 'contrast_digital', 'elastic_transform_digital',
                                 'pixelate_digital', 'jpeg_compression_digital', 'speckle_noise', 'gaussian_blur',
                                 'spatter_weather', 'saturate_digital'
                                 ],
                        help="corruption to apply",
                        choices=CORRUPTIONS.keys())

    parser.add_argument('--corruption_id',
                        type=int,
                        nargs='+',
                        default=[],
                        help="corruption to apply (number)",
                        choices=CORRUPTIONS.values())

    parser.add_argument('--severity',
                        nargs='+',
                        type=int,
                        default=[1, 2, 3, 4, 5],
                        help="strength of the corruption",
                        choices=SEVERITY)

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

    parser.add_argument('--CUDA',
                        action='store_true',
                        help="Use or not CUDA")

    parser.add_argument('--save_frequency',
                        type=int,
                        help="path to weights",
                        default=50)

    # Call main and parse arguments
    main(parser.parse_args())
