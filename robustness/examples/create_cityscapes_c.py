import importlib
import os
from argparse import ArgumentParser
import time

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

import torch
from torch.utils.data import DataLoader
import torchvision as tv

from corruptions import CORRUPTIONS, SEVERITY
from dataloader.pt_data_loader.specialdatasets import StandardDataset
from dataloader.definitions.labels_file import *
import dataloader.pt_data_loader.mytransforms as mytransforms

# Define ImageNet mean and var
MEAN = [0.485, 0.456, 0.406]
VAR = [0.229, 0.224, 0.225]

PATHS = {
    'motion_blur': 'blur/motion',
    'defocus_blur': 'blur/defocus',
    'glass_blur': 'blur/glass',
    'gaussian_blur': 'blur/gaussian',

    'gaussian_noise': 'noise/gaussian',
    'impulse_noise': 'noise/impulse',
    'shot_noise': 'noise/shot',
    'speckle_noise': 'noise/speckle',

    'brightness_weather': 'digital/brightness',
    'contrast_digital': 'digital/contrast',
    'saturate_digital': 'digital/saturate',
    'jpeg_compression_digital': 'digital/jpeg',

    'snow_weather': 'weather/snow',
    'spatter_weather': 'weather/spatter',
    'fog_weather': 'weather/fog',
    'frost_weather': 'weather/frost',
}

def main(args):
    transforms_list = [mytransforms.CreateScaledImage(),
                       mytransforms.ConvertSegmentation(),
                       mytransforms.CreateColoraug(),
                       mytransforms.ToTensor(),
                       mytransforms.RemoveOriginals()]

    dataset_val = StandardDataset(dataset='cityscapes',
                                  trainvaltest_split='validation',
                                  keys_to_load=['color', 'segmentation_trainid'],
                                  labels_mode='fromtrainid',
                                  data_transforms=transforms_list,
                                  output_filenames=True)

    data_loader = DataLoader(dataset_val,
                             1,
                             False,
                             num_workers=2,
                             pin_memory=True,
                             drop_last=True)

    corruptions = []

    if len(args.corruption) != 0:
        for k in range(len(args.corruption)):
            module = importlib.import_module(CORRUPTIONS[args.corruption[k]].classpath)
            corruption = getattr(module, CORRUPTIONS[args.corruption[k]].classname)

            try:
                corruption = corruption()
            except ImportError:
                raise Exception('The given corruption {} is not implemented.'.format(args.corruption))

            # Append corrution to the list
            corruptions.append(corruption)

    # Define paths where to save the images
    base_path = os.path.join(args.savedir, 'cityscapes_c')


    for step, data in enumerate(data_loader):
        # Print corruption computation status here and there
        print(f"Computing corruption(s) on sample number {step + 1} from",
              f"{len(data_loader)}, {100 * (step + 1) / len(data_loader)} % ", flush=True)

        # Fetch image and labels
        images = data["color_aug", 0, 0]
        filename = data['filename'][("color", 0, -1)][0]
        subfolder = filename.split('/')[:-1]
        filename = filename.split('/')[-1]

        # Corrupt the image, predict and compute IoU
        for i in range(len(args.corruption)):
            for j in range(len(args.severity)):

                filepath = os.path.join(base_path, *subfolder, PATHS[args.corruption[i]], f'severity_{args.severity[j]}', filename)
                save_path = filepath.split('/')[:-1]
                save_path = '/' + os.path.join(*save_path)
                print(save_path)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                else:
                    pass

                # Compute the corrupted image with corruption i and severity j
                # To be more aligned with Hendrycks implementation, the following operations are necessary
                # img_corrupted = (img_corrupted*255).type(torch.uint8)/255.
                # To be even 1e-8 close to Hendrycks implementation torch.set_default_dtype(torch.float64) needs
                # to be activated. Read more in the README.md.
                start_time = time.time()
                img_corrupted = corruptions[i](images, args.severity[j])
                img_corrupted = (img_corrupted * 255).type(torch.uint8) / 255.
                tv.utils.save_image(img_corrupted[0], filepath)
                print(args.corruption[i], f'severity_{args.severity[j]}', 'took', time.time() - start_time, 'seconds')


def colorize(tensor, trainid2label, NUM_CLASSES):
    tensor = tensor.cpu()
    new_tensor = torch.zeros((tensor.shape[0], tensor.shape[1], 3), dtype=torch.uint8)
    for trainid in range(NUM_CLASSES - 1):
        new_tensor[tensor == trainid] = torch.tensor(trainid2label[trainid].color, dtype=torch.uint8)
    new_tensor[tensor == -100] = torch.tensor(trainid2label[255].color, dtype=torch.uint8)
    return new_tensor.permute((2, 0, 1))


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--corruption',
                        type=str,
                        nargs='+',
                        default=[
                            'motion_blur', 'defocus_blur', 'glass_blur', 'gaussian_blur',
                            'gaussian_noise', 'impulse_noise', 'shot_noise', 'speckle_noise',
                            'brightness_weather', 'contrast_digital', 'saturate_digital', 'jpeg_compression_digital',
                            'snow_weather', 'spatter_weather', 'fog_weather', 'frost_weather', 
                        ],
                        help="corruption to apply",
                        choices=CORRUPTIONS.keys())

    parser.add_argument('--severity',
                        nargs='+',
                        type=int,
                        default=[1, 2, 3, 4, 5],
                        help="strength of the corruption",
                        choices=SEVERITY)

    parser.add_argument('--savedir',
                        type=str,
                        default='/beegfs/work/baer/attack_tool/robustness/examples/')

    # Call main and parse arguments
    main(parser.parse_args())
