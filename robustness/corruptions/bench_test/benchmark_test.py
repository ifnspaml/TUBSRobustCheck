"""
FILE:            benchmark_test.py
SW-COMPONENT:    Benchmarking our corruption tool with the Hendrycks's code base
DESCRIPTION:     Script containing a code to compare our results with the ones from Hendrycks's code base
COPYRIGHT:       (C) TU Braunschweig

14.06.2022, TU Braunschweig, Andreas Bär, Edgard Moreira Minete
Initial release.
"""

import sys
from PIL import Image
import numpy as np
import torch
import random
import importlib
import cv2
import requests
from io import BytesIO

# sys.path.append(...) # if necessary to import from imagenet_c
# sys.path.append(.../attack_tool) # if necessary if necessary to import from robustness
# sys.path.append(.../attack_tool/robustness/corruptions/) # if necessary if necessary to import from robustness
# sys.path.append(.../attack_tool/robustness/corruptions/frost/) # if necessary if necessary to import from robustness

from imagenet_c import corrupt
from robustness import corruptions
from collections import namedtuple

Corruption = namedtuple('Corruption', ['classpath', 'classname'])

CORRUPTIONS = {
    'gaussian_noise': Corruption('robustness.corruptions.gaussian_noise', 'GaussianNoise'),
    'shot_noise': Corruption('robustness.corruptions.shot_noise', 'ShotNoise'),
    'impulse_noise': Corruption('robustness.corruptions.impulse_noise', 'ImpulseNoise'),
    'defocus_blur': Corruption('robustness.corruptions.defocus_blur', 'DefocusBlur'),
    'glass_blur': Corruption('robustness.corruptions.glass_blur', 'GlassBlur'),
    'motion_blur': Corruption('robustness.corruptions.motion_blur', 'MotionBlur'),
    'zoom_blur': Corruption('robustness.corruptions.zoom_blur', 'ZoomBlur'),
    'snow_weather': Corruption('robustness.corruptions.snow_weather', 'SnowWeather'),
    'frost_weather': Corruption('robustness.corruptions.frost_weather', 'FrostWeather'),
    'fog_weather': Corruption('robustness.corruptions.fog_weather', 'FogWeather'),
    'brightness_weather': Corruption('robustness.corruptions.brightness_weather', 'BrightnessWeather'),
    'contrast_digital': Corruption('robustness.corruptions.contrast_digital', 'ContrastDigital'),
    'elastic_transform_digital': Corruption('robustness.corruptions.elastic_transform_digital',
                                            'ElasticTransformDigital'),
    'pixelate_digital': Corruption('robustness.corruptions.pixelate_digital', 'PixelateDigital'),
    'jpeg_compression_digital': Corruption('robustness.corruptions.jpeg_compression_digital', 'JpegCompressionDigital'),
    'speckle_noise': Corruption('robustness.corruptions.speckle_noise', 'SpeckleNoise'),
    'gaussian_blur': Corruption('robustness.corruptions.gaussian_blur', 'GaussianBlur'),
    'spatter_weather': Corruption('robustness.corruptions.spatter_weather', 'SpatterWeather'),
    'saturate_digital': Corruption('robustness.corruptions.saturate_digital', 'SaturateDigital'),
}

response = requests.get(
    'https://www.tu-braunschweig.de/fileadmin/Redaktionsgruppen/Institute_Fakultaet_5/IFN/bilderifn/mitarbeiter/Andreas_Baer_120x150.jpg')
im = Image.open(BytesIO(response.content)).resize((224, 224))
np_ = np.asarray(im)
torch_ = torch.from_numpy(np_).permute(2, 0, 1).unsqueeze(0)

float64_corruptions = ['defocus_blur',
                       'glass_blur',
                       'fog_weather',
                       'brightness_weather',
                       'speckle_noise',
                       'gaussian_blur',
                       'saturate_digital']

for corruption in CORRUPTIONS.keys():
    if corruption in float64_corruptions:
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)
    module = importlib.import_module(CORRUPTIONS[corruption].classpath)
    corrupt_ = getattr(module, CORRUPTIONS[corruption].classname)()

    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    if '_weather' in corruption:
        new_corruption = corruption.replace('_weather', '')
        hendrycks = corrupt(np_, severity=1, corruption_name=new_corruption) / 255.
    elif '_digital' in corruption:
        new_corruption = corruption.replace('_digital', '')
        hendrycks = corrupt(np_, severity=1, corruption_name=new_corruption) / 255.
    else:
        hendrycks = corrupt(np_, severity=1, corruption_name=corruption) / 255.
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    if corruption == 'gaussian_noise' or corruption == 'impulse_noise':
        ours = corrupt_(torch_.permute(0, 2, 3, 1) / 255., severity=1).squeeze(0)
    else:
        ours = corrupt_(torch_ / 255., severity=1).squeeze(0).permute(1, 2, 0)
    ours = np.asarray((ours * 255).type(torch.uint8) / 255.)
    print(corruption, ":",
          np.allclose(np.asarray(torch.from_numpy(hendrycks)), ours, rtol=1e-05, atol=1e-08, equal_nan=False))