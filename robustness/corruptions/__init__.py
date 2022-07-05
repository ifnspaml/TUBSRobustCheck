import ctypes
from collections import namedtuple
from wand.api import library as wandlibrary
from wand.image import Image as WandImage

# Define list of possible corruption severities
SEVERITY = [1, 2, 3, 4, 5]

# Attacks dictionary structure
Corruption = namedtuple('Corruption', ['classpath', 'classname', 'type', 'id'])

# Define dictionary of attacks
CORRUPTIONS = {
    # Training corruptions:
    'gaussian_noise': Corruption('robustness.corruptions.gaussian_noise', 'GaussianNoise', 'noise', 1),
    'shot_noise': Corruption('robustness.corruptions.shot_noise', 'ShotNoise', 'noise', 2),
    'impulse_noise': Corruption('robustness.corruptions.impulse_noise', 'ImpulseNoise', 'noise', 3),
    'defocus_blur': Corruption('robustness.corruptions.defocus_blur', 'DefocusBlur', 'blur', 4),
    'glass_blur': Corruption('robustness.corruptions.glass_blur', 'GlassBlur', 'blur', 5),
    'motion_blur': Corruption('robustness.corruptions.motion_blur', 'MotionBlur', 'blur', 6),
    'zoom_blur': Corruption('robustness.corruptions.zoom_blur', 'ZoomBlur', 'blur', 7),
    'snow_weather': Corruption('robustness.corruptions.snow_weather', 'SnowWeather', 'weather', 8),
    'frost_weather': Corruption('robustness.corruptions.frost_weather', 'FrostWeather', 'weather', 9),
    'fog_weather': Corruption('robustness.corruptions.fog_weather', 'FogWeather', 'weather', 10),
    'brightness_weather': Corruption('robustness.corruptions.brightness_weather', 'BrightnessWeather', 'weather', 11),
    'contrast_digital': Corruption('robustness.corruptions.contrast_digital', 'ContrastDigital', 'digital', 12),
    'elastic_transform_digital': Corruption('robustness.corruptions.elastic_transform_digital',
                                            'ElasticTransformDigital',
                                            'digital', 13),
    'pixelate_digital': Corruption('robustness.corruptions.pixelate_digital', 'PixelateDigital', 'digital', 14),
    'jpeg_compression_digital': Corruption('robustness.corruptions.jpeg_compression_digital', 'JpegCompressionDigital',
                                           'digital', 15),

    # Validation corruptions:
    'speckle_noise': Corruption('robustness.corruptions.speckle_noise', 'SpeckleNoise', 'noise', 16),
    'gaussian_blur': Corruption('robustness.corruptions.gaussian_blur', 'GaussianBlur', 'blur', 17),
    'spatter_weather': Corruption('robustness.corruptions.spatter_weather', 'SpatterWeather', 'weather', 18),
    'saturate_digital': Corruption('robustness.corruptions.saturate_digital', 'SaturateDigital', 'digital', 19),
}

# Tell Python about the C method
wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
                                              ctypes.c_double,  # radius
                                              ctypes.c_double,  # sigma
                                              ctypes.c_double)  # angle


# Extend wand.image.Image class to include method signature
class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)
