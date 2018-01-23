"""Contains a factory for building various models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import inception_preprocessing
from . import lenet_preprocessing

def get_preprocessing(name):
    preprocessing_fn_map = {
        'lenet': lenet_preprocessing,
        'inception_v3': inception_preprocessing,
    }
    if name not in preprocessing_fn_map:
        raise ValueError('Preprocessing name [%s] was not recognized' % name)
    def preprocessing_fn(image, output_height, output_width, is_training, **kwargs):
        return preprocessing_fn_map[name].preprocess_image(
            image, output_width, output_height, is_training, **kwargs
        )

    return preprocessing_fn

