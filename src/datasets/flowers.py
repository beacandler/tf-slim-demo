"""return a slime dataset object"""

import os
import tensorflow as tf
from tensorflow.contrib import slim

DEFAULT_DATASET_DIR = '/tmp/flower_photos/tfrecord/'

def get_config():
    return {
        'dataset_name': 'flowers',
        'splits': {
            'train': {
                'size': 3303,
                'pattern': 'train/train*'
            },
            'val': {
                'size': 367,
                'pattern': 'val/val*'
            }
        },
        'num_classes': 5,
        'items_to_descriptions': {
            'image': 'a color image',
            'label': 'a single integer rang in [0, 4]'
        }
    }

config = get_config()

def get_split(split_name, dataset_dir=None):
    """Returns a dataset tuple for a flower dataset

    Args:
        split_name: A train/val split name.
        dataset_dir: The base directory of the dataset sources, by it uses
        a predefined path (see DEFAULT_DATASET_DIR)

    Returns:
        A 'Dataset' namedtuple.
    Raises:
        ValueError: if 'split_name' is not a valid train/val split.
    """

    if not dataset_dir:
        dataset_dir = DEFAULT_DATASET_DIR
    file_pattern = os.path.join(dataset_dir, config['splits'][split_name]['pattern'])

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/class/label': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=tf.TFRecordReader,
        num_samples=config['splits'][split_name]['size'],
        decoder=decoder,
        items_to_descriptions = config['items_to_descriptions'],
        num_classes=config['num_classes'],
    )