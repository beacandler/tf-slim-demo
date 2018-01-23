"""Functions to read, decode, and pre-process input data for the Model"""

import collections
import tensorflow as tf
from tensorflow.contrib import slim
from preprocessing import preprocessing_factory
"""
    tuple to store data endpoints for the Model.
    It has following fields (tensors):
    images: input images, shape [batch_size x H x w x3]
    images_orig: raw images which are not preprocessed , shape [batch_size x H x w x3] 
    labels: ground truth label ids shape [batch_size x 1]
    labels: ground truth label for one hot [batch_size x num_classes ]
"""
# InputEndpoints = collections.namedtuple(
#     'InputEndpoints', ['images', 'images_orig', 'labels', 'labels_one_hot']
# )
InputEndpoints = collections.namedtuple(
    'InputEndpoints', ['images', 'labels', 'labels_one_hot']
)
"""
    tuple to ShuffleBatchConfig
"""
ShuffleBatchConfig = collections.namedtuple(
    'ShuffleBatchConfig', ['num_batching_threads', 'queue_capacity', 'min_after_dequeue']
)

DEFAULT_SHUFFLE_CONFIG = ShuffleBatchConfig(
    num_batching_threads=8, queue_capacity= 3000, min_after_dequeue=1000)

def get_data(dataset,
             model_name,
             batch_size = 32,
             shuffle_config = None,
             shuffle=None,
             is_training=True,
             height=0,
             width=0):
    """return input data for Model input
    Args:
        dataset: a slim Dataset object.
        model_name: specify Network.
        shuffle_config: a namedtuple to control shuffle queue.
         fields: {queue_capacity, num_batching_threads, min_after_dequeue}.
        shuffle: control data provider whether shuffle.
        is_training: if Ture preprocess image for train.
        width: excepted resized width
        height: excepted resized height
    """
    if not shuffle_config:
        shuffle_config = DEFAULT_SHUFFLE_CONFIG
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=shuffle,
        common_queue_capacity = 2 * batch_size,
        common_queue_min = batch_size
    )
    [image_orig, label] = provider.get(['image', 'label'])
    tf.summary.image('image_org', tf.expand_dims(image_orig, 0))
    tf.summary.scalar('label_orig', label)
    preprocessing_fn = preprocessing_factory.get_preprocessing(model_name)
    image = preprocessing_fn(image_orig,
                          width,
                          height,
                          is_training)
    label_one_shot = slim.one_hot_encoding(label, dataset.num_classes)
    images, labels, labels_one_hot = (tf.train.shuffle_batch(
        tensors=[image, label, label_one_shot],
        batch_size = batch_size,
        capacity=shuffle_config.queue_capacity,
        num_threads=shuffle_config.num_batching_threads,
        min_after_dequeue=shuffle_config.min_after_dequeue))

    return InputEndpoints(
        images=images,
        labels=labels,
        labels_one_hot=labels_one_hot)
