"""Define flags are common for both train.py and test.py"""

import sys
import logging
import collections
import tensorflow as tf

import model
import datasets

FLAGS = tf.app.flags.FLAGS
logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stderr,
    format='%(levelname)s '
    '%(asctime)s.%(msecs)06d: '
    '%(filename)s: '
    '%(lineno)d '
    '%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

# some namedtuple
OutputEndPoints = collections.namedtuple('OutputEndPoints',[
    'Logits', 'end_points'])

ConvTowerParams = collections.namedtuple('ConvTowerParams', ['final_endpoint'])

TrainingHparams = collections.namedtuple('TrainingHparams',
    ['learning_rate',
     'optimizer',
     'momentum'])

def define():
    tf.app.flags.DEFINE_integer(
        'num_readers', 4,
        'The number of parallel reader that read date from the dataset.')
    ######################
    # Optimization Flags #
    ######################

    tf.app.flags.DEFINE_float(
        'momentum', 0.9,
        'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

    ######################
    # Learning Rate Flags #
    ######################
    tf.app.flags.DEFINE_string(
        'learning_rate_decay_type', 'exponential',
        'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
        'or "polynomial" ')

    tf.app.flags.DEFINE_float(
        'end_learning_rate', 0.0001,
        'The minimal end learning rate used by a polynomial decay learning rate.')
    tf.app.flags.DEFINE_float(
        'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')
    tf.app.flags.DEFINE_float(
        'num_epochs_per_decay', 2.0,
        'Number of epochs after which learning rate decays.')
    tf.app.flags.DEFINE_bool(
        'sync_replicas', False,
        'Whether or not to synchronize the replicas during training.')
    tf.app.flags.DEFINE_integer(
        'replicas_to_aggregate', 1,
        'The number of gradients to collect before updating params.')
    tf.app.flags.DEFINE_float(
        'moving_average_decay', None,
        'The decay to use for the moving average.'
        'if left as None, then moving average is not used.')


    #####################
    # Dataset Flags #
    #####################
    tf.app.flags.DEFINE_string('dataset_name','flowers',
                               'The name of the dataset to load.')
    tf.app.flags.DEFINE_string('dataset_split_name', 'train',
                               'the name of the train/val split.')
    tf.app.flags.DEFINE_string('dataset_dir', '/mnt/data/classification/flowers',
                               'Directory where the dataset files are stored.')
    tf.app.flags.DEFINE_string('model_name', 'inception_v3',
                               'The name of the architecture to train.')

    tf.app.flags.DEFINE_integer('batch_size', 32,
                                'The name of examples in each batch.')
    tf.app.flags.DEFINE_integer('train_image_size', None,
                                'Train image size.')
    tf.app.flags.DEFINE_integer('num_classes', 5,
                                'The number of predicted classes')

    tf.app.flags.DEFINE_integer('height', 299,
                                'Height of the image for train/val')

    tf.app.flags.DEFINE_integer('width', 299,
                                'Width of the image for train/val')

    #####################
    # Models parameters #
    #####################
    tf.app.flags.DEFINE_string('final_endpoint', 'Mixed_7c',
                               'Endpoint to cut inception tower')

    tf.app.flags.DEFINE_string('train_dir', './log/train_dir',
                               'Directory where checkpoints and event logs are written to.')

    tf.app.flags.DEFINE_string('master', '',
                                'BNS name of the TensorFlow master to use.')

def create_mparams():
    """Some model parameters"""
    return {
        'conv_tower_fn': ConvTowerParams(final_endpoint=FLAGS.final_endpoint)
    }

def create_TrainingHparams(
        learning_rate,
        optimizer,
        momentum):

    return TrainingHparams(
        learning_rate=learning_rate,
        optimizer=optimizer,
        momentum=momentum)

def create_model(*args, **kwargs):
    flower_model = model.Model(mparams=create_mparams(), *args, ** kwargs)
    return flower_model

def create_dataset(dataset_name, dataset_split_name):
    ds_module = getattr(datasets, dataset_name)
    return ds_module.get_split(dataset_split_name)

