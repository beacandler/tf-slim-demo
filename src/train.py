"""train code"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf
from tensorflow.contrib import slim

import utils
import common_flags
import data_provider

FLAGS = tf.app.flags.FLAGS
common_flags.define()

####################
# training flags #
####################
tf.app.flags.DEFINE_boolean('reset_train_dir', False,
                         'Whether or not to reset train dir before training.')

tf.app.flags.DEFINE_boolean('show_graph_state', False,
                          'Whether or not to show graph states to stderr.')

tf.app.flags.DEFINE_bool('use_augment_input', True,
                         'If True process training input with augmentation')

tf.app.flags.DEFINE_float('clip_gradient_norm', 2.0,
                          'If greater than 0 then the gradients would be clipped by '
                          'it.')

tf.app.flags.DEFINE_integer('log_every_n_steps', 100,
                            'The fequency with which logs are saved in seconds.')

tf.app.flags.DEFINE_integer('save_summaries_secs', 600,
                            'The frequency with which summaries are saved in seconds.')

tf.app.flags.DEFINE_integer('save_interval_secs', 600,
                            'The frequency with which the model is saved in seconds.')

tf.app.flags.DEFINE_integer('max_number_of_steps', int(1e10),
                            'The maximum number of gradient step.')

####################
# optimizier flags #
####################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string('checkpoint_path', None,
                           'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string('checkpoint_inception', None,
                           'Checkpoint to recover inception weights from.')
tf.app.flags.DEFINE_string('checkpoint_exclude_scopes', None,
                           'Comma-separated list of scopes of variables to exclude when restoring.'
                           'from a checkpoint.')
tf.app.flags.DEFINE_string('trainabele_scopes', None,
                           'Comma-seprarated list of scopes to filter the set of variables to train.'
                           'By default, None would train all the variables.')
tf.app.flags.DEFINE_string('trainable_scopes', None,
                           'Comma-separated list of scopes to filter the set of variables to train.'
                           'By default, None would train all variables.')
tf.app.flags.DEFINE_string('ignore_missing_vars', False,
                           'When restoring a checkpoint would ignore missing variables.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')


def prepare_training_dir():
    if not tf.gfile.Exists(FLAGS.train_dir):
        logging.info('NOT EXIST so create a new training directory'.format(FLAGS.train_dir))
        tf.gfile.MakeDirs(FLAGS.train_dir)
    else:
        if FLAGS.reset_train_dir:
            logging.info('RESET the traning directory')
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
            tf.gfile.MakeDirs(FLAGS.train_dir)
        else:
            logging.info('ALREADY Use already existing training directory {}'.format(FLAGS.train_dir))

def train(loss, init_fn, variables_to_train):
    """Wraps slim.training.train to run a training loop.

    Args:
        loss: a loss tensor.
        init_fn: a callable to be executed after all other initialization is done.
        variables_to_train: an optional list of variables to train. If None, it will
        default to all tf.trainable_variables().
    """
    hparams = common_flags.create_TrainingHparams(
        learning_rate=FLAGS.learning_rate,
        optimizer=FLAGS.optimizer,
        momentum=FLAGS.momentum)

    optimizer = utils.create_optimizer(hparams)
    lr = hparams.learning_rate
    tf.summary.scalar('learning_rate', lr)

    train_op = slim.learning.create_train_op(
        loss,
        optimizer,
        summarize_gradients=True,
        variables_to_train=variables_to_train,
        clip_gradient_norm=FLAGS.clip_gradient_norm)

    slim.learning.train(
        train_op,
        logdir=FLAGS.train_dir,
        graph=loss.graph,
        number_of_steps=FLAGS.max_number_of_steps,
        log_every_n_steps=FLAGS.log_every_n_steps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs,
        init_fn=init_fn)

def main(_):

    prepare_training_dir()
    logging.info('dataset_name: {}, split_name: {}'.format(FLAGS.dataset_name, FLAGS.dataset_split_name))
    dataset = common_flags.create_dataset(FLAGS.dataset_name, FLAGS.dataset_split_name)
    model = common_flags.create_model(num_classes=FLAGS.num_classes)

    data = data_provider.get_data(dataset,
                                  FLAGS.model_name,
                                  batch_size=FLAGS.batch_size,
                                  is_training=True,
                                  height=FLAGS.height,
                                  width=FLAGS.width)

    logits, endpoints = model.create_model(data.images,
                             num_classes=dataset.num_classes,
                             weight_decay=FLAGS.weight_decay,
                             is_training=True)
    total_loss = model.create_loss(logits,
                                   endpoints,
                                   data.labels_one_hot,
                                   FLAGS.label_smoothing)
    model.create_summary(data, logits, is_training=True)
    init_fn = model.create_init_fn_to_restore(FLAGS.checkpoint_path,
                                              FLAGS.checkpoint_inception,
                                              FLAGS.checkpoint_exclude_scopes)
    variables_to_train = model.get_variables_to_train(FLAGS.trainable_scopes)
    if FLAGS.show_graph_state:
        logging.info('Total number of weights in the graph: %s',
                     utils.calculate_graph_metrics())
    train(total_loss, init_fn, variables_to_train)

if __name__ == '__main__':
    tf.app.run()