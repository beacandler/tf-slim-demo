"""Functions to build the classification model

Usage example:
    clss_model = model.Model()

    endpoints = model.create_base(data.images, data.labels_one_hot)
    #endpoints.class_scores is tensor with predicted score
    total_loss = model.create_loss(data, endpoints)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import logging
import tensorflow as tf
from tensorflow.contrib import slim

import utils
from nets import inception_v3

class Model():
    def __init__(self,
                 num_classes,
                 mparams=None):
        """Initialized model parameters.

        Args:
            num_classes: number of predicted classes.
            mparams: a dictionary with hyper parameters for methods,
                key - function names, values - corresponding nametuples.
        """
        self.num_classes = num_classes
        self.mparams = mparams

    def create_model(self,
                    images,
                    num_classes,
                    weight_decay=0.00004,
                    scope='Flowers',
                    reuse=None,
                    is_training=True):
        """Creates a base part of the Model (no gradients, no loss, no summaries).

        Args:
            images: A tensor of size [batch_size, height, width, channels].
            num_classes: The number of predicted classes.
            scope: Optional variable_scope.
            reuse: Whether or not the network or its variables should be reused. To
                be able to reuse 'scope' must be given.
            is_training: Whether is training or not.

        Returns:
            A named tuple OutputEndpoints.
        """
        with tf.variable_scope(scope, [images], reuse=reuse):
            with slim.arg_scope(inception_v3.inception_v3_arg_scope(weight_decay=weight_decay)):
                logits, endpoints = inception_v3.inception_v3(
                    inputs = images,
                    num_classes=num_classes,
                    is_training=is_training)
                return logits, endpoints

    def create_loss(self,
                    logits,
                    endpoints,
                    labels_one_hot,
                    label_smoothing):
        """Create a CrossEntropy loss of the Model (typical a softmax loss)

        Args:
            logits: logits inference by the model
            labels_one_hot: groundtruth label
            """
        if 'AuxLogits' in endpoints:
            slim.losses.softmax_cross_entropy(
                endpoints['AuxLogits'], labels_one_hot,
                label_smoothing=label_smoothing,
                weights=0.4,
                scope='aux_loss')
        slim.losses.softmax_cross_entropy(
            logits, labels_one_hot, label_smoothing=label_smoothing, weights=1.0)


        total_loss = slim.losses.get_total_loss()
        tf.summary.scalar('TotalLoss', total_loss)

        return total_loss

    def create_summary(self, data, logits, is_training=True):
        """Create all summary for the Model

        Args:
            data: InputEndpoints namedtuple:
            logits: logits inference by the model.
            is_training: Whether is training or not.

        Returns:
            A list of of evalution ops
            """

        def sname(label):
            prefix = 'train' if is_training else 'eval'
            return '{}/{}'.format(prefix, label)

        if is_training:
            # tf.summary.image(
            #     sname('Image/summary'), data.images, max_outputs=4)
            # tf.summary.histogram(
            #     sname('Image/label'), data.labels)
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)
            return None
        else:
            predictions = tf.argmax(logits, axis=1)
            labels = tf.squeeze(data.labels)
            names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
                'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
            })

            # define the summaries to write.
            for name, value in names_to_values.iteritems():
                summary_name = sname(name)
                tf.summary.scalar(summary_name, tf.Print(value, [value], summary_name))
            return names_to_updates.values()

    def create_init_fn_to_restore(self,
                                  master_checkpoint,
                                  inception_checkpoint=None,
                                  checkpoint_exclude_scopes=None):
        """Create an init operations to restore weights from various checkpoints.

        Args:
            master_checkpoint: path to a checkpoint which contains all weights for
            the whole model.
            inception_checkpoint: path to a checkpoint which contains weights for the
            inception part only.

        Returns:
            a function to run initialization to ops.
        """

        all_assign_ops = []
        all_feed_dict = {}
        def assign_from_checkpoint(variables, checkpoint):
            logging.info('Request to re-store {} weights from {}'.format(
                len(variables), checkpoint))
            if not variables:
                logging.error('can\'t find any variables to restore.')
                sys.exit(1)
            assign_op, feed_dict = slim.assign_from_checkpoint(checkpoint, variables)
            all_assign_ops.append(assign_op)
            all_feed_dict.update(feed_dict)

        if master_checkpoint:
            assign_from_checkpoint(utils.variabels_to_restore(), master_checkpoint)
        if inception_checkpoint:
            exclusions = []
            if checkpoint_exclude_scopes:
                exclusions = [scope.strip()
                              for scope in checkpoint_exclude_scopes.split(',')]
            variabels_to_restore = {}
            variables = utils.variabels_to_restore(
                'Flowers', strip_scope=True)
            for var in variables:
                exclude = False
                for exclusion in exclusions:
                    if var.startswith(exclusion):
                        exclude = True
                        break
                if not exclude:
                    variabels_to_restore[var] = variables[var]
            assign_from_checkpoint(variabels_to_restore, inception_checkpoint)

        def init_assign_fn(sess):
            logging.info('Restoring checkpoint(s)')
            sess.run(all_assign_ops, all_feed_dict)

        return init_assign_fn

    def get_variables_to_train(self, trainable_scopes=None):
        """Returns a list of variables to train.

        Returns:
            A list of variables to train by the optimizer.
        """
        if trainable_scopes is None:
            return tf.trainable_variables()
        else:
            scopes = ['Flowers/'+ scope.strip() for scope in trainable_scopes.split(',')]

        variables_to_train = []
        for scope in scopes:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            variables_to_train.extend(variables)
        return variables_to_train

