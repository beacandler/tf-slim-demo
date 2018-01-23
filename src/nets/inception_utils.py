"""Contains common code shared by all inception models

    Usage of arg scope:
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            logits, end_points = inception.inception_v3(images, num_classes,
                                                        is_training=is_training)

    attention: If use only inception_v3_base, Please make sure pass is_traing to inception_v3_base

    Usage:
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            with slim.arg_scope([slim.batch_norm, slim.drop_out],  # the same as inception.inception_v3 function
                                is_training=is_training):
                net, endpoints = inception.inception_v3(images, num_classes,
                                                        is_traning=is_training)
"""
import tensorflow as tf
slim = tf.contrib.slim

def inception_arg_scope(weight_decay=0.00004,
                        use_batch_norm=True,
                        batch_norm_decay=0.9997,
                        batch_norm_epsilon=0.001,
                        activation_fn=tf.nn.relu):
    """Defines the default arg scope for inception models
    Args:
        weight_decay: The weight decay to regularizing the model.
        use_batch_norm: If True, batch_norm is applied after each convolution
        batch_norm_decay: Decay for batch norm moving average.
        batch_norm_epsilon: Small float added to variance to avoid dividing by zero
        in batch norm
        activation_fn: Activation function for conv2d.

    Returns:
        An 'arg_scope' to use for the inception model.
        """

    batch_norm_param = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        # collection containing update_ops
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'fused': None,
    }

    if use_batch_norm:
        normalizer_fn = slim.batch_norm
        normalizer_params = batch_norm_param
    else:
        normalizer_fn = None
        normalizer_params = {}

    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
            [slim.conv2d],
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=activation_fn,
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params) as sc:
            return sc