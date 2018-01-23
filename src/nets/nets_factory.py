"""Contains a factory to build various model."""

import functools

from tensorflow.contrib import slim

import lenet
import inception_v3

networks_map = {'inception_v3': inception_v3.inception_v3,
                'lenet': lenet.lenet}

arg_scope_map = {'inception_v3': inception_v3.inception_v3_arg_scope,
                 'lenet': lenet.lenet_arg_scope}

def get_network_fn(name, num_classes, weight_decay=0.0, is_training=False):
    """Returns a network function such as 'logits, end_points = network_fn(images)

    Args:
        name: The name of networks.
        num_classes: The number of classes to use for classification. If 0 or None,
            the logits layer is omitted and inputs feature are returned instead.
        weight_decay: The l2 coefficient for the model weights.
        is_training: Whether is Training or not.

    Returns:
        network_fn: A function that applies the model to a batch of images. It has
            the following signature:
              net, end_points = network_fn(images)
            The `images` input is a tensor of shape [batch_size, height, width, 3]
            with height = width = network_fn.default_image_size. (The permissibility
          and treatment of other sizes depends on the network_fn.)
          The returned `end_points` are a dictionary of intermediate activations.
          The returned `net` is the topmost layer, depending on `num_classes`:
          If `num_classes` was a non-zero integer, `net` is a logits tensor
          of shape [batch_size, num_classes].
          If `num_classes` was 0 or `None`, `net` is a tensor with the input
          to the logits layer of shape [batch_size, 1, 1, num_features] or
          [batch_size, num_features]. Dropout has not been applied to this
          (even if the network's original classification does); it remains for
          the caller to do this or not.    '

    Raises:
        ValueError: if network 'name' is not recognized.
        """
    if name not in networks_map:
        ValueError('Name of network unknown {}'.format(name))

    func = networks_map[name]
    @functools.wraps(func)

    def network_fn(images, **kwargs):
        arg_scope = arg_scope_map[name](weight_decay=weight_decay)
        with slim.arg_scope(arg_scope):
            return func(images, num_classes, is_training, **kwargs)
    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size

    return network_fn