"""Functions to support building models for Flowers Classification"""

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.tfprof import model_analyzer

def variabels_to_restore(scope=None, strip_scope=False):
    """Returns a list of variabels to restore for the specified list method.

    It is supposed that variable name starts with the method's scope (a prefix
    returned by _method_scope function.)

    Args:
        scope: a scope for a whole model.
        strip_scope: If True will return variable names without method's scope.
    """
    if scope:
        variable_map = {}
        variables_to_restore = slim.get_variables_to_restore(include=[scope])
        for var in variables_to_restore:
            if strip_scope:
                var_name = var.op.name[len(scope) + 1:]
            else:
                var_name = var.op.name
            variable_map[var_name] = var
        return variable_map
    else:
        return {var.op.name: var for var in slim.get_variables_to_restore()}


def create_optimizer(hparams):
    """Creates optimizer based on the specified flags."""

    if hparams.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=hparams.learning_rate, momentum=hparams.momentum)
    elif hparams.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate=hparams.learning_rate)
    elif hparams.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            hparams.learning_rate)
    elif hparams.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            hparams.learning_rate)
    elif hparams.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            hparams.learning_rate, momentum=hparams.momentum)

    return optimizer

def calculate_graph_metrics():
    param_state = model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
    return param_state.total_params