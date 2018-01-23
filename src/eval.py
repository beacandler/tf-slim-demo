"""Script to evaluate a trained Flowers model.

A simple usage example:
python eval.py
"""

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow import app
from tensorflow.python.platform import flags

import common_flags
import data_provider

FLAGS = flags.FLAGS
common_flags.define()

flags.DEFINE_integer('num_evals', 1000,
                     'Number of batches to run eval for.')
flags.DEFINE_string('eval_log_dir', './log/eval_dir',
                    'Directory where the evaluation results are saved to.')
flags.DEFINE_integer('eval_interval_secs', 600,
                     'Frequency in seconds to run evaluations.')
flags.DEFINE_integer('number_of_steps', None,
                     'Number of times to run evaluation.')

def main(_):
    if not tf.gfile.Exists(FLAGS.eval_log_dir):
        tf.gfile.MakeDirs(FLAGS.eval_log_dir)

    dataset = common_flags.create_dataset(FLAGS.dataset_name, FLAGS.dataset_split_name)
    model = common_flags.create_model(num_classes=FLAGS.num_classes)
    data = data_provider.get_data(dataset,
                                  FLAGS.model_name,
                                  FLAGS.batch_size,
                                  is_training=False,
                                  height=FLAGS.height,
                                  width=FLAGS.width)
    logits, endpoints = model.create_model(data.images,
                                           num_classes=FLAGS.num_classes,
                                           is_training=False)
    eval_ops = model.create_summary(data, logits, is_training=False)
    slim.get_or_create_global_step()
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    slim.evaluation.evaluation_loop(
        master=FLAGS.master,
        checkpoint_dir=FLAGS.train_dir,
        logdir=FLAGS.eval_log_dir,
        eval_op=eval_ops,
        num_evals=FLAGS.num_evals,
        eval_interval_secs=FLAGS.eval_interval_secs,
        max_number_of_evaluations=FLAGS.number_of_steps,
        session_config=session_config)

if __name__=='__main__':
    app.run()
