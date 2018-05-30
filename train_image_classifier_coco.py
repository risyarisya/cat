# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic training script that trains a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
import re

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'master', '', 'Session master')

tf.app.flags.DEFINE_string(
    'train_dir', '/path/to/train_dir/',
    'Directory where checkpoints are written to.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

_LEARNING_RATE_DECAY_FACTOR = 0.94
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset.')

tf.app.flags.DEFINE_integer(
    'batch_size', 10, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

FLAGS = tf.app.flags.FLAGS

def get_and_convert_variables_in_checkpoint(variables,
                                          checkpoint_path,
                                          include_global_step=True):
  """Returns the subset of variables available in the checkpoint.
  Inspects given checkpoint and returns the subset of variables that are
  available in it.
  TODO(rathodv): force input and output to be a dictionary.
  Args:
    variables: a list or dictionary of variables to find in checkpoint.
    checkpoint_path: path to the checkpoint to restore variables from.
    include_global_step: whether to include `global_step` variable, if it
      exists. Default True.
  Returns:
    A list or dictionary of variables.
  Raises:
    ValueError: if `variables` is not a list or dict.
  """
  ckpt_reader = tf.train.NewCheckpointReader(checkpoint_path)
  ckpt_vars_to_shape_map = ckpt_reader.get_variable_to_shape_map()
  ckpt_vars_convert_tbl = {}
  for ckpt_var_name, variable in sorted(ckpt_vars_to_shape_map.items()):
    if ckpt_var_name.startswith('FeatureExtractor'):
      convert_var_name = (
        re.split('^' + 'FeatureExtractor' + '/', 
                 ckpt_var_name)[-1])
      ckpt_vars_convert_tbl[convert_var_name] = ckpt_var_name

  if not include_global_step:
    ckpt_vars_to_shape_map.pop(tf.GraphKeys.GLOBAL_STEP, None)
  
  vars_in_ckpt = {}
  for variable in tf.global_variables():
    var_name = variable.op.name
    if var_name in ckpt_vars_convert_tbl:
      if ckpt_vars_to_shape_map[ckpt_vars_convert_tbl[var_name]] == variable.shape.as_list():
        vars_in_ckpt[ckpt_vars_convert_tbl[var_name]] = var_name
      else:
        tf.logging.warning('Variable [%s] is available in checkpoint, but has an '
                        'incompatible shape with model variable.',
                        var_name)
    else:
      tf.logging.warning('Variable [%s] is not available in checkpoint',
                      var_name)
  return vars_in_ckpt


def _get_init_fn():
  """Returns a function run by the chief worker to warm-start the training.

  Note that the init_fn is only run when initializing the model during the very
  first global step.

  Returns:
    An init function run by the supervisor.
  """
  if FLAGS.checkpoint_path is None:
    return None

  # Warn the user if a checkpoint exists in the train_dir. Then we'll be
  # ignoring the checkpoint anyway.
  if tf.train.latest_checkpoint(FLAGS.train_dir):
    tf.logging.info(
        'Ignoring --checkpoint_path because a checkpoint already exists in %s'
        % FLAGS.train_dir)
    return None

  if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
  else:
    checkpoint_path = FLAGS.checkpoint_path

  available_var_map = get_and_convert_variables_in_checkpoint([], checkpoint_path)

  exclusions = []
  if FLAGS.checkpoint_exclude_scopes:
    exclusions = [scope.strip()
                  for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

  variables_to_restore = {}
  for old_var, new_var in available_var_map.items():
    excluded = False
    for exclusion in exclusions:
      if new_var.startswith(exclusion):
        excluded = True
        break
    if not excluded:
      variables_to_restore[old_var] = new_var

  tf.logging.info('Fine-tuning from %s' % checkpoint_path)
  return tf.train.init_from_checkpoint(checkpoint_path, variables_to_restore)

def _get_variables_to_train():
  """Returns a list of variables to train.

  Returns:
    A list of variables to train by the optimizer.
  """
  if FLAGS.trainable_scopes is None:
    return tf.trainable_variables()
  else:
    scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

  variables_to_train = []
  for scope in scopes:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    variables_to_train.extend(variables)
  return variables_to_train


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    #######################
    # Config model_deploy #
    #######################
    deploy_config = model_deploy.DeploymentConfig(
        num_clones=1,
        clone_on_cpu=False,
        replica_id=0,
        num_replicas=1,
        num_ps_tasks=0)

    # Create global_step
    with tf.device(deploy_config.variables_device()):
      global_step = slim.create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        'flowers', 'train', FLAGS.dataset_dir)

    ######################
    # Select the network #
    ######################
    network_fn = nets_factory.get_network_fn(
        'mobilenet_v1',
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        weight_decay=FLAGS.weight_decay,
        is_training=True)

    #####################################
    # Select the preprocessing function #
    #####################################
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        'mobilenet_v1',
        is_training=True)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    with tf.device(deploy_config.inputs_device()):
      provider = slim.dataset_data_provider.DatasetDataProvider(
          dataset,
          num_readers=4,
          common_queue_capacity=20 * FLAGS.batch_size,
          common_queue_min=10 * FLAGS.batch_size)
      [image, label] = provider.get(['image', 'label'])
      label -= FLAGS.labels_offset

      train_image_size = network_fn.default_image_size

      image = image_preprocessing_fn(image, train_image_size, train_image_size)

      images, labels = tf.train.batch(
          [image, label],
          batch_size=FLAGS.batch_size,
          num_threads=4,
          capacity=5 * FLAGS.batch_size)
      labels = slim.one_hot_encoding(
          labels, dataset.num_classes - FLAGS.labels_offset)
      batch_queue = slim.prefetch_queue.prefetch_queue(
          [images, labels], capacity=2 * deploy_config.num_clones)

    ####################
    # Define the model #
    ####################
    def clone_fn(batch_queue):
      """Allows data parallelism by creating multiple clones of network_fn."""
      images, labels = batch_queue.dequeue()
      logits, end_points = network_fn(images)

      #############################
      # Specify the loss function #
      #############################
      slim.losses.softmax_cross_entropy(
          logits, labels, label_smoothing=FLAGS.label_smoothing, weights=1.0)
      return end_points

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
    first_clone_scope = deploy_config.clone_scope(0)
    # Gather update_ops from the first clone. These contain, for example,
    # the updates for the batch_norm variables created by network_fn.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

    # Add summaries for end_points.
    end_points = clones[0].outputs
    for end_point in end_points:
      x = end_points[end_point]
      summaries.add(tf.summary.histogram('activations/' + end_point, x))
      summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                      tf.nn.zero_fraction(x)))

    # Add summaries for losses.
    for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
      summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

    # Add summaries for variables.
    for variable in slim.get_model_variables():
      summaries.add(tf.summary.histogram(variable.op.name, variable))

    moving_average_variables, variable_averages = None, None

    #########################################
    # Configure the optimization procedure. #
    #########################################
    with tf.device(deploy_config.optimizer_device()):

      num_epochs_per_decay = 2.5
      decay_steps = int(dataset.num_samples / FLAGS.batch_size *
                        num_epochs_per_decay)
      learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
                                  global_step,
                                  decay_steps,
                                  _LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True,
                                    name='exponential_decay_learning_rate')

      optimizer = tf.train.RMSPropOptimizer(
                           learning_rate,
                           decay=FLAGS.rmsprop_decay,
                           momentum=FLAGS.rmsprop_momentum,
                           epsilon=FLAGS.opt_epsilon)
      summaries.add(tf.summary.scalar('learning_rate', learning_rate))

    # Variables to train.
    variables_to_train = _get_variables_to_train()

    #  and returns a train_tensor and summary_op
    total_loss, clones_gradients = model_deploy.optimize_clones(
        clones,
        optimizer,
        var_list=variables_to_train)
    # Add total_loss to summary.
    summaries.add(tf.summary.scalar('total_loss', total_loss))

    # Create gradient updates.
    grad_updates = optimizer.apply_gradients(clones_gradients,
                                             global_step=global_step)
    update_ops.append(grad_updates)

    update_op = tf.group(*update_ops)
    with tf.control_dependencies([update_op]):
      train_tensor = tf.identity(total_loss, name='train_op')

    # Add the summaries from the first clone. These contain the summaries
    # created by model_fn and either optimize_clones() or _gather_clone_loss().
    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                       first_clone_scope))

    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False)

    ###########################
    # Kicks off the training. #
    ###########################
    slim.learning.train(
        train_tensor,
        logdir=FLAGS.train_dir,
        master=FLAGS.master,
        is_chief=True,
        session_config=session_config,
        init_fn=_get_init_fn(),
        summary_op=summary_op,
        number_of_steps=FLAGS.max_number_of_steps,
        log_every_n_steps=10,
        save_summaries_secs=300,
        save_interval_secs=300,
        sync_optimizer=optimizer if False else None)


if __name__ == '__main__':
  tf.app.run()
