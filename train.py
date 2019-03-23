# Copyright 2017 Zhongyi Han. All Rights Reserved.
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
# limitations under the License.6
import os
import time
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from nets import preprocessing, SpinePathNet,losses
import tf_utils

# Fold is one of five folds cross-validation.
Fold = 1
# =========================================================================== #
# Model saving Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'train_dir', 'tmp/tfmodels_%s/'%Fold,
    'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 1, 'GPU memory fraction to use.')

tf.app.flags.DEFINE_float(
    'weight', 1, 'The weight is between cross entropy loss and metric loss.')
# =========================================================================== #
# Dataset Flags.
# =========================================================================== #

tf.app.flags.DEFINE_string(
    'dataset_name', 'spine_segmentation', 'The name of the dataset to load.')
tf.app.flags.DEFINE_integer(
    'num_classes', 7, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train_%s_fold'%Fold, 'The name of the train/test split.')#spine_segmentation_train_7_class.tfrecord
tf.app.flags.DEFINE_string(
    'dataset_dir', './datasets/tfrecords_spine_segmentation/', 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_integer(
    'batch_size', 4, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'num_samples', 202, 'The number of samples in the training set.')
tf.app.flags.DEFINE_integer(
    'num_readers', 20,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 8,
    'The number of threads used to create the batches.')

# =========================================================================== #
# Optimization Flags.
# =========================================================================== #
tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')
tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.94,
    'The decay rate for adadelta.')
tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')
tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')
tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')
tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')
tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')
tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')
tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')
tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

# =========================================================================== #
# Learning Rate Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.00001,
    'The minimal end learning rate used by a polynomial decay learning rate.')
tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')
tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.96, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 10.0,
    'Number of epochs after which learning rate decays.')
tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')
tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')
tf.app.flags.DEFINE_integer('num_epochs', 300,
                            'The number of training epochs.')

# =========================================================================== #
# Fine-Tuning Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'checkpoint_model_scope', None,
    'Model scope in the checkpoint. None if the same as the trained model.')
tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')
tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')
tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', None,#['SpinePathNet/feature_embedding','SpinePathNet/logits/'],
    'When restoring a checkpoint would ignore missing variables.')

FLAGS = tf.app.flags.FLAGS
# =========================================================================== #
# Main training routine.
# =========================================================================== #


def main(_):

    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    ##logging tools
    tf.logging.set_verbosity(tf.logging.DEBUG)

    with tf.Graph().as_default():


        # Create global_step.
        with tf.device('/cpu:0'):
            global_step = tf.train.create_global_step()

        # Select the dataset.
        dataset = FLAGS.dataset_dir + '%s_%s.tfrecord' %(FLAGS.dataset_name,FLAGS.dataset_split_name)

        with tf.device('/cpu:0'):
            with tf.name_scope('input'):
                filename_queue = tf.train.string_input_producer([dataset], num_epochs=FLAGS.num_epochs)

                image,mask_class,mask_instance = tf_utils.read_and_decode(filename_queue, batch_size = FLAGS.batch_size,\
                                                                       capacity=20 * FLAGS.batch_size,\
                                                                       num_threads=FLAGS.num_readers,\
                                                                       min_after_dequeue=10 * FLAGS.batch_size)

                # reshape mask_class as [-1, num_classes]
                labels = tf.contrib.layers.one_hot_encoding(mask_class,FLAGS.num_classes)

                labels = tf.to_float(tf.reshape(labels, (-1, FLAGS.num_classes)))
               # onedim_class = tf.reshape(mask_class, (-1,))

            #image = preprocessing.data_augmentation(image, is_training=True)


        #image,mask_class,mask_instance = preprocessing(image,mask_class,mask_instance,is_train=True,data_format= 'NHWC')





        _, logits= SpinePathNet.g_net_graph(image, Fold, batch_size=FLAGS.batch_size,\
                                       class_num = FLAGS.num_classes, reuse=None,\
                                       is_training=True, scope='SpinePathNet')


        ###computer constrint and loss_logits.
        #with tf.name_scope('loss_constrint'):

            #loss_constrint = SpinePathNet.instance_loss(FLAGS,feature_embedding, mask_instance)
        #with tf.device('/gpu:1'):
         #with tf.name_scope('metric_loss'):
         #       metric_loss_1 = losses.metric_loss_one_dimension(FLAGS,feature_embedding, onedim_class)
          #      metric_loss = tf.losses.compute_weighted_loss(metric_loss_1,weights=1.0)

        with tf.name_scope('cross_entropy_loss'):

            cross_entropy_loss = losses.weighted_cross_entropy_with_logits(FLAGS, Fold, logits, labels)

        #with tf.name_scope('total_loss'):

        #    total_loss = tf.add(metric_loss, cross_entropy_loss)

        learning_rate = tf_utils.configure_learning_rate(FLAGS,FLAGS.num_samples, global_step)


        optimizer = tf_utils.configure_optimizer(FLAGS, learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)


        #Note: when training, the moving_mean and moving_variance need to be updated.
        with tf.control_dependencies(update_ops):
            train_op_cel = optimizer.minimize(cross_entropy_loss, global_step=global_step)
            #train_op_ml = optimizer.minimize(metric_loss,global_step=global_step)
        # The op for initializing the variables.
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())



        # Create a session for running operations in the Graph.
        config = tf.ConfigProto(allow_soft_placement = True)
        sess = tf.Session(config=config)

        # Initialize the variables (the trained variables and the
        # epoch counter).
        sess.run(init_op)


        #Include max_to_keep=5
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)

        if ckpt and ckpt.model_checkpoint_path:
            exclude_list = FLAGS.ignore_missing_vars
            variables_list  = tf.contrib.framework.get_variables_to_restore(exclude=exclude_list)
            restore = tf.train.Saver(variables_list)
            restore.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            global_step = int(global_step)

        else: global_step = 0
        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)


        # Save models.
        if not tf.gfile.Exists(FLAGS.train_dir):
                #tf.gfile.DeleteRecursively(FLAGS.train_dir)
                tf.gfile.MakeDirs(FLAGS.train_dir)


        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')


        try:
            step = global_step # TODO: Continue to train the model, if the checkpoints are exist.
            while not coord.should_stop():
                start_time = time.time()
                # To inspect the values
                # of your ops or variables, you may include them in
                # the list passed to sess.run() and the value tensors
                # will be returned in the tuple from the call.

                _, loss_value, lr = sess.run([train_op_cel, cross_entropy_loss, learning_rate])

                duration = time.time() - start_time

                if (step) % 10 == 0:
                    #total_loss = %.2f, metric_loss= %.2f,
                    print('Step %d:  cross_entropy_loss = %.2f, learning rate = %.5f (%.3f sec)' % (step, loss_value, lr, duration))

                step += 1


                if step % 1000 == 0: #or (step + 1) == FLAGS.max_steps


                    saver.save(sess, checkpoint_path, global_step=step)
                    ##Add the model evaluatation  in the future.

        except tf.errors.OutOfRangeError:

            print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
            saver.save(sess, checkpoint_path, global_step=step)
            ##Add the model evaluatation  in the future.
            print('Model is saved as %s') % checkpoint_path

        finally:

            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()


if __name__ == '__main__':
    tf.app.run()
