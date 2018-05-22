import os
import numpy as np

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

class SolverWrapper(object):
    """ A wrapper class for the training process """

    def __init__(self, sess, network, blobs_all, output_dir, tbdir, pretrained_model=None):
        self.net = network
        self.blobs_all = blobs_all
        self.output_dir = output_dir
        self.tbdir = tbdir
        self.pretrained_model = 'nets/vgg16.ckpt'

    def get_variables_in_checkpoint_file(self):
        reader = pywrap_tensorflow.NewCheckpointReader(self.pretrained_model)
        var_to_shape_map = reader.get_variable_to_shape_map()
        return var_to_shape_map

    def construct_graph(self, sess):
        with sess.graph.as_default():
            # Build the main computation graph
            layers = self.net.create_architecture('TRAIN',sess)
            # Define the loss
            loss = layers['total_loss']
            # Set learning rate and momentum
            lr = tf.Variable(0.001, trainable=False)
            self.optimizer = tf.train.MomentumOptimizer(lr, 0.9)

            # Compute the gradients with regard to the loss
            gvs = self.optimizer.compute_gradients(loss)
            train_op = self.optimizer.apply_gradients(gvs)

            self.saver = tf.train.Saver(max_to_keep=10000)
            # Write the train and validation information to tensorboard
            self.writer = tf.summary.FileWriter(self.tbdir, sess.graph)
            self.valwriter = tf.summary.FileWriter(self.tbvaldir)

        return lr, train_op

    def initialize(self, sess):
        # Fresh train directly from ImageNet weights
        print('Loading initial model weights from %s'.format(self.pretrained_model))
        variables = tf.global_variables()
        # Initialize all variables first
        sess.run(tf.variables_initializer(variables, name='init'))
        var_keep_dic = self.get_variables_in_checkpoint_file()
        # Get the variables to restore, ignoring the variables to fix
        variables_to_restore = self.net.get_variables_to_restore(variables, var_keep_dic)

        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(sess, self.pretrained_model)
        print('Loaded.')
        # Need to fix the variables before loading to
        # change the convolutional weights fc6 and fc7 to fully connected weights
        self.net.fix_variables(sess, self.pretrained_model)
        print('Fixed.')
        rate = 0.001

        return rate

    def train_model(self, sess):
        # Construct the computation graph
        lr, train_op = self.construct_graph(sess)

        for i in range(10):
            if i == 0:
                rate = self.initialize(sess)
            # Learning rate
            if i == 8:
                rate *= 0.1
                sess.run(tf.assign(lr, rate))
            for j in range(len(self.blobs_all)):
                # Get training data, one image at a time
                blobs = self.blobs_all[j]

                if i is not 0:
                    # Compute the graph with summary
                    rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss, summary = \
                        self.net.train_step_with_summary(sess, blobs, train_op)
                    self.writer.add_summary(summary, float(i))
                else:
                    # Compute the graph without summary
                    rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss = \
                        self.net.train_step(sess, blobs, train_op)

            # Display training information
            print('iter: %d / %d, total loss: %.6f\n >>> rpn_loss_cls: %.6f\n '
                    '>>> rpn_loss_box: %.6f\n >>> loss_cls: %.6f\n >>> loss_box: %.6f\n >>> lr: %f' % \
                    (i, 10, total_loss, rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, lr.eval()))

        restorer = tf.train.Saver()
        restorer.save(sess, self.output_dir)

        self.writer.close()
        self.valwriter.close()

def train_net(network, blobs_all, output_dir, tb_dir, pretrained_model=None):
    """Train a Faster R-CNN network."""

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    with tf.Session(config=tfconfig) as sess:
        sw = SolverWrapper(sess, network, blobs_all, output_dir, tb_dir,
                    pretrained_model=pretrained_model)

        print('Solving...')
        sw.train_model(sess)
    print('Done solving.')
