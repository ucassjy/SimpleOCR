from network import vgg16
from network import generate_anchors
from network import rpn

import tensorflow as tf

class Network(object):
    def __init__(self, mode, sess, blobs):
        self.is_training = mode == 'TRAIN'
        self._sess = sess
        self._blobs = blobs

    def build_network(self):
        self._img = self._blobs[0]['data'] #tf.placeholder(tf.float32, shape=[1, None, None, 3])
        self._img = self._img.reshape((1, self._img.shape[0], self._img.shape[1], 3))
        self._im_info = self._blobs[0]['im_info'] #tf.placeholder(tf.float32, shape=[3])
        self._gt_boxes = self._blobs[0]['gt_list'] #tf.placeholder(tf.float32, shape=[None, 5])

        # network architecture
        net = vgg16.img2fm(self._img, self.is_training)
        net = vgg16.get_pretrained_net(self._sess, net)
        net, cls_scores, reg_coords = rpn.rpn(net)
        anchors = generate_anchors.generate_anchors()
        anchors = rpn.anchors_in_image(self._im_info, anchors, net.shape[1:3])
        print(net)
        print(len(anchors))
        #d1 = tf.layers.dense(self.x, 512, activation=tf.nn.relu, name="dense1")
        #d2 = tf.layers.dense(d1, 10, name="dense2")

        #with tf.name_scope("loss"):
        #    self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=d2))
        #    self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cross_entropy,
        #                                                                                 global_step=self.global_step_tensor)
        #    correct_prediction = tf.equal(tf.argmax(d2, 1), tf.argmax(self.y, 1))
        #    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
#        self.saver = tf.train.Saver()
