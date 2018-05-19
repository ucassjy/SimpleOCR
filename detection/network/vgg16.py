import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python import pywrap_tensorflow

def get_pretrained_net(sess, net):
    reader = pywrap_tensorflow.NewCheckpointReader('network/vgg_16.ckpt')
    shape_map = reader.get_variable_to_shape_map()
    v_to_restore = []
    v_to_fix = {}

    variables = tf.global_variables()
    for v in variables:
        if v.name == 'image:0':
            continue

        if v.name.split(':')[0] in shape_map:
            print('Variables restored: ', v.name)
            v_to_restore.append(v)

    restorer = tf.train.Saver(v_to_restore)
    restorer.restore(sess, 'network/vgg_16.ckpt')

    return net

def img2fm(image, is_training):
    with tf.variable_scope('vgg_16', 'vgg_16'):
        image = tf.to_float(image)
        net = slim.repeat(image, 2, slim.conv2d, 64, [3, 3],
                            trainable=False, scope='conv1')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
                            trainable=False, scope='conv2')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3],
                            trainable=False, scope='conv3')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                            trainable=is_training, scope='conv4')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                            trainable=is_training, scope='conv5')

    return net

if __name__ == '__main__':
    img = slim.variable('image', shape=[1, 100, 100, 3],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
    net = img2fm(img, False)
    with tf.Session() as sess:
        net = get_pretrained_net(sess, net)
    print(net)
