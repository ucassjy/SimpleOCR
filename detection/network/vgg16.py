import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

def get_variables_to_restore(variables, shape_map):
    v_to_restore = []

    for v in shape_map:
        if v == 'vgg_16/fc6/weights:0' or \
           v == 'vgg_16/fc7/weights:0' or \
           v == 'vgg_16/conv1/conv1_1/weights:0':
            print('To be fixed.')
            continue

        print('Variables restored: ', v)
        v_to_restore.append(v)

    return v_to_restore


def _image_to_head(is_training, reuse=None):
    net = slim.repeat(image, 2, slim.conv2d, 64, [3, 3],
                        trainable=False, scope='conv1')
    net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
                        trainable=False, scope='conv2')
    net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')
    net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3],
                        trainable=False, scope='conv3')
    net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
    net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3],
                        trainable=is_training, scope='conv4')
    net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
    net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3],
                        trainable=is_training, scope='conv5')

    return net

def vgg16(img):
    reader = pywrap_tensorflow.NewCheckpointReader('vgg_16.ckpt')
    shape_map = reader.get_variable_to_shape_map()
    variables = tf.global_variables()

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=tfconfig)
    sess.run(tf.variables_initializer(variables, name='init'))
    v_to_restore = get_variables_to_restore(variables, shape_map)
    print(v_to_restore)
    feat_map = img
    return feat_map

if __name__ == '__main__':
    import numpy as np
    img = np.zeros((224, 224, 3))
    fm = vgg16(img)
