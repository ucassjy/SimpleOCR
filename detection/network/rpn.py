import tensorflow as tf

def reduce_anchors_and_get_iou(img, im_info, gt_boxes, anchors):
    for x in im_info[0]:
        for y in im_info[1]:
            for a in anchors:
                if
                a[0] = x
                a[1] = y
    return anchors

def rpn(net, shape):
    kernel = tf.random_normal(shape=[1, 3, 3, 512])
    net = tf.nn.conv2d_transpose(net, kernel, output_shape=[1, shape[0], shape[1], 3],
                                strides=[1,1,1,1], padding="SAME")
    return net
