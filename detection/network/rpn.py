import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

def anchors_in_image(im_info, anchors, fm_shape):
    anchors = [(x*16, y*16, a[2], a[3], a[4]) for x in range(int(fm_shape[0]))
                                              for y in range(int(fm_shape[1]))
                                              for a in anchors]

    x_max = im_info[0]
    y_max = im_info[1]

    for a in anchors:
        x = a[0]
        y = a[1]
        h = a[2]
        w = a[3]
        angle = a[4] * np.pi / 180
        y_top = y + (w * np.abs(np.sin(angle)) + h * np.abs(np.cos(angle))) / 2
        y_bot = y - (w * np.abs(np.sin(angle)) + h * np.abs(np.cos(angle))) / 2
        x_top = x + (w * np.abs(np.cos(angle)) + h * np.abs(np.sin(angle))) / 2
        x_bot = x - (w * np.abs(np.cos(angle)) + h * np.abs(np.sin(angle))) / 2
        if y_top > y_max or y_bot < 0 or x_top > x_max or x_bot < 0:
            anchors.remove(a)
    return anchors

def rpn(net):
    net = slim.conv2d(net, 512, [3, 3], scope='a')
    scale = int(net.shape[0]) * int(net.shape[1])
    cls_scores = slim.conv2d(net, scale * 2, [1, 1], scope='cls_scores')
    reg_coords = slim.conv2d(net, scale * 5, [1, 1], scope='reg_coords')
    return net, cls_scores, reg_coords
