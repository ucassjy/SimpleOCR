import tensorflow as tf
import numpy as np
from layer_utils.generate_anchors import generate_anchors

def generate_anchors_pre_tf(height, width):

    anchors = tf.py_func(generate_anchors,[height, width],tf.float32,stateful=False,name=None)
    A = tf.constant(54)

    return tf.cast(anchors, dtype=tf.float32), A
