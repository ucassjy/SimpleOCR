import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

from give_label import give_label

# MAX_CHARACTER = 50
# LOAD IN
Threshold = 50
MAX_LENGTH = 15 * Threshold

label_origin, filename = give_label()

#TENSORFLOW TIME

OUTPUT_SHAPE = (Threshold, MAX_LENGTH)
num_epochs = 1000

num_hidden = 64
num_layers = 1
INITIAL_LEARNING_RATE = 0.001
DECAY_STEP = 500
REPORT_STEP = 100
LEARNING_RATE_DECAY_FACTOR = 0.9
MOMENTUM = 0.9
BATCH_SIZE = 10

START = 3 # NEED TO BE CHANGE IN Session #NOTE


def get_input_label(BATCH_SIZE, START, label_origin):
    input = np.zeros([BATCH_SIZE, OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]])
    for i in range(BATCH_SIZE):
        img = cv2.imread(filename[i + START], cv2.IMREAD_GRAYSCALE)
        img = img[:,0:MAX_LENGTH]
        img = np.transpose(img)

        input[i,:,:] =  img
    target = label_origin[START:START+BATCH_SIZE]
    sparse_target = sparse_tuple_from(target)
    return input, sparse_target

def sparse_tuple_from(sequences, dtype = np.int32):
    indices = []
    values = []
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
    return indices, values, shape

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.5)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, stride=(1, 1), padding='SAME'):
    return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1],padding=padding)

def max_pool(x, ksize=(2, 2), stride=(2, 2)):
    return tf.nn.max_pool(x, ksize=[1, ksize[0], ksize[1], 1],strides=[1, stride[0], stride[1], 1], padding='SAME')

def avg_pool(x, ksize=(2, 2), stride=(2, 2)):
    return tf.nn.avg_pool(x, ksize=[1, ksize[0], ksize[1], 1],strides=[1, stride[0], stride[1], 1], padding='SAME')


get_input_label(BATCH_SIZE, START, label_origin)





















# for i in range(len(filename)):
#     img = cv2.imread(filename[i], cv2.IMREAD_GRAYSCALE)
#     img = img[:,0:MAX_LENGTH]
#     if img is None:
#         continue
#
