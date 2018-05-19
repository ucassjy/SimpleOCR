import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import os

from tensorflow.python import debug as tf_debug
from tensorflow.contrib import rnn
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from give_label import give_label

# MAX_CHARACTER = 50
# LOAD IN
Threshold = 50
MAX_LENGTH = 5 * Threshold

label_origin, filename = give_label()

#TENSORFLOW TIME

OUTPUT_SHAPE = (Threshold, MAX_LENGTH) #(50,750)
num_epochs = 300
#
# num_hidden = 256
# num_layers = 2
INITIAL_LEARNING_RATE = 0.01
DECAY_STEPS = 50
REPORT_STEPS = 10
LEARNING_RATE_DECAY_FACTOR = 0.9
MOMENTUM = 0.9
BATCH_SIZE = 300
BATCHES = int(len(filename)/BATCH_SIZE) + 1
TRAIN_SIZE = BATCHES * BATCH_SIZE
num_classes = 2240
 # NEED TO BE CHANGE IN Session #NOTE

def decode_dictionary():
    with open("dictionary.txt",'r') as f:
        dictionary = f.readlines()
        decode_dictionary = {}
        for line in dictionary:
            key = line.split(':')[0]
            value = line.split(':')[1]
            value = value.split('\n')[0]
            decode_dictionary[value] = key  #  Num->Character
    return decode_dictionary

def decode_sparse_tensor(sparse_tensor):
    #print("sparse_tensor = ", sparse_tensor)
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)
    #print("decoded_indexes = ", decoded_indexes)
    result = []
    for index in decoded_indexes:
        #print("index = ", index)
        result.append(decode_a_seq(index, sparse_tensor))
        #print(result)
    return result

def decode_a_seq(indexes, spars_tensor):
    decoded = []
    decode_dictionarys = decode_dictionary()
    for m in indexes:
        key = str(spars_tensor[1][m])
        #print(key)
        strs = decode_dictionarys[key]
        decoded.append(strs)
    return decoded

def report_accuracy(decoded_list, test_targets):

    original_list = decode_sparse_tensor(test_targets)

    detected_list = decode_sparse_tensor(decoded_list)
    print("original",original_list)
    print("DETECTED",detected_list)

    true_numer = 0

    if len(original_list) != len(detected_list):
        print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
              " test and detect length desn't match")
        return
    print("T/F: original(length) <-------> detectcted(length)")
    for idx, number in enumerate(original_list):
        detect_number = detected_list[idx]
        hit = (number == detect_number)
#        print(hit, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
        if hit:
            true_numer = true_numer + 1
    print("Test Accuracy:", true_numer * 1.0 / len(original_list))


def get_input_label(BATCH_SIZE, START, label_origin):
    input = np.zeros([BATCH_SIZE, OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]])
    for i in range(BATCH_SIZE):
        if START + BATCH_SIZE > len(filename):
            START = len(filename) - BATCH_SIZE
        img = cv2.imread(filename[i + START], cv2.IMREAD_GRAYSCALE)
        img = img[:,0:MAX_LENGTH]
        img = np.transpose(img)
        input[i,:,:] =  img
    target = label_origin[START:START+BATCH_SIZE]

    sparse_target = sparse_tuple_from(target)
    seq_len = np.ones(input.shape[0]) * OUTPUT_SHAPE[1]
    return input, sparse_target, seq_len

def sparse_tuple_from(sequences, dtype = np.int32):
    indices = []
    values = []
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)

    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
#    print(values)
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

def convolutional_layers():
    #输入数据，shape [batch_size, max_stepsize, num_features]

    train_inputs, _, train_seq_len = get_input_label(BATCH_SIZE,0, label_origin)
#    inputs = tf.placeholder(tf.float32, [None, None, OUTPUT_SHAPE[0]])
    inputs = np.float32(train_inputs)
    #第一层卷积层, 50*750*1 => 25*375*48
    W_conv1 = weight_variable([5, 5, 1, 48])
    b_conv1 = bias_variable([48])
    x_expanded = tf.expand_dims(inputs, 3)
    h_conv1 = tf.nn.relu(conv2d(x_expanded, W_conv1) + b_conv1)
#    h_conv1 = tf.layers.batch_normalization(h_conv1)
    h_pool1 = max_pool(h_conv1, ksize=(2, 2), stride=(2, 2))


    #第二层, 25*375*48 => 13*188*64
    W_conv2 = weight_variable([5, 5, 48, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_conv2 = tf.layers.batch_normalization(h_conv2)
    h_pool2 = max_pool(h_conv2, ksize=(2, 2), stride=(2, 2))

    #第三层, 13*188*64 => 7*94*128
    W_conv3 = weight_variable([5, 5, 64, 128])
    b_conv3 = bias_variable([128])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
#    h_conv3 = tf.layers.batch_normalization(h_conv3)
    h_pool3 = max_pool(h_conv3, ksize=(2, 2), stride=(2, 2))

    W_conv4 = weight_variable([5, 5, 128, 256])
    b_conv4 = bias_variable([256])
    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
#   h_conv4 = tf.layers.batch_normalization(h_conv4)
    h_pool4 = max_pool(h_conv4, ksize=(2, 2), stride=(2, 2))

    W_conv5 = weight_variable([5, 5, 256, 512])
    b_conv5 = bias_variable([512])
    h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
#    h_conv5 = tf.layers.batch_normalization(h_conv5)
    h_pool5 = max_pool(h_conv5, ksize=(2, 2), stride=(2, 2))


#    全连接

    W_fc1 = weight_variable([2 * 8 * 512, OUTPUT_SHAPE[1]])
    b_fc1 = bias_variable([OUTPUT_SHAPE[1]])

    conv_layer_flat = tf.reshape(h_pool5, [-1, 2 * 8 * 512])

    features = tf.nn.relu(tf.matmul(conv_layer_flat, W_fc1) + b_fc1)



    #（batchsize,256）
    shape = tf.shape(features)
    features = tf.reshape(features, [shape[0], OUTPUT_SHAPE[1], 1])  # batchsize * outputshape * 1

    return inputs, features, train_seq_len


def LSTM_layers(inputs, seq_len):


    with tf.variable_scope(None, default_name="bidirectional-rnn-1"):
        # Forward
        lstm_fw_cell_1 = rnn.BasicLSTMCell(256)
        # Backward
        lstm_bw_cell_1 = rnn.BasicLSTMCell(256)

        inter_output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_1, lstm_bw_cell_1, inputs, seq_len, dtype=tf.float32)

        inter_output = tf.concat(inter_output, 2)

    with tf.variable_scope(None, default_name="bidirectional-rnn-2"):
        # Forward
        lstm_fw_cell_2 = rnn.BasicLSTMCell(256)
        # Backward
        lstm_bw_cell_2 = rnn.BasicLSTMCell(256)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_2, lstm_bw_cell_2, inter_output, seq_len, dtype=tf.float32)

        outputs = tf.concat(outputs, 2)

    return outputs

def get_train_model():
    global_step = tf.Variable(0, trainable=False)

    inputs, features, train_seq_len = convolutional_layers()

    crnn_model = LSTM_layers(features, train_seq_len)

    logits = tf.reshape(crnn_model, [-1, 512])


    W = tf.Variable(tf.truncated_normal([512,num_classes],stddev=0.1),name="W",dtype=tf.float32)
    b = tf.Variable(tf.constant(0., shape=[num_classes]),name="b", dtype=tf.float32)

    logits = tf.matmul(logits,W)+b
    logits = tf.reshape(logits,[BATCH_SIZE, -1, num_classes])

    logits = tf.transpose(logits, (1, 0, 2))

    targets = tf.sparse_placeholder(tf.int32, name='targets')

    loss = tf.nn.ctc_loss(targets, logits, train_seq_len)

    cost = tf.reduce_mean(loss)

    optimizer = tf.train.AdamOptimizer(learning_rate = INITIAL_LEARNING_RATE).minimize(cost, global_step = global_step)

    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, train_seq_len)
#    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, train_seq_len)
    dense_decoded = tf.sparse_tensor_to_dense(decoded[0], default_value=-1)

    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

    init = tf.global_variables_initializer()


    return inputs, targets, train_seq_len, logits, decoded, optimizer, acc,cost, init, global_step



#
#     # inputs = tf.placeholder(tf.float32, [None, None, OUTPUT_SHAPE[0]])\
#     inputs, features, seq_len, targets = convolutional_layers() # IN THIS NETWORK, H_POOL3 IS INPUT
#     seq_len = np.int32(seq_len)
#     seq_len = (tf.convert_to_tensor(seq_len))
#     #targets = tf.convert_to_tensor(targets)
#
#     #定义ctc_loss需要的稀疏矩阵
#     targets = tf.sparse_placeholder(tf.int32)
#
#     #1维向量 序列长度 [batch_size,]
# #    seq_len = tf.placeholder(tf.int32, [None])
#
#     #定义LSTM网络
#     cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
#     stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
#     outputs, _ = tf.nn.dynamic_rnn(cell, features, seq_len, dtype=tf.float32)
#     # shape = tf.shape(inputs)
#     shape = tf.shape(features)
#     batch_s, max_timesteps = shape[0], shape[1]
#     outputs = tf.reshape(outputs, [-1, num_hidden])
#     W = tf.Variable(tf.truncated_normal([num_hidden,
#                                           num_classes],
#                                          stddev=0.1), name="W")
#     b = tf.Variable(tf.constant(0., shape=[num_classes]), name="b")
#     logits = tf.matmul(outputs, W) + b
#     logits = tf.reshape(logits, [batch_s, -1, num_classes])
#     logits = tf.transpose(logits, (1, 0, 2))
#
#     return logits, inputs, targets, seq_len, W, b

def train():
    with tf.Session() as session:
        tensorboard_dir = 'tensorboard'
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        writer = tf.summary.FileWriter(tensorboard_dir)


        inputs, targets, train_seq_len, logits, decoded, optimizer, acc,cost, init, global_step = get_train_model()
        session.run(init)
        START = 0


        learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                                    global_step,
                                                    DECAY_STEPS,
                                                    LEARNING_RATE_DECAY_FACTOR,
                                                    staircase=True)
        for curr_epoch in range(1):
            train_cost = 0
            START = 0
            print("EPOCH", curr_epoch)

            for i in range(BATCHES):
                print("BATCH", i)
                start = time.time()
                train_inputs, train_targets, train_seq_len = get_input_label(BATCH_SIZE, START, label_origin)
                feed = {targets: train_targets}
                [c, accuracy, original_list, steps, _] = session.run([cost, acc, targets, global_step, optimizer],feed)
                if steps % REPORT_STEPS == 0:
                    decoded_list = session.run(decoded[0])
                    decoded_list = decode_sparse_tensor(decoded_list)
                    original_list = decode_sparse_tensor(original_list)
                    print(c,accuracy)
                    print(decoded_list)
                    print(original_list)
                print("STEPS:", steps)

                START = START + BATCH_SIZE
                # train_cost += c * BATCH_SIZE
                seconds = time.time() - start

                print( "batch seconds:", seconds)

        writer.add_graph(session.graph)


    # logits, inputs, targets, seq_len, W, b = get_train_model()

    # loss = tf.nn.ctc_loss(labels=targets,inputs=logits, sequence_length=seq_len)
    # cost = tf.reduce_mean(loss)
    #
    # #optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=MOMENTUM).minimize(cost, global_step=global_step)
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step)
    # # decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
    # decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len, merge_repeated = False)
    # acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))
    #
    # init = tf.global_variables_initializer()

    # def do_report():
    #     test_inputs,test_targets,test_seq_len = get_input_label(BATCH_SIZE, START, label_origin)
    #     print("DIFFICULT?")
    #     test_feed = {
    #                  targets: test_targets
    #                  }
    #
    #     dd, log_probs, accuracy = session.run([decoded[0], log_prob, acc], test_feed)
    #     print("accuracy",accuracy)
    #     print("LOG", log_probs)
    #     report_accuracy(dd, test_targets)
    #     # decoded_list = decode_sparse_tensor(dd)
    #
    # def do_batch():
    #     train_inputs, train_targets, train_seq_len = get_input_label(BATCH_SIZE, START, label_origin)
    #     feed = {targets: train_targets}
    #     b_loss,b_targets, b_logits, b_seq_len,b_cost, steps, _ = session.run([loss, targets, logits, seq_len, cost, global_step, optimizer], feed)
    #
    #     #print b_loss
    #     #print b_targets, b_logits, b_seq_len
    #     print(b_cost, steps)
    #
    #
    #     if steps > 0 and steps % REPORT_STEPS == 0:
    #         do_report()
    #         print("DO REPORT")
    #         #save_path = saver.save(session, "ocr.model", global_step=steps)
    #         # print(save_path)
    #
    #     return b_cost, steps
    #
    # with tf.Session() as session:
    #
    #     session.run(init)
    #     saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
    #     for curr_epoch in range(num_epochs):
    #
    #         print("Epoch.......", curr_epoch)
    #         train_cost = train_ler = 0
    #         START = 0
    #         for batch in range(1):
    #             print(batch)
    #             start = time.time()
    #             c, steps = do_batch()
    #
    #             START = START + BATCH_SIZE
    #             train_cost += c * BATCH_SIZE
    #             seconds = time.time() - start
    #             print("Step:", steps, ", batch seconds:", seconds)
    #         # train_cost /= TRAIN_SIZE
            #
            # train_inputs, train_targets, train_seq_len = get_input_label(BATCH_SIZE, START, label_origin)
            #
            #
            #
            # val_feed = {
            #             targets: train_targets
            #             }
            #
            # val_cost, val_ler, lr, steps = session.run([cost, acc, learning_rate, global_step], feed_dict=val_feed)
            #
            # log = "Epoch {}/{}, steps = {}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}s, learning_rate = {}"
            # print(log.format(curr_epoch + 1, num_epochs, steps, train_cost, train_ler, val_cost, val_ler, time.time() - start, lr))

if __name__ == '__main__':
    train()
