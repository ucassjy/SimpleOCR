import numpy as np
import tensorflow as tf

from scipy.misc import imread, imresize, imsave

import config

def sparse_tuple_from(sequences, dtype=np.int32):
    """
        Inspired (copied) from https://github.com/igormq/ctc_tensorflow_example/blob/master/utils.py
    """

    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), [i for i in range(len(seq))]))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape

def resize_image(image, input_width):
    """
        Resize an image to the "good" input size
    """

    im_arr = image
    r, c = np.shape(im_arr)
    if c > input_width:
        c = input_width
        ratio = float(input_width / c)
        final_arr = imresize(im_arr, (int(32 * ratio), input_width))
    else:
        final_arr = np.zeros((32, input_width))
        ratio = float(32 / r)
        im_arr_resized = imresize(im_arr, (32, int(c * ratio)))
        final_arr[:, 0:np.shape(im_arr_resized)[1]] = im_arr_resized
    return final_arr, c

def label_to_array(label, dictionary):
    try:
        return [dictionary[x] for x in label if x != '\n']                        #TODO Use dictionary
    except Exception as ex:
        print(label)
        raise ex

def ground_truth_to_word(ground_truth, dictionary):
    """
        Return the word string based on the input ground_truth
    """

    try:
        word = []
        for i in ground_truth:
            if i != -1:
                word.append(dictionary[str(i)])
        return word
    except Exception as ex:
        print(ground_truth)
        print(ex)
        input()

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def read_dictionary(dictionary_path):
    with open(dictionary_path,'r') as f:
        dictionary = f.readlines()
        real_dict = {}
        inverse_dict = {}
        for line in dictionary:
            key = line.split(':')[0]
            value = line.split(':')[1]
            value = value.split('\n')[0]
            real_dict[key] = value
            inverse_dict[value] = key
        real_dict[':'] = len(real_dict)
        inverse_dict[len(inverse_dict)] = ':'
    return real_dict, inverse_dict, len(real_dict)
