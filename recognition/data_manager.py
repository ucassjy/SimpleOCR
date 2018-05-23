import re
import os
import numpy as np
import cv2

from utils import sparse_tuple_from, resize_image, label_to_array, read_dictionary

class DataManager(object):
    def __init__(self, batch_size, model_path, examples_picture_path, examples_label_path, dictionary_path,max_image_width, train_test_ratio, max_char_count):
        if train_test_ratio > 1.0 or train_test_ratio < 0:
            raise Exception('Incoherent ratio!')

        print(train_test_ratio)

        self.train_test_ratio = train_test_ratio
        self.max_image_width = max_image_width
        self.batch_size = batch_size
        self.model_path = model_path
        self.current_train_offset = 0

        self.examples_picture_path = examples_picture_path
        self.examples_label_path = examples_label_path
        self.max_char_count = max_char_count
        self.dictionary_path = dictionary_path
        self.data, self.data_len, self.NUM_CLASSES = self.__load_data()
        self.test_offset = int(train_test_ratio * self.data_len)
        self.current_test_offset = self.test_offset
        self.train_batches = self.__generate_all_train_batches()
        self.test_batches = self.__generate_all_test_batches()

    def __load_data(self):
        """
            Load all the images in the folder
        """

        print('Loading data')

        examples = []
        count = 0
        skipped = 0
        # for f in os.listdir(self.examples_picture_path):
        #     if len(f.split('_')[0]) > self.max_char_count:
        #         continue
        #     arr, initial_len = resize_image(
        #         os.path.join(self.examples_path, f),
        #         self.max_image_width
        #     )
        with open(self.examples_label_path,'r') as f: # Address of target_label.txt
            for line in f.readlines():
                address = line.split("__")[0]

                label = line.split("__")[1]
                if len(label) > self.max_char_count:
                    continue
                if list(label)[0]=='#':
                    continue
                img = cv2.imread(address, cv2.IMREAD_GRAYSCALE)
                arr, initial_len = resize_image(img, self.max_image_width)
                dictionary,_, dictionary_len = read_dictionary(self.dictionary_path)

                examples.append(
                    (
                        arr,
                        label,
                        label_to_array(label, dictionary)
                    )
                )
                count += 1
                dictionary_len = dictionary_len + 1 #!
        return examples, len(examples), dictionary_len



    def __generate_all_train_batches(self):
        train_batches = []
        while not self.current_train_offset + self.batch_size > self.test_offset:
            old_offset = self.current_train_offset

            new_offset = self.current_train_offset + self.batch_size

            self.current_train_offset = new_offset

            raw_batch_x, raw_batch_y, raw_batch_la = zip(*self.data[old_offset:new_offset])

            batch_y = np.reshape(
                np.array(raw_batch_y),
                (-1)
            )

            batch_dt = sparse_tuple_from(
                np.reshape(
                    np.array(raw_batch_la),
                    (-1)
                )
            )

            batch_x = np.reshape(
                np.array(raw_batch_x),
                (-1, self.max_image_width, 32, 1)
            )

            train_batches.append((batch_y, batch_dt, batch_x))
        return train_batches

    def __generate_all_test_batches(self):
        test_batches = []
        while not self.current_test_offset + self.batch_size > self.data_len:
            old_offset = self.current_test_offset

            new_offset = self.current_test_offset + self.batch_size

            self.current_test_offset = new_offset

            raw_batch_x, raw_batch_y, raw_batch_la = zip(*self.data[old_offset:new_offset])

            batch_y = np.reshape(
                np.array(raw_batch_y),
                (-1)
            )

            batch_dt = sparse_tuple_from(
                np.reshape(
                    np.array(raw_batch_la),
                    (-1)
                )
            )

            batch_x = np.reshape(
                np.array(raw_batch_x),
                (-1, self.max_image_width, 32, 1)
            )

            test_batches.append((batch_y, batch_dt, batch_x))
        return test_batches
