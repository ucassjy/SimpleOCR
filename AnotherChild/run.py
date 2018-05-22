from crnn import CRNN
import os

batch_size = 50
model_path = 'MyModel'
examples_picture_path = 'restore/'
examples_label_path = 'target_label.txt'
dictionary_path = 'dictionary.txt'
max_image_width = 256
train_test_ratio = 0.5
restore = False
NUM_CLASSES = 240
iteration_count = 100

crnn = CRNN(batch_size, model_path, examples_picture_path, examples_label_path, dictionary_path, max_image_width, train_test_ratio, restore, NUM_CLASSES)

if __name__ == '__main__':
    crnn.train(iteration_count)
