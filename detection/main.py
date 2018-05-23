import os
import sys
import tensorflow as tf

from data.data_loader import GetBlobs
from nets.vgg16 import vgg16
from model.train_val import train_net

def main():
    args = sys.argv
    print('Called with args:')
    print(args)

    if 'train' in args:
        # train set
        blobs_all = GetBlobs('../image_1000/')
        print('{:d} images'.format(len(blobs_all)))

        # output directory where the models are saved
        output_dir = 'output/models'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print('Output will be saved to `{:s}`'.format(output_dir))

        # tensorboard directory where the summaries are saved during training
        tb_dir = 'output/tensorboard'
        if not os.path.exists(tb_dir):
            os.makedirs(tb_dir)
        print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

        # load network
        net = vgg16()

        train_net(net, blobs_all, output_dir, tb_dir)

    else :
        pre_trained = 'output/models/'
        if os.file.exists(pre_trained):
            print('For test.')
        else:
            print('No pre_trained model yet.')

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()
