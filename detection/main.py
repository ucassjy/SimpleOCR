import os
import sys
import tensorflow as tf

from data_loader.data_generator import GetBlobs
from utils.dirs import create_dirs
from network.network import Network
from trainers.trainer import Trainer
#from utils.logger import Logger


def main():
    arg = sys.argv

    # train the network
    if 'train' in arg:
        #  create the needed dirs
        create_dirs(['output/'])
        tensorboard_dir = 'output/'
        writer = tf.summary.FileWriter(tensorboard_dir)
        #  get blobs
        blobs = GetBlobs('../image_1000/')
        #  create tensorflow session
        sess = tf.Session()
        #  create an instance of the network
        network = Network('TRAIN', sess, blobs)
        network = network.create_architecture()
        writer.add_graph(sess.graph)
        # create trainer and pass all the previous components to it
#        trainer = Trainer(sess, network, blobs, logger)

    # import the pretrained network and show the result for one image
    else :
        print('For test')

    #load model if exists
#    model.load(sess)
    # create tensorboard logger
#    logger = Logger(sess, config)

    # here you train your model
#    trainer.train()


if __name__ == '__main__':
    main()
