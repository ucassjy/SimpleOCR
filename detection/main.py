import os
import sys
import tensorflow as tf

from data_loader.data_generator import GetBlobs
from utils.dirs import create_dirs
#from models.example_model import ExampleModel
#from trainers.example_trainer import ExampleTrainer
#from utils.config import process_config
#from utils.logger import Logger
#from utils.utils import get_args


def main():
    arg = sys.argv

    # train the network
    if 'train' in arg:
        # create the needed dirs
        create_dirs(['output/'])
        # get blobs
        blobs = GetBlobs('../image_1000/')
        print(blobs[0])
        print(arg)

    # import the pretrained network and pre
    else :
        print('For test')

    # create tensorflow session
#    sess = tf.Session()
    # create an instance of the model you want
#    model = ExampleModel(config)
    #load model if exists
#    model.load(sess)
    # create your data generator
#    data = DataGenerator(config)
    # create tensorboard logger
#    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
#    trainer = ExampleTrainer(sess, model, data, config, logger)

    # here you train your model
#    trainer.train()


if __name__ == '__main__':
    main()
