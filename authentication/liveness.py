import tensorflow as tf


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

import pickle


def loadLiveness(path):
    model = tf.keras.models.load_model(path)
    return model
