import tensorflow as tf
import pickle


def loadLiveness(path):
    model = tf.keras.models.load_model(path)
    return model


def loadLabel(path):
	label = pickle.loads(open(path, 'rb').read())
	return label