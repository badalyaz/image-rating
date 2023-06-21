import random
import cv2
import os

import argparse
import tensorflow as tf
import tensorflow_hub as hub
from keras.layers import *
from keras.models import Model
from keras.applications import *
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

from PIL import Image
from glob import glob
from numpy.random import rand
from utils import *
    
def read_args():
	parser = argparse.ArgumentParser()	
	parser.add_argument('-d', '--data_path', required=True,
			             help='path of single image')
	parser.add_argument('-w', '--weights_path', required=False,
                                     help='weights path')
	parser.add_argument('-v', '--visualize', required=False)
    parser.add_argument('-xai', '')

	args = parser.parse_args()
	
	return args.data_path, args.weights_path, args.visualize

def evaluator(path, weights_path):
    model_gap = model_inceptionresnet_multigap()
    model = fc_model_softmax(input_num=9744)
    model.load_weights(weights_path)
    model_CNN = tf.keras.Sequential([hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/b7/feature-vector/1",trainable=False) ])
    
    predicted = predict_from_path(model_gap=model_gap, model=model, paths=[path], resize_func=resize_max, size=(996,996), for_all=False, save_results=None, save_to=None, model_CNN=model_CNN, PCA=True)
    return np.argmax(predicted, axis=-1)


if __name__ == '__main__':
	data_path, weights_path, visualize = read_args()
	
	if weights_path == None:
		weights_path = 'models/Softmax/MG_CNN/model_fc_softmax_MG_8k_B7_1k_600x600.hdf5'

	is_aesth = evaluator(data_path, weights_path)

	if is_aesth:
		print('Image is aesthetic')
	else:
		print('Image is NOT aesthetic')

	if visualize:
		img = Image.open(data_path)

		plt.imshow(img)
		plt.title(f'Prediction on this image: {is_aesth}')
		plt.show()
