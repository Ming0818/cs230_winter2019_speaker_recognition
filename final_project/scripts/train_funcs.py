'''
Training functions

This script includes all the functions requiered to train 
the differente models.
'''

##########
# Import #
##########

import os
import numpy as np
import pickle
from glob import glob
import argparse
from argparse import ArgumentParser
from collections import deque
from natsort import natsorted

# Includes
import sys
sys.path.append('/Users/j/deep-speaker-data/venv-speaker/amendezp-cs230-winter-2018/')

# Keras
import keras
import keras.backend as K
from keras import Input, Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, Callback
from keras.layers import Dense, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical

#########
# Local #
#########

from reports import plot_history
from model_builder import ModelBuilder

# Constants
from config import activation, optimizer, dropout_rate, epochs, batch_size
from constants import c



def get_arguments(parser: ArgumentParser):
	'''
	Get arguments from command.

	'''

	args = None
	try:
	    args = parser.parse_args()
	except Exception:
	    parser.print_help()
	    exit(1)
	return args

def get_script_arguments():
	'''
	Parse arguments and add flags to parser.

	'''

	parser = argparse.ArgumentParser()

	# generated from: python cli.py --generate_training_inputs --multi_threading
	#parser.add_argument('--data_filename', type=str, required=True)

	parser.add_argument('--ccn_1', action='store_true')
	parser.add_argument('--cnn_2', action='store_true')

	#parser.add_argument('--', action='store_true')
	#parser.add_argument('--', action='store_true')

	args = get_arguments(parser)

	return args

class SpeakersToCategorical:
	def __init__(self, data):
		from keras.utils import to_categorical
		self.speaker_ids = sorted(list(data.keys()))
		self.int_speaker_ids = list(range(len(self.speaker_ids)))
		self.map_speakers_to_index = dict([(k, v) for (k, v) in zip(self.speaker_ids, self.int_speaker_ids)])
		self.map_index_to_speakers = dict([(v, k) for (k, v) in zip(self.speaker_ids, self.int_speaker_ids)])
		self.speaker_categories = to_categorical(self.int_speaker_ids, num_classes=len(self.speaker_ids))

	def get_speaker_from_index(self, index):
		return self.map_index_to_speakers[index]

	def get_one_hot_vector(self, speaker_id):
		index = self.map_speakers_to_index[speaker_id]
		return self.speaker_categories[index]

	def get_speaker_ids(self):
		return self.speaker_ids


def data_to_keras(data):
    '''
    '''

    # Create class
    categorical_speakers = SpeakersToCategorical(data)

    # Init lists
    kx_train, ky_train, kx_test, ky_test = [], [], [], []

    # Iterate over speakers
    for speaker_id in categorical_speakers.get_speaker_ids():

        # get data from speaker i.
        d = data[speaker_id]

        # get one hot encoding for speaker i
        y = categorical_speakers.get_one_hot_vector(d['speaker_id'])

        # Iterate over trainin audio samples for current speaker
        # Features: Alrededor de 1000
        for x_train_elt in data[speaker_id]['train']:

            for x_train_sub_elt in x_train_elt:

                # Cada una de las matrices de [N x 390]
                kx_train.append(x_train_sub_elt)

                # One hot encoding vector corresponding to speaker X
                ky_train.append(y)

        for x_test_elt in data[speaker_id]['test']:

            for x_test_sub_elt in x_test_elt:

                # Cada una de las matrices de [N x 390]
                kx_test.append(x_test_sub_elt)

                # One hot encoding vector corresponding to speaker X
                ky_test.append(y)

    kx_train = np.array(kx_train)
    kx_test = np.array(kx_test)

    ky_train = np.array(ky_train)
    ky_test = np.array(ky_test)

    return kx_train, ky_train, kx_test, ky_test, categorical_speakers


def data_reshape(kx_train, ky_train, kx_test, ky_test):
	''' Data reshaping for convolutional architectures. '''

	kx_train = kx_train.reshape(kx_train.shape[0], kx_train.shape[1], 1)
	kx_test = kx_test.reshape(kx_test.shape[0], kx_test.shape[1], 1)

	ky_train = ky_train.reshape(ky_train.shape[0], -1)
	ky_test = ky_test.reshape(ky_test.shape[0], -1)

	return kx_train, ky_train, kx_test, ky_test

def get_architecture(cnn_1, cnn_2):
	'''
	Get model architecture. 

	'''
	architecture = None

	if cnn_1:
		architecture = 'cnn_1'
	elif cnn_2:
		architecture = 'cnn_2'
	else:
		architecture = 'baseline'

	return architecture

def fit_model(m, kx_train, ky_train, kx_test, ky_test, batch_size=batch_size, max_epochs=100, initial_epoch=0):
	'''

	'''

	# if the accuracy does not increase by 1.0% over 10 epochs, we stop the training.
	early_stopping = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=100, verbose=2, mode='max')

	# if the accuracy does not increase over 10 epochs, we reduce the learning rate by half.
	reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, min_lr=0.0001, verbose=2)

	max_len_train = len(kx_train) - len(kx_train) % batch_size
	kx_train = kx_train[0:max_len_train]
	ky_train = ky_train[0:max_len_train]

	max_len_test = len(kx_test) - len(kx_test) % batch_size
	kx_test = kx_test[0:max_len_test]
	ky_test = ky_test[0:max_len_test]

	history = m.fit(kx_train,
					ky_train,
					batch_size=batch_size,
					epochs=initial_epoch + max_epochs,
					initial_epoch=initial_epoch,
					verbose=2,
					validation_data=(kx_test, ky_test),
					callbacks=[early_stopping, reduce_lr],
					 shuffle = True)

	return m, history


