
'''
train text independent speaker recognition convolutional neural network.
save model and parameters.
'''

import os
import numpy as np
import pickle

# Keras
import keras
from keras.utils import to_categorical

# Local
from model_builder import ModelBuilder
from reports import plot_history
from train_funcs import data_to_keras, get_architecture, data_reshape, fit_model

# Import constants
from config import activation, optimizer, dropout_rate, epochs, batch_size

# Paths
from config import data_filename, model_dir, data_filename, data_dir, reports_dir

def main():
	'''Main function.'''

	# Parse arguments from command line
	#args = get_script_arguments()

	# Verify data file exist
	assert os.path.exists(data_filename), 'Data does not exist.'

	# Load input data
	print('Loading the inputs in memory. It might take a while...')
	data = pickle.load(open(data_filename, 'rb'))

	# Get train and test data
	kx_train, ky_train, kx_test, ky_test, categorical_speakers = data_to_keras(data)

	# Reshape data
	# Convolutional models require data reshaping
	# if args.cnn_1 or args.cnn_2
	if True or False:
		kx_train, ky_train, kx_test, ky_test = data_reshape(kx_train, ky_train, kx_test, ky_test)

	############################
	#           MODEL          #
	############################

	# Instantiate ModelBuilder(input_shape, num_categories)
	builder = ModelBuilder(kx_train.shape, ky_train.shape[1])

	# Create model object.
	model = builder(activation=activation, 
	                optimizer=optimizer, 
	                dropout_rate=dropout_rate, 
	                architecture=get_architecture(True, False), 
	                batch_size=batch_size)

	# Train model
	model, history = fit_model(model, kx_train, ky_train, kx_test, ky_test, max_epochs=60)

	# plot history
	plot_history(history)

	# Save the trained model and its weights
	builder.save(model)


if __name__ == '__main__':
    #logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()

'''
DEPRECATED SNIPPET

if not args.baseline and not args.cnn_1 and not args.cnn_2:
    print('Please provide at least --baseline, --cnn_1 or cnn_2')
    exit(1)
'''

