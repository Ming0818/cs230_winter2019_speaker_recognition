'''
stores application wide configurations. 
'''

import os

#############
#	PATHS	#
#############

# application root directory (relative to the /code dir) 
project_root = os.path.pardir

# resources dir
res_dir = os.path.join(project_root, "outputs")

# Raw dataset root dir
#raw_data_dir = os.path.join(res_dir, "")
#raw_data_dir = os.path.join(res_dir, "")


# dir where h5 file created from converting wav files
data_dir = os.path.join(res_dir, "data")
# data_filename = '/tmp/speaker-change-detection-data.pkl'
#data_filename = os.path.expanduser(args.data_filename)

data_filename = '/home/ubuntu/deep-speaker-data/cache/full_inputs.pkl'

# name of the h5 data file
#data_file = os.path.join(data_dir, "voxforge.h5")
#data_file = os.path.join(data_dir, "voxforge_mini.h5") 

# label mapping file
#labels_file = os.path.join(data_dir, "labels.json")

# dir where model and parameters is stored
model_dir = os.path.join(res_dir, "model")

# dir where results are stored
reports_dir = os.path.join(model_dir, "reports")

# quick test dir
quick_test_dir = os.path.join(res_dir, "testing")

#############
# Constants #
#############

# MFCC max_pad length
max_pad_len = 196

# Default activation: valid set: 'relu' and 'tanh'
activation = 'relu'

# Default optimizer: valid set: 'adadelta', 'adam' and 'rmsprop'
optimizer = 'adam'

# Default model architecture: valid set: 'baseline', 'cnn-1' and 'cnn-2'
architecture = 'baseline'

# Default Number of training epochs
epochs = 32

# Default batch size
batch_size = 512

# Default dropout rate
dropout_rate = 0.2

# lamda value of the regulizer. set it to zero for no regularizarion.
lambda_regularizer = 0.005

# Sample rate from phillip remy
sample_rate = 8000
