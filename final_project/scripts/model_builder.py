'''
Wrapper class around speaker regocnition cnn model.
provides utilities to save and load trained models.
'''

import os

# KERAS
import keras
from keras import Input, Model
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Lambda
from keras.layers.normalization import BatchNormalization
from keras import regularizers

# Constants
from config import activation, optimizer, dropout_rate, batch_size, architecture
# Paths
from config import data_filename, model_dir, data_filename, data_dir, reports_dir


class ModelBuilder:
    ''' 
    Builds a convolutional nueral network model. 
    '''

    def __init__(self, input_shape, num_categories):
        ''' 
        CONSTRUCTOR

        Attributes:
            input_shape(tuple): shape of X.  This is required for the input layer definition.
            num_categories(int): number of unique softmax output values expected. 
            In this case, number of unique speakers in the dataset.
            activation(string): activation function to use for all layers except the output layer.
                valid values are: 'relu' and 'tanh'
            optimizer(string): optimizer to use for gradient descent.
                valid values are: 'rmsprop', 'adam' and 'adadelta'
            dropout(float): dropout rate.
            architecture(string): Model arquitecture.
                valid values are: 'baseline', 'cnn-1' and 'cnn-2'
            batch_size(int):



        The default values are read from config.
        '''
        self.input_shape = input_shape
        self.num_categories = num_categories
        self.activation = activation
        self.optimizer = optimizer
        self.dropout_rate = dropout_rate
        self.architecture = architecture
        self.batch_size = batch_size
 
    def __call__(self, activation=activation, optimizer=optimizer, dropout_rate=dropout_rate, architecture=architecture, batch_size=batch_size):
        ''' 
        Default method on this object. 
        It was designed to support KerasClassifier definition.
        '''

        self.activation = activation
        self.optimizer = optimizer
        self.dropout_rate = dropout_rate
        self.architecture = architecture
        self.batch_size = batch_size

        # Select architecture.
        # Instantiate model with specific architecture.
        if self.architecture == 'cnn_1':
            model = self.cnn_model_1(batch_size)
        elif self.architecture == 'cnn_2':
            model = self.cnn_model_2(batch_size)
        else:
            model = self.baseline_model(batch_size, normalize_embeddings=False)

        # Compile model
        model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=self.optimizer_instance(),
              metrics=['acc'])

        return model


    def optimizer_instance(self):
        '''
        Select optimizing strategy.

        args:
            self: Model builder object.

        return:
            opt(obj): Optimizing function object.

        '''

        opt = None
        if self.optimizer == 'rmsprop':
            opt = keras.optimizers.rmsprop()
        elif self.optimizer == 'adam':
            opt = keras.optimizers.Adam()
        elif self.optimizer == 'adadelta':
            opt = keras.optimizers.Adadelta()
        return opt

    def save(self, model):
        ''' 
        Serialize and save model and parameters.
        
        args:
            self
            model(obj): Trained model.
        ''' 

        # Build file name for specific architecture.
        filename = "model_" + str(self.architecture) + ".json"
        model_file = os.path.join(model_dir, filename)

        # Veify directory for saving model and weights exist.
        # If not. Create it.
        if not os.path.exists(os.path.dirname(model_file)):
            os.makedirs(os.path.dirname(model_file))
            
        # Save model in JSON
        model_json = model.to_json()
        with open(model_file, "w") as json_file:
            json_file.write(model_json)

        # model weights - parameters (h5 format)
        filename_weights = 'model_' + str(self.architecture)  + '.h5'
        model_params = os.path.join(model_dir, filename_weights)
        
        # serialize weights to HDF5
        model.save_weights(model_params)
        print("Saved model to disk")

    def load(self):
        '''
        Load model and weights for specific architecture.

        args:
            self

        return:
            model(obj): Model loaded with stored weights.
        '''
        
        # Build file name for specific architecture.
        filename = "model_" + str(self.architecture) + ".json"
        model_file = os.path.join(model_dir, filename)

        # Load json and create model
        json_file = open(model_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        # Create model. Load from JSON.
        model = model_from_json(loaded_model_json)

        # model weights - parameters (h5 format)
        filename_weights = 'model_' + str(self.architecture)  + '.h5'
        model_params = os.path.join(model_dir, filename_weights)
        
        # Load weights into new model
        model.load_weights(model_params)
        print("Loaded model from disk")

        # Compile Model
        model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=self.optimizer_instance(),
              metrics=['acc'])

        return model

    def cnn_model_1(self, batch_size):
        '''
        Convolutional Neural Network 1
        manishpandit
        https://github.com/manishpandit/speaker-recognition

        Build CNN-1. This model has the following structure:

        Layers:
            0. INPUT (900, 390, 1) 

            1. CONV-1D (filters = 16, kernel = 3)
            2. MAXPOOL-1D (pool_size = 2, stride = 2) 
            3. Batch Normalization

            4. CONV-1D (filters = 32, kernel = 3)
            5. MAXPOOL-1D (pool_size = 2, stride = 2) 
            6. Batch Normalization

            7. CONV-1D (filters = 64, kernel = 3)
            8. MAXPOOL-1D (pool_size = 2, stride = 2) 
            9. Batch Normalization

            10. Flatten
            11. Dropout (drop_out_rate = .5)

            12. DENSE (units = num_categories * 2) 
            13. Batch Normalization
            14. Dropout (drop_out_rate = .5)

            15. SOFTMAX (units = num_speakers(105)) 

        args:
            num_categories(int): Number of speakers in dataset.
            batch_size(int): Training batch size. (DEFAULT = 900)

        return:
            Model(obj): CNN-1 model object.

        '''

        #############################
        #           INPUT           #
        #############################

        inp = Input(batch_shape=[batch_size, 39 * 10,1])

        ########################
        #       LAYER 1        #
        ########################

        # Convolution layer: 3 filter, 16 filters
        conv_1 = Conv1D(16, kernel_size=(3),  padding='same', 
                            activation=activation,
                            input_shape=self.input_shape)(inp)

        ########################
        #       LAYER 2        #
        ########################

        # Max pool layer (filter = 2)
        max_pool_1 = MaxPooling1D(pool_size=(2), strides=2)(conv_1)

        ########################
        #       LAYER 3        #
        ########################

        # Batch Normalization
        batch_norm_1 = BatchNormalization()(max_pool_1)

        ########################
        #       LAYER 4        #
        ########################

        # Convolution layer: 3 filter, 32 filters
        conv_2 = Conv1D(32, kernel_size=(3),  padding='same', 
                activation=activation)(batch_norm_1)

        ########################
        #       LAYER 5        #
        ########################

        # Max pool layer 2
        max_pool_2 = MaxPooling1D(pool_size=(2), strides=2)(conv_2)

        ########################
        #       LAYER 6        #
        ########################

        # Batch norm
        batch_norm_2 = BatchNormalization()(max_pool_2)

        ########################
        #       LAYER 7        #
        ########################

        # Convolution layer: 3 filter, 64 filters
        conv_3 = Conv1D(64, kernel_size=(3),  padding='same', 
                activation=activation)(batch_norm_2)

        ########################
        #       LAYER 8        #
        ########################

        # Max pool later 2
        max_pool_3 = MaxPooling1D(pool_size=(2), strides=2)(conv_3)

        ########################
        #       LAYER 9        #
        ########################

        # Batch norm
        batch_norm_3 = BatchNormalization()(max_pool_3)

        ########################
        #       LAYER 10       #
        ########################

        # Flatten
        flatten_4 = Flatten()(batch_norm_3)

        ########################
        #       LAYER 11       #
        ########################

        # Dropout for regularization
        drop_out_1 = Dropout(dropout_rate)(flatten_4)

        ########################
        #       LAYER 12       #
        ########################

        # Dense layer
        dense_5 = Dense(self.num_categories * 2, 
                        activation=activation,
                        kernel_regularizer=regularizers.l2(0.001))(drop_out_1)

        ########################
        #       LAYER 13       #
        ########################

        # Batch norm
        batch_norm_5 = BatchNormalization()(dense_5)

        ########################
        #       LAYER 14       #
        ########################

        # Dropout for regularization
        drop_out_2 = Dropout(dropout_rate)(batch_norm_5)

        ########################
        #       LAYER 15       #
        ########################

        # Densely-connected NN layer.
        softmax = Dense(self.num_categories, activation='softmax', name='softmax')(drop_out_2)

        return Model(inputs=[inp], outputs=[softmax])

    def cnn_model_2(self, batch_size):
        '''
        Convolutional Neural Network 2
        manishpandit
        https://github.com/manishpandit/speaker-recognition

        Build CNN-2. This model has the following structure:

        Layers:
            0. INPUT (900, 390, 1) 

            1. CONV-1D (filters = 32, kernel = 3)
            2. MAXPOOL-1D (pool_size = 2, stride = 2) 
            3. Batch Normalization

            4. CONV-1D (filters = 64, kernel = 3)
            5. MAXPOOL-1D (pool_size = 2, stride = 2) 
            6. Batch Normalization

            9. DENSE (units = num_categories * 5) 
            8. Dropout (drop_out_rate = .25)
            9. Flatten

            10. DENSE (units = num_categories * 10) 
            11. Batch Normalization

            12. SOFTMAX (num_categories = num_speakers(105)) 


        args:
            num_categories(int): Number of speakers in dataset.
            batch_size(int): Training batch size. (DEFAULT = 900)

        return:
            Model(obj): CNN-2 model object.

        '''

        dropout_rate = .25

        #############################
        #           INPUT           #
        #############################

        inp = Input(batch_shape=[batch_size, 39 * 10,1])

        ########################
        #       LAYER 1        #
        ########################

        # Convolution layer: 32 filters
        conv_1 = Conv1D(32, kernel_size=(3),  padding='same', 
                            activation=activation,
                            input_shape=self.input_shape)(inp)

        ########################
        #       LAYER 2        #
        ########################

        # Max pool later 2, 2 
        max_pool_2 = MaxPooling1D(pool_size=(2), strides=2)(conv_1)

        ########################
        #       LAYER 3        #
        ########################

        # Batch norm
        batch_norm_2 = BatchNormalization()(max_pool_2)

        ########################
        #       LAYER 4        #
        ########################

        # Convolution layer: 32 filters
        conv_3 = Conv1D(64, kernel_size=(3),  padding='same', 
                            activation=activation,
                            input_shape=self.input_shape)(batch_norm_2)

        ########################
        #       LAYER 5        #
        ########################

        # Max pool later 4, 4
        max_pool_4 = MaxPooling1D(pool_size=(2), strides=2)(conv_3)

        ########################
        #       LAYER 6        #
        ########################

        # Batch norm
        batch_norm_4 = BatchNormalization()(max_pool_4)


        ########################
        #       LAYER 7        #
        ########################

        # Dense layer
        dense_5 = Dense(self.num_categories * 10, 
                        activation=activation,
                        kernel_regularizer=regularizers.l2(0.01))(batch_norm_4)

        ########################
        #       LAYER 8        #
        ########################

        # Dropout for regularization
        drop_out_6 = Dropout(dropout_rate)(dense_5)

        ########################
        #       LAYER 9        #
        ########################

        # Flatten
        flatten_6 = Flatten()(drop_out_6)

        ########################
        #       LAYER 10       #
        ########################

        # Dense layer
        dense_7 = Dense(self.num_categories * 5, 
                        activation=activation,
                        kernel_regularizer=regularizers.l2(0.01))(flatten_6)

        ########################
        #       LAYER 11        #
        ########################

        # Batch norm
        batch_norm_7 = BatchNormalization()(dense_7)

        ########################
        #       LAYER 12        #
        ########################

        # Densely-connected NN layer.
        softmax = Dense(self.num_categories, activation='softmax', name='softmax')(batch_norm_7)

        return Model(inputs=[inp], outputs=[softmax])

    def baseline_model(self, batch_size=batch_size, normalize_embeddings=False):
        '''
        Base line model (Fully connected network)
        Phillip Remy
        https://github.com/philipperemy/deep-speaker

        Build baseline model. This model has the following structure:
        Layers
            1. INPUT (900,390) 
            2. DENSE (units = 200)
            3. L2 Normalization
            4. SOFTMAX (units = num_speakers)

        args:
            num_categories(int): Number of speakers in dataset.
            batch_size(int): Number elements to be proccessed in batch.
            normalize_embeddings(bool): Flag to indicate whether embeddings will be normalized.

        return:
            Model(obj): Baseline model object.

        '''

        ############################
        #           INPUT          #
        ############################

        inp = Input(batch_shape=[batch_size, 39 * 10])

        ########################
        #       LAYER 1        #
        ########################

        embeddings = Dense(200, activation='sigmoid', name='fc1')(inp)

        ############################
        #   Normalize embeddings   #
        ############################

        if normalize_embeddings:
            print('Embeddings will be normalized.')
            embeddings = Lambda(lambda y: K.l2_normalize(y, axis=1), name='normalization')(embeddings)

        # Rename embeddings
        # just a trick to name a layer after if-else.
        embeddings = Lambda(lambda y: y, name='embeddings')(embeddings)  

        ########################
        #       LAYER 2        #
        ########################
        
        softmax = Dense(self.num_categories, activation='softmax', name='softmax')(embeddings)

        return Model(inputs=[inp], outputs=[softmax])

