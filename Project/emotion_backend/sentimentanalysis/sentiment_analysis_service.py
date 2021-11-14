#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import librosa as lr
from keras.utils import Sequence, to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from keras.layers import Convolution2D 
from keras.layers import GlobalAveragePooling2D 
from keras.layers import BatchNormalization 
from keras.layers import Flatten
from keras.layers import GlobalMaxPool2D
from keras.layers import MaxPool2D
from keras.layers import concatenate
from keras.layers import Activation
from keras.layers import Input
from keras.layers import Dense
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras.activations import relu, softmax
from keras import losses, models, optimizers
from keras import backend as K
from keras.models import Model
from keras.models import model_from_json
import pickle
import gc


# Initialize Keras seed for consistent results

def init_keras_seed():
    import tensorflow as tf
    import random as rn

    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.

    np.random.seed(42)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.

    rn.seed(12345)

    # Force TensorFlow to use single thread.
    # Multiple threads are a potential source of non-reproducible results.
    # For further details, see: https://stackoverflow.com/questions/42022950/

    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)


    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see:
    # https://www.tensorflow.org/api_docs/python/tf/set_random_seed

    tf.set_random_seed(1234)

    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)


# Define System Configuration:

class Config(object):
    def __init__(self,
                 sampling_rate=16000, audio_duration=2, n_classes=8,
                 use_mfcc=False, n_folds=10, learning_rate=0.0001, 
                 max_epochs=100, n_mfcc=20):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.n_classes = n_classes
        self.use_mfcc = use_mfcc
        self.n_mfcc = n_mfcc
        self.n_folds = n_folds
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        self.audio_length = self.sampling_rate * self.audio_duration
        if self.use_mfcc:
            self.dim = (self.n_mfcc, 1 + int(np.floor(self.audio_length/512)), 1)
        else:
            self.dim = (self.audio_length, 1)


# Sentiment Analysis Service Class:
class SentimentAnalysisService:

    def __init__(self):

        init_keras_seed()
        print('hello 2')
        self.config = Config(sampling_rate=44100, audio_duration=3, n_folds=10,
                learning_rate=0.01, use_mfcc=True, n_mfcc=40)
        print('hello 2.5')
        print(self.config.dim[0])
        print(self.config.dim[1])
# In[9]:

    def prepare_data(self, fileList):
        print('hello 3')
        print(fileList)
        print(self.config.dim[0])
        print(self.config.dim[1])
        X = np.empty(shape=(len(fileList), self.config.dim[0], self.config.dim[1], 1))
        input_length = self.config.audio_length
        for i, fname in enumerate(fileList):
            print(fname)
            file_path = fname
            data, _ = lr.core.load(file_path, sr=self.config.sampling_rate, res_type="kaiser_fast", offset=0.5)

            # Random offset / Padding
            if len(data) > input_length:
                max_offset = len(data) - input_length
                offset = np.random.randint(max_offset)
                data = data[offset:(input_length+offset)]
            else:
                if input_length > len(data):
                    max_offset = input_length - len(data)
                    offset = np.random.randint(max_offset)
                else:
                    offset = 0
                data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

            data = lr.feature.mfcc(data, sr=self.config.sampling_rate, n_mfcc=self.config.n_mfcc)
            data = np.expand_dims(data, axis=-1)
            X[i,] = data
        return X


# In[10]:


# MakeFileList
#fileList = []

#Path of wavefile to be tested , one example here 
#fileName = '/home/ubuntu/ml-emotion-detection/test_data/03-01-01-01-01-01-16.wav'
#fileName = '/home/ubuntu/ml-emotion-detection/Audio_Speech_Actors_01-24/Actor_15/03-01-04-02-02-01-15.wav'
#fileList = []
#for root, dirs, files in os.walk("/home/ubuntu/ml-emotion-detection/test_data"):
    #for file in files:
        #if file.endswith(".wav"):
             #fileList.append(os.path.join(root, file))
#print(len(fileList))
#fileList.append(fileName)
#print(len(fileList))


# In[11]:

    def predict_result(self, fileList):

        #Prepare MFCC data from fileList
        print('hello 1')
        X = self.prepare_data(fileList)
        print(X.shape)

        # read  STD and MEAN FOR DATA NORMALIZATION
        mean = pickle.load( open( "mean.pkl", "rb" ) )
        std = pickle.load( open( "std.pkl", "rb" ) )

        X = (X - mean)/std


        # load json and create model
        json_file = open('/Users/adityaasthana/cs196/sentimentanalysis/model_data/model_final.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights('/Users/adityaasthana/cs196/sentimentanalysis/model_data/weights.best.from_scratch.hdf5')
        print("Loaded model from disk")

        #compile model
        opt = optimizers.Adam(self.config.learning_rate)
        loaded_model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])

        #Print Model Summary
        print(loaded_model.summary())

        # In[13]:


        #Now predict
        predicted_vector = loaded_model.predict(X, batch_size=64, verbose=1)
        print(predicted_vector)
        print(predicted_vector.shape)
        result = np.argmax(predicted_vector)
        print(result)


        # In[14]:


        #for i in range(len(fileList)):
            #print(fileList[i])
            #print(predicted_vector[i])
            #print(np.argmax(predicted_vector[i], axis=-1) + 1)


        # In[ ]:
        del loaded_model
        gc.collect()
        K.tensorflow_backend.clear_session()
        return result

    def __del__(self):
        print("GARBAGE COLLECTED")




