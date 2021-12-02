#encoding:utf-8
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

from keras.models import Sequential
from keras.layers import Input,Dense,Conv2D,MaxPooling2D,UpSampling2D,Dropout,Flatten 
from keras.layers import BatchNormalization,AveragePooling2D  
from keras.layers import ZeroPadding2D,add
from keras.layers import Dropout, Activation
from keras.models import Model,load_model
from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard
from keras import optimizers, regularizers 
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
import cv2, numpy as np
import math
import numpy, scipy
from scipy import interpolate
import scipy.ndimage
import time
import tensorflow as tf
import pdb
from vgg16 import *


def get_image_descriptor_for_image(image, model):
    im = cv2.resize(image, (224, 224)).astype(np.float32)
    #dim_ordering = K.image_dim_ordering()
    dim_ordering = 'th'
    if dim_ordering == 'th':
        # 'RGB'->'BGR'
        im = im[::-1, :, :]

    else:
        # 'RGB'->'BGR'
        im = im[:, :, ::-1]

    im = np.expand_dims(im, axis=0)
    with tf.Session(graph=g2) as sess:
    	inputs = [K.learning_phase()] + model.inputs
    	_convout1_f = K.function(inputs, [model.layers[20].output])
    return _convout1_f([0] + [im])



def get_loss1(image, model):
    im = cv2.resize(image, (224, 224)).astype(np.float32)
    #dim_ordering = K.image_dim_ordering()
    dim_ordering = 'th'
    if dim_ordering == 'th':
        # 'RGB'->'BGR'
        im = im[::-1, :, :]
        # Zero-center by mean pixel

    else:
        # 'RGB'->'BGR'
        im = im[:, :, ::-1]


    im = np.expand_dims(im, axis=0)

    with tf.Session(graph=g2) as sess1:
        with g2.as_default():
            vgg_params =tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='myvgg')
            sess1.run(tf.global_variables_initializer())
            model.load_weights('../fine-tune207/4030.h5')
            inputs = [K.learning_phase()] + model.inputs
            _convout1_f = K.function(inputs, [model.layers[23].output])
            score =  _convout1_f([0] + [im])[0]
    if score.max() == score[0][1]:
        return score.max()*100
    else:
        return 25
    



def obtain_compiled_vgg_16(vgg_weights_path):
    global g2
    g2 = tf.Graph()
    with tf.Session(graph=g2) as sess1:
        with g2.as_default():
            model = vgg_16(vgg_weights_path)
            sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(optimizer=sgd, loss='categorical_crossentropy')
    return model


def vgg_16(weights_path=None):
    with K.name_scope('myvgg'):
        model = Sequential()  
        model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(224, 224,3),padding='same',activation='relu',kernel_initializer='uniform')) 
        model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(MaxPooling2D(pool_size=(2,2)))  
        model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(MaxPooling2D(pool_size=(2,2)))  
        model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(MaxPooling2D(pool_size=(2,2)))  
        model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(MaxPooling2D(pool_size=(2,2)))  
        model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(MaxPooling2D(pool_size=(2,2)))  
        model.add(Flatten())  
        model.add(Dense(4096,activation='relu'))  
        model.add(Dropout(0.5))  
        model.add(Dense(4096,activation='relu'))  
        model.add(Dropout(0.5))  
        model.add(Dense(2,activation='softmax'))

        if weights_path:
            model.load_weights(weights_path)

    return model

	


