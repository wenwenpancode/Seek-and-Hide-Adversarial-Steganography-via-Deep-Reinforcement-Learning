import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras import initializers
from keras.initializers import random_normal, identity, VarianceScaling
from keras.layers import Dense, Dropout, Activation, Flatten, normalization

from keras.optimizers import RMSprop, SGD, Adam
from features import *
import os

g4 = tf.Graph()
# Different actions that the agent can do
number_of_actions = 9
# Actions captures in the history vector
actions_of_history = 4
# Visual descriptor size
visual_descriptor_size = 25088
# Reward movement action
reward_movement_action = 1
# Reward terminal action
reward_terminal_action = 3
# IoU required to consider a positive detection
loss_threshold = 55


def update_history_vector(history_vector, action):
    action_vector = np.zeros(number_of_actions)
    action_vector[action-1] = 1
    size_history_vector = np.size(np.nonzero(history_vector))
    updated_history_vector = np.zeros(number_of_actions*actions_of_history)
    if size_history_vector < actions_of_history:
        aux2 = 0
        for l in range(number_of_actions*size_history_vector, number_of_actions*size_history_vector+number_of_actions - 1):
            history_vector[l] = action_vector[aux2]
            aux2 += 1
        return history_vector
    else:
        for j in range(0, number_of_actions*(actions_of_history-1) - 1):
            updated_history_vector[j] = history_vector[j+number_of_actions]
        aux = 0
        for k in range(number_of_actions*(actions_of_history-1), number_of_actions*actions_of_history):
            updated_history_vector[k] = action_vector[aux]
            aux += 1
        return updated_history_vector
 
def get_state(image, history_vector, model_vgg):
    descriptor_image = get_conv_image_descriptor_for_image(image, model_vgg)
    descriptor_image = np.reshape(descriptor_image, (visual_descriptor_size, 1))
    history_vector = np.reshape(history_vector, (number_of_actions*actions_of_history, 1))
    state = np.vstack((descriptor_image, history_vector))
    return state

def get_reward_movement(last_loss, new_loss):
    #modify into losses
    if new_loss < last_loss:
        #reward = reward_movement_action
        reward = reward_movement_action*(last_loss - new_loss)
    else:
        #reward = - reward_movement_action
        reward = reward_movement_action*(last_loss - new_loss)
    return reward

def get_reward_trigger(new_loss):
    #modify into iter > 50
    if new_loss < loss_threshold:
        reward = reward_terminal_action
    else:
        reward = - reward_terminal_action
    return reward

def q_predict(model, state, path_model,batch_size=1):
    with tf.Session(graph=g4) as sess:
	    model.load_weights('tmp_qnet.h5')    
	    qval = model.predict(state, batch_size=1)
    return qval

def get_q_network(weights_path):
    with K.name_scope('myqnet'):
        with tf.Session(graph=g4) as sess:
            model = Sequential()
            model.add(Dense(1024,kernel_initializer=VarianceScaling(scale=0.01,mode='fan_in',
            distribution='normal',seed=None),activation='relu',input_shape=(25124,)))
            model.add(Dropout(0.2))
            #model.add(normalization.BatchNormalization(axis=1, momentum=0.99,epsilon=0.001)
            model.add(Dense(1024,kernel_initializer=VarianceScaling(scale=0.01, mode='fan_in', distribution='normal', seed=None),activation='relu'))
            model.add(Dropout(0.2))
            #model.add(normalization.BatchNormalization(axis=1,momentum=0.99,epsilon=0.001)
            model.add(Dense(9,kernel_initializer=VarianceScaling(scale=0.01, mode='fan_in', distribution='normal', seed=None),activation='linear'))
            adam = Adam(lr=1e-6)
            model.compile(loss='mean_squared_error',optimizer=SGD(lr=0.00001,momentum=0.5),metrics=['acc'])
            #adam = Adam(lr=1e-6)
            #model.compile(loss='mse', optimizer=Adam(lr=1e-6),metrics=['acc'])
            if weights_path != "0":
                model.load_weights(weights_path)
            if not os.path.exists('tmp_qnet.h5'):
                model.save_weights('tmp_qnet.h5')
    return model

def get_array_of_q_networks_for_pascal(weights_path):
    global g4
    #g4 = tf.Graph()
    with g4.as_default():
        if weights_path == "0":
            q_networks = get_q_network("0")
        else:
            print weights_path
            q_networks = get_q_network(weights_path)
    return q_networks
    #return np.array([q_networks])
