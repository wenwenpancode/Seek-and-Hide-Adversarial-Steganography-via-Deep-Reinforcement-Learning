#encoding:utf-8
import tensorflow as tf
import cv2, numpy as np
import time
import math as mth
from PIL import Image, ImageDraw, ImageFont
import scipy.io
from keras.models import Sequential
from keras import initializers
from keras.initializers import random_normal
from keras.layers import Dense, Dropout, Activation, Flatten

import random as myrandom
import argparse
from scipy import ndimage
from scipy import misc
from keras.preprocessing import image
from sklearn.preprocessing import OneHotEncoder
from features import get_image_descriptor_for_image, obtain_compiled_vgg_16, vgg_16
from vgg16 import  get_conv_image_descriptor_for_image

from image_helper import *
from reinforcement import *
#from numpy import *
import pdb
from hider import *
import time
from utils import *
from get_contain import *
import os
from vgg16 import *

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = tf.Session(config=config)

epochs_id = 0

if __name__ == "__main__":

    ######## PATHS definition ########
    path_secret = "../generaliza-without-classification/secret_4030/"
    path_cover = "../generaliza-without-classification/cover_4030/"
    path_model = "models_test/"
    path_testing_folder = 'testing_visualizations/'
    path_vgg = "../smallq/4030.h5"
    ######## PARAMETERS ########
    alpha = 0.1   # bigger -> quicker; small -> accuracy
    number_of_steps = 50
    epochs = 1000
    gamma = 0.5
    epsilon = 0.5
    batch_size = 10
    #scale_subregion = float(3)/4
    #scale_mask = float(1)/(scale_subregion*4)
    # Pointer to where to store the last experience in the experience replay buffer,
    #pdb.set_trace()
    h = np.zeros(1)
    # Each replay memory has a capacity of 100 experiences
    buffer_experience_replay = 100
    # Init replay memories
    replay = []
    '''
    if os.path.exists('replay.npy'):
        repl = np.load('replay.npy',allow_pickle=True)
        replay = list(repl)
#        print replay
#        pdb.set_trace()
    else:
        replay = []
    '''
    #pdb.set_trace()
    reward = 0

    ######## MODELS ########
    #pdb.set_trace()
    model_vgg_ex = myVGG()
    #pdb.set_trace()
#    tf.reset_default_graph()

    model_vgg = obtain_compiled_vgg_16(path_vgg)

    #model_vgg_ex = myVGG()
    #pdb.set_trace()
    if epochs_id == 0:
        model = get_array_of_q_networks_for_pascal("0")
    else:
        model = get_array_of_q_networks_for_pascal(path_model)

    ######## LOAD IMAGE NAMES ########
    secret_names = np.array([load_images_names_in_data_set(path_secret)])
    cover_names = np.array([load_images_names_in_data_set(path_cover)])
    #cover_images = get_all_images(path_cover)
    #secret_images = get_all_images(path_secret)
    
    #if len(secret_names)!=len(cover_names):
    #    print 'error! secret images not equal cover images'
    #
    for i in range(epochs_id, epochs_id + epochs):
        print '&'*30
        print 'epsilon',epsilon
        print 'epoch',i
        #for j in range(len(secret_names[0])):
        for j in range(1):   
            image_name = secret_names[0][j]
            print image_name
            image_cover = np.array(load_image(path_cover + cover_names[0][j]+'.jpg'))
            image_secret = np.array(load_image(path_secret + secret_names[0][j]+'.jpg'))
            image_cover = misc.imresize(image_cover,(256,256))
            image_secret = misc.imresize(image_secret, (256,256))
            image_noise = conceal_noise(image_secret) 
            #image_noise = read_noise('000030') 
            #noise_origi = np.array(image_noise)#*1.0/20
            noise_origi = image_noise
            last_loss = 300
            #size = (256, 256)
            #size = (256, 256)
            #offset = (0,0)
            #offset = (64,64)
            size = (128,128)
            offset = (random.randint(0,image_cover.shape[0]-128),random.randint(0,image_cover.shape[1]-128))
            history_vector = np.zeros([36])
            status = 1
            action = 0
            reward = 0
            step = 0
            # this for loss calcu
            image_contain, image_roi, image_ROI, image_rev = get_container(image_cover, noise_origi, offset, size)
            #t1 = time.time()
            hider_loss = get_loss(image_secret,image_cover,image_contain,image_rev,size)
            #pdb.set_trace()
            #t2 = time.time()
            #print 'time:',t2-t1
            class_loss = get_loss1(image_roi, model_vgg)
            new_loss =  hider_loss
            last_loss = new_loss
            print 'hider_loss ', hider_loss
            print '****************************new_loss', new_loss
            print 'class_loss ', class_loss
            print '\n'
            #print image_roi.shape
            #pdb.set_trace()
            state = get_state(image_ROI, history_vector, model_vgg_ex)
            
            if step > number_of_steps:
                background = Image.new('RGB', (image_cover.shape[1],image_cover.shape[0]), (255, 255, 255))
                draw = ImageDraw.Draw(background)
                draw.rectangle((offset[0],offset[1],offset[0]+size[0],offset[1]+size[1]), outline='blue')
                draw.text((offset[0]+1,offset[1]+10),str(new_loss),fill=(255,0,0))
                background.save(path_testing_folder+image_name+'_'+str(i)+'_50.jpg')    
                step += 1
            
            while (status == 1) & (step < number_of_steps):
                #model = models[0]
                #with tf.Session(graph=g4) as sess:
                #    qval = model.predict(state.T, batch_size=1)
                #pdb.set_trace()
                qval = q_predict(model, state.T, path_model,1)
                step += 1
                # we force terminal action in case loss is smaller than *0.5*, to train faster the agent
                if (i < 100) & (last_loss < 20):
                    action = 9
                # epsilon-greedy policy
                elif random.random() < epsilon:
                    action = random.randint(1, 10)
                else:
                    action = (np.argmax(qval))+1
                # terminal action
                #and offset[0] >0 and offset[1] >0 and offset[0]+size[1]<256 and offset[1]+size[0] < 256
                print 'action:***********',action
                if action == 9:
                    #size = int(size)
                    #offset = int(offset)
                    #hider_loss = get_loss(image_secret, image_cover,image_contain) * 1.0 / 10
                    image_contain, image_roi,image_ROI,image_rev = get_container(image_cover,noise_origi, offset, size)
                #    t3 = time.time()
                    hider_loss = get_loss(image_secret,image_cover,image_contain,image_rev,size)
                #    t4 = time.time()
                #    print 'time:',t4-t3
                    class_loss = get_loss1(image_roi,model_vgg)
                    new_loss =  hider_loss
                    reward = get_reward_trigger(new_loss)
                    print 'action = 9'
                    print 'hider_loss ', hider_loss
                    print '**************************new_loss',new_loss
                    print 'class_loss', class_loss
                    print '\n'

                    background = Image.new('RGB', (image_cover.shape[1],image_cover.shape[0]), (255, 255, 255))
                    draw = ImageDraw.Draw(background)
                    draw.rectangle((offset[0],offset[1],offset[0]+size[0],offset[1]+size[1]),outline='blue')
#                    draw.text((10, 1),'cls_loss='+str(class_loss),fill=(255,0,0))
                    draw.text((10, 10),'action='+str(action),fill=(255,0,0))
                    draw.text((10, 20),'reward='+str(reward),fill=(255,0,0))
                    draw.text((10, 30),'hide_loss='+str(hider_loss),fill=(255,0,0))

                    #draw.text((offset[0]+size[0]/2,offset[1]+size[1]/2+1),str(new_loss),fill=(255,0,0))
                    #draw.text((offset[0]+size[0]/2+10,offset[1]+size[1]/2+10),'action=5',fill=(255,0,0))                    
                    background.save(path_testing_folder+image_name+'_'+str(i)+'_'+str(step)+'.jpg')  
                    step += 1
                # movement action, we perform the crop of the corresponding subregion
                else:
                    alpha_size = (size[0]*alpha, size[1]*alpha)
                    size_tmp = size
                    offset_tmp = offset
                    if action == 1:
                        offset = (offset[0]+alpha_size[1], offset[1])
                    elif action == 2:
                        offset = (offset[0]-alpha_size[1], offset[1])
                    elif action == 3:
                        offset = (offset[0], offset[1]-alpha_size[0])
                    elif action == 4:
                        offset = (offset[0], offset[1]+alpha_size[0])
                    elif action == 5:
                        size = (size[0]+alpha_size[0], size[1]+alpha_size[1])
                        offset = (offset[0]-0.5*alpha_size[1],offset[1]-0.5*alpha_size[0])
                    elif action == 6:
                        size = (size[0]-alpha_size[0], size[1]-alpha_size[1])
                        offset=(offset[0]+0.5*alpha_size[1],offset[1]+0.5*alpha_size[0])
                    elif action == 7:
                        size = (size[0], size[1]-alpha_size[1])
                        offset = (offset[0], offset[1]+0.5*alpha_size[1])
                    else:
                        size = (size[0]-alpha_size[0], size[1])
                        offset = (offset[0]+0.5*alpha_size[0], offset[1])

                    size = (int(size[0]),int(size[1]))
                    offset = (int(offset[0]), int(offset[1]))
                    if offset[0] < 0 or offset[1] < 0 or size[0] < 0 \
                    or size[1] < 0 or offset[0]+size[1] > 256 or offset[1]+size[0]>256 \
                    or size[0]==0 or size[1]==0:
                        reward = -1
                        size = size_tmp
                        offset = offset_tmp
                    else:

                        print('action',action)
                        print('offset',offset)
                        #offset = (int(offset[0]), int(offset[1]))
                        image_contain, image_roi, image_ROI, image_rev = get_container(image_cover,noise_origi, offset, size)
                        hider_loss = get_loss(image_secret,image_cover,image_contain,image_rev,size)
                        class_loss = get_loss1(image_roi, model_vgg)
                        new_loss =  hider_loss
                        print 'hider_loss ', hider_loss
                        print '**************************new_loss', new_loss
#                        print 'class_loss ', class_loss
                        print '\n'

                        reward = get_reward_movement(last_loss, new_loss)
                        last_loss = new_loss
                        background = Image.new('RGB', (image_cover.shape[1],image_cover.shape[0]), (255, 255, 255))
                        draw = ImageDraw.Draw(background)
                        draw.rectangle((offset[0],offset[1],offset[0]+size[0],offset[1]+size[1]),outline='red')
#                        draw.text((offset[0]+ 10,offset[1]+1),'cls_loss='+str(class_loss),fill=(255,0,0))
                        draw.text((10, 10),'action='+str(action),fill=(255,0,0))
                        draw.text(( 10,20),'reward='+str(reward),fill=(255,0,0))
                        draw.text(( 10,30),'hide_loss='+str(hider_loss),fill=(255,0,0))
                        background.save(path_testing_folder+image_name+'_'+str(i)+'_'+str(step)+'.jpg')


                #pdb.set_trace()
                history_vector = update_history_vector(history_vector, action)
                new_state = get_state(image_ROI, history_vector, model_vgg_ex)
                #*******************************#
                f = open('replay.txt','a')
                f.write('size: '+str(size)+'\t')
                f.write('offset: '+str(offset)+'\n')
                f.write(str((state, action, reward, new_state))) 
                f.write('\n')

                if len(replay) < buffer_experience_replay:
                    replay.append((state, action, reward, new_state))
                    print 'replay_size: ' ,len(replay)
                else:
                    if h < (buffer_experience_replay-1):
                        h += 1
                    else:
                        h = 0
                    h_aux = h
                    h_aux = int(h_aux)
                    replay[h_aux] = (state, action, reward, new_state)
                    minibatch = myrandom.sample(replay, batch_size)
                    #minibatch = replay[:100]
                    X_train = []
                    y_train = []
                    # we pick from the replay memory a sampled minibatch and generate the training samples
                    #global g4
                    with tf.Session(graph=g4) as sess:
                        qnet_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='myqnet')
                        sess.run(tf.global_variables_initializer())
                        model.load_weights('tmp_qnet.h5')
                        #pdb.set_trace()
                        for memory in minibatch:
                            old_state, action, reward, new_state = memory
                            #print 'action:****************',action
                            #old_qval = q_predict(model,old_state.T,path_model,1)
                            old_qval = model.predict(old_state.T, batch_size=1)
                            print 'old_q****',np.argmax(old_qval),old_qval
                            print action,reward
                            #newQ = q_predict(model, new_state.T, path_model,1)
                            newQ = model.predict(new_state.T, batch_size=1)
                            print 'newQ:****',np.argmax(newQ),newQ

                            maxQ = np.max(newQ)

                            y = np.zeros([1, 9])
                            y = old_qval
                            y = y.T
                            if action != 9: #non-terminal state
                                update = (reward + (gamma * maxQ))
                            else: #terminal state
                                update = reward
                            #pdb.set_trace()
                            y[action-1] = update #target output
                            X_train.append(old_state)
                            y_train.append(y)
                        X_train = np.array(X_train)
                        y_train = np.array(y_train)
                        X_train = X_train.astype("float32")
                        y_train = y_train.astype("float32")
                        X_train = X_train[:, :, 0]
                        y_train = y_train[:, :, 0]
                        print 'training'
                        print '%'*30
                        print 'epsilon: ',epsilon
                        hist = model.fit(X_train, y_train,batch_size=batch_size, epochs=1, verbose=0)
                        print 'acc: ',hist.history['acc']
                        print 'loss: ',hist.history['loss']
                        model.save_weights('tmp_qnet.h5',overwrite=True)
                    #models[0] = model
                    state = new_state
                if action == 9:
                    status = 0

                replay_array = np.array(replay)
                np.save('replay.npy',replay_array)
        if epsilon > 0.1:
            epsilon = epsilon - 0.1


     
