#encoding:utf-8
import cv2, numpy as np
import time
import math as mth
from PIL import Image, ImageDraw, ImageFont
import scipy.io
from keras.models import Sequential
from keras import initializers
from keras.initializers import random_normal
from keras.layers import Dense, Dropout, Activation, Flatten

from keras.optimizers import RMSprop, SGD, Adam
import random
import argparse
from scipy import ndimage
from scipy import misc
from keras.preprocessing import image
from sklearn.preprocessing import OneHotEncoder
from features import get_image_descriptor_for_image, obtain_compiled_vgg_16, vgg_16
from image_helper import *
from reinforcement import *
from get_contain import *
import pdb
from hider import conceal_noise, get_loss, read_noise
from vgg16 import *
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
sess = tf.Session(config=config)


if __name__ == "__main__":
   
    ######## PATHS definition ########

    path_secret = "../generaliza-without-classification/secret_4030/"
    path_cover = "../generaliza-without-classification/cover_4030/"
    weights_path = "models/"
    path_testing_folder = 'testing_result_912/'
    path_vgg = "../smallq/4030.h5"

    model_vgg = obtain_compiled_vgg_16(path_vgg)
    model_vgg_ex = myVGG()
    model = get_array_of_q_networks_for_pascal('tmp_qnet1.h5')
    #model = get_array_of_q_networks_for_pascal('0')[0]

    secret_names = np.array([load_images_names_in_data_set(path_secret)])
    cover_names = np.array([load_images_names_in_data_set(path_cover)])
    alpha = 0.05   # bigger -> quicker; small -> accuracy
    number_of_steps = 50

    for j in range(0,3000):
   # for j in range(len(secret_names[0])):
        image_name = secret_names[0][j]
        image_cover = np.array(load_image(path_cover + cover_names[0][j]+'.jpg'))
        image_secret = np.array(load_image(path_secret + secret_names[0][j]+'.jpg'))
        image_cover = misc.imresize(image_cover,(256,256))
        image_secret = misc.imresize(image_secret,(256,256))

        noise_origi = conceal_noise(image_secret)
        #image_noise = read_noise(image_name)
        #noise_origi = np.array(image_noise)*1.0/20
		# init drawing for visualization
        #background = Image.new('RGBA', (10000, 2000), (255, 255, 255, 255))
        #background = Image.new('RGB',(image_cover.shape[1],image_cover.shape[0]), (255, 255, 255))
        background = Image.open(path_cover+cover_names[0][j]+'.jpg').convert("RGB")
        background = background.resize((256,256))

        #background = background_ori
        draw = ImageDraw.Draw(background)
        
        # offset of the region observed at each time step
        last_loss = 300
        offset = (64, 64)
        #offset = (random.randint(0, image_cover.shape[0]-128),random.randint(0,image_cover.shape[1]-128))
        size = (128, 128)

        action = 0
        step = 0
        qval = 0

        image_contain, image_roi, image_ROI,image_rev = get_container(image_cover, noise_origi, offset, size)


        hider_loss = get_loss(image_secret,image_cover,image_contain,image_rev,size)
        class_loss = get_loss1(image_rev, model_vgg)
        new_loss =  hider_loss + class_loss
        print 'hider_loss ', hider_loss
        print 'new_loss', new_loss
        print 'class_loss ', class_loss
        last_loss = new_loss
                
        draw.rectangle((offset[0],offset[1],offset[0]+size[0],offset[1]+size[1]),outline='red')
        draw.text((offset[0]+size[0]/2+1,offset[1]+size[1]/2+1),str(new_loss),fill=(255,0,0))
        background.save(path_testing_folder+image_name+'_'+str(step)+'.jpg')


        history_vector = np.zeros([36])
        state = get_state(image_ROI, history_vector, model_vgg_ex)
        status = 1
        
        while (status == 1) & (step < number_of_steps):
            step += 1
            with tf.Session(graph=g4) as sess:
                #qnet_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='myqnet')
                #sess.run(tf.global_variables_initializer())
                model.load_weights('tmp_qnet1.h5')
                qval = model.predict(state.T, batch_size=1)
            
                action = (np.argmax(qval))+1
                if action == 9:
                    status = 0
                    background =Image.open(path_cover+cover_names[0][j]+'.jpg').convert("RGB").resize((256,256))
                    draw = ImageDraw.Draw(background)
                    draw.rectangle((offset[0],offset[1],offset[0]+size[0],offset[1]+size[1]),outline='red')
                    draw.text((offset[0]+size[0]/2+1,offset[1]+size[1]/2+1),str(new_loss),fill=(255,0,0))
                    draw.text((offset[0]+size[0]/2+10,offset[1]+size[1]/2+10),'action='+str(action),fill=(255,0,0))
                    background.save(path_testing_folder+image_name+'_'+str(step)+'.jpg')
                else:

                    alpha_size = (size[0]*alpha, size[1]*alpha)
                    size_save = size
                    offset_save = offset
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
                        offset =(offset[0]+0.5*alpha_size[1],offset[1]+0.5*alpha_size[0])
                    elif action == 7:
                        size = (size[0], size[1]-alpha_size[1])
                        offset = (offset[0], offset[1]+0.5*alpha_size[1])
                    else:
                        size = (size[0]-alpha_size[0], size[1])
                        offset = (offset[0]+0.5*alpha_size[0],offset[1])

                    size_tmp = (int(size[0]),int(size[1]))
                    offset_tmp = (int(offset[0]), int(offset[1]))
                    if offset_tmp[0] < 0 or offset_tmp[1] < 0 or size_tmp[0] < 0 \
                    or size_tmp[1] < 0 or offset_tmp[0]+size_tmp[1] > 256 or offset_tmp[1]+size_tmp[0]>256 \
                    or size_tmp[0]==0 or size_tmp[1]==0:
                        reward = -1
                        size = size_save
                        offset = offset_save
                    else:
                        size = size_tmp
                        offset = offset_tmp
                        image_contain, image_roi, image_ROI, image_rev = get_container(image_cover,noise_origi, offset, size)
                        hider_loss = get_loss(image_secret,image_cover,image_contain,image_rev,size)
                        class_loss = get_loss1(image_rev, model_vgg)
                        new_loss =  hider_loss + class_loss
                        print 'hider_loss ', hider_loss
                        print '**************************new_loss', new_loss
                        print 'class_loss ', class_loss
                        print '\n'  
 
                        reward = get_reward_movement(last_loss, new_loss)
                        last_loss = new_loss
                        background = Image.open(path_cover+cover_names[0][j]+'.jpg').convert("RGB").resize((256,256))

                        draw = ImageDraw.Draw(background)
                        draw.rectangle((offset[0],offset[1],offset[0]+size[0],offset[1]+size[1]),outline='red')
                        draw.text((10,1),'offset='+str(offset),fill=(255,0,0))
                        draw.text((10,10),'action='+str(action),fill=(255,0,0))
                        draw.text((10,20),'reward='+str(reward),fill=(255,0,0))
                        draw.text((10,30),'hide_loss='+str(hider_loss),fill=(255,0,0))
                        draw.text((10,40),'cls_loss='+str(class_loss),fill=(255,0,0))
                        draw.text((10,50), 'size= '+str(size),fill=(255,0,0))
                        background.save(path_testing_folder+image_name+'_'+str(j)+'_'+str(step)+'.jpg')


            history_vector = update_history_vector(history_vector, action)
            new_state = get_state(image_ROI, history_vector, model_vgg_ex)
            state = new_state
