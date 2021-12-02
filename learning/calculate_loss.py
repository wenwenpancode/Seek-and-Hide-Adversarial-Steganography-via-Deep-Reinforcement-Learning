from __future__ import division
from os import listdir
from os.path import isfile, join
import argparse
import os
import shutil
import tensorflow as tf
import time
import tensorflow as tf
import numpy as np
import sys
from scipy.misc import imresize
import functools
import scipy.misc
from scipy import misc

from utils import *
import netdef_autoencoder as netdef
from utils import mkdir_if_not_exists
import pdb


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"



output=0


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--style', type=str, dest='style', help='style image path',
                        default='mask_data_1/mask1.jpg')#'secret.jpg')
    parser.add_argument('--batch_size', type=int, dest='batch_size', help='batch size', 
                        default=1)
    parser.add_argument('--max_iter', type=int, dest='max_iter', help='max iterations', 
                        default=1e5)
    parser.add_argument('--learning_rate', type=float, dest='learning_rate', 
                        default=1e-4)
    parser.add_argument('--iter_print', type=int, dest='iter_print', default=100)
    parser.add_argument('--checkpoint_iterations', type=int, dest='checkpoint_iterations',
                        help='checkpoint frequency', default=500)
    parser.add_argument('--train_path', type=str, dest='train_path',
                        help='path to training content images folder', default = '') 
    # Finetune Options
    parser.add_argument('--continue_train', type=bool, dest='continue_train', default=False)
    # Others
    parser.add_argument('--sample_path', type=str, dest="sample_path", 
                        default='')#'sample.jpg')
    parser.add_argument('--output_rev', type=str, dest="output_rev", 
                        default='output_rev_mask/')
    parser.add_argument('--output_path', type=str, dest="output_path", 
                        default='')
    return parser



def calculate(secret_object,cover_object,container_object,roi_object, size):
    parser = build_parser()
    args = parser.parse_args()

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    g1 = tf.Graph()

    with tf.Session(graph=g1) as sess1:
        with g1.as_default():
            style_image_basename = os.path.basename(args.style)
            style_image_basename = style_image_basename[:style_image_basename.find(".")]

            args.checkpoint_dir = os.path.join("", style_image_basename)
            args.serial = os.path.join("", style_image_basename)
            
            args.checkpoint_dir=''

            train_model = Model(sess1,g1,args,secret_object,cover_object,container_object,roi_object,size)
            mkdir_if_not_exists(args.serial, args.checkpoint_dir)
            

    return output
    

def gradient(pred):            
    D_dx = -pred[:, :, 1:, :] + pred[:, :, :-1, :]            
    D_dy = -pred[:, 1:, :, :] + pred[:, :-1, :, :]          
    return D_dx, D_dy

class Model(object):
    def __init__(self, sess, g1,args,secret_object,cover_object,container_object,roi_object,size):
        self.sess = sess
        self.g1 = g1
        self._build_train_model_slow(args)
        result = self._slow_optimize(args,secret_object,cover_object,container_object,roi_object,size)
        global output 
        output = result
        
    def _build_train_model_slow(self,args):

        self.secret_img = tf.placeholder(tf.float32, shape=(args.batch_size, image_h, image_w, 3), name = 'secret_img')
        self.cover_img = tf.placeholder(tf.float32, shape=(args.batch_size, image_h, image_w, 3), name = 'cover_img')
        self.temp_img = tf.placeholder(tf.float32, shape=(args.batch_size, image_h, image_w, 3), name = 'temp_img')
        self.roi_img = tf.placeholder(tf.float32, shape=(args.batch_size, image_h, image_w, 3), name = 'roi_img')
        self.noise_img, _ = netdef.autoencoder(self.secret_img/255.,'sender', reuse = False)

        self.revealed, self.revealed_latent = netdef.autoencoder(self.roi_img/255.,'autoencoder', reuse = False)
        self.loss_secret = tf.losses.absolute_difference(self.secret_img, self.revealed*255.)
        self.loss_perturbation = tf.reduce_sum(tf.multiply(self.noise_img, self.noise_img))
        self.loss_cover = tf.losses.absolute_difference(self.cover_img, self.temp_img)

        img=self.temp_img
        imgn=self.cover_img
        
        img_dx, img_dy = gradient(imgn)        
        disp_dx, disp_dy = gradient(img)         
        weight_x = tf.exp(-tf.reduce_mean(tf.abs(img_dx), 3, keep_dims=True))        
        weight_y = tf.exp(-tf.reduce_mean(tf.abs(img_dy), 3, keep_dims=True))         
        self.loss_tv = tf.reduce_mean(weight_x*tf.abs(disp_dx)) + tf.reduce_mean(weight_y*tf.abs(disp_dy))

        self.loss =  self.loss_secret + self.loss_perturbation + 10*self.loss_tv + 10*self.loss_cover
        
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('loss_secret', self.loss_secret)
        tf.summary.scalar('loss_perturbation', self.loss_perturbation)

    def _slow_optimize(self,args,secret_object,cover_object,container_object,roi_object,size):        
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('log/', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        variables_to_restore = [v for v in tf.global_variables() if v.name.split('/')[0] == 'sender' or v.name.split('/')[0] == 'autoencoder']
        self.saver = tf.train.Saver(variables_to_restore)
        self.load('')
            

	
        secret_input =secret_object
        cover_input = cover_object
        temp_img = container_object
        roi_img = roi_object
		
        noise,revealed,loss,loss_secret,loss_perturbation,loss_cover,cover_img,loss_tv= self.sess.run([self.noise_img,self.revealed,self.loss,self.loss_secret,self.loss_perturbation,self.loss_cover,self.cover_img,self.loss_tv], feed_dict={self.cover_img: [cover_input],self.secret_img: [secret_input],self.temp_img:[temp_img],self.roi_img:[roi_img] })
        loss_final = loss_secret + loss_perturbation +10*loss_tv + loss_cover*100*(1.0/size[0])*(1.0/size[1])

        print('loss secret:',loss_secret)
        print('loss perturbation:',loss_perturbation)
        print('loss cover:',loss_cover*100*(1.0/size[0])*(1.0/size[1]))
        print('loss tv: ',loss_tv*10)
        
        return loss_final
        
		
    def load(self, checkpoint_dir):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            try:
                self.saver.restore(self.sess, checkpoint_dir)
                return True
            except:
                return False

