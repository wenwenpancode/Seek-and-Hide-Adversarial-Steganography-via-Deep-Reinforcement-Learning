from __future__ import division
from os import listdir
from os.path import isfile, join
import argparse
import os
import shutil
import tensorflow as tf
import time
import tensorflow as tf
#tf.losses = tf
import numpy as np
import sys
from scipy.misc import imresize
import functools
import scipy.misc
from scipy import misc
# import vgg19.vgg as vgg
sys.path.append('/nfs/yyl/AAAI_steganography/')
from utils import *
import netdef_autoencoder as netdef
from utils import mkdir_if_not_exists
import pdb
os.system("rm -rf log/*")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

#temp = np.array(load_image('secret.jpg',shape = [256,256]))

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
                        help='path to training content images folder', default = '/nfs/yyl/AAAI_steganography/MSCOCO_train2k/') 
    # Finetune Options
    parser.add_argument('--continue_train', type=bool, dest='continue_train', default=False)
    # Others
    parser.add_argument('--sample_path', type=str, dest="sample_path", 
                        default='/nfs/yyl/AAAI_steganography/test.jpg')#'sample.jpg')
    parser.add_argument('--output_rev', type=str, dest="output_rev", 
                        default='output_rev_mask/')
    parser.add_argument('--output_path', type=str, dest="output_path", 
                        default='/nfs/yyl/AAAI_steganography/test_output/')
    return parser

def sender_noise(secret_object):
    parser = build_parser()
    args = parser.parse_args()

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    #sess = tf.Session(config=config)
    g0 = tf.Graph()
    with tf.Session(graph=g0) as sess:

        style_image_basename = os.path.basename(args.style)
        style_image_basename = style_image_basename[:style_image_basename.find(".")]

        args.checkpoint_dir = os.path.join("/nfs/yyl/AAAI_steganography/examples/checkpoint", style_image_basename)
        args.serial = os.path.join("/nfs/yyl/AAAI_steganography/examples/serial", style_image_basename)
        
        args.checkpoint_dir = '/nfs/yyl/AAAI_steganography/checkpoint_15-20-2/'
        
        train_model = Model(sess,g0, args,secret_object)

    #    print("[*] Checkpoint Directory: {}".format(args.checkpoint_dir))
    #    print("[*] Serial Directory: {}".format(args.serial))
        mkdir_if_not_exists(args.serial, args.checkpoint_dir)
    #sess.close()
    #print(output)
    tf.reset_default_graph()
    return output

def gradient(pred):            
    D_dx = -pred[:, :, 1:, :] + pred[:, :, :-1, :]            
    D_dy = -pred[:, 1:, :, :] + pred[:, :-1, :, :]          
    return D_dx, D_dy

class Model(object):
    def __init__(self,sess,g0, args, secret_object):
        self.sess = sess
        #pdb.set_trace()
        self.g0 = g0
        self._build_train_model_slow(args)
        result = self._slow_optimize(args,secret_object)
        global output
        output = result
        
    def _build_train_model_slow(self,args):

        with self.g0.as_default():
            self.secret_img = tf.placeholder(tf.float32, shape=(args.batch_size, 256, 256, 3), name = 'secret_img')
            self.cover_img = tf.placeholder(tf.float32, shape=(args.batch_size, 256, 256, 3), name = 'cover_img')
            self.noise_img, _ = netdef.autoencoder(self.secret_img/255.,'sender', reuse = False)
            self.container_img = self.noise_img*255. + self.cover_img
            self.revealed, self.revealed_latent = netdef.autoencoder(self.container_img/255.,'autoencoder', reuse = False)
            self.loss_secret = tf.losses.absolute_difference(self.secret_img, self.revealed*255.)
            self.loss_perturbation = tf.reduce_sum(tf.multiply(self.noise_img, self.noise_img))
            self.loss_cover = tf.losses.absolute_difference(self.cover_img, self.container_img)

            img=self.container_img
            imgn=self.cover_img
            
            img_dx, img_dy = gradient(imgn)        
            disp_dx, disp_dy = gradient(img)         
            weight_x = tf.exp(-tf.reduce_mean(tf.abs(img_dx), 3, keep_dims=True))        
            weight_y = tf.exp(-tf.reduce_mean(tf.abs(img_dy), 3, keep_dims=True))         
            self.loss_tv = tf.reduce_mean(weight_x*tf.abs(disp_dx)) + tf.reduce_mean(weight_y*tf.abs(disp_dy))

            self.loss = 20 * self.loss_secret + self.loss_perturbation
            
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('loss_cover', self.loss_cover)
            tf.summary.scalar('loss_secret', self.loss_secret)
            tf.summary.scalar('loss_perturbation', self.loss_perturbation)

    def _slow_optimize(self, args,secret_object):        
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('log/', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        #pdb.set_trace()
        #for  v in tf.global_variables():
        #    print v.name
        #    if v.name.split('/')[0] == 'sender' or v.name.split('/')[0] == 'autoencoder':
        #        print 'hhh'
        #pdb.set_trace()
        variables_to_restore = [v for v in tf.global_variables() if \
            v.name.split('/')[0] == 'sender' or v.name.split('/')[0] == 'autoencoder']
        #pdb.set_trace()
        self.saver = tf.train.Saver(variables_to_restore)
        #myvariables = [v for v in tf.global_variables() if \
        #            v.name.split('/')[0] == 'myvgg' or v.name.split('/')[0] == 'myqnet']
        #print myvariables
        #saver_out = tf.train.Saver(myvariables)
        self.load('/nfs/yyl/AAAI_steganography/checkpoint_15-20-2/')
        

		# wendy
        secret_input =secret_object
        cover_input = np.array(load_image('sample.jpg',shape = [256,256]))
		
        noise, container, revealed,loss,loss_secret,loss_perturbation,loss_cover = self.sess.run([self.noise_img, self.container_img, self.revealed,self.loss,self.loss_secret,self.loss_perturbation,self.loss_cover], feed_dict={self.cover_img: [cover_input],self.secret_img: [secret_input] })
		
        #print('loss:',loss)
        #print('loss secret:',loss_secret)
        #print('loss perturbation:',loss_perturbation)
        #print('loss cover:',loss_cover)
        #tf.reset_default_graph()
        #scipy.misc.imsave('noise.jpg',noise[0]*255)
        noise2 = np.array(load_image('noise.jpg'))
        return noise[0]*255
		
    def load(self, checkpoint_dir):
        #print("flag")
        print (" [*] Reading checkpoint...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        #pdb.set_trace()
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            #print os.path.join(checkpoint_dir, ckpt_name)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            try:
                self.saver.restore(self.sess, checkpoint_dir)
                return True
            except:
                return False

##if __name__ == "__main__":
##    main(temp)
