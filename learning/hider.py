import sender_noise as sendernoise
import calculate_loss as closs
import os
import glob
import numpy as np
from keras.preprocessing import image
import random
import cv2
def conceal_noise(secret_img):

    noise = sendernoise.sender_noise(secret_img)
    return noise
	
def get_loss(secret_img, cover_image,container_img, image_roi,size):

    loss = closs.calculate(secret_img, cover_image,container_img,image_roi,size)
    return loss

def read_noise(image_name):
	index = int(str(image_name))
	noise_name = os.path.join('noise/', '*.jpg')
	for image_file in glob.glob(noise_name):
		_, index_tmp = image_file.split('/')
		index_tmp,_ = index_tmp.split('-')
		if int(index_tmp) == index:
			noise = image.load_img(image_file, False)
	return noise
