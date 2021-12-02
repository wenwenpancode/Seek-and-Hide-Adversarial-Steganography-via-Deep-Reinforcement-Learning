#encoding:utf-8
import os
import numpy as np
import cv2
from numpy import *

def get_container(image_cover, noise_origi, offset, size):
	#shape[0],shape[1]可能写反了
    shape = image_cover.shape
#    print shape
    image_contain = image_cover
    image_noise = cv2.resize(noise_origi, size, interpolation=cv2.INTER_AREA)
    part1 = image_contain[0:shape[0],0:offset[1]]
#    print part1.shape
    center = image_contain[offset[0]:offset[0]+size[1],offset[1]:offset[1]+size[0]] + image_noise
#    print center.shape
    part2 = image_contain[0:offset[0],offset[1]:offset[1]+size[0]]
#    print part2.shape
    part3 = image_contain[offset[0]+size[1]:shape[0],offset[1]:offset[1]+size[0]]
#    print part3.shape
    part4 = image_contain[0:shape[0],offset[1]+size[0]:shape[1]]
#    print part4.shape

    mid0 = vstack((part2,center))
    mid = vstack((mid0,part3))
    top_mid = hstack((part1,mid))
    updown = hstack((top_mid,part4))

    size_roi = (int(size[0]* 1.3), int(size[1]*1.3))
    offset_roi = (int(offset[0] * 0.85), int(offset[1]*0.85))

    if offset_roi[0] + size_roi[1] > shape[0]:
	size_roi_tmp1 = shape[0] - offset_roi[0]
    else:
	size_roi_tmp1 = size_roi[1]  
    if offset_roi[1] + size_roi[0] > shape[1]:
	size_roi_tmp0 = shape[1] - offset_roi[1]
    else:
	size_roi_tmp0 = size_roi[0]

    size_roi = (size_roi_tmp0,size_roi_tmp1)
    roi = updown[offset_roi[0]:offset_roi[0]+size_roi[1],offset_roi[1]:offset_roi[1]+size_roi[0]]    
    roi = cv2.resize(roi, (256,256), interpolation=cv2.INTER_AREA)
    
    img_test = image_cover[offset[0]:offset[0]+size[1],offset[1]:offset[1]+size[0]]
    img_test = cv2.resize(img_test,(256,256))
    img_test = img_test + noise_origi


    return updown,center,roi, img_test


def get_container_ex(image,  offset, size):

	
	pass
