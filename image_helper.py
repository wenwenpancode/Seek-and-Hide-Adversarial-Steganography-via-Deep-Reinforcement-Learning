from keras.preprocessing import image
import numpy as np
import os
import glob


def get_all_images(path_voc):
    images = []
    new_path = os.path.join(path_voc,'*.jpg')
    for image_file in glob.glob(new_path):
        images.append(image.load_img(image_file, False))
    return images


def load_images_names_in_data_set(path_voc):
    image_names = []
    new_path = os.path.join(path_voc,'*.jpg')
    for image_file in glob.glob(new_path):
	_,tmp_file = os.path.split(image_file)
	file_name,_ = tmp_file.split('.')
        image_names.append(file_name)
    return image_names
