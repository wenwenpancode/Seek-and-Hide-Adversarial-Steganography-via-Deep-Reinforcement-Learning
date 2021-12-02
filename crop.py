import numpy as np
import matplotlib
from PIL import Image
import os
import pdb
import cv2
import glob
import tensorflow as tf
"""
Example Usage:
---------------
python crop.py \
    --image_name: Path to the croping image.
    --save_dir: Path to save the croped images(dir).
"""
flags = tf.app.flags
flags.DEFINE_string('image_name', None, 'Path to image.')
flags.DEFINE_string('noise_name',None,'Path to noise')
flags.DEFINE_string('save_dir', None, 'Path to save croped images.')
FLAGS = flags.FLAGS

def img_seg(path, name):
    img = Image.open(os.path.join(name))
    hight, width = img.size
    #pdb.set_trace()
    w = 224
    id = 1
    i = 0   
    f = open(path+'location.txt',mode='w')
    while(i+w < hight):
        j=0
        while(j+w < width):
            new_img = img.crop((i,j,i+w,j+w))
            rename = path + 'neg/'+str(id) + '_0.jpg'
            new_img.save(rename)
            id += 1
	    f.write(str(id-1) + '\t' +str(i)+ ' ' + '\t' + str(j) + '\n')
            j += 5
        i += 5
    #f = open('location.txt',mode='w') 
    print(id-1)


def get_train_data(images_path, noise_path):

    if not os.path.exists(images_path+'neg/'):
        raise ValueError('images_path is not exist.')

    images = []
    new_path = os.path.join(images_path+'neg/', '*.jpg')
    
    count = 0
    noise = cv2.imread(noise_path)
    noise = cv2.resize(noise, (224,224), interpolation=cv2.INTER_AREA)
#    noise = cv2.cvtColor(noise, cv2.COLOR_BGR2RGB)
    for image_file in glob.glob(new_path):
        count += 1
        if count % 100 == 0:
            print('Load {} images.'.format(count))
        image = cv2.imread(image_file)
 #       image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print noise
        #print image
        container = image + noise*1.0/20
        #print container
        #exit()
        save_path = images_path + 'pos/' + str(count) + '_1.jpg'
        cv2.imwrite(save_path,container)


if __name__ == '__main__':
    path = FLAGS.save_dir
    if not os.path.exists(path):
        os.mkdir(path)
    os.mkdir(path+'neg/')
    os.mkdir(path+'pos/')
    image = FLAGS.image_name
    noise = FLAGS.noise_name
    img_seg(path, image)
    get_train_data(path, noise)
