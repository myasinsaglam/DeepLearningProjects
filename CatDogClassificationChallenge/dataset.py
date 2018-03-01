from __future__ import division, print_function, absolute_import
from scipy.misc import imresize
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from glob import glob
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.metrics import Accuracy
files_path = '/home/student_1/Desktop/catdog/train2/'
def file_operations():
    cat_files_path = os.path.join(files_path, 'cat*.jpg')
    dog_files_path = os.path.join(files_path, 'dog*.jpg')
    
    cat_files = sorted(glob(cat_files_path))
    dog_files = sorted(glob(dog_files_path))
    
    n_files = len(cat_files) + len(dog_files)
    print(n_files)
    
    size_image = 64
    
    allX = np.zeros((n_files, size_image, size_image, 3), dtype='float64')
    ally = np.zeros(n_files)
    count = 0
    for f in cat_files:
        try:
            img = Image.open(f)
            new_img = imresize(img, (size_image, size_image, 3))
            allX[count] = np.array(new_img)
            ally[count] = 0
            count += 1
        except:
            continue

    for f in dog_files:
        try:
            img = Image.open(f)
            new_img = imresize(img, (size_image, size_image, 3))
            allX[count] = np.array(new_img)
            ally[count] = 1
            count += 1
        except:
            continue


###################################
# Prepare train & test samples
###################################

# test-train split
    X, X_test, Y, Y_test = train_test_split(allX, ally, test_size=0.1, random_state=42)
    
    # encode the Ys
    Y = to_categorical(Y, 2)
    Y_test = to_categorical(Y_test, 2)
    np.save("ALL_X.npy",allX)
    np.save("ALL_Y.npy",ally)
    np.save("TRAIN.npy",X)
    np.save("TRAIN_LABEL.npy",Y)
    np.save("TEST.npy",X_test)
    np.save("TEST_LABEL.npy",Y_test)
