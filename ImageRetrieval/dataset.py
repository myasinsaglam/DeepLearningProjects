from __future__ import division, print_function, absolute_import
from scipy.misc import imresize
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from glob import glob
from tflearn.data_utils import shuffle, to_categorical

files_path = "/Users/Macbook/Desktop/datas/train/"


def file_operations():
    all_files_path = os.path.join(files_path, 'image*.*')

    all_files = sorted(glob(all_files_path))

    n_files = len(all_files)
    print("Total image", n_files)

    size_image = 128

    allX = np.zeros((n_files, size_image, size_image, 3), dtype='float64')
    ally = np.zeros(n_files)
    count = 0
    for f in all_files:
        try:
            img = Image.open(f)
            new_img = imresize(img, (size_image, size_image, 3))
            allX[count] = np.array(new_img)
            ally[count] = count
            count += 1
        except:
            continue
    return allX, ally, n_files

load_test_path="/Users/Macbook/Desktop/image_retrieval_son_deneme/datas/test/"
def load_test_image():
    sample_path=os.path.join(load_test_path,'*.*')
    all_files = sorted(glob(sample_path))
    n_files = len(all_files)
    size_image = 128
    allX = np.zeros((n_files, size_image, size_image, 3), dtype='float64')
    count = 0
    for f in all_files:
        try:
            img = Image.open(f)
            new_img = imresize(img, (size_image, size_image, 3))
            allX[count] = np.array(new_img)
            count += 1
        except:
            continue
    return allX


#X_test_64=load_test_image()
#np.save("TEST.npy",X_test_64)

###################################
# Prepare train & test samples
###################################
allX, ally, n_files = file_operations()
# test-train split
#X, X_test, Y, Y_test = train_test_split(allX, ally, test_size=0.1, random_state=42)

# encode the Ys
#Y = to_categorical(Y, n_files)
np.save("128_X.npy", allX)
np.save("128_Y.npy", ally)
