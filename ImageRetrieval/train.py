
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

###################################
### Import picture files
###################################
size_image = 64
files_path = '/Users/Macbook/Desktop/imageretrievaldataset'

X=np.load("64_X.npy")
Y=np.load("64_Y.npy")
print("Total size :",len(X))
class_size = len(X)
Y=to_categorical(Y,class_size)
###################################
# Image transformations
###################################

# normalisation of images
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Create extra synthetic training data by flipping & rotating images
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)



###################################
# Define network architecture
###################################

# Input is a 64x64 image with 3 color channels (red, green and blue)
network = input_data(shape=[None, size_image, size_image, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

# 1: Convolution layer with 64 filters, each 3x3x3
conv_1 = conv_2d(network, 64, 3, activation='relu', name='conv_1')

# 2: Max pooling layer
network = max_pool_2d(conv_1, 2)

# 3: Convolution layer with 128 filters
conv_2 = conv_2d(network, 128, 3, activation='relu', name='conv_2')

# 4: Convolution layer with 128 filters
conv_3 = conv_2d(conv_2, 128, 3, activation='relu', name='conv_3')

# 5: Max pooling layer
network = max_pool_2d(conv_3, 2,name='features')

# 6: Fully-connected 1024 node layer
network = fully_connected(network, 1024, activation='relu')

# 7: Dropout layer to combat overfitting
network = dropout(network, 0.75)

# 8: Fully-connected layer with 1400 outputs
network = fully_connected(network, class_size, activation='softmax')

# Configure how the network will be trained
acc = Accuracy(name="Accuracy")
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.0001, metric=acc)

# Wrap the network in a model object
model = tflearn.DNN(network, checkpoint_path='model_cbir_9.tflearn', max_checkpoints=3,
                    tensorboard_verbose=3, tensorboard_dir='/Users/Macbook/Desktop/image_retrieval_son_deneme/log/')
###################################
# Train model for 200 epochs
###################################
model.fit(X, Y, batch_size=32,
          n_epoch=200, run_id='model_cbir_9', show_metric=True)

model.save('model_cbir_9_final.tflearn')



#MODEL VGG

"""

network = input_data(shape=[None, size_image, size_image, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 128, 3, activation='relu')
network = conv_2d(network, 128, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, class_size, activation='softmax')

network = regression(network, optimizer='rmsprop',
                     loss='categorical_crossentropy',
                     learning_rate=0.0005)

# Training
model = tflearn.DNN(network, checkpoint_path='model_vgg',
                    max_checkpoints=1, tensorboard_verbose=3,tensorboard_dir='/Users/Macbook/Desktop/tensorboard/tmp/tflearn_logs/')
model.fit(X, Y, n_epoch=250, shuffle=True,
          show_metric=True, batch_size=32, snapshot_step=500,
          snapshot_epoch=False, run_id='vgg_net')
model.save('vgg_model.tflearn')

"""
