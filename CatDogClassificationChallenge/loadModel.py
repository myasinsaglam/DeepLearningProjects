"""
Based on the tflearn CIFAR-10 example at:
https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_cifar10.py
"""
from __future__ import division, print_function, absolute_import
from sklearn.metrics import confusion_matrix,accuracy_score
from scipy.misc import imresize
import numpy as np
from PIL import Image
from matplotlib.pyplot import imshow,imread
import matplotlib.pyplot as plt
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
import random
train=np.load("/Users/Macbook/Desktop/CatDog Challenge/records/TRAIN.npy")
X_test = np.load("/Users/Macbook/Desktop/CatDog Challenge/records/TEST.npy")
label_Y =np.load("/Users/Macbook/Desktop/CatDog Challenge/records/TEST_LABEL.npy")
test_count = len(X_test)
print("Train Images : ",len(train))
print("Test Images : ",test_count)
print("Total Images : ",len(train)+test_count)
size_image = 64

#Y_test = to_categorical(label_Y, 2)
x=X_test.astype(np.uint8)
labelY=label_Y.astype(np.uint8)
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
network = input_data(shape=[None, 64, 64, 3],data_preprocessing=img_prep,data_augmentation=img_aug)

# 1: Convolution layer with 32 filters, each 3x3x3
conv_1 = conv_2d(network, 32, 3, activation='relu', name='conv_1')

# 2: Max pooling layer
network = max_pool_2d(conv_1, 2)

# 3: Convolution layer with 64 filters
conv_2 = conv_2d(network, 64, 3, activation='relu', name='conv_2')

# 4: Convolution layer with 64 filters
conv_3 = conv_2d(conv_2, 64, 3, activation='relu', name='conv_3')

# 5: Max pooling layer
network = max_pool_2d(conv_3, 2)

# 6: Fully-connected 512 node layer
network = fully_connected(network, 512, activation='relu')

# 7: Dropout layer to combat overfitting
network = dropout(network, 0.5)

# 8: Fully-connected layer with two outputs
network = fully_connected(network, 2, activation='softmax')

# Configure how the network will be trained
acc = Accuracy(name="Accuracy")
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.0005, metric=acc)

# Wrap the network in a model object
model = tflearn.DNN(network, checkpoint_path='model_cat_dog_6.tflearn', max_checkpoints=3,
                    tensorboard_verbose=3, tensorboard_dir='tmp/tflearn_logs/')

model.load('/Users/Macbook/Desktop/CatDog Challenge/model_cat_dog_7_final.tflearn')

img_shape = (64,64,3)

def plot_images(images, cls_true, cls_pred=None):
    fig, axes = plt.subplots(3,3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    error= []
    for index in range(len(x)):
        if cls_true[index]!=cls_pred[index]:
            error.append(index)
            random.shuffle(error)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[error[i]].reshape(img_shape))
        if cls_pred is None:
            xlabel = "True: {0} ".format(cls_true[error[i]])
        else:
            xlabel = "True: {0}, Pred: {1} ".format(cls_true[error[i]],cls_pred[error[i]])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

score = model.evaluate(X_test, label_Y)
print('Training test score', score)
model_predict = model.predict(X_test)
predict = np.array([label.argmax() for label in model_predict])
labelY = np.array([label.argmax() for label in label_Y])
plot_images(images=x, cls_true=labelY, cls_pred=predict)
print(confusion_matrix(labelY,predict))
print(accuracy_score(labelY,predict))
print(len(x))
plt.show()
for i in range(40):
    plot_images(images=x, cls_true=labelY, cls_pred=predict)
    plt.show()
