from __future__ import division, print_function, absolute_import
from sklearn.metrics import confusion_matrix,accuracy_score
from scipy.misc import imresize
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
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
load_test_path="/Users/Macbook/Desktop/datas/test/"

all =np.load("128_X.npy")
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


X_test_128=load_test_image()
#np.save("TEST.npy",X_test)
X_test=np.load("TEST.npy")
n_files=len(X_test)
print("Total test images : ",n_files)
#X_test = np.load("/Users/Macbook/Desktop/CatDog Challenge/records/ALL_X.npy")
#label_Y =np.load("/Users/Macbook/Desktop/CatDog Challenge/records/ALL_Y.npy")
#test_count = len(X_test)+len(label_Y)


#Y_test = to_categorical(label_Y, 2)
test=X_test_128.astype(np.uint8)
x =all.astype(np.uint8)

imsize=64

#labelY=label_Y.astype(np.uint8)
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
img_aug.add_random_rotation(max_angle = 89.)
img_aug.add_random_blur(sigma_max=3.)
img_aug.add_random_flip_updown()
img_aug.add_random_90degrees_rotation(rotations = [0, 1, 2, 3])


###################################
# Define network architecture
###################################

# Input is a 64x64 image with 3 color channels (red, green and blue)
network = input_data(shape=[None,64,64,3],data_preprocessing=img_prep,data_augmentation=img_aug)

# 1: Convolution layer with 32 filters, each 3x3x3
conv_1 = conv_2d(network, 64, 3, activation='relu', name='conv_1')

# 2: Max pooling layer
network = max_pool_2d(conv_1, 2)

# 3: Convolution layer with 64 filters
conv_2 = conv_2d(network, 128, 3, activation='relu', name='conv_2')

# 4: Convolution layer with 64 filters
conv_3 = conv_2d(conv_2, 128, 3, activation='relu', name='conv_3')

# 5: Max pooling layer
network = max_pool_2d(conv_3, 2)

# 6: Fully-connected 512 node layer
network = fully_connected(network, 1024, activation='relu')

# 7: Dropout layer to combat overfitting
network = dropout(network, 0.75)

# 8: Fully-connected layer with two outputs
network = fully_connected(network, 1400, activation='softmax')

# Configure how the network will be trained
acc = Accuracy(name="Accuracy")
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.0001, metric=acc)

# Wrap the network in a model object
model = tflearn.DNN(network, checkpoint_path='model_cbir_17.tflearn', max_checkpoints=3,
                    tensorboard_verbose=3)

model.load('/Users/Macbook/Desktop/image_retrieval_son_deneme/model_cbir_15_final.tflearn')


def selectionSort(alist, k):
    newList = []
    for fillslot in range(k):
        positionOfMax = 0
        for location in range(fillslot, len(alist)):
            if alist[location] > alist[positionOfMax] and (location not in newList):
                positionOfMax = location
        newList.append(positionOfMax)
    return newList

img_shape = (64,64,3)

k=3
model_predict = model.predict(X_test)
#print(model_predict)
predict = np.array([label.argmax() for label in model_predict])
predict2 = np.array([np.argpartition(label,-k)[-k:] for label in model_predict])
print(predict2[0][0])
print(predict[0])
#plot_images(x,test)
true = 0
for i in range(len(test)):
    fig = plt.figure()
    #fig.suptitle('Result : '+ str(i))
    a = fig.add_subplot(4,4,1)
    plt.imshow(test[i])
    a.set_title('Test : '+ str(i))
    predict3 = selectionSort(model_predict[i], k)
    for j in range(k):
        predict2[i] = predict2[i][::-1]
        print("Test :",i,"Similiar :",j+1,"Index : ",predict3[j])
        print("Probability : ",model_predict[i][predict3[j]]*100)
        title="Similiar :"+str(j+1)+"\n%"+str(round(model_predict[i][predict3[j]]*100,2))
        a = fig.add_subplot(4,4,j+2)
        #plt.imshow(x[predict3[j]])
        a.set_title(title)
        limit = int(i/20)
        print("Mod : ",limit)
        if predict3[j]>=limit*200 and predict3[j] < (limit+1)*200:
            true+=1
            #print("Test index : ",i)
            #print("index : ",predict3[j])
            print("True : ",true)
    #plt.show()
result = (true/140*k)*100
print("Acc = ",result,"True : ",true)

##p =np.argpartition(model_predict,-k)[-k:]
#predict= np.array([label.argpartition()] for label in model_predict)

"""
score = model.evaluate(X_test, label_Y)
print('Training test score', score)
labelY = np.array([label.argmax() for label in label_Y])
plot_images(images=x, cls_true=labelY, cls_pred=predict)
print(confusion_matrix(labelY,predict))
print(accuracy_score(labelY,predict))
print(len(x))
plt.show()
for i in range(10):
    plot_images(images=x, cls_true=labelY, cls_pred=predict)
    plt.show()

"""
