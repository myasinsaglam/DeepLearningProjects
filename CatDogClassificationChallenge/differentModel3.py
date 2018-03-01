"""
Based on the tflearn CIFAR-10 example at:
https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_cifar10.py
"""
from __future__ import division, print_function, absolute_import
from sklearn.metrics import confusion_matrix
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
size_image = 64
test_count = 25000
###################################
### Import picture files
###################################


"""
test_files_path = '/Users/Macbook/PycharmProjects/kagglecatsanddogs_3367a/imnettest'
def create_test_data():
    count = 0
    allTest = np.zeros((test_count, size_image, size_image, 3), dtype='float64')
    allTest_labels = np.zeros(test_count)
    path = os.path.join(test_files_path,'*.JPEG')
    for img in path:
        try:
            image = Image.open(img)
            new_image = imresize(image,(size_image,size_image,3))
            allTest[count]= np.array(new_image)
            if img.split('_')[0] == "n02121620":
                allTest_labels[count] = 0
            else:
                allTest_labels[count] = 1
            count+=1
        except:
            continue
    print("test image count : ",count)
    allTest_labels = to_categorical(allTest_labels,2)
    np.save("alltest.npy",allTest)
    np.save("alltest_label.npy",allTest_labels)
    return allTest,allTest_labels

files_path = '/Users/Macbook/PycharmProjects/kagglecatsanddogs_3367a/train/'

cat_files_path = os.path.join(files_path, 'cat*.jpg')
dog_files_path = os.path.join(files_path, 'dog*.jpg')

cat_files = sorted(glob(cat_files_path))
dog_files = sorted(glob(dog_files_path))

n_files = len(cat_files) + len(dog_files)
print(n_files)
allX = np.zeros((test_count, size_image, size_image, 3), dtype='float64')
ally = np.zeros(test_count)
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
label_Y =Y_test
Y_test = to_categorical(Y_test, 2)
np.save("X_test.npy",X_test)
np.save("Y_test.npy",label_Y)

"""
X_test = np.load("X_test.npy")
label_Y =np.load("Y_test.npy")
Y_test = to_categorical(label_Y, 2)
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

###################################
# Train model for 100 epochs
###################################
#model.fit(X, Y, validation_set=(X_test, Y_test), batch_size=500,
#          n_epoch=100, run_id='model_cat_dog_6', show_metric=True)

#model.save('model_cat_dog_6_final.tflearn')
new_dir ="/Users/Macbook/PycharmProjects/kagglecatsanddogs_3367a/cattt8.jpeg"
model.load('/Users/Macbook/PycharmProjects/kagglecatsanddogs_3367a/differentmodelsave/model_cat_dog_6_final.tflearn')

img_shape = (64,64,3)

def plot_images(images, cls_true, cls_pred=None):
    fig, axes = plt.subplots(3,3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    error= []
    for index in range(len(x)):
        if cls_true[index]!=cls_pred[index]:
            error.append(index)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[error[i]].reshape(img_shape))
        if cls_pred is None:
            xlabel = "True: {0} ".format(cls_true[error[i]])
        else:
            xlabel = "True: {0}, Pred: {1} ".format(cls_true[error[i]],cls_pred[error[i]])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])


model_predict = model.predict(X_test)
predict = np.array([label.argmax() for label in model_predict])
plot_images(images=x, cls_true=labelY, cls_pred=predict)
print(confusion_matrix(labelY,predict))
plt.show()

"""
new_data = np.zeros((10,size_image,size_image,3),dtype='float64')
lojik = Image.open(new_dir)
new_img = imresize(lojik,(size_image,size_image,3))
imshow(new_img)
plt.show()
new_data[0] = np.array(new_img)
imshow(new_data[0])
plt.show()
predict = model.predict(new_data)
print(predict)
if np.argmax(predict[0])== 0:
    print("cat")
else:
    print("dog")
"""


"""
n_files = 0
for img_name in os.listdir(new_dir):
    n_files+=1
print("Test image number",n_files)
new_data = np.zeros((n_files,size_image, size_image, 3), dtype='float64')
new_data_label = np.zeros(n_files)

i = 0
for img_name in os.listdir(new_dir):
    try:
        img_path = os.path.join(new_dir, img_name)
        img = Image.open(img_path)
        new_img = imresize(img, (size_image, size_image, 3))
        new_data[i] = np.array(new_img)
        imshow(new_data[i])
        plt.show()
        i += 1
    except:
        continue
cat = 0
dog = 0
model_predict = model.predict(new_data)
for i in range(n_files):
    print(model_predict[i])
    if np.argmax(model_predict[i]) == 0:
        print("lojikkk")
        cat +=1
    else:
        dog+=1
        imshow(new_data[i])
        plt.show()
        print("dogg")

print(" cat number : ",cat)
print("dog number : ",dog)


"""
