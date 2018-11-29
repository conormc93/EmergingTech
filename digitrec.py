import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import numpy as np

# the data, split between train and test sets
(image_train, label_train), (image_test, label_test) = mnist.load_data()

batch_size = 128
num_labels = 10
epochs = 2


def rsDataset():
    global image_train
    global image_test
    global label_train
    global label_test

    # reshape the data set
    image_train = image_train.reshape(60000, 784)
    image_test = image_test.reshape(10000, 784)

    # convert the data from the images to type float 32
    image_train = image_train.astype('float32')
    image_test = image_test.astype('float32')

    # values are either RGB(0-255)
    # convert the value to either 0 or 1
    image_train /= 255
    image_test /= 255

    # convert class vectors to binary class matrices and output details
    label_train = keras.utils.to_categorical(label_train, num_labels)
    label_test = keras.utils.to_categorical(label_test, num_labels)
    print(label_test, 'label test.')
    print(image_train.shape[0], 'train samples.')
    print(image_test.shape[0], 'test samples.')

