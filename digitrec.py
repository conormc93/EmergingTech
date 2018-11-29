import timeit
# Import Cv2  and Image for image processing
import cv2
from PIL import Image

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import numpy as np

# the data, split between train and test sets
(image_train, label_train), (image_test, label_test) = mnist.load_data()

num_labels = 10


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
    print('\n', image_train.shape[0], 'train samples.')
    print('\n', image_test.shape[0], 'test samples.')


def nnModel():
    batch_size = 128
    epochs = 1

    # Linear stack of layers
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_labels, activation='softmax'))

    # Print a string summary of the model
    model.summary()

    # Compile, train, and evaluate the model
    # Have looked at SGD, Adagrad, Adelta here https://keras.io/optimizers/
    # RMS seems to be more efficient
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    start_timer = timeit.default_timer()

    log = model.fit(image_train, label_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(image_test, label_test))

    end_timer = timeit.default_timer() - start_timer

    score = model.evaluate(image_test, label_test, verbose=0)

    print('\n\n\t\t\tTest loss(%):', score[0] * 100 / 1)
    print('\n\t\t\tTest accuracy(%):', score[1] * 100 / 1)
    print("\n\t\t\tHow long it took to train model at (", epochs, ") epochs:", end_timer, 'seconds\n')


def prediction():
    input = "\n\t\t\tEnter file name -- Do not enter file extension (.png .jpeg)" \
            "\n\t\t\t'exit' to Main Menu: \n"
    while input != "exit":
        # Get user input
        input = input("\t\t\tFile(Image) Name: \n")

        # check for exit condition
        if input == "exit":
            print("Returning to main menu...")
            break

        # img = Image.open("images/" + input)


def menu():
    option = True
    while option:
        print("\n\t\tMNIST DIGIT RECOGNITION SCRIPT - Machine Learning Script\n"
              "\n\t\t\t1.\tLoad, Reshape, Model, & Evaluate the MNIST Dataset\n"
              "\t\t\t2.\tQuit\n")
        option = input("\n\t\t\tChoose (1) or (2)\n")
        if option == "1":
            rsDataset()
            nnModel()
        elif option == "2":
            print("\n\t\t\tQuiting the program...")
            exit()
        elif option != "":
            print("\n\t\t\tInvalid Option! Enter either (1) or (2).")


menu()
