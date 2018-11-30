import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.models import load_model
import numpy as np
import timeit
import cv2
from PIL import Image
import os.path

# the data, split between train and test sets
(image_train, label_train), (image_test, label_test) = mnist.load_data()

num_labels = 10


def reshape_dataset():
    global image_train
    global image_test
    global label_train
    global label_test

    # Print out the shapes of the images and set variables values to the shapes
    print("Training/testing Images shape:", image_train.shape, "/", image_test.shape)
    print("Training/testing Labels shape:", label_train.shape, "/", label_test.shape)
    print("First 5 training labels and testing labels:", label_train[:5], "/", label_test[:5])

    # Set no_rows and no_cols to the number of rows and columns in an image
    no_rows = image_train.shape[1]
    no_cols = image_train.shape[2]

    print("There are", no_rows, "rows and", no_cols, "columns in an image")

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


def neural_net_model():
    global model
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

    model.save('model/seq_nn_model')


def prediction():
    global model

    print('\n\t\t\tEnter file name -- Including extension (.png .jpeg)'
          '\n\t\t\tEnter "exit" to return to the Main Menu: \n')

    # Get user input
    user_input = input("\n\n\t\t\tFile(Image) Name: ")

    while user_input != "exit":

        # check for exit condition
        if user_input == "exit":
            print("Returning to main menu...")
            break

        try:
            image = np.invert(Image.open("images/" + user_input))

            # convert images from one color space to another
            # in this instance we want to convert our images to GREY
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # resize our image to 28x28 pixels
            image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)

            # inverts the bits of our image array
            image = cv2.bitwise_not(image)
            image = image.reshape(1, 784)
            image = image.astype('float32')
            image /= 255

            # Loads a model saved via save_model
            model = load_model('model/seq_nn_model')

            # predicts the handwritten digit in the image array
            prediction_array = model.predict(np.array(image, dtype=float))
            # print("\n\t\t\tPrediction:  ", prediction_array, '\n')

            print('\n\t\t\tProbability of a specific number')
            counter = 0
            for predicted in prediction_array[0]:
                print('\t\t\t\tNumber', counter, '=', "%.5f" % (predicted * 100/1), '%')
                counter += 1

            # Returns the indices of the maximum values along an axis.
            # Get index of closest(MAX) prediction
            number_predicted = prediction_array.argmax(axis=1)
            percent = format(max(prediction_array[0]) * 100, '.2f')

            print("\n\t\t\tThe number predicted is:", number_predicted, ",with", percent, '% accuracy')
            print('\n===============================================================================================\n')
            print('\n\t\t\tEnter file name -- Including extension (.png .jpeg)'
                  '\n\t\t\tEnter "exit" to return to the Main Menu: \n')

            # Get user input
            user_input = input('\n\n\t\t\tFile(Image) Name: ')

        except FileNotFoundError:
            print('(ERROR)--> ', user_input, 'file not found.')
            print('\n===============================================================================================\n')
            print('\n\t\t\tEnter file name -- Including extension (.png .jpeg)'
                  '\n\t\t\tEnter "exit" to return to the Main Menu: \n')

            # Get user input
            user_input = input("\n\n\t\t\tFile(Image) Name: ")


def menu():
    model_is_built = False
    if os.path.isfile('model/seq_nn_model'):
        model_is_built = True

    option = True
    while option:

        print("\n\t\tMNIST DIGIT RECOGNITION SCRIPT - Machine Learning Script\n"
              "\n\t\t\t1.\tLoad, Reshape, Model, & Evaluate the MNIST Dataset\n"
              "\t\t\t2.\tMake a prediction on an image\n"
              "\t\t\t3.\tQuit\n")

        option = input("\n\t\t\tChoose (1), (2), or (3)\n")
        if option == "1":
            if model_is_built:
                print('\n\t\tNeural Network Model already built....\n'
                      '\t\tReturning to main menu.\n')
                menu()
            else:
                reshape_dataset()
                neural_net_model()
                model_is_built = True
        elif option == "2":
            if model_is_built:
                prediction()
            else:
                print('\n\t\tNeural Network Model isn`t built....\n'
                      '\t\tSelect option (1)', '\n')
                menu()
        elif option == "3":
            print("\n\t\t\tQuiting the program...")
            exit()
        elif option != "":
            print('\n\t\t\tInvalid Option! Enter either (1), (2), or (3)')


# Launches menu
menu()
