from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D


def RLConvNet(input_shape=(3, 224, 224)):
    model = Sequential()
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 4, 4, subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(20, activation='softmax'))

    return model
