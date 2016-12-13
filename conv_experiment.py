from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

model = Sequential()

model.add(Convolution2D(16, 4, 4, input_shape=(1, 80, 80)))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))

sgd = SGD()

model.compile(optimizer=sgd, loss='binary_crossentropy')

print model
