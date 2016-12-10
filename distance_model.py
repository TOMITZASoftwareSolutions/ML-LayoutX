from keras.callbacks import ProgbarLogger
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard

import numpy as np

from models import vggnet

if __name__ == "__main__":
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255, )

    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='sparse',
    )

    validation_generator = test_datagen.flow_from_directory(
        'data/validate',
        target_size=(224, 224),
        batch_size=32,
        class_mode='sparse')

    weights_path = '/deeplearningresources/vgg16_weights.h5'
    model = vggnet.VGG_16(weights_path=None)
    model.layers.pop()
    model.add(Dense(1, activation='linear'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='mse')

    tensorboard_callback = TensorBoard()

    model.fit_generator(train_generator, samples_per_epoch=500, nb_epoch=100, validation_data=validation_generator,
                        nb_val_samples=100, callbacks=[tensorboard_callback], nb_worker=1)
