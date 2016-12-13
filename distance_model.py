from keras.callbacks import ProgbarLogger
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
import models

import numpy as np

from models import vggnet

if __name__ == "__main__":
    train_datagen = ImageDataGenerator(

    )

    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(84, 84), classes=['1', '10'],
        batch_size=128,
        class_mode='categorical',
    )

    validation_generator = test_datagen.flow_from_directory(
        'data/validate',
        target_size=(84, 84), classes=['1', '10'],
        batch_size=128,
        class_mode='categorical'
    )

    # weights_path = '/deeplearningresources/vgg16_weights.h5'
    # model = vggnet.VGG_16(weights_path=weights_path)
    # model.layers.pop()
    # model.add(Dense(2, activation='softmax'))
    #
    import models.rlconvnet as rlc

    model = rlc.RLConvNet(input_shape=(3, 84, 84))
    model.layers.pop()
    model.add(Dense(2, activation='softmax'))

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    rmsprop = RMSprop(lr=0.001, decay=1e-6)
    adam = Adam()

    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    tensorboard_callback = TensorBoard()
    progbar_callback = ProgbarLogger()


    model.fit_generator(train_generator, samples_per_epoch=3000, nb_epoch=100, validation_data=validation_generator,
                        nb_val_samples=100, verbose=2,
                        callbacks=[tensorboard_callback, progbar_callback],
                        nb_worker=1)
