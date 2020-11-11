from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import cv2, numpy as np
import matplotlib.pyplot as plt
from keras.applications import Xception
from keras.utils import to_categorical
import os
from utils import get_train_x_y
from os.path import isfile, join
import enum

tiles = {"right":0, "left":1, "straight":2, "three_cross":3, "four_cross":4, "empty":5}

if __name__ == "__main__":
    #im = cv2.resize(cv2.imread('/home/andrey/hacaton/orig_2019-12-07-15-24-05/1575732276.099659919.jpg'), (224, 224)).astype(np.float32)

    weight_saver = ModelCheckpoint('duckietown.h5', monitor='val_accuracy',
                                   save_best_only=True, save_weights_only=True)

    model = model = Sequential([
    Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=2),

    Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=2),

    Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=2),

    Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=2),

    Flatten(),
    Dense(units=6, activation='softmax'),
])

    x_data, y_data = get_train_x_y('./data')
   # x_data = x_data[:400]
    x_data = x_data / 255.0
   # y_data = y_data[:400]
    y_data = [tiles[i] for i in y_data]

    y_data = to_categorical(y_data, 6)

    print(x_data[0].shape)
    print(y_data[0])

    train_X, test_X, train_Y, test_Y = train_test_split(x_data, y_data, test_size = 0.2)

    print(train_X.shape)
    print(train_Y.shape)

    sgd = Adam()

    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    hist = model.fit(train_X, train_Y, validation_data=(test_X, test_Y),
                               epochs=15, verbose=1, batch_size=24,
                               callbacks=[weight_saver])
    model.load_weights('duckietown.h5')

    plt.plot(hist.history['loss'], color='b')
    plt.plot(hist.history['val_loss'], color='r')
    plt.show()
    plt.plot(hist.history['accuracy'], color='b')
    plt.plot(hist.history['val_accuracy'], color='r')
    plt.show()

    out = model.predict(im)
    plt.imshow(out)
