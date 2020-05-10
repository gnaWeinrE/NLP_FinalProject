import random
import Model
from Preprocess import load_files, load_preprocessed_data
import numpy as np
import tensorflow as tf
import spacy
from Model import batch_generator

import time

import keras

from keras.models import Sequential

from keras.layers import Embedding, Activation, Bidirectional, LSTM, Dense, TimeDistributed, Dropout, Conv1D, \
    MaxPooling2D, Flatten, MaxPooling1D, GlobalMaxPooling1D, BatchNormalization, Conv2D, GRU, LeakyReLU

if __name__ == "__main__":

    tf.config.list_physical_devices('GPU')

    batch_size = 64
    epochs = 20

    train_data = load_preprocessed_data('train.txt')
    test_data = load_preprocessed_data('test.txt')
    dev_data = load_preprocessed_data('dev.txt')

    t0 = time.perf_counter()

    glove = dict()

    with open('glove.6B.100d.txt', encoding="utf-8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            glove[word] = coefs

    print(time.perf_counter() - t0)
    print('glove pretrained embedding loaded')
    t0 = time.perf_counter()


    for i in range(10):
        print(train_data[i])

    x_train, y_train = Model.create_data(train_data, glove)

    x_dev, y_dev = Model.create_data(dev_data, glove)

    train_batch = Model.batch_generator(train_data, glove, batch_size=batch_size)

    dev_batch = Model.batch_generator(dev_data, glove, batch_size=batch_size)

    print(x_train.shape)
    print(y_train.shape)

    print(time.perf_counter() - t0)
    print('train data created')
    t0 = time.perf_counter()

    model = Model.correction()

    model.compile(optimizer=keras.optimizers.Adadelta(rho=0.9), loss='binary_crossentropy', metrics=['accuracy'])

    History = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_dev, y_dev))

    # model.fit_generator(train_batch,    epochs=epochs, steps_per_epoch=len(train_data)/batch_size,
    # validation_data = dev_batch , validation_steps = len(dev_data))

    model.save('model.h5')

    print(History.history['val_loss'])
    print(History.history['val_accuracy'])
