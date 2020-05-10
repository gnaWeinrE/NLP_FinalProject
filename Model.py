import keras

import Wordpool
from keras.models import Sequential
from spellchecker import SpellChecker
from keras.layers import Embedding, Activation, Bidirectional, LSTM, Dense, TimeDistributed, Dropout, Conv1D, \
    MaxPooling2D, Flatten, MaxPooling1D, GlobalMaxPooling1D, BatchNormalization, Conv2D, GRU, LeakyReLU

import numpy as np

import random


def embed(sent, glove):
    embedded = []
    for word in sent:
        if word.lower() in glove:
            embedded.append(glove[word.lower()])
        else:
            unk = []
            for i in range(100):
                unk.append(random.random() * 2 - 1)

            embedded.append(unk)

    return embedded


def pad_sequences(batch_x, pad_value):
    pad_length = len(max(batch_x, key=lambda x: len(x)))
    for i, x in enumerate(batch_x):
        if len(x) < pad_length:
            batch_x[i] = x + ([pad_value] * (pad_length - len(x)))

    return batch_x


def batch_generator(data, glove, batch_size=1):
    while True:
        batch_x = []
        batch_y = []
        for i in range(len(data)):
            if i % 2 == 1:
                sent1 = data[i - 1]
                sent2 = data[i]

                if len(sent2) == 0:

                    batch_x.append(embed(sent1, glove))
                    batch_y.append([1])

                else:
                    batch_x.append(embed(sent1, glove))
                    batch_y.append([0])

                    batch_x.append(embed(sent2, glove))
                    batch_y.append([1])

            if len(batch_x) >= batch_size:
                batch_x = pad_sequences(batch_x, [0] * 100)

                yield np.array(batch_x), np.array(batch_y)
                batch_x = []
                batch_y = []


def create_data(data, glove):
    batch_x = []
    batch_y = []

    for i in range(len(data)):
        if i % 2 == 1:
            sent1 = data[i - 1]
            sent2 = data[i]

            if len(sent1) <= 40:
                if len(sent2) == 0:

                    batch_x.append(embed(sent1, glove))
                    batch_y.append([1])

                else:
                    batch_x.append(embed(sent1, glove))
                    batch_y.append([0])

                    batch_x.append(embed(sent2, glove))
                    batch_y.append([1])

    for i, x in enumerate(batch_x):
        if len(x) < 40:
            batch_x[i] = x + [[0] * 100] * (40 - len(x))
        else:
            batch_x[i] = x[:40]

    return np.array(batch_x), np.array(batch_y)


def correction():
    model = Sequential()


    '''

    model.add(Conv1D(32, 3, padding='valid', activation='relu',strides=1))


    model.add(MaxPooling1D(pool_size=4))
    

  

    


    '''

    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(BatchNormalization())

    model.add(Bidirectional(LSTM(64, return_sequences=False)))
    model.add(BatchNormalization())

    model.add(Dropout(0.3))

    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))


    model.add(Dense(1, activation='sigmoid'))

    return model


def predict_sentence(sentence, glove, model, nlp):
    sent = nlp(sentence)
    nsent = embed([i.text for i in sent], glove)
    nsent = nsent + [[0] * 100] * (40 - len(nsent))

    p_data = np.array([nsent])

    p = model.predict(p_data)

    return p


def correct(sentence, glove, model, nlp):
    spell = SpellChecker()

    new_sentence = sentence

    sent = nlp(new_sentence)
    val = predict_sentence(new_sentence, glove, model, nlp)

    temp_sentence = ''
    temp_val = 0

    new_val = val

    i = 0

    while i < len(sent):

        s1 = ' '.join([t.text for t in sent[0:i]])
        s2 = ' '.join([t.text for t in sent[i + 1:len(sent)]])

        word = sent[i]
        new_sent = sent[i]

        former_pos = ''

        if i >0:
            former_pos = sent[i-1].pos_

        pool = Wordpool.wordpool(word.text,word.pos_,former_pos,word.dep_,spell)

        for w in pool:
            #print(w)
            temp_sentence = s1 + ' ' + w + ' ' + s2
            temp_val = predict_sentence(temp_sentence, glove, model, nlp)

            if temp_val > new_val:
                new_val = temp_val
                new_sentence = temp_sentence

        if new_val > val:
            val = new_val
            i = 0
            sent = nlp(new_sentence)

        i = i + 1

    return new_sentence
