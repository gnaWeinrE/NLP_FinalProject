import keras
import numpy as np
from keras.models import Sequential,load_model
import Model
import spacy
import sys
import Wordpool
from keras.layers import Embedding, Activation, Bidirectional, LSTM, Dense, TimeDistributed, Dropout, Conv1D, \
    MaxPooling2D, Flatten, MaxPooling1D, GlobalMaxPooling1D, BatchNormalization, Conv2D, GRU, LeakyReLU




if __name__ == "__main__":


    model = load_model('model.h5')

    nlp = spacy.load("en_core_web_sm")


    glove = dict()


    with open('glove.6B.100d.txt', encoding="utf-8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            glove[word] = coefs

    
    act = sys.argv[1]

    sentence = ' '.join(sys.argv[2:])

    #print(act,sentence)


    if act == 'p':
        print(Model.predict_sentence(sentence, glove, model, nlp))


    elif act == 'c':
        print(Model.correct(sentence, glove, model, nlp))


