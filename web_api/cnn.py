#coding: utf-8
__author__ = 'hs'
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.layers.embeddings import Embedding
from keras.constraints import unitnorm
from keras.regularizers import l2

import pickle
import jieba
from string import punctuation


def load_pickle(filename):
    out = pickle.load(open(filename, "rb"))
    return out


def clean_str(sentence):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    for p in list(punctuation):
        sentence = sentence.replace(p, '')
    return sentence.strip().lower()


def get_idx_from_sent(sent, word_idx_map, max_l=200, kernel_size=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = kernel_size - 1
    for i in range(pad):
        x.append(0)
    if type(sent) is not list:
        words = sent.split()
    else:
        words = sent
    for num, word in enumerate(words, 1):
        if word in word_idx_map:
            x.append(word_idx_map[word])
        if num > max_l:
            break
    while len(x) < max_l + 2 * pad:
        x.append(0)
    return x


def cnn(text=None):
    ########################### file_path ##############################
    embedding_matrix = './embedding_matrix_tweets.p'
    word_idx_map = './word_idx_map_tweets.p'
    cnn_model_weights = './cnn_model_weights.hdf5'
    ####################################################################
    request_text = text
    W = load_pickle(embedding_matrix)
    # print(len(W[1]))
    if request_text is None:
        request_text = 'why you are not happy'
    request_text = clean_str(request_text)
    # print(request_text)
    word_idx_map = load_pickle(word_idx_map)

    idx_request_text = get_idx_from_sent(request_text, word_idx_map)
    # print(idx_request_text)  # type: list
    max_len = len(idx_request_text)
    idx_request_text = np.array(idx_request_text).reshape((1, max_len))
    # print(idx_request_text.shape)

    def cnn_model():
        N_fm = 100  # number of filters
        kernel_size = 5
        conv_input_height, conv_input_width = max_len, len(W[1])

        model = Sequential()
        model.add(Embedding(input_dim=W.shape[0], output_dim=W.shape[1], weights=[W], W_constraint=unitnorm()))
        model.add(Reshape(dims=(1, conv_input_height, conv_input_width)))
        model.add(Convolution2D(nb_filter=N_fm,
                                nb_row=kernel_size,
                                nb_col=conv_input_width,
                                border_mode='valid',
                                W_regularizer=l2(0.0001)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(conv_input_height - kernel_size + 1, 1), ignore_border=True))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('linear'))
        sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='mse', optimizer='adagrad')
        return model

    model = cnn_model()
    model.load_weights(cnn_model_weights)
    predict_value = model.predict(idx_request_text)

    return [predict_value[0][0], 5.0]

def cnn_Chinese(text=None):
    ########################### file_path ##############################
    embedding_matrix = './embedding_matrix_CVAT.p'
    word_idx_map = './word_idx_map_CVAT.p'
    cnn_model_weights_Valence = './CVAT_cnn_model_weights_Valence.hdf5'
    cnn_model_weights_Arousal = './CVAT_cnn_model_weights_Arousal.hdf5'
    ####################################################################
    request_text = text
    W = load_pickle(embedding_matrix)
    # print(len(W[1]))
    if request_text is None:
        request_text = '中文斷詞前言自然語言處理的其中一個重要環節就是中文斷詞的'
    # request_text = clean_str(request_text)
    # print(request_text)
    request_text = list(jieba.cut(request_text))
    word_idx_map = load_pickle(word_idx_map)

    idx_request_text = get_idx_from_sent(request_text, word_idx_map)
    # print(idx_request_text)  # type: list
    max_len = len(idx_request_text)
    idx_request_text = np.array(idx_request_text).reshape((1, max_len))
    # print(idx_request_text.shape)

    def cnn_model():
        N_fm = 400  # number of filters
        kernel_size = 8
        conv_input_height, conv_input_width = max_len, len(W[1])

        model = Sequential()
        model.add(Embedding(input_dim=W.shape[0], output_dim=W.shape[1], weights=[W], W_constraint=unitnorm()))
        model.add(Reshape(dims=(1, conv_input_height, conv_input_width)))
        model.add(Convolution2D(nb_filter=N_fm,
                                nb_row=kernel_size,
                                nb_col=conv_input_width,
                                border_mode='valid',
                                W_regularizer=l2(0.0001)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(conv_input_height - kernel_size + 1, 1), ignore_border=True))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('linear'))
        model.compile(loss='mse', optimizer='adagrad')
        return model

    model = cnn_model()
    model.load_weights(cnn_model_weights_Valence)
    valence = model.predict(idx_request_text)

    model.load_weights(cnn_model_weights_Arousal)
    arousal = model.predict(idx_request_text)

    return [valence[0][0], arousal[0][0]]
#
def CNN_VA_prediction(text=None):
    def is_English(text=None):
        if text is not None:
            try:
                text.encode('ascii')
            except UnicodeEncodeError:
                # print("it was not a ascii-encoded unicode string")
                return False
            else:
                # print("It may have been an ascii-encoded unicode string")
                return True
        else:
            # print('The input string is None.')
            return

    if is_English(text):
        return cnn(text)
    else:
        return cnn_Chinese(text)
if __name__ == "__main__":
    text = '我今天特別高興'
    print(CNN_VA_prediction(text))

    text = 'appy B-day Jim Price!! :-) (you are more awesome than you could dream) Hope today was the best ever!!  :-D'
    print(CNN_VA_prediction(text))