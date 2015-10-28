__author__ = 'hs'
from load_data import load_pickle
from preprocess_imdb import clean_str
from word2vec_fn import get_idx_from_sent

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.layers.embeddings import Embedding
from keras.constraints import unitnorm
from keras.regularizers import l2

def cnn(text=None):
    request_text = text
    # Test
    [idx_data, ratings] = load_pickle('./data/corpus/vader/vader_processed_data_tweets.p')
    # print(idx_data[2])
    # print(ratings[2])

    W = load_pickle('./data/corpus/vader/embedding_matrix_tweets.p')
    # print(len(W[1]))
    if request_text is None:
        request_text = 'why you are not happy'
    request_text = clean_str(request_text)
    # print(request_text)
    word_idx_map = load_pickle('./data/corpus/vader/word_idx_map_tweets.p')

    idx_request_text = get_idx_from_sent(request_text, word_idx_map)
    # print(idx_request_text)  # type: list
    max_len = len(idx_request_text)
    idx_request_text = np.array(idx_request_text).reshape((1,max_len))
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
    model.load_weights('./data/corpus/vader/cnn_model_weights.hdf5')
    predict_value = model.predict(idx_request_text)

    print(predict_value)
