from __future__ import absolute_import
from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.layers.embeddings import Embedding
from keras.constraints import unitnorm, maxnorm
from keras.regularizers import l2

import numpy as np
import os
import pickle
from sklearn import cross_validation

########################################## config ########################################
vec_dim = 300
max_len = 200
kernel_size = 8
filename = './data/corpus/vader/vader_processed_data_movie_reviews.p'
embedding_maxtrix = './data/corpus/vader/embedding_matrix_movie_reviews.p'
##########################################################################################
idx_data, ratings = pickle.load(open(filename, "rb"))
W = pickle.load(open(embedding_maxtrix, "rb"))
print(W.shape)
print(idx_data.shape)
conv_input_width = W.shape[1]  # embedding dimension
conv_input_height = int(idx_data.shape[1])  # max_len
print(conv_input_width, conv_input_height)

Y = np.array(ratings) + np.ones(len(ratings), dtype=float) * 5
print(Y.shape)
print(Y)

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(idx_data, Y, test_size=0.3,
                                                                     random_state=1)

print(X_train.shape)
print(Y_train.shape)
print(len(Y_test))

maxlen = max_len
size = vec_dim
print(X_train.shape)

# Number of feature maps (outputs of convolutional layer)
N_fm = 300

batch_size = 50
nb_epoch = 100


###################################### model #######################################
def cnn_model_default():
    kernel_size = 8
    model = Sequential()
    model.add(Embedding(input_dim=W.shape[0], output_dim=W.shape[1], weights=[W], W_constraint=unitnorm()))
    model.add(Reshape(1, conv_input_height, conv_input_width))
    model.add(Convolution2D(N_fm, 1, kernel_size, conv_input_width, border_mode='valid', W_regularizer=l2(0.0001)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(conv_input_height - kernel_size + 1, 1), ignore_border=True))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(N_fm, 1))
    # SoftMax activation; actually, Dense+SoftMax works as Multinomial Logistic Regression
    model.add(Activation('linear'))
    # Custom optimizers could be used, though right now standard adadelta is employed
    model.compile(loss='mean_squared_error', optimizer='adagrad')
    return model


def cnn_model_default_improve():
    N_fm = 50
    kernel_size = 5
    model = Sequential()
    model.add(Embedding(input_dim=W.shape[0], output_dim=W.shape[1], weights=[W], W_constraint=unitnorm()))
    model.add(Reshape(1, conv_input_height, conv_input_width))
    model.add(Convolution2D(N_fm, 1, kernel_size, conv_input_width, border_mode='valid', W_regularizer=l2(0.0001)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(conv_input_height - kernel_size + 1, 1), ignore_border=True))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(N_fm, 1))
    model.add(Activation('linear'))
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model


def cnn_0():
    N_fm = 50
    kernel_height = 8
    kernel_width = conv_input_width
    model = Sequential()
    model.add(Embedding(input_dim=W.shape[0], output_dim=W.shape[1], weights=[W], W_constraint=unitnorm()))
    model.add(Reshape(1, conv_input_height, conv_input_width))

    model.add(Convolution2D(N_fm, 1, kernel_height, kernel_width, border_mode='valid', W_regularizer=l2(0.0001)))
    model.add(Activation('relu'))
    model.add(Convolution2D(N_fm, N_fm, kernel_height, 1, border_mode='valid', W_regularizer=l2(0.0001)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(poolsize=(conv_input_height - 2 * kernel_height + 2, 1)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    # model.add(Dense(N_fm, N_fm))
    # model.add(Activation('relu'))
    model.add(Dense(N_fm, 1))
    model.add(Activation('linear'))
    model.compile(loss='mean_squared_error', optimizer='adagrad')
    return model


def cnn_1():
    N_fm = 50
    model = Sequential()
    model.add(Embedding(input_dim=W.shape[0], output_dim=W.shape[1], weights=[W], W_constraint=unitnorm()))
    model.add(Reshape(1, conv_input_height, conv_input_width))
    output_size = (conv_input_height, conv_input_width)

    kernel_height, kernel_width = 8, output_size[1]
    model.add(Convolution2D(N_fm, 1, kernel_height, kernel_width, border_mode='valid', W_regularizer=l2(0.0001)))
    model.add(Activation('relu'))
    output_size = (output_size[0] - kernel_height + 1, output_size[1] - kernel_width + 1)
    model.add(Dropout(0.25))
    kernel_height, kernel_width = 5, 1
    model.add(Convolution2D(N_fm, N_fm, kernel_height, kernel_width, border_mode='valid', W_regularizer=l2(0.0001)))
    model.add(Activation('relu'))
    output_size = (output_size[0] - kernel_height + 1, output_size[1] - kernel_width + 1)

    poolsize = (output_size[0], 1)
    model.add(MaxPooling2D(poolsize=poolsize))
    h = output_size[0] / poolsize[0]
    w = output_size[1] / poolsize[1]
    model.add(Flatten())
    # model.add(Dense(N_fm, N_fm))
    # model.add(Activation('relu'))
    model.add(Dense(N_fm * h * w, 1))
    model.add(Activation('linear'))
    model.add(Dropout(0.25))
    model.compile(loss='mean_squared_error', optimizer='adagrad')
    return model


####################################################################################

model = cnn_model_default_improve()
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test)
print('The score:', score)
predict = model.predict(X_test, batch_size=batch_size).reshape((1, len(Y_test)))[0]

pickle.dump((Y_test, predict), open('./data/corpus/cnn_result.p', "wb"))
