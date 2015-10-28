from __future__ import absolute_import
from __future__ import print_function
print('OK')
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
import math

########################################## config ########################################
vec_dim = 300
max_len = 200
kernel_size = 8
filename = './data/corpus/vader/vader_processed_data_tweets.p'
embedding_maxtrix = './data/corpus/vader/embedding_matrix_tweets.p'
##########################################################################################
idx_data, ratings = pickle.load(open(filename, "rb"))
W = pickle.load(open(embedding_maxtrix, "rb"))
print('词向量大小: %s' % str(list(W.shape)))
print('句向量空间: %s' % str(idx_data.shape))
conv_input_width = W.shape[1]  # embedding dimension
conv_input_height = int(idx_data.shape[1])  # max_len
print('词向量维度：%s；句子长度：%s' % (conv_input_width, conv_input_height))

# Y = np.array(ratings) + np.ones(len(ratings), dtype=float) * 5
Y = ratings
print('语料库大小: %s' % str(Y.shape))
print('标记示例：%s' % str(Y[:100]))

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(idx_data, Y, test_size=0.2,
                                                                     random_state=7)

print('训练数据： %s' % str(X_train.shape))
print(Y_train.shape)
print('测试数据数量：%s' % str(len(Y_test)))
print('测试数据shape： %s' % str(X_test.shape))
print(X_test[0].shape)

maxlen = max_len
size = vec_dim

# Number of feature maps (outputs of convolutional layer)
N_fm = 150

batch_size = 64
nb_epoch = 20


###################################### model #######################################
def cnn_model_default():
    N_fm = 150
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
    N_fm = 300 # number of filters
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
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer='adagrad')
    return model

def cnn_model_default_improve_2():
    N_fm = 300 # number of filters
    kernel_size = 5
    model = Sequential()
    model.add(Embedding(input_dim=W.shape[0], output_dim=W.shape[1], weights=[W], W_constraint=unitnorm()))
    model.add(Reshape(dims=(1, conv_input_height, conv_input_width)))
    model.add(Convolution2D(nb_filter=N_fm,
                            nb_row=kernel_size,
                            nb_col=conv_input_width,
                            border_mode='valid',
                            W_regularizer=l2(0.0001)))
    model.add(Activation("sigmoid"))
    model.add(MaxPooling2D(pool_size=(conv_input_height - kernel_size + 1, 1), ignore_border=True))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('linear'))
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer='adagrad')
    return model

def cnn_model_default_improve_3():
    N_fm = 300 # number of filters
    kernel_size = 5
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
    model.add(Dropout(0.6))
    model.add(Dense(1))
    model.add(Activation('linear'))
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer='adagrad')
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


def cnn_model_2():
    N_fm = 100
    model = Sequential()
    model.add(Embedding(input_dim=W.shape[0], output_dim=W.shape[1], weights=[W], W_constraint=unitnorm()))
    model.add(Reshape(1, conv_input_height, conv_input_width))
    model.add(Convolution2D(N_fm, 1, 5, 5, border_mode='valid', W_regularizer=l2(0.0001)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(2, 2), ignore_border=True))

    model.add(Convolution2D(N_fm, N_fm, 5, 5, border_mode='valid', W_regularizer=l2(0.0001)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(2, 2), ignore_border=True))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(N_fm * (((conv_input_height - 4) / 2 - 4) / 2) * (((conv_input_width - 4) / 2 - 4) / 2), 1))
    model.add(Activation('linear'))
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer='adagrad')
    return model


def cnn_model_3():
    N_fm = 100
    model = Sequential()
    model.add(Embedding(input_dim=W.shape[0], output_dim=W.shape[1], weights=[W], W_constraint=unitnorm()))
    model.add(Reshape(1, conv_input_height, conv_input_width))
    model.add(Convolution2D(N_fm, 1, 7, 7, border_mode='valid', W_regularizer=l2(0.0001)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(3, 3), ignore_border=True))

    model.add(Convolution2D(N_fm, N_fm, 5, 5, border_mode='valid', W_regularizer=l2(0.0001)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(2, 2), ignore_border=True))

    model.add(Flatten())
    model.add(Dropout(0.5))
    h = math.floor((math.floor((conv_input_height - 6) / 3) - 4) / 2)
    w = math.floor((math.floor((conv_input_width - 6) / 3) - 4) / 2)
    model.add(Dense(N_fm * h * w, 1))
    model.add(Activation('linear'))
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer='adagrad')
    return model


def cnn_model_simple():
    N_fm = 10
    model = Sequential()
    model.add(Embedding(input_dim=W.shape[0], output_dim=W.shape[1], weights=[W], W_constraint=unitnorm()))
    model.add(Reshape(1, conv_input_height, conv_input_width))
    model.add(Convolution2D(N_fm, 1, 5, conv_input_width, border_mode='valid', W_regularizer=l2(0.0001)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(conv_input_height - 5 + 1, 1), ignore_border=True))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(N_fm, 1))
    model.add(Activation('linear'))
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer='adagrad')
    return model

####################################################################################
if __name__ == '__main__':
    model = cnn_model_default_improve_3()
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_test, Y_test))
    model.save_weights('./data/corpus/vader/cnn_model_weights.hdf5', overwrite=True)
    print('The weights of CNN have been saved!')
    print('Starting to predicting...')
    score = model.evaluate(X_test, Y_test)

    print('The score:', score)
    predict = model.predict(X_test, batch_size=batch_size).reshape((1, len(Y_test)))[0]

    pickle.dump((Y_test, predict), open('./data/corpus/vader/cnn_result.p', "wb"))
