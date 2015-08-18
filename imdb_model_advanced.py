from __future__ import absolute_import
from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD

import numpy as np
import os
import pickle
from sklearn import cross_validation

########################################## config ########################################
vec_dim = 300
max_len = 200
kernel_size = 5
filename = './data/tmp/imdb_processed_data.p'
##########################################################################################
database, pos_length, neg_length = pickle.load(open(filename, "rb"))
Y = np.concatenate((np.ones((1, pos_length)), np.zeros((1, neg_length))), axis=1).T

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(database, Y, test_size=0.2,
                                                                     random_state=0)
nb_classes = 2
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

print(X_train.shape)
print(len(Y_test))

maxlen = max_len
size = vec_dim
X_train = X_train.reshape(X_train.shape[0], 1, maxlen, size)
X_test = X_test.reshape(X_test.shape[0], 1, maxlen, size)
print(X_train.shape)

batch_size = 128
nb_epoch = 1

###################################### model #######################################
model = Sequential()
# Embedding layer (lookup table of trainable word vectors)
model.add(Embedding(input_dim=W.shape[0], output_dim=W.shape[1], weights=[W], W_constraint=unitnorm()))
# Reshape word vectors from Embedding to tensor format suitable for Convolutional layer
model.add(Reshape(1, conv_input_height, conv_input_width))

# first convolutional layer
model.add(Convolution2D(N_fm, 1, kernel_size, conv_input_width, border_mode='valid', W_regularizer=l2(0.0001)))
# ReLU activation
model.add(Activation('relu'))

# aggregate data in every feature map to scalar using MAX operation
model.add(MaxPooling2D(poolsize=(conv_input_height - kernel_size + 1, 1), ignore_border=True))

model.add(Flatten())
model.add(Dropout(0.5))
# Inner Product layer (as in regular neural network, but without non-linear activation function)
model.add(Dense(N_fm, 2))
# SoftMax activation; actually, Dense+SoftMax works as Multinomial Logistic Regression
model.add(Activation('softmax'))

# Custom optimizers could be used, though right now standard adadelta is employed
model.compile(loss='categorical_crossentropy', optimizer='adadelta')
####################################################################################





model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test)
print('The score:', score)
exit()
predict = model.predict(X_test, batch_size=batch_size).reshape((1, len(Y_test)))[0]

pickle.dump((Y_test, predict), open(os.path.join('.', 'data', 'tmp', 'NN_output_CVAT.p'), "wb"))

'''
from evaluate import evaluate
print(Y_test)
print(predict)
evaluate(Y_test, predict, 'Result of CNN')
'''
