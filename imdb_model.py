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
# windows_dir = 'E:/研究/Data/IMDB/aclImdb/train/'
windows_dir = 'D:/data/'
file_dir = windows_dir if os.name == 'nt' else '/home/hs/Data/imdb/aclImdb/train/'
vec_dim = 300
##########################################################################################
pos_sentences_matrix, pos_length = pickle.load(open(os.path.join(windows_dir, 'pos.p'), 'rb'))
print(pos_sentences_matrix[2], pos_sentences_matrix.shape)
neg_sentences_matrix, neg_length = pickle.load(open(os.path.join(windows_dir, 'neg.p'), 'rb'))
print(neg_sentences_matrix[3], neg_sentences_matrix.shape)
print('load complete ')
sentence_embedding_matrix = np.concatenate((pos_sentences_matrix, neg_sentences_matrix), axis=0)
valence = np.concatenate((np.ones((1, pos_length)), np.zeros((1, neg_length))), axis=1).T

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(sentence_embedding_matrix, valence, test_size=0.2,
                                                                     random_state=0)
nb_classes = 2
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

print(X_train.shape)
print(len(Y_test))

maxlen = 200
size = vec_dim
X_train = X_train.reshape(X_train.shape[0], 1, maxlen, size)
X_test = X_test.reshape(X_test.shape[0], 1, maxlen, size)
print(X_train.shape)

batch_size = 128
nb_epoch = 1

model = Sequential()

model.add(Convolution2D(32, 1, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(32, 32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(72128, 128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(128, nb_classes))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='mean_squared_error', optimizer='adagrad')
model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode="binary")

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
