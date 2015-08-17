from __future__ import absolute_import
from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD

import os
import pickle
from sklearn import cross_validation

# sentence_embedding_matrix, valence = pickle.load(open(os.path.join('.', 'data', 'tmp', 'NN_input_CVAT.p'), 'rb'))
sentence_embedding_matrix, valence = pickle.load(open('D:/chinese_word2vec/CVAT_sentence_matrix_400.p', 'rb'))

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(sentence_embedding_matrix, valence, test_size=0.2,
                                                                     random_state=0)
print(X_train.shape)
print(len(Y_test))

maxlen = 200  # number of words to count in one sentence
size = 400  # dimension of word embeddings
X_train = X_train.reshape(X_train.shape[0], 1, maxlen, size)
X_test = X_test.reshape(X_test.shape[0], 1, maxlen, size)
print(X_train.shape)

batch_size = 128
nb_epoch = 200
nb_filter = [100, 50]
filter_row = [7, 5]
filter_col = [size, 1]
nb_neuron = 256

model = Sequential()

model.add(Convolution2D(nb_filter[0], 1, filter_row[0], filter_col[0], border_mode='valid'))
model.add(Activation('relu'))
# model.add(Convolution2D(nb_filter[1], nb_filter[0], filter_row[1], filter_col[1], border_mode = 'full'))
# model.add(Activation('sigmoid'))
model.add(MaxPooling2D(poolsize=(2, 1)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(9700, nb_neuron))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))

# model.add(Dense(nb_neuron, nb_neuron))
# model.add(Activation('sigmoid'))
# model.add(Dropout(0.2))

model.add(Dense(nb_neuron, 1))
model.add(Activation('linear'))

sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='mean_squared_error', optimizer='adagrad')
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test)
print('The score:', score)
predict = model.predict(X_test, batch_size=batch_size).reshape((1, len(Y_test)))[0]

pickle.dump((Y_test, predict), open(os.path.join('.', 'data', 'tmp', 'NN_output_CVAT.p'), "wb"))

'''
from evaluate import evaluate
print(Y_test)
print(predict)
evaluate(Y_test, predict, 'Result of CNN')
'''
