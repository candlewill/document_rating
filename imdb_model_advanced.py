from __future__ import absolute_import
from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.constraints import unitnorm, maxnorm
from keras.layers.embeddings import Embedding
from keras.regularizers import l2

import numpy as np
import os
import pickle
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score

########################################## config ########################################
vec_dim = 300
max_len = 200
kernel_size = 8
filename = './data/tmp/imdb_processed_data.p'
embedding_maxtrix = 'D:/DATA/embedding_matrix.p'
##########################################################################################
dataset, pos_length, neg_length = pickle.load(open(filename, "rb"))
W = pickle.load(open(embedding_maxtrix, "rb"))
print(W.shape)
print(dataset.shape)
conv_input_width = W.shape[1]  # embedding dimension
conv_input_height = int(dataset.shape[1])  # max_len
print(conv_input_width, conv_input_height)

Y = np.concatenate((np.ones((1, pos_length)), np.zeros((1, neg_length))), axis=1).T

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(dataset, Y, test_size=0.2,
                                                                     random_state=0)
nb_classes = 2
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

print(X_train.shape)
print(Y_train.shape)
print(len(Y_test))

maxlen = max_len
size = vec_dim
print(X_train.shape)

# Number of feature maps (outputs of convolutional layer)
N_fm = 300

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


epoch = 0
val_acc = []
val_auc = []

# Train model for N_epoch epochs (could be run as many times as needed)

N_epoch = 3

for i in range(N_epoch):
    model.fit(X_train, Y_train, batch_size=50, nb_epoch=1, verbose=1, show_accuracy=True)
    output = model.predict_proba(X_test, batch_size=10, verbose=1)
    # find validation accuracy using the best threshold value t
    vacc = np.max(
        [np.sum((output[:, 1] > t) == (Y_test[:, 1] > 0.5)) * 1.0 / len(output) for t in np.arange(0.0, 1.0, 0.01)])
    # find validation AUC
    vauc = roc_auc_score(Y_test, output[:, 0])
    val_acc.append(vacc)
    val_auc.append(vauc)
    print('Epoch {}: validation accuracy = {:.3%}, validation AUC = {:.3%}'.format(epoch, vacc, vauc))
    epoch += 1

print('{} epochs passed'.format(epoch))
print('Accuracy on validation dataset:')
print(val_acc)
print('AUC on validation dataset:')
print(val_auc)

# Save model
model.save_weights('cnn_3epochs.model')

'''
from evaluate import evaluate
print(Y_test)
print(predict)
evaluate(Y_test, predict, 'Result of CNN')
'''
