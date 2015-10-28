from __future__ import absolute_import
from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD

from sklearn import cross_validation
import os
from load_data import load_embeddings, load_mark, load_pickle
from load_data import load_corpus
from file_name import get_file_path
from word2vec_fn import build_embedding_matrix, build_sentence_matrix, gold_valence_arousal
from collections import defaultdict
from save_data import dump_picle
from word2vec_fn import make_idx_data
import numpy as np


def get_vocab(corpus):
    vocab = defaultdict(float)
    for sent in corpus:
        for word in sent:
            vocab[word] += 1
    print(len(vocab))
    return vocab

# 注意： 这个文件就是CVAT构造cnn输入数据的代码
########################################## config ########################################
vec_dim = 400
##########################################################################################
corpus = load_corpus(get_file_path('cn_corpus'))
print(corpus[:2])
vocab = get_vocab(corpus)
dump_picle(vocab, get_file_path('CVAT_Vocab'))
print('Dump CVAT vocab OK')
# vocab = load_pickle(get_file_path('CVAT_Vocab'))
for i in vocab:
    print(i)
print(len(vocab))

W, word_idx_map = build_embedding_matrix(load_embeddings('zh_tw'), vocab, k=400)
dump_picle(word_idx_map, get_file_path('word_idx_map_CVAT'))
print('dump word_idx_map successful')
dump_picle(W, './data/tmp/embedding_matrix_CVAT.p')
print('OK')

# word_idx_map = load_pickle(get_file_path('word_idx_map_CVAT'))
mark = load_mark(get_file_path('mark'))
valence, arousal = gold_valence_arousal(corpus, mark)
idx_data = make_idx_data(corpus, word_idx_map, max_len=200, kernel_size=5)

dump_picle([idx_data, valence, arousal], get_file_path('CVAT_processed_data'))
# idx_data, valence, arousal = load_pickle(get_file_path('CVAT_processed_data'))
print(idx_data.shape)
exit()

word_vecs = load_embeddings('zh_tw')

dim = len(word_vecs['我們'])  # 400

embedding_matrix, idx_map = build_embedding_matrix(word_vecs, k=dim)
print(embedding_matrix[1])
print(idx_map['我們'])

print(len(word_vecs['我們']))
print(word_vecs['我們'].shape)

print(build_sentence_matrix(model=word_vecs, sententces=corpus[:2], dim=dim))

print('Result')
sentence_embedding_matrix = build_sentence_matrix(word_vecs, corpus, dim=dim)
print(sentence_embedding_matrix.shape)
print(sentence_embedding_matrix[3], valence[3], arousal[3])

from save_data import dump_picle

dump_picle((sentence_embedding_matrix, valence), get_file_path('CVAT_sentence_matrix_400'))

exit()

'''
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(sentence_embedding_matrix, valence, test_size=0.2,
                                                                     random_state=0)
print(X_train.shape)
print(len(Y_test))

maxlen = 200
size = 50
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

model.add(Dense(128, 1))
model.add(Activation('linear'))

sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='mean_squared_error', optimizer='adagrad')
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test)
print('The score:', score)
predict = model.predict(X_test, batch_size=batch_size).reshape((1, len(Y_test)))[0]

from evaluate import evaluate
print(Y_test)
print(predict)
evaluate(Y_test, predict, 'Result of CNN')
'''
