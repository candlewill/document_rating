__author__ = 'NLP-PC'
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD

from load_data import load_embeddings, load_mark
from load_data import load_corpus
from file_name import get_file_path
from word2vec_fn import build_embedding_matrix, build_sentence_matrix, gold_valence_arousal
from sklearn import cross_validation

word_vecs = load_embeddings('CVAT')

embedding_matrix, idx_map = build_embedding_matrix(word_vecs, k=50)
print(embedding_matrix[:1])
print(idx_map['我們'])

corpus = load_corpus(get_file_path('cn_corpus'))
print(corpus[:2])
print(build_sentence_matrix(model=word_vecs, sententces=corpus[:2]))

mark = load_mark(get_file_path('mark'))
valence, arousal = gold_valence_arousal(corpus, mark)

print('Result')
sentence_embedding_matrix = build_sentence_matrix(word_vecs, corpus)
print(sentence_embedding_matrix.shape)
print(sentence_embedding_matrix[3], valence[3], arousal[3])

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(sentence_embedding_matrix, valence, test_size=0.2,
                                                                     random_state=0)
print(X_train.shape)

maxlen = 200
size = 50
X_train = X_train.reshape(X_train.shape[0], 1, maxlen, size)
X_test = X_test.reshape(X_test.shape[0], 1, maxlen, size)
print(X_train.shape)

batch_size = 128
nb_epoch = 12

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

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test)
print('Test score:', score[0])
print('Test accuracy:', score[1])
