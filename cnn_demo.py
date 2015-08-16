from __future__ import absolute_import
from __future__ import print_function
import codecs
import re
import csv
import pickle

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from sklearn import cross_validation
import gensim


def load_corpus(corpus_dir):
    file_dir = os.listdir(corpus_dir)
    corpus_data = []
    file_id = []
    for file_name in file_dir:
        file_id.append(int(file_name.strip().split('.')[0]))
    file_id.sort()
    for id in file_id:
        text = codecs.open(os.path.join(corpus_dir, str(id) + '.txt'), 'r', 'utf-8').readlines()
        file_text = []
        for sentence in text:
            words = sentence.split(u'\u3000')  # blank space
            for word in words:
                word = re.sub(r'\(.*\)', '', word).strip().replace(u'\u3000', u'')
                if word is not u'':
                    file_text.append(word)
        corpus_data.append(file_text)
    return corpus_data


def load_mark(filename):
    fr = codecs.open(filename)

    mark_data = []
    for line in fr.readlines():
        line = line.strip().split(',')

        mark_data.append([int(line[0]), float(line[1]), float(line[2]), \
                          int(line[3]), line[4]])
    return mark_data


def load_lexicon(filename):
    fr = codecs.open(filename, 'r', 'utf-8')

    lexicon_data = []
    for line in fr.readlines():
        line = line.strip().split(',')

        lexicon_data.append([line[0], float(line[1]), float(line[2])])
    return lexicon_data


def combine_lexicon(lexicon_name, expand_name):
    lexicon_data = load_lexicon(lexicon_name)

    fr = codecs.open(expand_name, 'r', 'utf-8')
    for line in fr.readlines():
        line = line.strip().split()
        lexicon_data.append([line[0], float(line[1]), float(line[2])])

    return lexicon_data


def load_anew(filepath=None):
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        words, arousal, valence = [], [], []
        for line in reader:
            words.append(line[0])
            valence.append(float(line[1]))
            arousal.append(float(line[2]))
    return words, valence, arousal


def load_pickle(filename):
    out = pickle.load(open(filename, "rb"))
    return out


def load_embeddings(arg=None):
    if arg == 'zh_tw':  # dim = 400
        model = gensim.models.Word2Vec.load_word2vec_format(get_file_path('cn_word2vec'), binary=False)
    elif arg == 'CVAT':  # dim = 50
        model = gensim.models.Word2Vec.load(get_file_path('wordvecs_CVAT'))
    elif arg == 'IMDb':  # dim = 100
        model = Doc2Vec.load(get_file_path('test_doc2vec_model'))
    elif arg == 'CVAT_docvecs':  # dim = 50
        model = Doc2Vec.load(get_file_path('docvecs_CVAT'))
    else:
        raise Exception('Wrong Argument.')
    print('Load Model Complete.')
    return model


import os


def get_file_path(filename=None):
    out = None
    os_name = os.name
    if filename == 'cn_corpus':
        out = os.path.join('.', 'data', 'corpus', 'cn', 'corpus_raw')
    elif filename == 'mark':
        out = os.path.join('.', 'data', 'corpus', 'cn', 'mark.csv')
    elif filename == 'lexicon':
        out = os.path.join('.', 'data', 'corpus', 'cn_lexicon', 'lexicon.txt')
    elif filename == 'neural_cand':
        out = os.path.join('.', 'data', 'corpus', 'cn_lexicon', 'expand', 'neural_cand.txt')
    elif filename == 'log':
        out = os.path.join('.', 'log', 'logs.log')
    elif filename == 'anew':
        out = os.path.join('.', 'data', 'corpus', 'anew_seed.txt')
    elif filename == 'normalized_lexicon':
        out = os.path.join('.', 'data', 'corpus', 'cn_lexicon', 'normalized_lexicon.txt')
    elif filename == 'normalized_mark':
        out = os.path.join('.', 'data', 'corpus', 'cn', 'normalized_mark.csv')
    elif filename == 'normalized_onezero_lexicon':
        out = os.path.join('.', 'data', 'corpus', 'cn_lexicon', 'normalized_onezero_lexicon.txt')
    elif filename == 'normalized_onezero_mark':
        out = os.path.join('.', 'data', 'corpus', 'cn', 'normalized_onezero_mark.csv')
    elif filename == 'test_doc2vec':
        if os_name == 'posix':  # ubuntu
            out = os.path.join('/', 'home', 'hs', 'Data', 'test_doc2vec')
        elif os_name == 'nt':  # windows
            out = os.path.join('D:\\', 'chinese_word2vec', 'test_doc2vec')
    elif filename == 'test_doc2vec_model':
        if os_name == 'posix':
            out = os.path.join('/', 'home', 'hs', 'Data', 'test_doc2vec', 'imdb.d2v')
        elif os_name == 'nt':
            out = os.path.join('D:\\', 'chinese_word2vec', 'test_doc2vec', 'docvecs', 'imdb.d2v')
    elif filename == 'cn_word2vec':
        posix = os.path.join('/', 'home', 'hs', 'Data', 'test_doc2vec', 'cn_word2vec', 'wiki.zh.fan.vector')
        nt = os.path.join('D:\\', 'chinese_word2vec', 'wiki.zh.fan.vector')
        if os_name == 'posix':
            out = posix
        elif os_name == 'nt':
            out = nt
    elif filename == 'words_in_wordvec':
        out = os.path.join('.', 'data', 'tmp', 'words_in_wordvec.p')
    elif filename == 'wordvecs_CVAT':
        out = os.path.join('.', 'data', 'tmp', 'wordvecs_CVAT.w2v')
    elif filename == 'docvecs_CVAT':
        out = os.path.join('.', 'data', 'tmp', 'docvecs_CVAT.d2v')
    else:
        raise Exception('Wrong filename')
    return out


from gensim.models.word2vec import Word2Vec
import random
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec


# Build word vector for training set by using the average value of all word vectors in the tweet, then scale
def buill_word_vector(text, model):
    size = 400
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += model[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


def build_doc_vector(corpus, model):
    size = 50
    vecs = [model.docvecs['SENT_%s' % id].reshape((1, size)) for (id, _) in enumerate(corpus)]
    return np.concatenate(vecs)


def train_wordvecs(Sentence):
    model = Word2Vec(size=50, min_count=2)
    model.build_vocab(Sentence.toarray())
    for epoch in range(10):
        print('epoch: %s' % epoch)
        model.train(Sentence.rand())
    model.save(get_file_path('wordvecs_CVAT'))
    print('Training model complete, saved successful.')


def train_docvecs(Sentences):
    model = Doc2Vec(min_count=2, window=10, size=50, sample=1e-5, negative=5, workers=7)
    model.build_vocab(Sentences.to_array())
    for epoch in range(100):
        print('epoch: %s' % epoch)
        model.train(Sentences.sentences_rand())
    model.save(get_file_path('docvecs_CVAT'))
    print('Training model complete, saved successful.')


class Sentence(object):
    def __init__(self, file_dir):
        self.file_dir = file_dir
        self.corpus = load_corpus(file_dir)

    def toarray(self):
        return self.corpus

    def rand(self):
        random.shuffle(self.corpus)
        return self.corpus


class TaggedLineSentence(object):
    def __init__(self, corpus):
        self.sentences = corpus
        self.tagged_sentences = []

    def __iter__(self):
        for (id, sentence) in enumerate(self.sentences):
            yield TaggedDocument(sentence, tags=['SENT_%s' % str(id)])

    def to_array(self):
        for (id, sentence) in enumerate(self.sentences):
            self.tagged_sentences.append(
                TaggedDocument(words=sentence, tags=['SENT_%s' % str(id)]))
        return self.tagged_sentences

    def sentences_rand(self):
        random.shuffle(self.tagged_sentences)
        return self.tagged_sentences


def gold_valence_arousal(corpus, mark):
    valence, arousal = [], []
    for (i, _) in enumerate(corpus):
        try:
            ind = [item[0] for item in mark].index(i + 1)
        except ValueError:
            raise Exception('File not found. NO. %i' % (i + 1))
        valence.append(mark[ind][1])
        arousal.append(mark[ind][2])
    return valence, arousal


# word_vecs is the model of word2vec
def build_embedding_matrix(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs.vocab.keys())
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size, k))
    for i, word in enumerate(word_vecs.vocab.keys()):
        W[i] = word_vecs[word]
        word_idx_map[word] = i  # dict
    return W, word_idx_map


# maxlen is the fixed length to align sentence, padding zero if the number of word is less than maxlen,
# and cut off if more than maxlen
def build_sentence_matrix(model, sententces, maxlen=200):
    size = 50  # dimension
    sentences_matrix = []
    for text in sententces:
        text_matrix = []
        for word in text:
            try:
                text_matrix.append(model[word].reshape((1, size)))
            except KeyError:
                continue
        text_matrix = np.concatenate(text_matrix)
        len_text_matrix = text_matrix.shape[0]
        # print(len_text_matrix)
        if len_text_matrix > maxlen:
            text_matrix = text_matrix[: maxlen]
        if len_text_matrix < maxlen:
            # print(text_matrix)
            # text_matrix = np.lib.pad(text_matrix, ((0, maxlen-len_text_matrix), (0, 0)), mode = 'constant', constant_values = 0)
            # print(text_matrix.shape)
            text_matrix = np.concatenate((text_matrix, np.zeros((maxlen - len_text_matrix, size))), axis=0)
        sentences_matrix.append(text_matrix)
    return np.array(sentences_matrix)


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def log_performance(MSE, MAE, Pearson_r, R2, Spearman_r, sqrt_MSE):
    # create a file handler
    handler = logging.FileHandler(get_file_path('log'))
    handler.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(handler)
    logger.info('MSE: %s, MAE: %s, Pearson_r: %s, R2: %s, Spearman_r: %s, sqrt_MSE: %s', MSE, MAE, Pearson_r, R2,
                Spearman_r, sqrt_MSE)
    logger.removeHandler(handler)  # remove the Handler after you finish your job


def log_state(msg):
    # create a file handler
    handler = logging.FileHandler(get_file_path('log'))
    handler.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(handler)
    logger.info(msg)
    logger.removeHandler(handler)


def evaluate(true, pred, msg):
    true, pred = np.array(true), np.array(pred)
    MAE = mean_absolute_error(np.array(true), np.array(pred))
    MSE_sqrt = math.sqrt(mean_squared_error(np.array(true), np.array(pred)))
    MSE = mean_squared_error(np.array(true), np.array(pred))
    R2 = r2_score(np.array(true), np.array(pred))
    Pearson_r = pearsonr(np.array(true), np.array(pred))
    Spearman_r = spearmanr(np.array(true), np.array(pred))
    log_state(msg)
    log_performance(MSE, MAE, Pearson_r, R2, Spearman_r, MSE_sqrt)
    return None


print('Start from here..............')

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

print(Y_test)
print(predict)
evaluate(Y_test, predict, 'Result of CNN')
