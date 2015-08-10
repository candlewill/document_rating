__author__ = 'hs'
import numpy as np
from gensim.models.word2vec import Word2Vec
from file_name import get_file_path
from gensim import utils
from load_data import load_corpus
import random


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


def train_wordvecs(Sentence):
    model = Word2Vec(size=50, min_count=2)
    model.build_vocab(Sentence.toarray())
    for epoch in range(10):
        model.train(Sentence.rand())
    model.save(get_file_path('wordvecs_CVAT'))
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


#
# class LabeledLineSentence(object):
#     def __init__(self, sources):
#         self.sources = sources
#
#         flipped = {}
#
#         # make sure that keys are unique
#         for key, value in sources.items():
#             if value not in flipped:
#                 flipped[value] = [key]
#             else:
#                 raise Exception('Non-unique prefix encountered')
#
#     def __iter__(self):
#         for source, prefix in self.sources.items():
#             with utils.smart_open(source) as fin:
#                 for item_no, line in enumerate(fin):
#                     yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])
#
#     def to_array(self):
#         self.sentences = []
#         for source, prefix in self.sources.items():
#             with utils.smart_open(source) as fin:
#                 for item_no, line in enumerate(fin):
#                     self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
#         return self.sentences
#
#     def sentences_perm(self):
#         return numpy.random.permutation(self.sentences)
#

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
