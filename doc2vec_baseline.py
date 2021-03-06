__author__ = 'hs'
from gensim import utils
# from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

import numpy
from sklearn.linear_model import LogisticRegression
from file_name import get_file_path
from gensim.models.doc2vec import TaggedDocument as LabeledSentence
import random

class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources
        self.sentences = []

        flipped = {}
        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(words=utils.to_unicode(line).split(), tags=[prefix + '_%s' % str(item_no)])

    def to_array(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(
                        LabeledSentence(words=utils.to_unicode(line).split(), tags=[prefix + '_%s' % str(item_no)]))
        return self.sentences

    def sentences_rand(self):
        # out = numpy.random.permutation(self.sentences)
        # return out
        random.shuffle(self.sentences)
        return self.sentences


sources = {'test-neg.txt': 'TEST_NEG', 'test-pos.txt': 'TEST_POS', 'train-neg.txt': 'TRAIN_NEG',
           'train-pos.txt': 'TRAIN_POS', 'train-unsup.txt': 'TRAIN_UNS'}
##########################################################
dir = get_file_path('test_doc2vec')
keys = list(sources.keys())
for old_key in keys:
    sources[dir + '/' + old_key] = sources.pop(old_key)
##############################################################
############################# model training #################
'''
sentences = LabeledLineSentence(sources)
model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7)
model.build_vocab(sentences.to_array())
for epoch in range(10):
    print('epoch: %s' % epoch)
    model.train(sentences.sentences_rand())

print(model.most_similar('good'))
# print(model['TRAIN_NEG_0'])

model.save(get_file_path('test_doc2vec_model'))
print('model been saved!')
'''
############################# model training #################

model = Doc2Vec.load(get_file_path('test_doc2vec_model'))
print(model.most_similar('good'))
print(model.docvecs[1])
print(model.docvecs['TRAIN_NEG_1'])

train_arrays = numpy.zeros((25000, 100))
train_labels = numpy.zeros(25000)

for i in range(12500):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_pos]
    train_arrays[12500 + i] = model.docvecs[prefix_train_neg]
    train_labels[i] = 1
    train_labels[12500 + i] = 0

print(train_arrays)
print(train_labels)

test_arrays = numpy.zeros((25000, 100))
test_labels = numpy.zeros(25000)

for i in range(12500):
    prefix_test_pos = 'TEST_POS_' + str(i)
    prefix_test_neg = 'TEST_NEG_' + str(i)
    test_arrays[i] = model.docvecs[prefix_test_pos]
    test_arrays[12500 + i] = model.docvecs[prefix_test_neg]
    test_labels[i] = 1
    test_labels[12500 + i] = 0

classifier = LogisticRegression()
classifier.fit(train_arrays, train_labels)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

print(classifier.score(test_arrays, test_labels))
