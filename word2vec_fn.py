__author__ = 'hs'
import numpy as np
from gensim.models.word2vec import Word2Vec
from file_name import get_file_path
from gensim import utils
from load_data import load_corpus
import random
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec


# Build word vector for training set by using the average value of all word vectors in the tweet, then scale
def buill_word_vector(text, model, size = 400):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    if text is not list:
        text = text.split()
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


def train_wordvecs(Sentence, save_path=None):
    model = Word2Vec(size=50, min_count=1)
    model.build_vocab(Sentence.toarray())
    for epoch in range(10):
        print('epoch: %s' % epoch)
        model.train(Sentence.rand())
    if save_path is None:
        model.save(get_file_path('wordvecs_CVAT'))
    else:
        model.save(save_path)
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
def build_embedding_matrix(word_vecs, vocab, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    union = (set(word_vecs.vocab.keys()) & set(vocab))
    vocab_size = len(union)
    print(vocab_size)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size + 1, k))
    W[0] = np.zeros(k, dtype=np.float32)
    for i, word in enumerate(union, start=1):
        print(word, i)
        W[i] = word_vecs[word]
        word_idx_map[word] = i  # dict
    return W, word_idx_map


# abandon for it may lead to memory error as its big size
# maxlen is the fixed length to align sentence, padding zero if the number of word is less than maxlen,
# and cut off if more than maxlen
def build_sentence_matrix(model, sententces, maxlen=200, dim=50):
    size = dim  # dimension
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


def get_idx_from_sent(sent, word_idx_map, max_l=200, kernel_size=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = kernel_size - 1
    for i in range(pad):
        x.append(0)
    if type(sent) is not list:
        words = sent.split()
    else:
        words = sent
    for num, word in enumerate(words, 1):
        if word in word_idx_map:
            x.append(word_idx_map[word])
        if num > max_l:
            break
    while len(x) < max_l + 2 * pad:
        x.append(0)
    return x


def make_idx_data(sentences, word_idx_map, max_len=200, kernel_size=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    idx_data = []
    for sent in sentences:
        idx_sent = get_idx_from_sent(sent, word_idx_map, max_len, kernel_size)
        idx_data.append(idx_sent)
    idx_data = np.array(idx_data, dtype=np.int)
    return idx_data
