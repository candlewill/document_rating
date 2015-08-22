# -*- coding: utf-8 -*-
from load_data import load_corpus
from load_data import load_mark
from load_data import combine_lexicon
from load_data import load_lexicon
from file_name import get_file_path
import codecs
import re
import os
import sys
import logging
import math
import numpy as np
from evaluate import evaluate
import scipy as sp
from log_manager import log_state
import nltk
from load_data import load_anew, load_extend_anew
from save_data import dump_picle
from load_data import load_pickle

# from multiprocessing import Pool
# from multiprocessing.dummy import Pool as ThreadPool


def evaluate_mean(corpus, lexicon, mark):
    valence_pred = []
    valence_true = []

    def VA_mean(text):
        sum_valence = 0
        count = 0
        word_list = text.split()
        for word in word_list:
            for line in lexicon:
                if word == line:
                    count = count + 1
                    sum_valence = sum_valence + lexicon[line]
        return 5 if count == 0 else sum_valence / count

    for i, text in enumerate(corpus):
        V = VA_mean(text)
        valence_pred.append(V)
        valence_true.append(mark[i])
    print(valence_true[:200])
    print(valence_pred[:200])
    evaluate(valence_true, valence_pred, 'valence')


idfs = load_pickle('./data/vocab_idf.p')
def tfidf(t, d, D):
    d = d.split()
    tf = float(d.count(t)) / sum(d.count(w) for w in set(d))
    # idf = sp.log(float(len(D)) / (len([doc.split() for doc in D if t in doc.split()])))
    return tf * idfs[t]


def tf(t, d):
    d = d.split()
    tf = float(d.count(t)) / float(len(d))
    return tf


def evaluate_tfidf_geo(corpus, lexicon, mark):
    valence_pred = []
    valence_true = []

    def VA_mean(text):
        sum_valence = 1.
        count = 0
        word_list = text.split()
        for word in word_list:
            for line in lexicon:
                word_tfidf = tfidf(word, corpus[i], corpus)
                if word == line:
                    count = count + word_tfidf
                    sum_valence = sum_valence * (lexicon[word] ** word_tfidf)
        return 5 if count == 0 else sum_valence ** (1. / count)

    for i, text in enumerate(corpus):
        V = VA_mean(text)
        valence_pred.append(V)
        valence_true.append(mark[i])
    print(valence_true[:200])
    print(valence_pred[:200])
    evaluate(valence_true, valence_pred, 'valence')


def evaluate_tf_geo(corpus, lexicon, mark):
    valence_pred = []
    valence_true = []

    def VA_mean(text):
        sum_valence = 1
        count = 0
        word_list = text.split()
        for word in word_list:
            for line in lexicon:
                if word == line:
                    word_tf = tf(word, corpus[i])
                    count = count + word_tf
                    sum_valence = sum_valence * (lexicon[word] ** word_tf)
        return 5 if count == 0 else sum_valence ** (1. / count)

    for i, text in enumerate(corpus):
        V = VA_mean(text)
        valence_pred.append(V)
        valence_true.append(mark[i])
    print(valence_true[:200])
    print(valence_pred[:200])
    evaluate(valence_true, valence_pred, 'valence')


def evaluate_geo(corpus, lexicon, mark):
    valence_pred = []
    valence_true = []

    def VA_mean(text):
        sum_valence = 1
        count = 0
        word_list = text.split()
        for word in word_list:
            for line in lexicon:
                if word == line:
                    count = count + 1
                    sum_valence = sum_valence * lexicon[line]
        return 5 if count == 0 else sum_valence ** (1. / count)

    for i, text in enumerate(corpus):
        V = VA_mean(text)
        valence_pred.append(V)
        valence_true.append(mark[i])
    print(valence_true[:200])
    print(valence_pred[:200])
    evaluate(valence_true, valence_pred, 'valence')


def evaluate_tf_mean(corpus, lexicon, mark):
    valence_pred = []
    valence_true = []

    def VA_mean(text):
        sum_valence = 0
        count = 0
        word_list = text.split()
        for word in word_list:
            for line in lexicon:
                if word == line:
                    word_tf = tf(word, corpus[i])
                    count = count + word_tf
                    sum_valence = sum_valence + word_tf * lexicon[word]
        return 5 if count == 0 else sum_valence / count

    for i, text in enumerate(corpus):
        V = VA_mean(text)
        valence_pred.append(V)
        valence_true.append(mark[i])
    print(valence_true[:200])
    print(valence_pred[:200])
    evaluate(valence_true, valence_pred, 'valence')


def evaluate_tfidf_mean(corpus, lexicon, mark):
    valence_pred = []
    valence_true = []

    def VA_mean(text):
        sum_valence = 0
        count = 0
        word_list = text.split()
        for word in word_list:
            for line in lexicon:
                if word == line:
                    word_tfidf = tfidf(word, corpus[i], corpus)
                    count = count + word_tfidf
                    sum_valence = sum_valence + word_tfidf * lexicon[word]
        return 5 if count == 0 else sum_valence / count

    for i, text in enumerate(corpus):
        V = VA_mean(text)
        valence_pred.append(V)
        valence_true.append(mark[i])
    print(valence_true[:200])
    print(valence_pred[:200])
    evaluate(valence_true, valence_pred, 'valence')


from preprocess_imdb import clean_str


def process(corpus):
    return [clean_str(sent) for sent in corpus]


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))
    from load_data import load_vader
    # corpus, ratings = load_vader(['tweets', 'movie_reviews', 'product_reviews', 'news_articles'])
    corpus, ratings = load_vader(['movie_reviews'])
    corpus = process(corpus)
    print(corpus[:2])
    from collections import defaultdict
    # idf = sp.log(float(len(D)) / (len([doc.split() for doc in D if t in doc.split()])))

    # vocab = load_pickle('./data/corpus/vader/vocab_moview_reviews.p')
    # idf=defaultdict(float)
    # length = len(vocab)
    # for i, word in enumerate(vocab):
    #     idf[word] = sp.log(float(len(corpus)) / (len([doc for doc in corpus if word in doc.split()])))
    #     if i%50 == 0:
    #         print('%i/%i:'%(i, length),word, idf[word])
    #
    # dump_picle(idf, './data/vocab_idf.p')
    # exit()
    lexicon_name = get_file_path('anew')
    logger.info(r"loading lexicon form : " + lexicon_name)



    # words, valences, _ = load_anew(lexicon_name)
    words, valences, _ = load_extend_anew()
    mark = np.array(ratings) + np.ones(len(ratings), dtype=float) * 5
    lexicon = dict()
    for i, word in enumerate(words):
        lexicon[word] = valences[i]

    log_state('mean')
    evaluate_mean(corpus, lexicon, mark)

    log_state('tf_mean')
    evaluate_tf_mean(corpus, lexicon, mark)

    log_state('tfidf_mean')
    evaluate_tfidf_mean(corpus, lexicon, mark)

    log_state('geo')
    evaluate_geo(corpus, lexicon, mark)

    log_state('tfidf_geo')
    evaluate_tfidf_geo(corpus, lexicon, mark)

    log_state('tf_geo')
    evaluate_tf_geo(corpus, lexicon, mark)