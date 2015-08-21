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


def tfidf(t, d, D):
    tf = float(d.count(t)) / sum(d.count(w) for w in set(d))
    idf = sp.log(float(len(D)) / (len([doc for doc in D if t in doc])))
    return tf * idf


def tf(t, d):
    tf = float(d.count(t)) / float(len(d))
    return tf


def evaluate_tfidf_geo(corpus, lexicon, mark):
    valence_pred = []
    valence_true = []
    arousal_pred = []
    arousal_true = []

    num = len(corpus)
    for (i, text) in enumerate(corpus):
        sum_valence = 1.
        sum_arousal = 1.
        count = 0.

        for word in text:
            for l in lexicon:
                if word == l[0]:
                    word_tfidf = tfidf(word, corpus[i], corpus)
                    # logger.info("tfidf of word %s is %f" % (word, word_tfidf))
                    if l[1] > 9:
                        l[1] = 9
                    if l[1] < 1:
                        l[1] = 1
                    if l[2] > 9:
                        l[2] = 9
                    if l[2] < 1:
                        l[2] = 1

                    count = count + word_tfidf
                    sum_valence = sum_valence * (l[1] ** word_tfidf)
                    sum_arousal = sum_arousal * (l[2] ** word_tfidf)

        if count == 0.:
            valence_pred.append(5.)
            arousal_pred.append(5.)
        else:
            # logger.info("%f %f" % (sum_valence ** (1. / count), sum_arousal ** (1. / count)))
            valence_pred.append(sum_valence ** (1. / count))
            arousal_pred.append(sum_arousal ** (1. / count))

        for item in mark:
            if (i + 1) == item[0]:
                valence_true.append(item[1])
                arousal_true.append(item[2])
                break

        if (i + 1) % 10 == 0:
            logger.info("evaluate for text : %i/%i..." % ((i + 1), num))

    evaluate(valence_true, valence_pred, 'valence')
    evaluate(arousal_true, arousal_pred, 'arousal')


def evaluate_tf_geo(corpus, lexicon, mark):
    valence_pred = []
    valence_true = []
    arousal_pred = []
    arousal_true = []

    num = len(corpus)
    for (i, text) in enumerate(corpus):
        sum_valence = 1.
        sum_arousal = 1.
        count = 0.

        for word in text:
            for l in lexicon:
                if word == l[0]:
                    word_tf = tf(word, corpus[i])
                    # logger.info("tfidf of word %s is %f" % (word, word_tfidf))
                    if l[1] > 9:
                        l[1] = 9
                    if l[1] < 1:
                        l[1] = 1
                    if l[2] > 9:
                        l[2] = 9
                    if l[2] < 1:
                        l[2] = 1

                    count = count + word_tf
                    sum_valence = sum_valence * (l[1] ** word_tf)
                    sum_arousal = sum_arousal * (l[2] ** word_tf)

        if count == 0.:
            valence_pred.append(5.)
            arousal_pred.append(5.)
        else:
            # logger.info("%f %f" % (sum_valence ** (1. / count), sum_arousal ** (1. / count)))
            valence_pred.append(sum_valence ** (1. / count))
            arousal_pred.append(sum_arousal ** (1. / count))

        for item in mark:
            if (i + 1) == item[0]:
                valence_true.append(item[1])
                arousal_true.append(item[2])
                break

        if (i + 1) % 10 == 0:
            logger.info("evaluate for text : %i/%i..." % ((i + 1), num))

    evaluate(valence_true, valence_pred, 'valence')
    evaluate(arousal_true, arousal_pred, 'arousal')


def evaluate_geo(corpus, lexicon, mark):
    valence_pred = []
    valence_true = []
    arousal_pred = []
    arousal_true = []

    num = len(corpus)
    for (i, text) in enumerate(corpus):
        sum_valence = 1.
        sum_arousal = 1.
        count = 0.

        for word in text:
            for l in lexicon:
                if word == l[0]:
                    if l[1] > 9:
                        l[1] = 9
                    if l[1] < 1:
                        l[1] = 1
                    if l[2] > 9:
                        l[2] = 9
                    if l[2] < 1:
                        l[2] = 1

                    count = count + 1
                    sum_valence = sum_valence * l[1]
                    sum_arousal = sum_arousal * l[2]

        if count == 0.:
            valence_pred.append(5.)
            arousal_pred.append(5.)
        else:
            # logger.info("%f %f" % (sum_valence ** (1. / count), sum_arousal ** (1. / count)))
            valence_pred.append(sum_valence ** (1. / count))
            arousal_pred.append(sum_arousal ** (1. / count))

        for item in mark:
            if (i + 1) == item[0]:
                valence_true.append(item[1])
                arousal_true.append(item[2])
                break

        if (i + 1) % 10 == 0:
            logger.info("evaluate for text : %i/%i..." % ((i + 1), num))

    evaluate(valence_true, valence_pred, 'valence')
    evaluate(arousal_true, arousal_pred, 'arousal')


def evaluate_tf_mean(corpus, lexicon, mark):
    valence_pred = []
    valence_true = []
    arousal_pred = []
    arousal_true = []

    num = len(corpus)
    for (i, text) in enumerate(corpus):
        sum_valence = 0.
        sum_arousal = 0.
        count = 0.

        for word in text:
            for l in lexicon:
                if word == l[0]:
                    word_tf = tf(word, corpus[i])
                    # logger.info("tfidf of word %s is %f" % (word, word_tfidf))
                    if l[1] > 9:
                        l[1] = 9
                    if l[1] < 1:
                        l[1] = 1
                    if l[2] > 9:
                        l[2] = 9
                    if l[2] < 1:
                        l[2] = 1

                    count = count + word_tf
                    sum_valence = sum_valence + word_tf * l[1]
                    sum_arousal = sum_arousal + word_tf * l[2]

        if count == 0:
            valence_pred.append(5.)
            arousal_pred.append(5.)
        else:
            valence_pred.append(sum_valence / count)
            arousal_pred.append(sum_arousal / count)

        for item in mark:
            if (i + 1) == item[0]:
                valence_true.append(item[1])
                arousal_true.append(item[2])
                break

        if i % 10 == 0:
            logger.info("evaluate for text : %i/%i..." % (i, num))

    evaluate(valence_true, valence_pred, 'valence')
    evaluate(arousal_true, arousal_pred, 'arousal')


def evaluate_tfidf_mean(corpus, lexicon, mark):
    valence_pred = []
    valence_true = []
    arousal_pred = []
    arousal_true = []

    num = len(corpus)
    for (i, text) in enumerate(corpus):
        sum_valence = 0.
        sum_arousal = 0.
        count = 0.

        for word in text:
            for l in lexicon:
                if word == l[0]:
                    word_tfidf = tfidf(word, corpus[i], corpus)
                    # logger.info("tfidf of word %s is %f" % (word, word_tfidf))
                    if l[1] > 9:
                        l[1] = 9
                    if l[1] < 1:
                        l[1] = 1
                    if l[2] > 9:
                        l[2] = 9
                    if l[2] < 1:
                        l[2] = 1

                    count = count + word_tfidf
                    sum_valence = sum_valence + word_tfidf * l[1]
                    sum_arousal = sum_arousal + word_tfidf * l[2]

        if count == 0:
            valence_pred.append(5.)
            arousal_pred.append(5.)
        else:
            valence_pred.append(sum_valence / count)
            arousal_pred.append(sum_arousal / count)

        for item in mark:
            if (i + 1) == item[0]:
                valence_true.append(item[1])
                arousal_true.append(item[2])
                break

        if i % 10 == 0:
            logger.info("evaluate for text : %i/%i..." % (i, num))

    evaluate(valence_true, valence_pred, 'valence')
    evaluate(arousal_true, arousal_pred, 'arousal')


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

    lexicon_name = get_file_path('anew')
    logger.info(r"loading lexicon form : " + lexicon_name)

    from load_data import load_anew, load_extend_anew

    words, valences, _ = load_anew(lexicon_name)
    # words, valences, _ = load_extend_anew()
    mark = np.array(ratings) + np.ones(len(ratings), dtype=float) * 5
    from collections import defaultdict

    lexicon = dict()
    for i, word in enumerate(words):
        lexicon[word] = valences[i]
    log_state('mean')
    # evaluate_mean(corpus, lexicon, mark)


    log_state('tf_mean')
    evaluate_tf_mean(corpus, lexicon, mark)
    exit()

    log_state('tfidf_mean')
    evaluate_tfidf_mean(corpus, lexicon, mark)

    log_state('geo')
    evaluate_geo(corpus, lexicon, mark)
    log_state('tfidf_geo')
    evaluate_tfidf_geo(corpus, lexicon, mark)
    log_state('tf_geo')
    evaluate_tf_geo(corpus, lexicon, mark)
