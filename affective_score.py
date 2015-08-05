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


# from multiprocessing import Pool
# from multiprocessing.dummy import Pool as ThreadPool


def evaluate_mean(corpus, lexicon, mark):
    valence_pred = []
    valence_true = []
    arousal_pred = []
    arousal_true = []

    def VA_mean(text):
        sum_valence = 0
        sum_arousal = 0
        count = 0
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
                    sum_valence = sum_valence + l[1]
                    sum_arousal = sum_arousal + l[2]
        return [5., 5.] if count == 0 else [sum_valence / count, sum_arousal / count]

    num = len(corpus)
    for (i, text) in enumerate(corpus):
        V, A = VA_mean(text)
        valence_pred.append(V)
        arousal_pred.append(A)
        try:
            ind = [item[0] for item in mark].index(i + 1)
        except ValueError:
            raise Exception('File not found. NO. %i' % (i + 1))

        valence_true.append(mark[ind][1])
        arousal_true.append(mark[ind][2])

        # for item in mark:
        #     if (i + 1) == item[0]:
        #         valence_true.append(item[1])
        #         arousal_true.append(item[2])
        #         break
        #     else:
        #         raise Exception('File not found. NO. %i' % (i + 1))

        if i % 10 == 0:
            logger.info("evaluate for text : %i/%i..." % (i, num))

    evaluate(valence_true, valence_pred, 'valence')
    evaluate(arousal_true, arousal_pred, 'arousal')


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


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    corpus_name = get_file_path('cn_corpus')
    logger.info(r"loading corpus from : " + corpus_name)

    lexicon_name = get_file_path('lexicon')
    logger.info(r"loading lexicon form : " + lexicon_name)

    expand_name = get_file_path('neural_cand')
    logger.info(r"loading expand_word from : " + expand_name)

    mark_name = get_file_path('mark')
    logger.info(r"loading mark from : " + mark_name)

    corpus = load_corpus(corpus_name)
    lexicon = load_lexicon(lexicon_name)
    mark = load_mark(mark_name)
    # log_state('use extend lexicon')
    lexicon = combine_lexicon(lexicon_name, expand_name)

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
