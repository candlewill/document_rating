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
from preprocess_imdb import clean_str
from functools import reduce


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


def screen_data(corpus, ratings, words):
    processed_texts, processed_ratings = [], []
    size = len(corpus)
    # sums = 0
    for i, sentence in enumerate(corpus):
        sentence = clean_str(sentence)
        same_words = list(set(sentence.split()).intersection(set(words)))
        nb_same_words = len(same_words)
        if nb_same_words >= 1:
            # max_freq = max([sentence.split().count(w) for w in same_words])
            # if max_freq >= 6:
            #     sums = sums + 1
            #     print('*'*100)
            #     print(sentence, str(same_words),max_freq, sums)
            processed_texts.append(sentence)
            processed_ratings.append(ratings[i])
        if i % 500 == 0:
            print(sentence)
            print('the %i/%i is processing: %s' % (i, size, str(same_words)))
    print('size of corpus is %i' % len(processed_texts))
    # print(sums)
    return processed_texts, processed_ratings


def calculate_ratings(corpus, ratings, lexicon):
    words = lexicon.keys()
    mean_ratings, tf_means, tfidf_means, geos, tf_geos, tfidf_geos = [], [], [], [], [], []
    for sentence, rating in zip(corpus, ratings):
        word_rating_list = []
        word_tf_list = []
        word_tfidf_list = []
        same_words = list(set(sentence.split()).intersection(set(words)))
        for same_word in same_words:
            tf = sentence.split().count(same_word)
            idf = idfs[same_word]
            word_rating_list.append(lexicon[same_word])
            word_tf_list.append(tf)
            word_tfidf_list.append(tf * idf)
        mean_ratings.append(np.average(np.array(word_rating_list)))
        tf_mean = sum(wr * wtf / sum(word_tf_list) for wr, wtf in zip(word_rating_list, word_tf_list))
        tf_means.append(tf_mean)
        tfidf_mean = sum(wr * wtfidf / sum(word_tfidf_list) for wr, wtfidf in zip(word_rating_list, word_tfidf_list))
        tfidf_means.append(tfidf_mean)
        geos.append(reduce(lambda x, y: x * y, word_rating_list, 1) ** (1 / len(word_rating_list)))
        tf_geos.append(
            reduce(lambda x, y: x * y, [wr ** wtf for wr, wtf in zip(word_rating_list, word_tf_list)], 1) ** (
                1 / sum(word_tf_list)))
        tfidf_geos.append(
            reduce(lambda x, y: x * y, [wr ** wtfidf for wr, wtfidf in zip(word_rating_list, word_tfidf_list)], 1) ** (
                1 / sum(word_tfidf_list)))

    return np.array(mean_ratings), np.array(tf_means), np.array(tfidf_means), np.array(geos), np.array(
        tf_geos), np.array(tfidf_geos)


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
    corpus, ratings = load_vader(['news_articles'])
    lexicon_name = get_file_path('anew')
    logger.info(r"loading lexicon form : " + lexicon_name)
    words, valences, _ = load_anew(lexicon_name)
    corpus, ratings = screen_data(corpus, ratings, words)
    ratings = np.array(ratings) + np.ones(len(ratings), dtype=float) * 5
    print(len(corpus))
    # for i in corpus[:100]:
    #     print(i)

    lexicon = dict()
    for i, word in enumerate(words):
        lexicon[word] = valences[i]
    mean_ratings, tf_means, tfidf_means, geos, tf_geos, tfidf_geos = calculate_ratings(corpus, ratings, lexicon)
    dump_picle([mean_ratings, tf_means, tfidf_means, geos, tf_geos, tfidf_geos, ratings], './data/vader_out.p')
    exit()
    from collections import defaultdict
    # idf = sp.log(float(len(D)) / (len([doc.split() for doc in D if t in doc.split()])))

    # vocab = load_pickle('./data/corpus/vader/vocab_moview_reviews.p')

    # length = len(vocab)
    ##################################### IDF ####################################################
    # idf=dict()
    # for i, word in enumerate(words):
    #     denominator = sum(1 for doc in corpus if word in doc.split())
    #     if denominator != 0:
    #         idf[word] = sp.log(float(len(corpus)) / denominator)
    #         if i%50 == 0:
    #             print('%i/%i:'%(i, len(words)),word, idf[word])
    #
    # dump_picle(idf, './data/vocab_idf.p')
    # exit()
    ################################################################################################




    # words, valences, _ = load_extend_anew()
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
