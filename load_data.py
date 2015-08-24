# coding: utf-8
# !/usr/bin/python

import codecs
import re
import os
import csv
import pickle

import gensim
from gensim.models import Doc2Vec


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


def load_extend_anew(D=False):
    print('Loading extend_anew lexicon')
    data_dir = './data/corpus/'
    with open(os.path.join(data_dir, "extend_anew.csv"), 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        words, arousal, valence, dominance = [], [], [], []
        for line in reader:
            if reader.line_num == 1:
                continue
            words.append(line[1])
            arousal.append(float(line[5]))
            valence.append(float(line[2]))
            if D == True:
                dominance.append(float(line[8]))
    print('Loading extend_anew lexicon complete')
    if D == True:
        return words, valence, arousal, dominance
    else:
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
    elif arg == 'google_news':
        model = gensim.models.Word2Vec.load_word2vec_format(get_file_path('google_news'), binary=True)
    elif arg == 'vader':
        model = gensim.models.Word2Vec.load('./data/vader_wordvecs.w2v')
    else:
        raise Exception('Wrong Argument.')
    print('Load Model Complete.')
    return model


def load_vader(name):
    def load_text(filename):
        with open(filename, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            texts, ratings = [], []
            for line in reader:
                texts.append(line[2])
                ratings.append(float(line[1]))
        return texts, ratings

    texts, ratings = [], []
    for filename in name:
        text, rating = load_text('./data/corpus/vader/' + filename + '.txt')
        texts.extend(text)
        ratings.extend(rating)
    return texts, ratings


if __name__ == '__main__':
    from file_name import get_file_path

    words = load_corpus(get_file_path('cn_corpus'))
    print(words[719])
    # for i, w in enumerate(words):
    #     print(i,w)
