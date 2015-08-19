__author__ = 'NLP-PC'
from load_data import load_vader
import numpy as np
from collections import defaultdict
from imdb_processing import clean_str
from save_data import dump_picle


def get_vocab(corpus):
    vocab = defaultdict(float)
    for sent in corpus:
        for word in clean_str(sent).split():
            vocab[word] += 1
    print(len(vocab))
    return vocab


vec_dim = 300
corpus, ratings = load_vader(['tweets', 'movie_reviews', 'product_reviews', 'news_articles'])
# name: tweets, movie_reviews, product_reviews, news_articles
print(len(corpus), len(corpus))
print(corpus[:2])
vocab = get_vocab(corpus)
dump_picle(vocab, './data/corpus/vader/vocab_all.p')
