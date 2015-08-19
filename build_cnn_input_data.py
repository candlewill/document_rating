__author__ = 'NLP-PC'
from load_data import load_vader
import numpy as np
from collections import defaultdict
from imdb_processing import clean_str
from save_data import dump_picle
from load_data import load_pickle, load_embeddings
from word2vec_fn import build_embedding_matrix
from word2vec_fn import make_idx_data

def get_vocab(corpus):
    vocab = defaultdict(float)
    for sent in corpus:
        for word in clean_str(sent).split():
            vocab[word] += 1
    print(len(vocab))
    return vocab


def process(corpus):
    return [clean_str(sent) for sent in corpus]


vec_dim = 300
corpus, ratings = load_vader(['tweets', 'movie_reviews', 'product_reviews', 'news_articles'])
# # name: tweets, movie_reviews, product_reviews, news_articles
# print(len(corpus), len(corpus))
# print(corpus[:2])
# vocab = get_vocab(corpus)
# dump_picle(vocab, './data/corpus/vader/vocab_all.p')
# vocab = load_pickle('./data/corpus/vader/vocab_all.p')
# print(len(vocab))
# W, word_idx_map = build_embedding_matrix(load_embeddings('google_news'), vocab, k=300)
# dump_picle(word_idx_map, './data/corpus/vader/word_idx_map_all.p')
# print('dump word_idx_map successful')
# dump_picle(W, './data/corpus/vader/embedding_matrix_all.p')
# print('OK')
print(corpus[2328])
corpus = process(corpus)
print(corpus[2328])
word_idx_map = load_pickle('./data/corpus/vader/word_idx_map_all.p')
idx_data = make_idx_data(corpus, word_idx_map, max_len=200, kernel_size=5)
print(idx_data[2328])

dump_picle([idx_data, ratings], './data/corpus/vader/vader_processed_data.p')
