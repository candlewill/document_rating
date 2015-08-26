__author__ = 'NLP-PC'
from load_data import load_vader
import numpy as np
from collections import defaultdict
from imdb_processing import clean_str
from save_data import dump_picle
from load_data import load_pickle, load_embeddings
from word2vec_fn import build_embedding_matrix
from word2vec_fn import make_idx_data
from affective_score_vader import screen_data
from file_name import get_file_path
from load_data import load_anew

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
############################################## all ###############################################
corpus, ratings = load_vader(['tweets', 'movie_reviews', 'product_reviews', 'news_articles'])
# corpus, ratings = load_vader(['movie_reviews'])
# # name: tweets, movie_reviews, product_reviews, news_articles
lexicon_name = get_file_path('anew')
words, valences, _ = load_anew(lexicon_name)
corpus, ratings = screen_data(corpus, ratings, words)
ratings = np.array(ratings) + np.ones(len(ratings), dtype=float) * 5
print(len(corpus), len(corpus))

vocab = get_vocab(corpus)
# dump_picle(vocab, './data/corpus/vader/vocab_moview_reviews.p')
# vocab = load_pickle('./data/corpus/vader/vocab_moview_reviews.p')
print(len(vocab))
W, word_idx_map = build_embedding_matrix(load_embeddings('google_news'), vocab, k=300)
# dump_picle(word_idx_map, './data/corpus/vader/word_idx_map_movie_reviews.p')
# print('dump word_idx_map successful')
dump_picle(W, './data/corpus/vader/embedding_matrix_all.p')
print('dump embedding matrix file OK')
# word_idx_map = load_pickle('./data/corpus/vader/word_idx_map_movie_reviews.p')
idx_data = make_idx_data(corpus, word_idx_map, max_len=200, kernel_size=5)
dump_picle([idx_data, ratings], './data/corpus/vader/vader_processed_data_all.p')
print(idx_data[0])
print(ratings[0])

############################################## tweets ###############################################
# corpus, ratings = load_vader(['tweets', 'movie_reviews', 'product_reviews', 'news_articles'])
corpus, ratings = load_vader(['tweets'])
# # name: tweets, movie_reviews, product_reviews, news_articles
lexicon_name = get_file_path('anew')
words, valences, _ = load_anew(lexicon_name)
corpus, ratings = screen_data(corpus, ratings, words)
ratings = np.array(ratings) + np.ones(len(ratings), dtype=float) * 5
print(len(corpus), len(corpus))

vocab = get_vocab(corpus)
# dump_picle(vocab, './data/corpus/vader/vocab_moview_reviews.p')
# vocab = load_pickle('./data/corpus/vader/vocab_moview_reviews.p')
print(len(vocab))
W, word_idx_map = build_embedding_matrix(load_embeddings('google_news'), vocab, k=300)
# dump_picle(word_idx_map, './data/corpus/vader/word_idx_map_movie_reviews.p')
# print('dump word_idx_map successful')
dump_picle(W, './data/corpus/vader/embedding_matrix_tweets.p')
print('dump embedding matrix file OK')
# word_idx_map = load_pickle('./data/corpus/vader/word_idx_map_movie_reviews.p')
idx_data = make_idx_data(corpus, word_idx_map, max_len=200, kernel_size=5)
dump_picle([idx_data, ratings], './data/corpus/vader/vader_processed_data_tweets.p')
print(idx_data[0])
print(ratings[0])

############################################## movie_reviews ###############################################
# corpus, ratings = load_vader(['tweets', 'movie_reviews', 'product_reviews', 'news_articles'])
corpus, ratings = load_vader(['movie_reviews'])
# # name: tweets, movie_reviews, product_reviews, news_articles
lexicon_name = get_file_path('anew')
words, valences, _ = load_anew(lexicon_name)
corpus, ratings = screen_data(corpus, ratings, words)
ratings = np.array(ratings) + np.ones(len(ratings), dtype=float) * 5
print(len(corpus), len(corpus))

vocab = get_vocab(corpus)
# dump_picle(vocab, './data/corpus/vader/vocab_moview_reviews.p')
# vocab = load_pickle('./data/corpus/vader/vocab_moview_reviews.p')
print(len(vocab))
W, word_idx_map = build_embedding_matrix(load_embeddings('google_news'), vocab, k=300)
# dump_picle(word_idx_map, './data/corpus/vader/word_idx_map_movie_reviews.p')
# print('dump word_idx_map successful')
dump_picle(W, './data/corpus/vader/embedding_matrix_movie_reviews.p')
print('dump embedding matrix file OK')
# word_idx_map = load_pickle('./data/corpus/vader/word_idx_map_movie_reviews.p')
idx_data = make_idx_data(corpus, word_idx_map, max_len=200, kernel_size=5)
dump_picle([idx_data, ratings], './data/corpus/vader/vader_processed_data_movie_reviews.p')
print(idx_data[0])
print(ratings[0])

############################################## product_reviews ###############################################
# corpus, ratings = load_vader(['tweets', 'movie_reviews', 'product_reviews', 'news_articles'])
corpus, ratings = load_vader(['product_reviews'])
# # name: tweets, movie_reviews, product_reviews, news_articles
lexicon_name = get_file_path('anew')
words, valences, _ = load_anew(lexicon_name)
corpus, ratings = screen_data(corpus, ratings, words)
ratings = np.array(ratings) + np.ones(len(ratings), dtype=float) * 5
print(len(corpus), len(corpus))

vocab = get_vocab(corpus)
# dump_picle(vocab, './data/corpus/vader/vocab_moview_reviews.p')
# vocab = load_pickle('./data/corpus/vader/vocab_moview_reviews.p')
print(len(vocab))
W, word_idx_map = build_embedding_matrix(load_embeddings('google_news'), vocab, k=300)
# dump_picle(word_idx_map, './data/corpus/vader/word_idx_map_movie_reviews.p')
# print('dump word_idx_map successful')
dump_picle(W, './data/corpus/vader/embedding_matrix_product_reviews.p')
print('dump embedding matrix file OK')
# word_idx_map = load_pickle('./data/corpus/vader/word_idx_map_movie_reviews.p')
idx_data = make_idx_data(corpus, word_idx_map, max_len=200, kernel_size=5)
dump_picle([idx_data, ratings], './data/corpus/vader/vader_processed_data_product_reviews.p')
print(idx_data[0])
print(ratings[0])

############################################## news_articles ###############################################
# corpus, ratings = load_vader(['tweets', 'movie_reviews', 'product_reviews', 'news_articles'])
corpus, ratings = load_vader(['news_articles'])
# # name: tweets, movie_reviews, product_reviews, news_articles
lexicon_name = get_file_path('anew')
words, valences, _ = load_anew(lexicon_name)
corpus, ratings = screen_data(corpus, ratings, words)
ratings = np.array(ratings) + np.ones(len(ratings), dtype=float) * 5
print(len(corpus), len(corpus))

vocab = get_vocab(corpus)
# dump_picle(vocab, './data/corpus/vader/vocab_moview_reviews.p')
# vocab = load_pickle('./data/corpus/vader/vocab_moview_reviews.p')
print(len(vocab))
W, word_idx_map = build_embedding_matrix(load_embeddings('google_news'), vocab, k=300)
# dump_picle(word_idx_map, './data/corpus/vader/word_idx_map_movie_reviews.p')
# print('dump word_idx_map successful')
dump_picle(W, './data/corpus/vader/embedding_matrix_news_articles.p')
print('dump embedding matrix file OK')
# word_idx_map = load_pickle('./data/corpus/vader/word_idx_map_movie_reviews.p')
idx_data = make_idx_data(corpus, word_idx_map, max_len=200, kernel_size=5)
dump_picle([idx_data, ratings], './data/corpus/vader/vader_processed_data_news_articles.p')
print(idx_data[0])
print(ratings[0])

##########################################################################################################
print('OK')